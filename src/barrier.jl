"""
    DikinWalk(; r = 0.5)

The Dikin walk (Kannan & Narayanan 2012): each step proposes a Gaussian move shaped by the
Dikin ellipsoid of the log barrier at the current point, `N(z, (r²/k) H(z)⁻¹)` with
`H(z) = Gᵀ S⁻² G` for slacks `S`, and accepts it by a Metropolis filter, so the chain has the
uniform distribution as its stationary distribution, or the density of a [`Target`](@ref),
whose log density joins the Metropolis ratio. The walk is affine-invariant, so it needs no
rounding, and it has a mixing bound of `O(mk)` steps from a warm start for `m` constraints
in `k` dimensions, whatever the shape of the polytope. A step costs `O(mk²)`. On rounded
polytopes it needed 340 to 490 steps per effective sample, about 100 times the
[`BilliardWalk`](@ref); use it as an independent check, or when rounding is not
available.

Kannan, R. & Narayanan, H. (2012). Random walks on polytopes and an affine interior point
method for linear programming. Mathematics of Operations Research 37, 1–20.
<https://doi.org/10.1287/moor.1110.0519>
"""
struct DikinWalk <: PolytopeSampler
    r::Float64
    function DikinWalk(r)
        r > 0 || throw(ArgumentError("r must be positive, got $r"))
        return new(r)
    end
end
DikinWalk(; r::Real = 0.5) = DikinWalk(Float64(r))

"""
    VaidyaWalk(; r = 0.7)

The Vaidya walk (Chen, Dwivedi, Wainwright & Yu 2018): the [`DikinWalk`](@ref) with each
constraint weighted by its leverage score plus `k/m` (Vaidya's volumetric barrier), and
proposals `N(z, (r²/√(km)) H(z)⁻¹)`. Its moves per effective sample grow as `O(m^½ k^1.5)`
instead of `O(mk)`, which helps when there are many more constraints than dimensions; a step
costs about twice a Dikin step.

Chen, Y., Dwivedi, R., Wainwright, M. J. & Yu, B. (2018). Fast MCMC sampling algorithms on
polytopes. Journal of Machine Learning Research 19, 1–86.
<https://jmlr.org/papers/v19/18-158.html>
"""
struct VaidyaWalk <: PolytopeSampler
    r::Float64
    function VaidyaWalk(r)
        r > 0 || throw(ArgumentError("r must be positive, got $r"))
        return new(r)
    end
end
VaidyaWalk(; r::Real = 0.7) = VaidyaWalk(Float64(r))

const BarrierWalk = Union{DikinWalk, VaidyaWalk}

Base.show(io::IO, m::DikinWalk) = print(io, "DikinWalk(", m.r == 0.5 ? "" : "r = $(m.r)", ")")
Base.show(io::IO, m::VaidyaWalk) = print(io, "VaidyaWalk(", m.r == 0.7 ? "" : "r = $(m.r)", ")")

# The state of a barrier walk, with the Cholesky factor of the barrier Hessian at the current
# point, so that a step factors only at the proposal
mutable struct BarrierState{M <: BarrierWalk, D}
    const method::M
    const warmup::Warmup
    const density::D # nothing for the uniform distribution
    logp::Float64 # log density at z
    const z::Vector{Float64}
    const x::Vector{Float64}
    const zprop::Vector{Float64}
    const slack::Vector{Float64}
    const slackprop::Vector{Float64}
    const step::Vector{Float64} # proposed move
    const work::Vector{Float64}
    const B::Matrix{Float64} # S⁻¹ G, rows scaled for the Vaidya weights
    const K::Matrix{Float64} # Bᵀ B, for leverage scores
    const X::Matrix{Float64} # U⁻ᵀ Bᵀ, for leverage scores
    U::Matrix{Float64} # upper Cholesky factor of H at z
    Uprop::Matrix{Float64} # the same at the proposal
    logdet::Float64 # half the log-determinant of H at z
    const scale::Float64 # √D, the proposal scale
    ready::Bool # whether U and logdet belong to z
    proposed::Int
    accepted::Int
    last_accepted::Bool
end

chain_state(method::BarrierWalk, t::Target) = barrier_state(method, t.warmup, chain_density(t))
chain_state(method::BarrierWalk, w::Warmup) = barrier_state(method, w, nothing)

AbstractMCMC.getparams(c::BarrierState) = copy(c.x)
function AbstractMCMC.getstats(c::BarrierState{<:BarrierWalk, Nothing})
    return (; is_accept = c.last_accepted, acceptance_rate = c.accepted / max(c.proposed, 1))
end
function AbstractMCMC.getstats(c::BarrierState)
    return (; lp = c.logp, is_accept = c.last_accepted, acceptance_rate = c.accepted / max(c.proposed, 1))
end

function barrier_state(method::BarrierWalk, w::Warmup, density)
    k = size(w.transform, 2)
    m = length(w.h)
    D = method isa DikinWalk ? method.r^2 / k : method.r^2 / sqrt(k * m)
    return BarrierState(
        method, w, density, 0.0, zeros(k), copy(w.center), zeros(k), copy(w.h), zeros(m), zeros(k), zeros(k),
        zeros(m, k), zeros(k, k), zeros(k, m), zeros(k, k), zeros(k, k), 0.0, sqrt(D), false,
        0, 0, false,
    )
end

# Factor the barrier Hessian at slacks `s` into the upper triangle of `U`; return half its
# log-determinant, or -Inf if the factorization fails
function factor_hessian!(U, c::BarrierState, s)
    G = c.warmup.G
    B = c.B
    m, k = size(B)
    for j in 1:k, i in 1:m
        B[i, j] = G[i, j] / s[i]
    end
    if c.method isa VaidyaWalk
        # Leverage scores σᵢ = bᵢᵀ (BᵀB)⁻¹ bᵢ = ‖U⁻ᵀ bᵢ‖², then rows scaled by √(σᵢ + k/m)
        mul!(c.K, B', B)
        issuccess(cholesky!(Hermitian(c.K, :U); check = false)) || return -Inf
        transpose!(c.X, B)
        ldiv!(UpperTriangular(c.K)', c.X)
        for i in 1:m
            σ = 0.0
            for j in 1:k
                σ += c.X[j, i]^2
            end
            weight = sqrt(σ + k / m)
            for j in 1:k
                B[i, j] *= weight
            end
        end
    end
    mul!(U, B', B)
    issuccess(cholesky!(Hermitian(U, :U); check = false)) || return -Inf
    logdet = 0.0
    for j in 1:k
        logdet += log(U[j, j])
    end
    return logdet
end

# ‖U v‖² for the upper factor `U`, the quadratic form vᵀ H v
function quadratic_form!(work, U, v)
    copyto!(work, v)
    lmul!(UpperTriangular(U), work)
    return sum(abs2, work)
end

# Adapted from polytopewalk, src/dense/BarrierWalk.cpp, DikinWalk.cpp and VaidyaWalk.cpp
# (commit 39c7183, MIT; Copyright (c) 2023 Yuansi Chen, Benny Sun): the barrier weights, the
# proposal scales and the Metropolis ratio. The proposal solves `U y = ξ` for the upper
# factor `H = UᵀU`, so that it has covariance `H⁻¹`; polytopewalk solves with the lower
# factor `L = Uᵀ`, which gives covariance `L⁻¹L⁻ᵀ` while its Metropolis ratio assumes `H⁻¹`.
# hops also solves with the lower factor (DikinProposal.hpp), but its ratio uses the same
# metric `LᵀL`, so hops is a valid chain whose metric is not the Dikin ellipsoid.
# Like polytopewalk, and unlike volesti, the chain is not lazy.
function step!(rng::AbstractRNG, c::BarrierState)
    k = length(c.z)
    k == 0 && return update_x!(c)
    w = c.warmup
    if !c.ready
        update_slack!(c)
        move_inside!(c.z, c.slack, w.h) && update_slack!(c)
        c.logdet = factor_hessian!(c.U, c, c.slack)
        isfinite(c.logdet) || throw(ArgumentError("the chain must start strictly inside the polytope"))
        c.logp = logdensity_or_zero(c.density, c.z)
        c.ready = true
    end
    randn!(rng, c.step)
    ldiv!(UpperTriangular(c.U), c.step)
    c.step .*= c.scale
    for i in eachindex(c.z)
        c.zprop[i] = c.z[i] + c.step[i]
    end
    mul!(c.slackprop, w.G, c.zprop)
    inside = true
    for i in eachindex(c.slackprop)
        c.slackprop[i] = w.h[i] - c.slackprop[i]
        inside &= c.slackprop[i] > 0
    end
    c.proposed += 1
    c.last_accepted = false
    if inside
        logdet = factor_hessian!(c.Uprop, c, c.slackprop)
        if isfinite(logdet)
            qz = quadratic_form!(c.work, c.U, c.step)
            qprop = quadratic_form!(c.work, c.Uprop, c.step)
            logp = logdensity_or_zero(c.density, c.zprop)
            logα = logp - c.logp + logdet - c.logdet - (qprop - qz) / (2 * c.scale^2)
            if log(rand(rng)) < logα
                c.logp = logp
                copyto!(c.z, c.zprop)
                copyto!(c.slack, c.slackprop)
                c.U, c.Uprop = c.Uprop, c.U
                c.logdet = logdet
                c.accepted += 1
                c.last_accepted = true
            end
        end
    end
    return update_x!(c)
end
