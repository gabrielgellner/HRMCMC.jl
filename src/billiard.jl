"""
    BilliardWalk(; trajectory_length = nothing, max_reflections = nothing)

The billiard walk (Gryazina & Polyak 2014) in its accelerated form (Chalkis et al. 2021):
each step draws a direction uniformly from the sphere and a length from the exponential
distribution with mean `trajectory_length`, and moves that far, reflecting off the facets
of the polytope. The chain is reversible with the uniform distribution as its stationary
distribution, and crosses a rounded polytope in a few steps, as a trajectory is not
stopped by the first facet it meets.

`trajectory_length` defaults to `2√k r` in the coordinates of the warmup, for dimension `k`
and inscribed radius `r` ([`Warmup`](@ref)`.radius`, near 1 after rounding), as in volesti.
On a warmup built with `rounding = :none` the inscribed ball says little about the length
of the polytope, so the default is raised to the largest distance between the ends of
`10 + log k` short walks from the center, as volesti's burn-in does, repeated with the new
length until it grows by less than 10%. On the unrounded box `[0, 100] × [0, 1]⁹` this gave
R̂ 1.001 and 7 times the effective samples per second of `2√k r`, which gave R̂ 1.40 after
4000 steps per chain. The walks use a fixed seed, so all chains of a warmup share the length.

A trajectory that makes more than `max_reflections` reflections (default `1000k`) is
abandoned and the chain stays put for that step. `getstats` reports `is_abandoned`, and a
chain warns once when more than 5% of its steps after the first 100 were abandoned. A reflection costs `O(m)` for `m`
constraints, as the rates `G v` are updated from the precomputed `G Gᵀ`, an `m × m` matrix
that the chains on one [`Warmup`](@ref) share.

Gryazina, E. & Polyak, B. (2014). Random sampling: billiard walk algorithm. European Journal
of Operational Research 238, 497–504. <https://doi.org/10.1016/j.ejor.2014.03.041>

Chalkis, A., Fisikopoulos, V., Tsigaridas, E. & Zafeiropoulos, H. (2021). Geometric
algorithms for sampling the flux space of metabolic networks. 37th International Symposium
on Computational Geometry (SoCG 2021), 21:1–21:16.
<https://doi.org/10.4230/LIPIcs.SoCG.2021.21>
"""
Base.@kwdef struct BilliardWalk <: PolytopeSampler
    trajectory_length::Union{Nothing, Float64} = nothing
    max_reflections::Union{Nothing, Int} = nothing
    function BilliardWalk(trajectory_length, max_reflections)
        trajectory_length === nothing || trajectory_length > 0 ||
            throw(ArgumentError("trajectory_length must be positive, got $trajectory_length"))
        max_reflections === nothing || max_reflections >= 1 ||
            throw(ArgumentError("max_reflections must be positive, got $max_reflections"))
        return new(trajectory_length, max_reflections)
    end
end

function Base.show(io::IO, m::BilliardWalk)
    args = String[]
    m.trajectory_length === nothing || push!(args, "trajectory_length = $(m.trajectory_length)")
    m.max_reflections === nothing || push!(args, "max_reflections = $(m.max_reflections)")
    print(io, "BilliardWalk(", join(args, ", "), ")")
    return
end

mutable struct BilliardState
    const method::BilliardWalk
    const warmup::Warmup
    const z::Vector{Float64}
    const x::Vector{Float64}
    const v::Vector{Float64} # direction of the trajectory
    const slack::Vector{Float64} # h - G z
    const Gv::Vector{Float64}
    const zstart::Vector{Float64} # start of the trajectory, restored if it is abandoned
    const Gt::Matrix{Float64} # Gᵀ, whose columns are the facet normals (shared)
    const GG::Matrix{Float64} # G Gᵀ (shared)
    const trajectory_length::Float64
    const max_reflections::Int
    steps::Int # steps taken, for refreshing the slacks
    reflections::Int # reflections in the last step
    abandoned::Int # trajectories abandoned so far
    last_abandoned::Bool
    warned::Bool # whether the chain has warned about abandoned trajectories
end

AbstractMCMC.getparams(c::BilliardState) = copy(c.x)
AbstractMCMC.getstats(c::BilliardState) = (; n_reflections = c.reflections, is_abandoned = c.last_abandoned)

# Adapted from hops, include/hops/MarkovChain/Proposal/BilliardWalkProposal.hpp (commit
# b62592e, MIT; Copyright (c) 2020 Johann Fredrik Jadebeck, Axel Theorell, Samuel Leweke,
# Katharina Nöh), which counts an abandoned trajectory as a rejection: warn once per chain
# when they become common, outside the allocation-free step
function check_state(c::BilliardState)
    if !c.warned && c.steps > 100 && c.abandoned > 0.05 * c.steps
        @warn "BilliardWalk: $(c.abandoned) of $(c.steps) trajectories were abandoned after max_reflections reflections, and the chain stayed put for them. Use a rounded warmup, or a shorter trajectory_length."
        c.warned = true
    end
    return c
end

function chain_state(::BilliardWalk, ::Target)
    throw(ArgumentError("BilliardWalk samples only the uniform distribution"))
end

function chain_state(method::BilliardWalk, w::Warmup)
    k = size(w.transform, 2)
    m = length(w.h)
    L = method.trajectory_length === nothing ? default_trajectory_length(w) : method.trajectory_length
    Gt, GG = reflection_matrices(w)
    return BilliardState(
        method, w, zeros(k), copy(w.center), zeros(k), copy(w.h), zeros(m), zeros(k), Gt, GG,
        L, something(method.max_reflections, 1000 * k), 0, 0, 0, false, false,
    )
end

# Adapted from volesti, include/random_walks/uniform_accelerated_billiard_walk.hpp (commit
# 064e18c, LGPL-3.0; Copyright (c) 2012-2024 Vissarion Fisikopoulos, 2018-2024 Apostolos
# Chalkis), `parameters_burnin` with the `10 + log k` walks of 10 steps that mmcs.hpp uses:
# `2√k r`, raised on an unrounded warmup to the largest distance between the center and the
# ends of the walks. Unlike volesti, the burn-in is repeated with the new length until it
# grows by less than 10%: one round from `2√k r` reached 11 on the box [0, 100] × [0, 1]⁹,
# and the rounds reached about 100, which raised the effective samples per second 7 times.
# Rounded polytopes keep `2√k r`, which mixed faster than longer lengths on the cube and
# simplex (#71).
function default_trajectory_length(w::Warmup)
    k = size(w.transform, 2)
    L = 2 * sqrt(k) * w.radius
    (w.rounded || k == 0) && return L
    cache = w.cache
    return lock(cache.lock) do
        cached = cache.trajectory_length
        cached === nothing || return cached
        burned = burnin_length(w, L)
        cache.trajectory_length = burned
        burned
    end
end

function burnin_length(w::Warmup, L::Float64)
    rng = Xoshiro(BURNIN_SEED)
    for _ in 1:BURNIN_ROUNDS
        Lnew = burnin_spread(rng, w, L)
        grown = Lnew > 1.1 * L
        L = max(L, Lnew)
        grown || break
    end
    return L
end

# The largest distance between the center and the ends of `10 + log k` walks of 10 steps
# with trajectory length `L`, from random points of the ball around the center
function burnin_spread(rng::AbstractRNG, w::Warmup, L::Float64)
    k = size(w.transform, 2)
    method = BilliardWalk(trajectory_length = L)
    ends = [zeros(k)]
    for _ in 1:(10 + floor(Int, log(k)))
        c = start_state(method, w, rng)
        for _ in 1:10
            step!(rng, c)
        end
        push!(ends, copy(c.z))
    end
    return maximum(norm(a - b) for a in ends, b in ends)
end

const BURNIN_ROUNDS = 30

const BURNIN_SEED = 0x6d6d6373

# The distance along the trajectory to the first facet it meets, and that facet. Slacks are
# clamped at zero against rounding error, and facets the trajectory moves away from or
# along are skipped.
function first_hit(slack, Gv)
    λ = Inf
    facet = 0
    for i in eachindex(slack, Gv)
        if Gv[i] > ZERO_TOL
            l = max(slack[i], 0.0) / Gv[i]
            if l < λ
                λ = l
                facet = i
            end
        end
    end
    facet == 0 && throw(ArgumentError("the polytope is unbounded along a sampled direction"))
    return λ, facet
end

function move!(c::BilliardState, t)
    for i in eachindex(c.z)
        c.z[i] += t * c.v[i]
    end
    for i in eachindex(c.slack)
        c.slack[i] -= t * c.Gv[i]
    end
    return c
end

# Reflect the direction off `facet`, whose normal is the unit row `G[facet, :]`, and update
# the rates `G v` with a column of `G Gᵀ`.
function reflect!(c::BilliardState, facet)
    a = c.Gv[facet]
    for i in eachindex(c.v)
        c.v[i] -= 2a * c.Gt[i, facet]
    end
    for i in eachindex(c.Gv)
        c.Gv[i] -= 2a * c.GG[i, facet]
    end
    return c
end

# Adapted from volesti, include/random_walks/uniform_accelerated_billiard_walk.hpp (commit
# 064e18c, LGPL-3.0; Copyright (c) 2012-2024 Vissarion Fisikopoulos, 2018-2024 Apostolos
# Chalkis): exponential trajectory lengths, rates updated through G Gᵀ after each
# reflection, and abandoned trajectories. Unlike volesti, which stops at 0.995 of the
# distance to a facet, the trajectory reflects on the facet itself and its slack is set to
# zero, which keeps the walk reversible; clamping slacks at zero in `first_hit` keeps
# rounding error from moving it outside.
function advance!(rng::AbstractRNG, c::BilliardState)
    c.reflections = 0
    c.last_abandoned = false
    isempty(c.z) && return c
    w = c.warmup
    randn!(rng, c.v)
    normalize!(c.v)
    refresh_slack!(c)
    mul!(c.Gv, w.G, c.v)
    copyto!(c.zstart, c.z)
    T = randexp(rng) * c.trajectory_length
    for _ in 1:c.max_reflections
        λ, facet = first_hit(c.slack, c.Gv)
        if T <= λ
            move!(c, T)
            return c
        end
        move!(c, λ)
        T -= λ
        c.slack[facet] = 0.0
        reflect!(c, facet)
        c.reflections += 1
    end
    copyto!(c.z, c.zstart)
    update_slack!(c)
    c.abandoned += 1
    c.last_abandoned = true
    return c
end
