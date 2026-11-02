"""
    CRHMC(; initial_step_size = 0.2, effective_step_size = 1.0, max_ode_steps = 30,
          implicit_tol = 1e-5)

Constrained Riemannian Hamiltonian Monte Carlo (Kook, Lee, Shen & Vempala 2022): Hamiltonian
dynamics in the metric of the log barrier, which slows the chain down only in the
directions in which a facet is near, so it takes long steps through the whole polytope.

CRHMC writes the polytope as `{x : A x = b, l ≤ x ≤ u}` with a diagonal barrier metric
projected onto `A x = b`, which keeps large sparse problems cheap. Here the dynamics run in
the coordinates `z` of the [`Warmup`](@ref): with slacks `s = h - G z` as the bounded
variables, the projected metric is `g(z) = Gᵀ S⁻² G`, the Hessian of the log barrier of
`G z ≤ h`, and the Hamiltonian is

    H(z, p) = ½ log det g(z) + ½ pᵀ g(z)⁻¹ p,

minus the log density for a [`Target`](@ref).

Each step refreshes the momentum partially (`p ← √μ p + √(1 - μ) g^½ ξ` with
`μ = 1 - h / effective_step_size`), takes one step of the Strang splitting of the
potential and kinetic parts, solving the kinetic part by the implicit midpoint rule with
fixed-point iterations (up to `max_ode_steps`, to a local-norm tolerance `implicit_tol`),
and accepts by a Metropolis filter on `H`; a rejection reverses the momentum. The step size
`h` grows from 0.001 to `initial_step_size` over the first 10 effective steps, then shrinks
by a factor of 1.1 whenever rejections, failed integrations or integration effort exceed
their targets, as in the reference implementation. A step factors the `k × k` metric once
or twice, `O(mk² + k³)` each, and each implicit iteration costs `O(mk)`.

The metric is affine-invariant, so in exact arithmetic the chain behaves the same in any
coordinates, but its condition number grows as the square of the polytope's aspect ratio,
and badly scaled polytopes need the default rounding: on a rotated box with sides from
1e-3 to 1e3 without rounding, chains reached points 1e-6 from a facet where every
integration failed, and stayed there.

A step whose implicit integration does not converge is rejected. As failures happen more
often near the boundary, they bias the chain slightly toward the interior; the metric is
refactored during slow integrations to keep them below about 0.1% of steps, where the bias
was not detectable over 20 seeds.

Kook, Y., Lee, Y. T., Shen, R. & Vempala, S. S. (2022). Sampling with Riemannian Hamiltonian
Monte Carlo in a constrained space. Advances in Neural Information Processing Systems 35.
<https://arxiv.org/abs/2202.01908>
"""
struct CRHMC <: PolytopeSampler
    initial_step_size::Float64
    effective_step_size::Float64
    max_ode_steps::Int
    implicit_tol::Float64
    function CRHMC(initial_step_size, effective_step_size, max_ode_steps, implicit_tol)
        initial_step_size > 0 || throw(ArgumentError("initial_step_size must be positive"))
        effective_step_size > 0 || throw(ArgumentError("effective_step_size must be positive"))
        max_ode_steps >= 1 || throw(ArgumentError("max_ode_steps must be positive"))
        implicit_tol > 0 || throw(ArgumentError("implicit_tol must be positive"))
        return new(initial_step_size, effective_step_size, max_ode_steps, implicit_tol)
    end
end

function CRHMC(;
        initial_step_size::Real = 0.2, effective_step_size::Real = 1.0,
        max_ode_steps::Integer = 30, implicit_tol::Real = 1.0e-5,
    )
    return CRHMC(initial_step_size, effective_step_size, max_ode_steps, implicit_tol)
end

Base.show(io::IO, ::CRHMC) = print(io, "CRHMC()")

# Settings of the step size adaptation, from the defaults of PolytopeSamplerMatlab
const CRHMC_WARMUP = 10 # effective steps over which the step size grows
const CRHMC_MAX_BAD = 10 # consecutive bad steps before shrinking
const CRHMC_TARGET_ODE = 10 # average implicit iterations per step before shrinking
const CRHMC_SHRINK = 1.1
const CRHMC_MIN_STEP = 1.0e-3
# Iterations of the implicit midpoint rule between refactorizations of the metric
const CRHMC_REFACTOR = 5
# Ridge added to the metric, as the reference adds 1e-20 to its Hessian
const CRHMC_RIDGE = 1.0e-20

mutable struct CRHMCState{D}
    const method::CRHMC
    const warmup::Warmup
    const density::D # nothing for the uniform distribution
    const z::Vector{Float64}
    const x::Vector{Float64}
    const p::Vector{Float64} # momentum
    # integrator
    const z1::Vector{Float64}
    const p1::Vector{Float64}
    const z2::Vector{Float64}
    const p2::Vector{Float64}
    const zmid::Vector{Float64}
    const pmid::Vector{Float64}
    const zold::Vector{Float64}
    const dKdz::Vector{Float64}
    const dKdp::Vector{Float64}
    const dU::Vector{Float64}
    const dU_z::Vector{Float64} # point at which dU was computed
    const work::Vector{Float64}
    const w::Vector{Float64} # ≈ g(zmid)⁻¹ pmid, refined once per iteration
    const smid::Vector{Float64} # slacks at zmid
    const Gd::Vector{Float64}
    # metric at `metric_z`
    const metric_z::Vector{Float64}
    const s::Vector{Float64} # slacks h - G z
    const Gw::Vector{Float64}
    const B::Matrix{Float64} # S⁻¹ G
    const U::Matrix{Float64} # upper Cholesky factor of g
    const X::Matrix{Float64} # U⁻ᵀ Bᵀ, for leverage scores
    logdet::Float64 # ½ log det g
    metric_ok::Bool
    # step size adaptation
    h::Float64
    momentum::Float64
    accumulated_momentum::Float64
    effective_steps::Float64
    warmed_up::Bool
    consecutive_bad::Int
    iter_since_shrink::Float64
    reject_since_shrink::Float64
    ode_since_shrink::Float64
    started::Bool
    stuck::Bool # steps kept failing at the smallest step size
    proposed::Int
    accepted::Int
    last_accepted::Bool
    last_step_size::Float64 # step size of the last step
end

# Warn, once per chain, when a chain may be stuck; called outside the allocation-free step
function check_state(c::CRHMCState)
    if c.stuck
        @warn "CRHMC: steps keep failing at the smallest step size; the chain may be stuck. Use a rounded warmup for badly scaled polytopes; a log density that diverges at a face, such as a gamma density with shape below 1, has no smooth gradient there, and hit-and-run suits it better."
        c.stuck = false
    end
    return c
end

chain_state(method::CRHMC, w::Warmup) = crhmc_state(method, w, nothing)

AbstractMCMC.getparams(c::CRHMCState) = copy(c.x)
function AbstractMCMC.getstats(c::CRHMCState{Nothing})
    return (;
        is_accept = c.last_accepted, acceptance_rate = c.accepted / max(c.proposed, 1),
        step_size = c.last_step_size,
    )
end
# The state keeps no log density, so this evaluates it once, outside the step
function AbstractMCMC.getstats(c::CRHMCState)
    return (;
        lp = logdensity_z(c.density, c.z), is_accept = c.last_accepted,
        acceptance_rate = c.accepted / max(c.proposed, 1), step_size = c.last_step_size,
    )
end

function chain_state(method::CRHMC, t::Target)
    density = chain_density(t)
    has_gradient(density) ||
        throw(ArgumentError("CRHMC needs the gradient of the log density; pass `gradient` to Target"))
    return crhmc_state(method, t.warmup, density)
end

function crhmc_state(method::CRHMC, w::Warmup, density)
    k = size(w.transform, 2)
    m = length(w.h)
    v() = zeros(k)
    h = CRHMC_MIN_STEP
    return CRHMCState(
        method, w, density,
        v(), copy(w.center), v(), # z, x, p
        v(), v(), v(), v(), v(), v(), v(), # z1, p1, z2, p2, zmid, pmid, zold
        v(), v(), v(), fill(NaN, k), # dKdz, dKdp, dU, dU_z
        v(), v(), zeros(m), zeros(m), # work, w, smid, Gd
        fill(NaN, k), zeros(m), zeros(m), # metric_z, s, Gw
        zeros(m, k), zeros(k, k), zeros(k, m), 0.0, false, # B, U, X, logdet, metric_ok
        h, 1 - min(1, h / method.effective_step_size), 0.0, 0.0, false, # h, momentum, …
        0, 0.0, 0.0, 0.0, false, false, 0, 0, false, NaN,
    )
end

# Compute the slacks and the Cholesky factor of the metric at `z`, unless they are cached;
# return false when `z` is outside the polytope or the factorization fails
function metric!(c::CRHMCState, z)
    z == c.metric_z && return c.metric_ok
    copyto!(c.metric_z, z)
    G = c.warmup.G
    mul!(c.s, G, z)
    ok = true
    for i in eachindex(c.s)
        c.s[i] = c.warmup.h[i] - c.s[i]
        ok &= c.s[i] > 0
    end
    if ok
        m, k = size(G)
        for j in 1:k, i in 1:m
            c.B[i, j] = G[i, j] / c.s[i]
        end
        mul!(c.U, c.B', c.B)
        for j in 1:k
            c.U[j, j] += CRHMC_RIDGE
        end
        ok = issuccess(cholesky!(Hermitian(c.U, :U); check = false))
        if ok
            c.logdet = 0.0
            for j in 1:k
                c.logdet += log(c.U[j, j])
            end
        end
    end
    c.metric_ok = ok
    return ok
end

# ∇U = Gᵀ (σ ./ s) - ∇ℓ, with σ the leverage scores of S⁻¹ G and ℓ the log density
function grad_potential!(out, c::CRHMCState, z)
    metric!(c, z) || return false
    transpose!(c.X, c.B)
    ldiv!(UpperTriangular(c.U)', c.X)
    m, k = size(c.B)
    for i in 1:m
        σ = 0.0
        for j in 1:k
            σ += c.X[j, i]^2
        end
        c.Gw[i] = σ / c.s[i]
    end
    mul!(out, c.warmup.G', c.Gw)
    c.density === nothing || add_gradient_z!(out, c.density, z, -1.0)
    return true
end

# ∂K/∂p = g⁻¹ p and ∂K/∂z = -½ Dg[g⁻¹p, g⁻¹p] = -Gᵀ ((G g⁻¹ p)² ./ s³)
function grad_kinetic!(dKdp, dKdz, c::CRHMCState, z, p)
    metric!(c, z) || return false
    copyto!(dKdp, p)
    ldiv!(UpperTriangular(c.U)', dKdp)
    ldiv!(UpperTriangular(c.U), dKdp)
    mul!(c.Gw, c.warmup.G, dKdp)
    for i in eachindex(c.Gw)
        c.Gw[i] = -c.Gw[i]^2 / c.s[i]^3
    end
    mul!(dKdz, c.warmup.G', c.Gw)
    return true
end

# H(z, p) = -ℓ(z) + ½ log det g + ½ pᵀ g⁻¹ p; Inf outside the polytope
function hamiltonian(c::CRHMCState, z, p)
    metric!(c, z) || return Inf
    copyto!(c.work, p)
    ldiv!(UpperTriangular(c.U)', c.work)
    return c.logdet + sum(abs2, c.work) / 2 - logdensity_or_zero(c.density, z)
end

# Slacks `h - G z` into `s`; false if any is not positive
function slacks!(s, c::CRHMCState, z)
    mul!(s, c.warmup.G, z)
    ok = true
    for i in eachindex(s)
        s[i] = c.warmup.h[i] - s[i]
        ok &= s[i] > 0
    end
    return ok
end

# g v = Gᵀ ((G v) ./ s²) for the metric at slacks `s`, into `out`, in O(mk)
function metric_product!(out, c::CRHMCState, s, v)
    mul!(c.Gd, c.warmup.G, v)
    for i in eachindex(c.Gd)
        c.Gd[i] /= s[i]^2
    end
    mul!(out, c.warmup.G', c.Gd)
    return out
end

# Adapted from PolytopeSamplerMatlab, code/implicit_midpoint.m and the approxDK method of
# code/Hamiltonian.m (commit f85505c, GPL-3.0; Kook, Lee, Shen & Vempala): a half step of the
# potential, the implicit midpoint rule for the kinetic part by fixed-point iteration, and a
# half step of the potential. As in the reference, the metric is factored at the start of
# the step and each iteration refines `w ≈ g(zmid)⁻¹ pmid` by one solve with that factor,
# `w ← w + g(z)⁻¹ (pmid - g(zmid) w)`, so an iteration costs O(mk + k²) instead of a
# factorization; every `CRHMC_REFACTOR` iterations the factor is renewed at the midpoint. Returns the number of iterations, or 0 when the iteration fails to
# converge or leaves the polytope.
function implicit_midpoint!(c::CRHMCState, h)
    # The gradient at the end of an accepted step is the one needed at the start of the next
    if c.z == c.dU_z
        metric!(c, c.z) || return 0
    else
        grad_potential!(c.dU, c, c.z) || return 0
        copyto!(c.dU_z, c.z)
    end
    # c.U now holds the factor at z, used for every refinement below
    copyto!(c.z1, c.z)
    for i in eachindex(c.p1)
        c.p1[i] = c.p[i] - (h / 2) * c.dU[i]
    end
    copyto!(c.z2, c.z1)
    copyto!(c.p2, c.p1)
    fill!(c.w, 0.0)
    tol = c.method.implicit_tol
    for iteration in 1:c.method.max_ode_steps
        for i in eachindex(c.zmid)
            c.zmid[i] = (c.z1[i] + c.z2[i]) / 2
            c.pmid[i] = (c.p1[i] + c.p2[i]) / 2
        end
        slacks!(c.smid, c, c.zmid) || return 0
        # When the refinement is slow to converge, factor the metric at the current midpoint
        # and refine with that. A step that fails to converge is rejected, and failures
        # concentrated near the boundary bias the chain toward the interior: with the
        # start-of-step factor alone, 2% of steps failed on a cube and standard deviations
        # came out 4 Monte Carlo standard errors low.
        if iteration % CRHMC_REFACTOR == 0
            metric!(c, c.zmid) || return 0
        end
        U = UpperTriangular(c.U)
        metric_product!(c.work, c, c.smid, c.w)
        for i in eachindex(c.work)
            c.work[i] = c.pmid[i] - c.work[i]
        end
        ldiv!(U', c.work)
        ldiv!(U, c.work)
        for i in eachindex(c.w)
            c.w[i] += c.work[i]
        end
        # ∂K/∂z = -Gᵀ ((G w)² ./ s³) at the midpoint
        mul!(c.Gd, c.warmup.G, c.w)
        for i in eachindex(c.Gd)
            c.Gd[i] = -c.Gd[i]^2 / c.smid[i]^3
        end
        mul!(c.dKdz, c.warmup.G', c.Gd)
        for i in eachindex(c.z2)
            c.zold[i] = c.z2[i]
            c.z2[i] = c.z1[i] + h * c.w[i]
            c.p2[i] = c.p1[i] - h * c.dKdz[i]
            c.zold[i] = c.z2[i] - c.zold[i]
        end
        # ‖Δz‖² in the metric at the midpoint
        mul!(c.Gd, c.warmup.G, c.zold)
        dist = 0.0
        for i in eachindex(c.Gd)
            dist += (c.Gd[i] / c.smid[i])^2
        end
        dist /= h
        if dist < tol
            grad_potential!(c.dU, c, c.z2) || return 0
            copyto!(c.dU_z, c.z2)
            for i in eachindex(c.p2)
                c.p2[i] -= (h / 2) * c.dU[i]
            end
            return iteration
        end
        (isfinite(dist) && dist < 1.0e16) || return 0
    end
    return 0
end

# Draw p ← √μ p + √(1 - μ) Uᵀ ξ, whose fresh part has covariance g
function refresh_momentum!(rng, c::CRHMCState, μ)
    metric!(c, c.z)
    randn!(rng, c.work)
    lmul!(UpperTriangular(c.U)', c.work)
    for i in eachindex(c.p)
        c.p[i] = sqrt(μ) * c.p[i] + sqrt(1 - μ) * c.work[i]
    end
    return c.p
end

# Adapted from PolytopeSamplerMatlab, code/module/Sampler.m and
# code/module/DynamicStepSize.m (commit f85505c, GPL-3.0): the generalized HMC step with
# partial momentum refreshment and momentum reversal on rejection, and the step size
# schedule. The reference adapts the step size jointly over its parallel chains; here each
# chain adapts its own. On rejection the reference reverses the momentum from before the
# refresh; here the refreshed momentum is reversed, as in generalized HMC (Horowitz 1991):
# the refresh keeps the momentum distribution N(0, g(z)) at fixed z and the Metropolis step
# with reversal keeps the joint distribution, so their composition does.
function step!(rng::AbstractRNG, c::CRHMCState)
    k = length(c.z)
    k == 0 && return update_x!(c)
    if !c.started
        slacks!(c.s, c, c.z) || move_inside!(c.z, c.s, c.warmup.h)
        metric!(c, c.z) || throw(ArgumentError("the chain must start strictly inside the polytope"))
        refresh_momentum!(rng, c, 0.0)
        c.started = true
    end
    h = c.h
    refresh_momentum!(rng, c, c.momentum)
    H1 = hamiltonian(c, c.z, c.p)
    iterations = implicit_midpoint!(c, h)
    prob = 0.0
    if iterations > 0
        H2 = hamiltonian(c, c.z2, c.p2)
        prob = isfinite(H2) ? min(1.0, exp(H1 - H2)) : 0.0
    end
    c.proposed += 1
    accept = rand(rng) < prob
    c.last_accepted = accept
    c.last_step_size = h
    if accept
        copyto!(c.z, c.z2)
        copyto!(c.p, c.p2)
        c.accepted += 1
    else
        c.p .*= -1
    end
    c.accumulated_momentum = prob * c.momentum * c.accumulated_momentum + h
    accept && (c.effective_steps += h * c.accumulated_momentum)
    adapt_step_size!(c, prob, iterations == 0 ? c.method.max_ode_steps : iterations)
    return update_x!(c)
end

function adapt_step_size!(c::CRHMCState, prob, iterations)
    method = c.method
    bad = prob < 0.5 || iterations >= method.max_ode_steps
    c.consecutive_bad = bad ? c.consecutive_bad + 1 : 0
    ratio = c.effective_steps / CRHMC_WARMUP
    if ratio < 1 && !c.warmed_up && c.consecutive_bad < CRHMC_MAX_BAD
        c.h = method.initial_step_size * min(ratio + 0.01, 1.0)
        c.momentum = 1 - min(1.0, c.h / method.effective_step_size)
        return
    end
    c.warmed_up = true
    c.iter_since_shrink += 1
    c.reject_since_shrink += 1 - prob
    c.ode_since_shrink += iterations
    shifted = c.iter_since_shrink + 20 / (1 - c.momentum)
    target = (1 - c.momentum)^(2 / 3) / 4
    if c.reject_since_shrink > target * shifted || c.consecutive_bad > CRHMC_MAX_BAD ||
            c.ode_since_shrink > CRHMC_TARGET_ODE * shifted
        c.iter_since_shrink = 0.0
        c.reject_since_shrink = 0.0
        c.ode_since_shrink = 0.0
        c.consecutive_bad = 0
        c.h == CRHMC_MIN_STEP && bad && (c.stuck = true)
        c.h = max(c.h / CRHMC_SHRINK, CRHMC_MIN_STEP)
        c.momentum = 1 - min(0.999, c.h / method.effective_step_size)
    end
    c.iter_since_shrink += 1
    return
end
