"""
    sample_until_converged([rng], w::Warmup, method = BilliardWalk(); target_ess = 1000,
                           max_rhat = 1.01, nchains = 4, n = 1000, max_n = 1_000_000,
                           thinning = 1)
    sample_until_converged([rng], t::Target, method = HitAndRun(); kwargs...)

Run `nchains` chains, on the uniform distribution of a [`Warmup`](@ref) or the density of
a [`Target`](@ref), and double their length until the second half of the chains has a
smallest effective sample size of at least `target_ess` and a largest R̂ below `max_rhat`
over the varying coordinates; the first half is discarded as burn-in (Gelman & Rubin
1992). The first round keeps `n` points per chain; rounds stop once `max_n` points per
chain are kept.

Each chain starts at a random point of the largest ball around the warmup center inside
the polytope and has its own random number generator, seeded from `rng`, so the result
does not depend on the number of threads; chains run on separate threads when Julia has them.

Returns a named tuple: `chains` (one matrix per chain, a point per column), `diagnostics`
(see [`diagnostics`](@ref)), `n` (points kept per chain), `converged` and `fixed`, the
coordinates that do not vary on the slice. `converged` covers the varying coordinates only;
`fixed` lists the others, which [`warmup`](@ref) holds at its center. A warning is logged
when the chains have not converged by `max_n`.
"""
function sample_until_converged(
        rng::AbstractRNG, model::Union{Warmup, Target}, method::PolytopeSampler = default_method(model);
        target_ess::Real = 1000, max_rhat::Real = 1.01, nchains::Integer = 4,
        n::Integer = 1000, max_n::Integer = 1_000_000, thinning::Integer = 1,
    )
    nchains >= 2 || throw(ArgumentError("nchains must be at least 2 for R̂, got $nchains"))
    n >= 4 || throw(ArgumentError("n must be at least 4, got $n"))
    max_n >= n || throw(ArgumentError("max_n must be at least n"))
    thinning >= 1 || throw(ArgumentError("thinning must be positive, got $thinning"))
    rngs = [Xoshiro(rand(rng, UInt64)) for _ in 1:nchains]
    states = [start_state(method, model, rngs[j]) for j in 1:nchains]
    draws = [zeros(nvars(geometry(model).polytope), 0) for _ in 1:nchains]
    kept = n
    while true
        total = 2kept
        Threads.@threads for j in 1:nchains
            draws[j] = hcat(draws[j], run_chain!(rngs[j], states[j], total - size(draws[j], 2), thinning))
        end
        chains = [X[:, (kept + 1):end] for X in draws]
        d = diagnostics(chains)
        converged = all(isnan, d.ess) || (d.min_ess >= target_ess && d.max_rhat < max_rhat)
        if converged || kept >= max_n
            converged || @warn "chains have not converged after $kept points per chain" d.min_ess d.max_rhat
            return (; chains, diagnostics = d, n = kept, converged, fixed = fixed_coordinates(geometry(model)))
        end
        kept = min(2kept, max_n)
    end
    return
end

function sample_until_converged(
        model::Union{Warmup, Target}, method::PolytopeSampler = default_method(model); kwargs...,
    )
    return sample_until_converged(default_rng(), model, method; kwargs...)
end

default_method(::Warmup) = BilliardWalk()
default_method(::Target) = HitAndRun()

# `count` more points of a chain, keeping every `thinning`-th step
function run_chain!(rng, state, count, thinning)
    X = Matrix{Float64}(undef, length(state.x), count)
    for j in 1:count
        for _ in 1:thinning
            step!(rng, state)
        end
        X[:, j] = state.x
    end
    check_state(state)
    return X
end

"""
    mmcs([rng], P::Polytope; target_ess = 1000, optimizer = HiGHS.Optimizer, kwargs...)

Multiphase Monte Carlo sampling (Chalkis et al. 2021): prepare `P` with one linear program
and rounding from billiard-walk samples (`warmup(rng, P; extremes = false, rounding =
:samples)`), then run billiard-walk chains until they reach `target_ess` with R̂ below 1.01
([`sample_until_converged`](@ref), which takes the other keyword arguments).

Returns the named tuple of [`sample_until_converged`](@ref) with the `warmup` added.

This differs from the volesti implementation in its stopping rule: volesti sums the ESS of
single chains across phases, including the rounding phases, and restarts each window of
100 points near the center; here the rounding samples are discarded and several chains run
until R̂ also shows they agree.

Chalkis, A., Fisikopoulos, V., Tsigaridas, E. & Zafeiropoulos, H. (2021). Geometric
algorithms for sampling the flux space of metabolic networks. 37th International Symposium
on Computational Geometry (SoCG 2021), 21:1–21:16.
<https://doi.org/10.4230/LIPIcs.SoCG.2021.21>
"""
function mmcs(
        rng::AbstractRNG, P::Polytope; target_ess::Real = 1000, optimizer = HiGHS.Optimizer,
        kwargs...,
    )
    w = warmup(rng, P; extremes = false, rounding = :samples, optimizer)
    result = sample_until_converged(rng, w, BilliardWalk(); target_ess, kwargs...)
    return (; result..., warmup = w)
end

mmcs(P::Polytope; kwargs...) = mmcs(default_rng(), P; kwargs...)
