"""
    sample_until_converged([rng], w::Warmup, method = BilliardWalk(); target_ess = 1000,
                           max_rhat = 1.01, nchains = 4, n = 1000, max_n = 1_000_000,
                           thinning = 1, multivariate = false)
    sample_until_converged([rng], t::Target, method = HitAndRun(); kwargs...)

Run `nchains` chains, on the uniform distribution of a [`Warmup`](@ref) or the density of
a [`Target`](@ref), and lengthen them until the second half of the chains has a smallest
effective sample size of at least `target_ess` and a largest R̂ below `max_rhat` over the
varying coordinates; the first half is discarded as burn-in (Gelman & Rubin 1992). With
`multivariate = true`, the multivariate R̂ of the chains (see [`diagnostics`](@ref)) must
also be below `max_rhat`; it grows with the number of coordinates, so above about 20 varying
coordinates this needs an ESS of many thousands.

The first round keeps `n` points per chain. Each later round sets the kept length from the
effective samples per point of the last one, to 1.2 times the length that reaches
`target_ess`, at most twice the last length (as volesti's `mmcs` extrapolates its run from
the ESS), and doubles it while R̂ is too high. Rounds stop once `max_n` points per chain are
kept, counted at the given `thinning`; the chains have then run twice as many, the first
half as burn-in.

Only the kept half of each chain is stored. When it holds more than 20 points per
effective sample (200 while the chains have fewer than 20 effective samples each), every
other point is dropped and later rounds keep every second step, as PolytopeSamplerMatlab
thins its stored chain; the same happens when the stored points would exceed 4 GiB. Long
runs of slowly mixing chains thus keep a few thousand points per chain, about as
informative as all of them.

Each chain starts at a random point of the largest ball around the warmup center inside
the polytope and has its own random number generator, seeded from `rng`, so the result
does not depend on the number of threads; chains run on separate threads when Julia has them.

Returns a named tuple: `chains` (one matrix per chain, a point per column), `diagnostics`
(see [`diagnostics`](@ref)), `n` (points kept per chain), `thinning` (the steps between
kept points, `thinning` times the automatic thinning), `converged` and `fixed`, the
coordinates that do not vary on the slice. `converged` covers the varying coordinates only;
`fixed` lists the others, which [`warmup`](@ref) holds at its center. A warning is logged
when the chains have not converged by `max_n`.
"""
function sample_until_converged(
        rng::AbstractRNG, model::Union{Warmup, Target}, method::PolytopeSampler = default_method(model);
        target_ess::Real = 1000, max_rhat::Real = 1.01, nchains::Integer = 4,
        n::Integer = 1000, max_n::Integer = 1_000_000, thinning::Integer = 1,
        multivariate::Bool = false,
    )
    nchains >= 2 || throw(ArgumentError("nchains must be at least 2 for R̂, got $nchains"))
    n >= 10 || throw(ArgumentError("n must be at least 10 for ESS and R̂, got $n"))
    max_n >= n || throw(ArgumentError("max_n must be at least n"))
    thinning >= 1 || throw(ArgumentError("thinning must be positive, got $thinning"))
    rngs = [Xoshiro(rand(rng, UInt64)) for _ in 1:nchains]
    states = [start_state(method, model, rngs[j]) for j in 1:nchains]
    nv = nvars(geometry(model).polytope)
    # The chains have run `total` records of `rec * thinning` steps each, of which the
    # records `head:total` are stored; a round runs to `2kept` records and keeps the last
    # `kept`
    draws = [zeros(nv, 0) for _ in 1:nchains]
    total, head, rec, kept = 0, 1, 1, n
    while true
        Threads.@threads for j in 1:nchains
            draws[j] = extend_chain(rngs[j], states[j], draws[j], head, total, kept, rec * thinning)
        end
        total, head = 2kept, kept + 1
        d = diagnostics(draws; multivariate)
        mixed = d.max_rhat < max_rhat && (!multivariate || d.multivariate_rhat < max_rhat)
        converged = all(isnan, d.ess) || (d.min_ess >= target_ess && mixed)
        result = (;
            chains = draws, diagnostics = d, n = kept, thinning = rec * thinning, converged,
            fixed = fixed_coordinates(geometry(model)),
        )
        done = converged || kept * rec >= max_n
        if !done
            # Halve the stored records while there are too many per effective sample
            # (PolytopeSamplerMatlab's MemoryStorage.m); `next` is the length that reaches
            # `target_ess`, before the factor of 1.2
            per_ess = kept * nchains / d.min_ess
            limit = d.min_ess < 20nchains ? 200 : 20
            next = mixed ? kept * target_ess / d.min_ess : 2kept
            while iseven(total) && per_ess > 2 &&
                    (per_ess > limit || 8 * nv * nchains * 1.2next > STORED_BYTES)
                draws = [X[:, (iseven(head) ? 1 : 2):2:end] for X in draws]
                total, head, rec = total ÷ 2, cld(head, 2), 2rec
                per_ess, next = per_ess / 2, next / 2
            end
            half = total ÷ 2
            kept = clamp(ceil(Int, 1.2next), half + 1, 2half)
            kept = min(kept, max_n ÷ rec, STORED_BYTES ÷ (8 * nv * nchains))
            done = kept <= half
        end
        if done
            converged || @warn "chains have not converged after $(result.n) points per chain" d.min_ess d.max_rhat
            return result
        end
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

# Stored points of the chains of `sample_until_converged`, 4 GiB in all
const STORED_BYTES = 2^32

# The records `kept + 1:2kept` of a chain whose records `head:total`, of `steps` steps each,
# are the columns of `X`
function extend_chain(rng, state, X, head, total, kept, steps)
    Y = Matrix{Float64}(undef, size(X, 1), kept)
    old = max(total - kept, 0) # stored records past the new burn-in
    copyto!(Y, 1, X, (kept - head + 1) * size(X, 1) + 1, old * size(X, 1))
    for i in (total + 1):(2kept)
        # the steps between records skip computing `x`, as `sample` does when thinning
        for _ in 2:steps
            advance!(rng, state)
        end
        step!(rng, state)
        i > kept && (Y[:, i - kept] = state.x)
    end
    check_state(state)
    return Y
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
