# Parallel sampling

Start Julia with threads (`julia --threads=auto`), and two kinds of work run in parallel.

## Chains

[`sample_until_converged`](@ref), [`mmcs`](@ref) and `sample(..., MCMCThreads(), n, nchains)`
run each chain on its own thread. Each chain has its own random number generator seeded
from `rng`, so the result is the same for any number of threads. Billiard-walk chains on
one [`Warmup`](@ref) share one copy of the `m × m` matrix `G Gᵀ`; ACHR chains share the
extreme points.

## Many polytopes

[`warmup`](@ref) builds its own JuMP model, every function takes an explicit `rng`, and a
[`Warmup`](@ref) is read-only while sampling, so `Threads.@threads` over independent
problems is safe:

```julia
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)  # threads already keep every core busy

results = Vector{Float64}(undef, N)
Threads.@threads for i in 1:N
    rng = Xoshiro(100 + i)            # one generator per problem
    P = make_polytope(rng, i)
    r = sample_until_converged(rng, warmup(rng, P); target_ess = 1_000)
    results[i] = summarize(r)
end
```

`examples/foodweb_replicates.jl` in the repository runs 80 food webs this way.

## Measured speedups

On a 10-core Apple machine (4 performance and 6 efficiency cores):

| Work | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| 80 food web replicates (15 species, to 1000 ESS each) | 54.4 s | 15.3 s | 10.4 s |
| 4 billiard chains of 4000 steps, 200 dimensions | 5.3 s | 1.6 s | |

Chains in one run gain little beyond the number of performance cores, as the slowest
chain sets the time; batches of independent problems keep gaining.
