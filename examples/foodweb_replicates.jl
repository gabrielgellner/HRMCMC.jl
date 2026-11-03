# Many replicates of the constrained food web study in parallel, one per thread: a web,
# pyramid biomasses, growth rates redrawn until the equilibrium is feasible, billiard-walk
# chains until 1000 effective samples, and the fraction of stable community matrices. Each
# replicate seeds its own generator, so the results do not depend on the number of threads.
# Run with `just example foodweb_replicates` (all threads).

include("foodweb_study.jl")

using LinearAlgebra: BLAS
using PolytopeSamplers: BilliardWalk, sample_until_converged, warmup
using Random: Xoshiro
using Statistics: mean

# Each replicate runs on one thread; BLAS threads would compete with them
BLAS.set_num_threads(1)

function replicate(seed; S = 15, C = 0.15)
    rng = Xoshiro(seed)
    web = generalized_cascade_network(rng, S, C)
    x = pyramid_biomass(rng, web)
    (; fw, P) = feasible_food_web(rng, web, x)
    result = sample_until_converged(rng, warmup(rng, P), BilliardWalk(); target_ess = 1_000)
    return mean(is_stable(community_matrix(fw, θ)) for X in result.chains for θ in eachcol(X))
end

let N = 80
    replicate(1) # compile before timing
    fractions = zeros(N)
    t = @elapsed Threads.@threads for i in 1:N
        fractions[i] = replicate(100 + i)
    end
    println(
        N, " replicates on ", Threads.nthreads(), " threads in ", round(t; digits = 1),
        " s; mean stable fraction ", round(mean(fractions); digits = 4),
    )
end
