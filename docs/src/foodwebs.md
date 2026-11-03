# Example: constrained food webs

The package began as the sampler for
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs): finding the
interaction strengths of a food web that are consistent with an observed equilibrium. The
repository's `examples/foodweb_study.jl` builds the webs and the polytope with the settings of
the original study, in about 140 lines adapted from
[FoodWebs.jl](https://github.com/gabrielgellner/FoodWebs.jl), which holds the general version.

## The polytope

Lotka–Volterra dynamics `dxᵢ/dt = xᵢ (rᵢ + Σⱼ aᵢⱼ xⱼ)` have an equilibrium at the biomasses
`x` when `rᵢ + Σⱼ aᵢⱼ xⱼ = 0` for every species. For a given web, biomasses and growth rates,
the magnitudes of the interactions form a polytope. Each feeding link, prey `i` eaten by
predator `j`, has the prey's loss `λ` (`aᵢⱼ = -λ`) and the predator's gain `γ` (`aⱼᵢ = γ`),
and every species eaten by another has a self-regulation `σ` (`aᵢᵢ = -σ`). The constraints are

- one equality per species, `rᵢ + Σⱼ aᵢⱼ xⱼ = 0`;
- bounds `0 ≤ λ, γ, σ ≤ 5`;
- `γ ≤ λ` on each link: a predator gains at most what its prey loses.

The study drew webs from the generalized cascade model, biomasses falling linearly with
trophic level, and growth rates `rᵢ ~ U(0, 5)` for basal species and `-U(0, 1)` for
consumers, redrawn until the polytope is nonempty. A 15-species web gives about 80
interaction strengths in a 65-dimensional slice.

## Sampling

```@setup foodweb
using PolytopeSamplers: PolytopeSamplers
include(joinpath(pkgdir(PolytopeSamplers), "examples", "foodweb_study.jl"))
```

```@example foodweb
using PolytopeSamplers, Random

rng = Xoshiro(2026)
web = generalized_cascade_network(rng, 15, 0.15)
x = pyramid_biomass(rng, web)
(; fw, P) = feasible_food_web(rng, web, x)

result = sample_until_converged(rng, warmup(rng, P); target_ess = 1_000)
result.diagnostics.min_ess
```

Each sample `θ` gives one community matrix `J = diag(x) A`, from `community_matrix(fw, θ)`,
consistent with the equilibrium. The fraction of them with all eigenvalues in the left
half-plane is the probability that the equilibrium is stable:

```@example foodweb
using Statistics: mean
mean(is_stable(community_matrix(fw, θ)) for X in result.chains for θ in eachcol(X))
```

The [`ranges`](@ref) of the interaction strengths show which ones the equilibrium and the
constraints nearly determine before any sampling:

```@example foodweb
lower, upper = ranges(P)
narrowest = sortperm(upper .- lower)[1:5]
[fw.coordinates[narrowest] (upper .- lower)[narrowest]]
```

## Results

On this web (four chains of 2000 points, `just example constrained_foodweb`):

| Sampler | min ESS | max R̂ | ESS/s | stable fraction |
|---|---|---|---|---|
| HitAndRun, unrounded (the 2019 version) | 18 | 2.38 | 12 | 0.720 |
| ACHR, unrounded (as in the original study) | 22 | 1.74 | 16 | 0.680 |
| CoordinateHitAndRun | 3591 | 1.002 | 5177 | 0.655 |
| BilliardWalk | 7140 | 1.001 | 6683 | 0.659 |

The methods used before this package had not converged at this length, and their stable
fractions disagree; the converged samplers agree with each other. Results from the
original study were computed with ACHR on single chains without convergence diagnostics
and should be rechecked.

`examples/foodweb_replicates.jl` runs 80 webs to 1000 effective samples each, one per
thread (see [Parallel sampling](@ref)), with a mean stable fraction of 0.711.
