# Example: constrained food webs

The package began as the sampler for
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs): finding the
interaction strengths of a food web that are consistent with an observed equilibrium.
[FoodWebs.jl](https://github.com/gabrielgellner/FoodWebs.jl) builds the webs and the
polytope, and its PolytopeSamplers.jl extension samples it. The repository's
`examples/foodweb_study.jl` holds the settings of the original study.

## The polytope

Lotka–Volterra dynamics `dxᵢ/dt = xᵢ (rᵢ + Σⱼ aᵢⱼ xⱼ)` have an equilibrium at the biomasses
`x` when `rᵢ + Σⱼ aᵢⱼ xⱼ = 0` for every species. For a given web, biomasses and growth rates,
the magnitudes `αᵢⱼ = |aᵢⱼ|` of the links that exist form a polytope:

- one equality per species, `Σⱼ sᵢⱼ xⱼ αᵢⱼ + rᵢ = 0`, with `sᵢⱼ` the sign of the interaction
  (−1 for predation on `i` and for self-regulation, +1 for consumption by `i`);
- bounds `lb ≤ αᵢⱼ ≤ ub`;
- optionally, for each predator `j` of prey `i`, `αᵢⱼ ≥ αⱼᵢ`: a predator gains less from a
  unit of prey than the prey loses.

FoodWebs.jl's `FeasibleCommunities(web, x; rates, ...)` holds these constraints, and
`Polytope(fc)` converts them for this package. The constrained food web studies used fixed
growth rates, self-regulation for every species but the top predators, the energetic
constraint, bounds `[0, 5]`, and biomasses falling linearly with trophic level. A 15-species web gives 78 interaction strengths in a
63-dimensional slice.

## Sampling

```julia
include("examples/foodweb_study.jl")
using FoodWebs, PolytopeSamplers, Random

rng = Xoshiro(2026)
web = generalized_cascade_network(rng, 15, 0.15; basal_paths = true)
x = pyramid_biomass(rng, web)
(; communities) = feasible_growth_rates(rng, web, x; constraints...)

result = sample_until_converged(rng, warmup(Polytope(communities)); target_ess = 1_000)
```

`feasible_growth_rates` draws growth rates until the polytope is nonempty, as in the
original studies. Each sample `θ` gives one community matrix `J = diag(x) A`, from
`community_matrix(communities, θ)`, consistent with the equilibrium. The fraction of them
with all eigenvalues in the left half-plane is the probability that the equilibrium is
stable. FoodWebs.jl's `sample_communities` runs the sampling and returns `(A, r, J)` for
each point.

The [`ranges`](@ref) of the interaction strengths show which ones the equilibrium and the
constraints nearly determine before any sampling:

```julia
P = Polytope(communities)
lower, upper = ranges(P)
narrowest = sortperm(upper .- lower)[1:5]   # indices of the five narrowest ranges
```

## Results

On this web (four chains of 2000 points, `just example constrained_foodweb`):

| Sampler | min ESS | max R̂ | ESS/s | stable fraction |
|---|---|---|---|---|
| HitAndRun, unrounded (the 2019 version) | 18 | 2.35 | 19 | 0.490 |
| ACHR, unrounded (as in the original study) | 21 | 1.67 | 38 | 0.678 |
| CoordinateHitAndRun | 2666 | 1.003 | 4637 | 0.714 |
| BilliardWalk | 6874 | 1.001 | 8387 | 0.703 |

The methods used before this package had not converged at this length, and their stable
fractions disagree; the converged samplers agree with each other. Results from the
original study were computed with ACHR on single chains without convergence diagnostics
and should be rechecked.

`examples/foodweb_replicates.jl` runs 80 webs to 1000 effective samples each, one per
thread (see [Parallel sampling](@ref)), with a mean stable fraction of 0.686.
