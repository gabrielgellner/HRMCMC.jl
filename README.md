# PolytopeSamplers.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://gabrielgellner.github.io/PolytopeSamplers.jl/dev/)
[![CI](https://github.com/gabrielgellner/PolytopeSamplers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/gabrielgellner/PolytopeSamplers.jl/actions/workflows/CI.yml)

Markov chain Monte Carlo sampling of the uniform distribution, or a density, on a convex
polytope: the billiard walk, coordinate hit-and-run with rounding (CHRR), hit-and-run,
artificial centering hit-and-run (ACHR), the Dikin and Vaidya walks, and constrained
Riemannian Hamiltonian Monte Carlo (CRHMC), with convergence diagnostics. Linear programs
for the warmup are solved with [HiGHS](https://highs.dev). The package was called HRMCMC.jl
until 2026.

```julia
using Pkg
Pkg.add(url = "https://github.com/gabrielgellner/PolytopeSamplers.jl")
```

## Citing

Cite PolytopeSamplers.jl with the version you used; `CITATION.cff` holds the metadata, and
GitHub's "Cite this repository" button formats it.

## Documentation

The [manual](https://gabrielgellner.github.io/PolytopeSamplers.jl/dev/) covers defining
polytopes, sampling and convergence, choosing a sampler (with benchmarks), non-uniform
targets, parallel sampling, how the samplers are tested, and the full API. Build it locally
with `just docs`.

## Usage

A `Polytope` is the set of `x` with `lower .<= x .<= upper`, `A * x .<= b` and `C * x .== d`.
The simplest use samples until the chains have converged:

```julia
using PolytopeSamplers, Random

# The triangle x ≥ 0, x1 + x2 ≤ 1
P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])

r = mmcs(Xoshiro(1), P; target_ess = 1_000)
r.converged, r.n                      # converged after r.n points per chain
r.diagnostics.mean, r.diagnostics.mcse_mean
X = reduce(hcat, r.chains)            # one point per column
```

`mmcs` is multiphase Monte Carlo sampling (Chalkis et al. 2021): a linear program finds
the largest ball in the polytope, billiard-walk samples round it, and four billiard-walk
chains run, doubling in length, until the smallest effective sample size reaches
`target_ess` and R̂ is below 1.01. Rounding from samples avoids the dense `m × m`
factorizations of ellipsoid rounding, but up to 2000 constraints it was the slower of the
two (121 s against 10 s with 2000 constraints in 400 dimensions), so
`sample_until_converged(rng, warmup(P))` is the faster route there.

The steps can also be run separately:

```julia
w = warmup(P)                       # prune redundant rows, find the slice, round
r = sample_until_converged(Xoshiro(1), w, BilliardWalk(); target_ess = 1_000)

X = sample(Xoshiro(1), w, BilliardWalk(), 10_000; discard_initial = 1_000)
chains = sample(Xoshiro(1), w, BilliardWalk(), MCMCThreads(), 2_000, 4)
d = diagnostics(chains)
d.min_ess, d.max_rhat     # aim for max_rhat < 1.01
```

`warmup` prepares the polytope:

- `extremes = false` (default) solves one linear program for the largest inscribed ball in
  the slice of `C * x .== d` and the fixed coordinates, and one more to check that the
  polytope is bounded. A polytope with room for a ball has no other equalities; when the
  ball has radius zero, one more program finds the inequalities that hold with equality,
  and only a polytope thinner than `atol` falls back to `extremes = true`. That solves `2n`
  linear programs for the extremes of each coordinate and finds the equalities that hold
  throughout from them. With 1000 dense constraints in 200 dimensions and
  `prune = false`, warmup and sampling to 1000 ESS take 4 s with the default and 59 s with
  `extremes = true`.
- `prune = true` (default) solves one linear program per bound and inequality and drops
  those that do not change the polytope, which makes every step cheaper: 502 of 796
  constraints on a 282-variable food web. Pass `prune = false` when `A` is dense, where
  these programs take far longer than the rest of the warmup.
- `rounding = :ellipsoid` (default) maps the maximum-volume inscribed ellipsoid to the unit
  ball until the polytope is close to round; `:samples` rounds from billiard-walk samples;
  `:none` skips rounding.

`sample` implements the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl)
interface; `diagnostics` reports ESS, R̂, means and standard deviations with their Monte
Carlo standard errors.

### Samplers

- `BilliardWalk()` (Gryazina & Polyak 2014, in the accelerated form of Chalkis et al. 2021)
  moves along a random direction for a random length, reflecting off the facets. It needs a
  few steps per effective sample in every dimension tested (10 to 200) and is the
  recommended method.
- `CoordinateHitAndRun()` on the rounded warmup is CHRR (Haraldsdóttir et al. 2017), the
  COBRA Toolbox default. Each step makes one coordinate move per dimension at `O(m)` each,
  in cyclic order by default; moves per effective sample grow as the square of the
  dimension.
- `DikinWalk()` (Kannan & Narayanan 2012) and `VaidyaWalk()` (Chen et al. 2018) propose
  Gaussian moves shaped by the log-barrier and volumetric-barrier ellipsoids at the current
  point and accept them by a Metropolis filter. They are affine-invariant, so they need no
  rounding, and carry polynomial mixing guarantees. In practice they are the slowest
  methods on rounded polytopes: with 10 dimensions and 100 to 2000 constraints they needed
  360 to 490 steps per effective sample, and reached 50 to 450 effective samples per second
  where the billiard walk reached 6500 to 44 000. Use them as independent checks, and when
  rounding is not available: on a 20-dimensional box with sides from 1e-3 to 1e3, the
  Dikin walk reaches the same rate with or without rounding, while the billiard walk does
  not converge without it.

  | Box with sides 1e-3 to 1e3 | BilliardWalk | DikinWalk |
  |---|---|---|
  | `rounding = :none` | R̂ 2.66 after 256 000 points per chain | 225 ESS/s |
  | `rounding = :ellipsoid` | 82 689 ESS/s | 218 ESS/s |
- `CRHMC()` is constrained Riemannian Hamiltonian Monte Carlo (Kook et al. 2022): Hamiltonian
  dynamics in the metric of the log barrier, integrated by the implicit midpoint rule, with
  a Metropolis filter and an adaptive step size. Its steps per effective sample grow slowly
  with dimension (15 to 23 on simplices of 10 to 80 dimensions), but each step factors the
  dense `k × k` metric, so on the problems tested it reached 2 to 4500 effective samples per
  second where the billiard walk reached 560 to 150 000. It needs the default rounding on
  badly scaled polytopes.
- `HitAndRun()` draws directions uniformly from the sphere. `ACHR()` draws them from the
  running mean to an extreme point of each coordinate, solving the `2n` linear programs for
  them on first use; it is only approximately Markov and the slowest method on rounded
  polytopes.

On the 63-dimensional food web polytope of the example (four chains of 2000 points):

| Sampler | min ESS | max R̂ | ESS/s |
|---|---|---|---|
| HitAndRun, unrounded (the 2019 version) | 19 | 2.03 | 17 |
| CoordinateHitAndRun | 1464 | 1.005 | 2210 |
| BilliardWalk | 7262 | 1.001 | 6690 |

Estimates agree with hopsy 1.7 on the same polytopes within their Monte Carlo standard
errors.

### Non-uniform targets

`Target(P, ℓ)` is the distribution with density proportional to `exp(ℓ(x))` on the
polytope, for a function `ℓ` or any object implementing
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl):

```julia
# Dirichlet(2, 3, 4) on the simplex x ≥ 0, sum(x) = 1
P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])
α = [2.0, 3.0, 4.0]
t = Target(P, x -> sum((α .- 1) .* log.(x)); gradient = (g, x) -> (g .= (α .- 1) ./ x))
r = sample_until_converged(Xoshiro(1), t, CoordinateHitAndRun())
```

- `HitAndRun()` and `CoordinateHitAndRun()` draw each point on a chord from the density
  along it by slice sampling (Neal 2003), which is exact for any `ℓ` and needs no tuning.
- `DikinWalk()` and `VaidyaWalk()` add `ℓ` to their Metropolis ratio.
- `CRHMC()` adds `-ℓ` to its potential and needs the gradient of `ℓ`.
- `BilliardWalk()` and `ACHR()` sample only the uniform distribution.

### Threads

Start Julia with threads (`julia --threads=auto`) and two kinds of work run in parallel:

- Chains: `sample_until_converged`, `mmcs` and `sample(..., MCMCThreads(), n, nchains)`
  run each chain on its own thread. Each chain has its own generator seeded from `rng`, so
  results are identical for any number of threads, and billiard chains share one copy of
  `G Gᵀ`.
- Independent problems: `warmup` builds its own JuMP model and every function takes an
  explicit `rng`, so `Threads.@threads` over polytopes is safe;
  `examples/foodweb_replicates.jl` runs one food web per thread.

Set `BLAS.set_num_threads(1)` when Julia threads keep every core busy. Measured on a
10-core Apple machine (4 performance and 6 efficiency cores):

| Work | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| 80 food web replicates (63 dimensions, to 1000 ESS each) | 42.0 s | 17.9 s | 10.4 s |
| 4 billiard chains of 4000 steps, 200 dimensions | 5.3 s | 1.6 s | |

Chains in one run gain little past the number of performance cores, as the slowest chain
sets the time; batches of independent problems keep gaining.

### Other details

- `isempty(P)` checks feasibility with one linear program. `warmup` throws an
  `ArgumentError` for empty or unbounded polytopes.
- Every function takes an optional leading `rng`. Chains have their own generators seeded
  from it, so results do not depend on the number of threads. Each `warmup` builds its own
  JuMP model, so separate polytopes can be set up in parallel tasks. Pass `optimizer` to use
  a different JuMP solver.

`examples/constrained_foodweb.jl` samples the interaction strengths of a food web consistent
with a given equilibrium, the problem of
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs), with the webs and
the polytope from [FoodWebs.jl](https://github.com/gabrielgellner/FoodWebs.jl). Run it with
`just example constrained_foodweb`; `just example foodweb_replicates` runs 80 webs in
parallel.

## Changes from the 2019 version (HRMCMC.jl)

- `ACHRSampler(problem)` and `hr_sample!(sampler, n)` are replaced by
  `sample(rng, warmup(P), ACHR(), n)`. The problem `E * α + r = 0, L * α >= 0`
  is `Polytope(lower, upper; C = E, d = -r, A = -L, b = zeros(size(L, 1)))`.
- `n` is now the number of points returned; the chain runs
  `discard_initial + (n - 1) * thinning + 1` steps.
- Plain hit-and-run drew its directions from the positive orthant of the nullspace, as
  `rand` replaced `randn`; it now draws them uniformly from the sphere.
- The step bounded each coordinate by `(1 - tol) * bound`, which loosens negative bounds; it
  now clamps slacks at zero.
- GLPK and Gurobi are replaced by HiGHS, and RecursiveArrayTools, Distributions and
  Parameters are no longer dependencies.

## Development

Tasks are run with [just](https://github.com/casey/just):

```sh
just setup      # install Runic and JETLS as Julia apps, instantiate environments
just fmt        # format with Runic
just ci         # formatting check, JETLS lint and tests; run before every push and merge
just docs       # build the documentation and open it
just stress     # convergence stress tests and mixing-cost scaling (about 10 minutes)
```

## License

GPL-3.0-or-later, see `LICENSE.md`. Parts of the package adapt code from other open-source
samplers; each such file names its source and license.
