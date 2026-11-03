# Sampling and convergence

## `sample`

[`sample`](@ref) implements the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl)
interface. It takes a model (a [`Warmup`](@ref) for the uniform distribution, or a
[`Target`](@ref) for a density), a sampler and a number of points, and returns one point
per column:

```@example sampling
using PolytopeSamplers, Random

P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
w = warmup(P)
X = sample(Xoshiro(1), w, BilliardWalk(), 1_000; discard_initial = 100, progress = false)
size(X)
```

- `discard_initial` drops the first steps, and `thinning` keeps every `thinning`-th point.
- Several chains run with `MCMCThreads()`, `MCMCSerial()` or `MCMCDistributed()` and come
  back as a vector of matrices.
- Passing a [`Polytope`](@ref) instead of a warmup calls `warmup` once for all chains.

```@example sampling
chains = sample(Xoshiro(1), w, BilliardWalk(), MCMCThreads(), 1_000, 4; progress = false)
length(chains), size(first(chains))
```

With [MCMCChains](https://github.com/TuringLang/MCMCChains.jl) loaded, `chain_type =
Chains` returns the chains as one `Chains` object, for its summaries, its plots with
StatsPlots and its conversion to ArviZ. Name the coordinates with `param_names`:

```@example sampling
using MCMCChains

c = sample(
    Xoshiro(1), w, BilliardWalk(), MCMCThreads(), 1_000, 4;
    chain_type = Chains, param_names = ["x₁", "x₂"], progress = false,
)
summarystats(c)
```

The `internals` section holds the statistics of each step, from `AbstractMCMC.getstats`:

| Sampler | Statistics |
|---|---|
| [`BilliardWalk`](@ref) | `n_reflections`, the reflections in the step, and `is_abandoned`, whether the trajectory reached `max_reflections` and the chain stayed put |
| [`DikinWalk`](@ref), [`VaidyaWalk`](@ref) | `is_accept`, and `acceptance_rate` so far |
| [`CRHMC`](@ref) | `is_accept`, `acceptance_rate` and `step_size` |
| [`HitAndRun`](@ref), [`CoordinateHitAndRun`](@ref) with a [`Target`](@ref) | `n_evaluations`, the log-density evaluations of the slice sampler, and `n_exhausted`, the slice updates that gave up after 200 shrinkage steps and left the point where it was |

With a [`Target`](@ref), every sampler also reports `lp`, the log density at the point.
The uniform hit-and-run methods report none.

```@example sampling
summarystats(c; sections = :internals)
```

## Diagnostics

Consecutive points of a chain are correlated, so `n` points carry less information than
`n` independent draws. [`diagnostics`](@ref) measures this for every coordinate:

```@example sampling
d = diagnostics(chains)
(; d.min_ess, d.max_rhat)
```

- `ess`: the effective sample size, the number of independent draws the chain is worth.
- `rhat`: the rank split-R̂ (Vehtari et al. 2021), which compares the chains with each
  other and with themselves, in location and in spread; values above 1.01 mean the chains
  have not converged.
- `mean`, `std` and their Monte Carlo standard errors `mcse_mean`, `mcse_std`: report
  estimates with these as error bars.

R̂ needs several chains that start in different places. [`sample`](@ref) starts each chain
at a uniform random point of the largest ball around the warmup center inside the
polytope, or at the point given by `initial_params` (one per chain with an ensemble). A
single chain can look converged while it has explored only part of the polytope, so run at
least four.

## Sampling until converged

[`sample_until_converged`](@ref) automates the usual protocol: run several chains, discard
the first half as burn-in, and lengthen them until the targets are met. Each round sets
the length from the effective samples per point of the last round, and slowly mixing
chains are thinned so that they keep at most about 20 points per effective sample.

```@example sampling
r = sample_until_converged(Xoshiro(2), w, CoordinateHitAndRun(); target_ess = 2_000)
(; r.converged, r.n, r.thinning, r.diagnostics.min_ess, r.diagnostics.max_rhat)
```

Chains start at random points of the largest ball around the warmup center inside the
polytope, each with its own random number generator seeded from `rng`, so the result does
not depend on the number of threads. When the chains have not converged by `max_n`
points, `converged` is `false` and a warning is logged.

[`mmcs`](@ref) runs multiphase Monte Carlo sampling: a one-program warmup with rounding from
samples, then billiard-walk chains until convergence.

## What R̂ catches

R̂ is the check against chains that have not mixed. In the stress tests, chains started near
the corners of a 49-dimensional simplex had R̂ above 1.05 after 100 steps for every method,
and every run that reached R̂ below 1.01 had means and standard deviations within their
standard errors of the exact values. A run that stops at `max_n` with R̂ above 1.01 should
not be used.

R̂ of each coordinate can miss chains that disagree along a direction such as `x₁ - x₂`
that is narrow compared with the coordinates' ranges. `diagnostics(chains; multivariate =
true)` adds the multivariate R̂ of Brooks & Gelman (1998), the largest R̂ of any linear
combination of the coordinates.

The multivariate R̂ grows with the number of coordinates even when the chains agree, as the
eight half-chain means of four chains estimate a covariance in every direction. For 4 chains
of 1000 independent normal draws it was 1.006 in 10 dimensions, 1.025 in 50 and 1.087 in 200.
`diagnostics` also returns `multivariate_rhat_iid`, its mean for independent draws with the
same dimension, chain length and ESS, which was within 0.002 of those values.
`sample_until_converged(...; multivariate = true)` runs until the multivariate R̂ exceeds
that reference by less than `max_rhat - 1`:

```@example sampling
r = sample_until_converged(Xoshiro(2), w, CoordinateHitAndRun(); multivariate = true)
r.diagnostics.multivariate_rhat, r.diagnostics.multivariate_rhat_iid
```

Two of four chains shifted by 0.3 standard deviations along one direction of 200 exceed the
reference by more than 0.01 (50 of 50 seeds), while every coordinate's R̂ stays below 1.01.
On a 282-variable food web the billiard walk ran to 8000 points per chain with this rule,
against 2000 for the coordinates' R̂ alone, whose chains exceeded the reference by 0.057.
