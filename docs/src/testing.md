# How the samplers are tested

A sampler can run, produce plausible points and still be wrong. The test suite checks
every sampler against answers known in advance, and against the error bars it reports.

## Calibration

For each sampler and test polytope, means and standard deviations are compared with their
exact values in units of the Monte Carlo standard errors from [`diagnostics`](@ref). For a
correct sampler with honest error bars these z-scores have root mean square near 1. Over
20 to 40 seeds on a pentagon, cubes, a skinny cube and simplices:

| Sampler | RMS of z-scores |
|---|---|
| BilliardWalk, CoordinateHitAndRun, HitAndRun, ACHR | 0.89–1.11 |
| DikinWalk, VaidyaWalk | 0.94–1.11 |
| CRHMC | 0.71–1.17 |
| independent draws, for reference | 1.00–1.02 |

The same check on non-uniform targets (Dirichlet, truncated exponentials, a correlated
Gaussian) gave 0.73–1.15 for every supported sampler.

## Test suites

- `test/` (`just ci`): exact moments on squares, simplices, a pentagon and cubes; checks
  adapted from volesti, hopsy, hops and PolyRound; reproducibility; type stability and zero
  allocations of every step (JET and AllocCheck); gradients against finite differences.
- `test/stress.jl` (`just stress`): R̂ flags chains started near corners; runs lengthened until
  R̂ < 1.01 give correct moments on a box of aspect ratio `10⁶`, 560 redundant constraints,
  implicit equalities, a 256-facet cross-polytope, the Birkhoff polytope and a
  200-dimensional simplex; 100 000 steps of every sampler stay inside four polytopes; the
  growth of mixing cost with dimension matches the literature.

## Cross-checks

- hopsy 1.7, on the same polytopes: estimates agree within their standard errors, and CHRR
  effective samples per move are within a factor 1.1 to 1.7, explained by hopsy's cyclic
  coordinate order and its rounding basis.
- Two converged samplers with different mechanisms agree on polytopes with no exact
  answer.

## Problems the tests found

- Output bundling copied the sample matrix once per point (quadratic in the number of
  samples).
- A first CRHMC integrator that reused the start-of-step factorization failed to converge
  on 2% of steps, mostly near facets; the resulting rejections biased standard deviations
  4 standard errors low. Refactoring during slow integrations removed the bias.
- polytopewalk draws its Dikin and Vaidya proposals with the lower Cholesky factor, which
  has the wrong covariance for its Metropolis ratio; on a pentagon that version put means
  9 to 13 standard errors from the truth.
- A boundedness check based only on Stiemke's theorem missed unbounded strips such as
  `0 ≤ x₁ - x₂ ≤ 1`; it now also requires full column rank.
