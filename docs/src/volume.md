# Volume

[`volume`](@ref) estimates the volume of a polytope with a standard error. For a polytope
with equalities it gives the volume of the slice the polytope lies on, in that slice's own
dimension: the simplex `x ≥ 0, Σx = 1` in 3 variables has area `√3 / 2`.

```@example volume
using PolytopeSamplers, Random

P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])
v = volume(Xoshiro(1), P)
v.volume, v.log_stderr, sqrt(3) / 2
```

`log_stderr` is the standard error of `log_volume`, close to the relative standard error of
`volume`. Volumes in high dimensions under- or overflow, so work with `log_volume`:

```@example volume
cube = Polytope(zeros(50), fill(2.0, 50))
v = volume(Xoshiro(2), cube; error = 0.05)
v.log_volume, v.log_stderr, 50 * log(2)
```

## Method

The estimate multiplies volume ratios (multiphase Monte Carlo with cooling balls, Chalkis,
Emiris & Fisikopoulos 2023). In the rounded coordinates of the warmup, balls centered at the
warmup center shrink from one holding the polytope to one of which 10–15% lies inside it.
Each body `K ∩ B(ρᵢ₊₁)` holds about an eighth of `K ∩ B(ρᵢ)`, so a 63-dimensional food web
polytope needs about 10 phases.

- The share of the last ball inside the polytope comes from independent uniform points of
  the ball.
- Each ratio `vol(K ∩ B(ρᵢ₊₁)) / vol(K ∩ B(ρᵢ))` comes from coordinate hit-and-run chains
  on `K ∩ B(ρᵢ)`, continued from the chains whose points chose `ρᵢ₊₁`. The ratio is the
  share of the chains' states inside `B(ρᵢ₊₁)` after each coordinate move, as volesti
  counts them. The chains lengthen until R̂ is below 1.01 and the ratio's Monte Carlo
  standard error meets its share of `error`, each round extrapolating the length from the
  last one's standard error.

The phases are independent, so the variance of the log volume is the sum of the squared
relative standard errors of the ratios. As in volesti, the share of the last ball, whose
independent points are cheap, gets `error² / 4L` of that variance for `L` phases, and the
ratios from chains split the rest.

## Calibration

Over 50 seeds on a 40-dimensional cube, a 20-dimensional simplex, a 14-dimensional simplex
slice and a 60-dimensional simplex (10 phases), the z-scores of the log volume had root mean
square between 0.86 and 1.09, and 47 to 49 of the 50 estimates were within two standard
errors. The mean z-score was between -0.03 and 0.23: the logarithm of a ratio estimate is
biased by about half its squared relative error, small next to the standard error.

## Food webs

For [constrained food webs](@ref "Example: constrained food webs") with fixed growth rates,
the volume of the polytope of interaction strengths measures how many communities share the
equilibrium. Under a uniform prior on the growth rates and interaction strengths jointly,
each growth-rate vector is weighted by this volume.
