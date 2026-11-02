# Non-uniform targets

[`Target`](@ref) is the distribution on a polytope with density proportional to
`exp(ℓ(x))`. `ℓ` is a function of the vector `x`, or any object implementing
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl).

## A Dirichlet distribution on the simplex

The Dirichlet distribution with parameters `α` has log density `Σ (αᵢ - 1) log xᵢ` on the
simplex, and Beta marginals with means `αᵢ / Σα`:

```@example targets
using PolytopeSamplers, Random

α = [2.0, 3.0, 4.0]
P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])
t = Target(P, x -> sum((α .- 1) .* log.(x)); gradient = (g, x) -> (g .= (α .- 1) ./ x))

r = sample_until_converged(Xoshiro(1), t, CoordinateHitAndRun(); target_ess = 2_000)
r.diagnostics.mean, α ./ sum(α)
```

## Gaussian and exponential targets

[`GaussianTarget`](@ref) restricts a Gaussian with mean `μ` and covariance `Σ` to the
polytope, and [`ExponentialTarget`](@ref) has density proportional to `exp(-cᵀx)`. On a
chord their densities are a truncated normal and a truncated exponential, which
[`HitAndRun`](@ref) and [`CoordinateHitAndRun`](@ref) draw exactly, by the rejection
samplers of Robert (1995), without evaluating `ℓ`:

```@example targets
box = Polytope(fill(-1.0, 2), fill(1.0, 2))
t = GaussianTarget(box, [0.5, 0.0], [1.0 0.8; 0.8 1.0])
r = sample_until_converged(Xoshiro(3), t, CoordinateHitAndRun(); target_ess = 2_000)
r.diagnostics.mean, r.diagnostics.mcse_mean
```

With `whiten = true`, `GaussianTarget` stretches the chains' coordinates in each direction
where the Gaussian is narrower than the polytope, as hops samples truncated Gaussians in
coordinates whitened by `Σ`; directions where the Gaussian is wider are left as they are.
Coordinate hit-and-run, smallest effective sample size of 4 chains (2000 sweeps each on the
boxes, 20 000 on the food web):

| Gaussian | not whitened | whitened | fully whitened (hops) |
|---|---|---|---|
| correlation 0.99 on `[-10, 10]²` | 67 | 7 797 | 7 797 |
| σ 100 and 0.05 at 45° on `[0, 10]²` | 19 | 7 765 | 7 765 |
| σ 50 and 20 on `[0, 1]²` (wider than the box) | 7 847 | 7 847 | 6 947 |
| σ from 0.01 to 10 on `[0, 1]¹⁰` | 31 | 4 342 | 4 265 |
| food web, 282 variables, σ a fifth of each coordinate's range | 1 019 | 729 | 770 |

Whitening helps a correlated Gaussian that lies mostly inside the polytope. On the food web,
where the constraints shape the target more than the Gaussian does, stretching the
coordinates elongates the polytope and slows the chains, so it is off by default.

On a 10-dimensional box, coordinate hit-and-run gave 6 times the effective samples per
second of slice sampling on the same Gaussian, and 13 times on an exponential density;
hit-and-run gave 1.5 times on the Gaussian and the same on the exponential, where computing
the chord dominates its step. Both targets are LogDensityProblems objects with gradients, so
[`DikinWalk`](@ref), [`VaidyaWalk`](@ref) and [`CRHMC`](@ref) take them too.

Robert, C. P. (1995). Simulation of truncated normal variables. Statistics and Computing 5,
121–125. <https://doi.org/10.1007/BF00143942>

## Which samplers support densities

- [`HitAndRun`](@ref) and [`CoordinateHitAndRun`](@ref) draw each point on a chord from the
  density along it by slice sampling with shrinkage (Neal 2003), starting from the whole
  chord, or exactly for [`GaussianTarget`](@ref) and [`ExponentialTarget`](@ref). Slice
  sampling is exact for any `ℓ` and needs no tuning; it calls `ℓ` a few times per move.
- [`DikinWalk`](@ref) and [`VaidyaWalk`](@ref) add `ℓ(z') - ℓ(z)` to their Metropolis ratio.
- [`CRHMC`](@ref) adds `-ℓ` to its potential and needs the gradient of `ℓ`: pass
  `gradient = (g, x) -> ...`, which fills `g` with `∇ℓ(x)`, or a LogDensityProblems object
  of order 1, for example one built with LogDensityProblemsAD.jl.
- [`BilliardWalk`](@ref) and [`ACHR`](@ref) sample only the uniform distribution and throw
  an `ArgumentError` for a target.

The default method of [`sample_until_converged`](@ref) for a target is
[`HitAndRun`](@ref); [`CoordinateHitAndRun`](@ref) was faster on the targets tested.

## A LogDensityProblems object

```@example targets
using LogDensityProblems

struct Gaussian
    μ::Vector{Float64}
end
LogDensityProblems.dimension(g::Gaussian) = length(g.μ)
LogDensityProblems.capabilities(::Type{Gaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(g::Gaussian, x) = -sum(abs2, x .- g.μ) / 2
LogDensityProblems.logdensity_and_gradient(g::Gaussian, x) = (-sum(abs2, x .- g.μ) / 2, g.μ .- x)

box = Polytope(fill(-1.0, 2), fill(1.0, 2))
r = sample_until_converged(Xoshiro(2), Target(box, Gaussian([0.5, 0.0])), CRHMC(); target_ess = 1_000)
r.diagnostics.mean, r.diagnostics.mcse_mean
```

Truncated to `[-1, 1]`, the exact means are `0.5 - (φ(0.5) - φ(-1.5)) / (Φ(0.5) - Φ(-1.5)) ≈ 0.144`
and 0, for the standard normal density `φ` and distribution function `Φ`.

## Densities that diverge at a face

A gamma density with shape below 1, `(k - 1) log(xᵢ) - xᵢ/θ`, is infinite at the face
`xᵢ = 0`. Samplers move in the coordinates of the warmup, and `x = center + T z` resolves a
coordinate only to about `1e-16` of its scale, so chains reach points where a coordinate
rounds to 0 or just below. `ℓ` is evaluated at `x` clamped to the polytope's bounds, so
`log(xᵢ)` stays defined, and a log density of `+Inf` or `NaN` counts as outside the support:
those points have probability zero, and a chain that accepted one would never move again.

With these, hit-and-run, coordinate hit-and-run, CRHMC and the barrier walks match the
moments of truncated gamma densities with shape 1/4 on a box. Products of such densities on
a polytope of much lower dimension than its number of variables concentrate near vertices,
where many coordinates are 0 together; chains then settle near different vertices, and R̂
stays high however long they run.

## Calibration

Over 20 seeds, on a Dirichlet distribution on a simplex, truncated exponentials on a box
with rates 1, 5 and 20, and a Gaussian with correlation 0.8, every supported sampler had
z-scores of its means and standard deviations against the exact values with root mean
square between 0.73 and 1.15.

`ℓ` is called from every chain, on separate threads when Julia has them, so it must be
safe to call concurrently.
