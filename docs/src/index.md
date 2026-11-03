# PolytopeSamplers.jl

PolytopeSamplers.jl draws points from the uniform distribution, or from a density, on a
convex polytope

```math
P = \{x \in \mathbb{R}^n : \ell \le x \le u,\ A x \le b,\ C x = d\}
```

by Markov chain Monte Carlo, and reports how far the results can be trusted: effective
sample sizes, R̂ and Monte Carlo standard errors for every coordinate.

It offers the billiard walk, coordinate hit-and-run with rounding (CHRR), hit-and-run,
artificial centering hit-and-run (ACHR), the Dikin and Vaidya walks, and constrained
Riemannian Hamiltonian Monte Carlo (CRHMC). Each is calibrated against exact answers and
stress-tested on hard geometry; see [How the samplers are tested](@ref).

The package began as HRMCMC.jl, a sampler for the interaction strengths of food webs
consistent with an equilibrium ([Example: constrained food webs](@ref)).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/gabrielgellner/PolytopeSamplers.jl")
```

The package needs Julia 1.13. Linear programs are solved with
[HiGHS](https://highs.dev) through JuMP.

## Quick start

Define the polytope, then sample until the chains have converged:

```@example quickstart
using PolytopeSamplers, Random

# The triangle x ≥ 0, x₁ + x₂ ≤ 1
P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])

r = sample_until_converged(Xoshiro(1), warmup(P); target_ess = 1_000)
r.converged, r.n
```

`sample_until_converged` runs four [`BilliardWalk`](@ref) chains, lengthening them,
until every coordinate has an effective sample size of at least `target_ess` and R̂ below
1.01. The estimates come with their Monte Carlo standard errors:

```@example quickstart
d = r.diagnostics
d.mean, d.mcse_mean
```

The exact means are `1/3`. The points themselves are the columns of the chains:

```@example quickstart
X = reduce(hcat, r.chains)
size(X)
```

## Where to go next

- [Defining polytopes and warmup](@ref) covers constraints, equalities and what `warmup`
  does before sampling.
- [Sampling and convergence](@ref) covers `sample`, multiple chains and the diagnostics.
- [Choosing a sampler](@ref) compares the methods with benchmarks.
- [Non-uniform targets](@ref) samples densities `exp(ℓ(x))` on a polytope.
- [Parallel sampling](@ref) runs chains, or many polytopes, on threads.
- [Volume](@ref) estimates the volume of a polytope with a standard error.
