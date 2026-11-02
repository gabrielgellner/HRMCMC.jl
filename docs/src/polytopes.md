# Defining polytopes and warmup

## Polytopes

A [`Polytope`](@ref) holds bounds, inequalities and equalities:

```math
\ell \le x \le u, \qquad A x \le b, \qquad C x = d.
```

```@example polytopes
using PolytopeSamplers, Random

# The simplex x ≥ 0, x₁ + x₂ + x₃ = 1
simplex = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])

# A pentagon: bounds and three inequalities
pentagon = Polytope([0.0, 0.0], [5.5, 5.5]; A = [1.5 -1.0; 1.0 1.0; -2.0 -1.0], b = [5.0, 10.0, -3.0])
```

- Bounds may be infinite, as long as the other constraints bound the polytope.
- A coordinate with `lower[i] == upper[i]` is fixed at that value.
- Constraints of the form `L x ≥ f` are `A = -L, b = -f`.
- `isempty(P)` solves one linear program to check that `P` has points, and `x in P` checks
  a point to within `1e-8` ([`isfeasible`](@ref) takes another tolerance).

```@example polytopes
isempty(pentagon), [2.0, 3.0] in pentagon, [0.0, 0.0] in pentagon
```

## Warmup

Samplers do not move in the coordinates `x` of the polytope. [`warmup`](@ref) writes the
polytope as

```math
x = c + T z, \qquad G z \le h,
```

where `z` has one coordinate for each direction in which the polytope has width and the
rows of `G` have unit norm. Equalities, fixed coordinates and any inequalities that hold
with equality over the whole polytope disappear into `T`, so a chain can never leave the
slice they define.

```@example polytopes
w = warmup(simplex)
```

### Finding the slice: `extremes`

With `extremes = false` (the default), `warmup` solves one linear program for the largest
ball inside the polytope, within the slice of `C x = d` and the fixed coordinates. A
polytope with room for a ball of positive radius has no other equalities, and a second
linear program checks that it is bounded. When the ball has radius zero, the polytope has
equalities that are not written as equalities (for example `x₁ + x₂ ≤ 1` together with
`x₁ + x₂ ≥ 1`), and `warmup` falls back to `extremes = true`.

With `extremes = true`, `warmup` solves `2n` linear programs, minimizing and maximizing each
coordinate, and finds every equality that holds over the polytope from the extreme points.
This costs far more for large `n`: with 1000 constraints in 200 dimensions the `2n`
programs take 58 s, against 1.3 s for the whole warmup with `prune = false`.

### Redundant constraints: `prune`

A constraint is redundant when removing it leaves the polytope unchanged, as for a bound
that the other constraints never reach, or reach only at a single vertex. Each step of every
sampler costs time in proportion to the number of constraints, so with `prune = true` (the
default) `warmup` solves one linear program per finite bound and inequality and leaves the
redundant ones out of `G` and `h`. It uses the test of PolyRound: raise the right-hand side
by 1 and check whether the maximum over the polytope still meets the original. `w.polytope`
keeps every constraint, and `w.removed` counts those left out. The upper bounds of the
triangle are reached only at its vertices:

```@example polytopes
triangle = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
w = warmup(triangle)
size(w.G), w.removed
```

On a food web with 282 variables, pruning removes 502 of 796 constraints and the warmup
takes 6.4 s instead of 16.2 s, as ellipsoid rounding factors matrices of the size of the
number of constraints. The programs are slow when `A` is dense: with 200 variables and 1000
dense inequalities, all of them needed, they took 90 times as long as the rest of the
warmup. Pass `prune = false` for such polytopes.

### Rounding

Long thin polytopes slow every sampler that moves along lines. `rounding` changes the
coordinates `z` so that the polytope is close to round:

- `:ellipsoid` (default): map the maximum-volume inscribed ellipsoid to the unit ball,
  repeatedly, until it is nearly a ball (Zhang's interior-point method, as in the COBRA
  Toolbox, PolyRound and volesti).
- `:samples`: draw billiard-walk samples and map their principal axes to equal lengths,
  repeatedly, as in multiphase Monte Carlo sampling ([`mmcs`](@ref)). Ellipsoid rounding
  was faster at every size tested (10 s against 121 s with 2000 constraints in 400
  dimensions); sample rounding avoids the dense `m × m` matrices of the ellipsoid method.
  Each round runs one billiard-walk chain per thread (start Julia with `--threads`), so a
  seeded warmup depends on the number of threads.
- `:none`.

A box with sides from `1e-3` to `1e3` rounds to a cube:

```@example polytopes
sides = 10.0 .^ range(-3, 3; length = 10)
box = warmup(Polytope(zeros(10), sides))
box.h'
```

Every facet of the rounded box is at distance one from its center. Without rounding the
[`BilliardWalk`](@ref) did not converge on a 20-dimensional version of this box after
256 000 points per chain; with it, it reached 82 000 effective samples per second.

### Errors

`warmup` throws an `ArgumentError` for an empty polytope, an unbounded one, or equalities
with no solution:

```@example polytopes
try
    warmup(Polytope([0.0, 0.0], [1.0, Inf]))
catch e
    e
end
```
