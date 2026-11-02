# Choosing a sampler

Use [`BilliardWalk`](@ref) unless you sample a density, in which case use
[`CoordinateHitAndRun`](@ref) or, with a gradient, [`CRHMC`](@ref). The other methods are
independent checks. The tables below give the measurements behind this advice; all ran on
one thread of a 10-core Apple machine, to 1000 effective samples on rounded polytopes
unless noted.

| Sampler | Moves per step | Uniform | Densities | Use it for |
|---|---|---|---|---|
| [`BilliardWalk`](@ref) | a trajectory reflecting off facets | yes | no | the default |
| [`CoordinateHitAndRun`](@ref) | `k` moves, one per coordinate (CHRR) | yes | slice sampling | densities; very cheap moves |
| [`HitAndRun`](@ref) | one move along a random direction | yes | slice sampling | a simple reference |
| [`CRHMC`](@ref) | Hamiltonian dynamics in the barrier metric | yes | needs the gradient | smooth densities; corners |
| [`DikinWalk`](@ref), [`VaidyaWalk`](@ref) | a Metropolis-filtered barrier step | yes | yes | independent checks; no rounding |
| [`ACHR`](@ref) | direction from the running mean | yes | no | comparison with COBRA Toolbox results |

A step of `CoordinateHitAndRun` sweeps all `k` coordinates, while a step of `HitAndRun` is
one move, as in volesti, hops and the COBRA Toolbox. The two cost about the same per step,
but `HitAndRun` needs about `k` times as many steps, so the same `n`, `max_n` or `thinning`
does not give the two the same amount of mixing.

## Speed

Effective samples per second:

| Polytope | Billiard | CHRR | CRHMC | Dikin |
|---|---|---|---|---|
| simplex, 10 dimensions | 150 608 | 87 108 | 4 527 | |
| simplex, 80 dimensions | 4 348 | 773 | 61 | |
| random, 100 dimensions, 500 constraints | 2 214 | 385 | 14 | |
| random, 200 dimensions, 1000 constraints | 559 | 51 | 2 | |
| 10 dimensions, 100 constraints | 44 027 | 23 995 | | 443 |
| 10 dimensions, 2000 constraints | 6 538 | 41 287 | | 50 |

Steps per effective sample as the dimension grows from 10 to 80 (simplices):

| Sampler | Cost per effective sample | Growth |
|---|---|---|
| [`BilliardWalk`](@ref) | 3.6 → 5.6 steps | `k^0.21` |
| [`HitAndRun`](@ref) | 133 → 452 steps | `k^0.6` |
| [`CoordinateHitAndRun`](@ref) | 81 → 4690 coordinate moves | `k^2.1`, as published |
| [`DikinWalk`](@ref), [`VaidyaWalk`](@ref) | 340 → 470 steps | `k^0.15` |
| [`CRHMC`](@ref) | 15 → 23 steps | slow, but a step factors a `k × k` matrix |

## Robustness

- **Corners.** From starting points near the corners of a 49-dimensional simplex, the
  billiard walk converged in 1000 steps and CRHMC in 10 000; hit-and-run, CHRR, ACHR and the
  barrier walks had not converged after 10 000.
- **Vertices.** A CHRR chain started exactly at a vertex of a simplex never moves, as no
  coordinate axis points into the polytope there. The samplers start chains in the
  interior.
- **Rounding.** Without rounding, the billiard walk, hit-and-run and CHRR stall on long thin
  polytopes, and CRHMC's metric becomes too ill-conditioned for double precision near
  facets. The Dikin and Vaidya walks behave the same with or without rounding. Keep the
  default rounding.
- **ACHR** is only approximately Markov: its directions depend on the history of the chain.

## Parameters

- [`BilliardWalk`](@ref)`(; trajectory_length, max_reflections)`: the mean trajectory length
  defaults to `2√k r` for inscribed radius `r` (about 1 after rounding).
- [`CoordinateHitAndRun`](@ref)`(; moves, scan)`: moves per step (default `k`) and the order
  of coordinates (`:cyclic`, the default, or `:random`).
- [`DikinWalk`](@ref)`(; r)`, [`VaidyaWalk`](@ref)`(; r)`: the proposal radius, 0.5 and 0.7 by
  default, giving acceptance rates of 0.5 to 0.65.
- [`CRHMC`](@ref)`(; initial_step_size, effective_step_size, max_ode_steps, implicit_tol)`:
  the step size adapts by itself.
