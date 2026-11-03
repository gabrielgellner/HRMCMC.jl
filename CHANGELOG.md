# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-11-03

### Documentation

- **warmup**: Record how the preprocessing compares with the references after the audit (#85)
- **rounding**: Report the cost and mixing of barrier rounding on food webs (#48)
- **crhmc**: Record the measured effect of the PolytopeSamplerMatlab safeguards (#51)

### Features

- **rounding**: Add rounding by the Dikin ellipsoid at the analytic center, rounding = :barrier (#48)
- **density**: Sample Gaussian targets on unbounded polyhedra with hit-and-run (#93)

### Performance

- **barrier**: Compute the forward term of the Metropolis ratio from the Gaussian draw, without a triangular multiply (#55)

## [0.5.0] - 2026-11-03

Results of `sample_until_converged` and `volume` differ from 0.4.0 for the same seed. `sample_until_converged` returns a new field `thinning`, and slowly mixing runs keep every `thinning`-th step. `volume` defaults to `n = 500`. `n` below 10 now throws an `ArgumentError`.

### Bug Fixes

- **volume**: Double a phase when its standard error is NaN, and state the first round length (#54)
- **diagnostics**: Require 10 points per chain, as shorter chains gave NaN ESS and sample_until_converged reported convergence

### Documentation

- **volume**: Report the calibration with a 10-phase simplex (#54)
- State how the multivariate R̂ grows with dimension and when volume without rounding fails

### Features

- **adaptive**: Set run lengths from the ESS rate, thin stored chains to bound memory, and add a multivariate R̂ (#53)

### Miscellaneous

- **examples**: Pin FoodWebs v0.9.2, which allows PolytopeSamplers 0.4

### Performance

- **volume**: Count every move, continue the pilot chains, split the error as volesti, and test ball points in batches (#54)

## [0.4.0] - 2026-11-02

### Breaking Changes

- **density**: Turn whitening of `GaussianTarget` off by default again (`whiten = false`, as before 0.3.0): on a 282-variable food web it lowered the effective sample size of coordinate hit-and-run from 1019 to 729; pass `whiten = true` for a correlated Gaussian that lies mostly inside the polytope, where it raised it from 67 to 7797 (#52)

## [0.3.0] - 2026-11-02

### Breaking Changes

- **density**: Whiten `GaussianTarget` by default (`whiten = true`): the chains move in coordinates stretched where the Gaussian is narrower than the polytope, so a seed gives different chains from the same distribution; `whiten = false` keeps the polytope's coordinates (#52)

### Features

- **warmup**: Find the inequalities that hold with equality by one linear program (Freund, Roundy & Todd 1985) instead of the 2n coordinate programs (#46)
- **warmup**: Time out every solver through `TimeLimitSec`, warn when a solver other than HiGHS keeps its default tolerances, and warn for polytopes wider than 1e9 (#88)

### Performance

- **rounding**: Factor the ellipsoid's Newton system by Cholesky, cap its iterations as PolyRound does, and start rounds from the last ellipsoid's center; rounding the 1082-variable food web takes 14 s instead of 33 s (#47)
- **sampler**: Compute x only for the points that sample keeps, and keep ACHR's running mean out of the other samplers (#50)
- **density**: Whiten Gaussian targets, evaluate slice densities in O(n) instead of O(nk), and keep density steps allocation free; a coordinate hit-and-run step with a density on a 282-variable food web takes 129 μs instead of 2657 μs (#52)

### Bug Fixes

- **sampler**: Give ACHR at least 2k warmup points spanning the polytope, as the COBRA Toolbox does; with few extreme points its means were biased while R̂ looked converged (#50)
- **rounding**: Keep :samples rounding in units of the inscribed radius, so that the body does not shrink below the solver's tolerance

## [0.2.0] - 2026-11-02

### Breaking Changes

- **warmup**: Remove redundant bounds and inequalities by default (`prune = true`); pass `prune = false` to keep them, as for polytopes with a dense `A`, where the linear programs are slow (#35)
- **warmup**: Default `atol` to `1e-8` (was `1e-6`), so that directions as thin as `1e-7` are sampled instead of fixed (#31)
- **diagnostics**: Report the rank-normalized R̂, which also compares the spread of the chains; `sample_until_converged` stops on it (#44)
- **density**: Throw when a chain starts where the density is zero, infinite or NaN (#70)
- **billiard**: Set the default trajectory length on unrounded warmups by burn-in instead of `2√k r` (#71)
- **polytope**: Reject infinite or NaN constraints, and treat bounds of `1e20` or more as infinite, as HiGHS does (#83)

### Features

- **warmup**: Add `ranges`, the smallest and largest value of each coordinate over a polytope (#34)
- **warmup**: Report the number of redundant constraints removed in `w.removed` and `show` (#35)
- **density**: Report slice updates that give up in `getstats` as `n_exhausted` (#70)
- **billiard**: Report abandoned trajectories in `getstats` as `is_abandoned`, and warn when more than 5% of a chain's steps are abandoned (#71)

### Performance

- **warmup**: Run the `:samples` rounding phases on one chain per thread, continuing each chain from its last point (#36)
- **warmup**: Keep `:samples` rounding going on skinny polytopes and draw fewer points in early phases; on a food web with 1082 variables the `:samples` warmup takes 290 s instead of 86 min (#45, #37, #33)

### Bug Fixes

- **warmup**: Rescale the variables by geometric-mean scaling before the linear programs and rounding, so that polytopes whose coefficients span many orders of magnitude, such as food webs with biomasses from `1e-3` to `1e3`, warm up (#83)
- **warmup**: Check every linear program's result, retry failed programs, drop dependent equality rows, and guard ellipsoid and `:samples` rounding as PolyRound, PolytopeSamplerMatlab, volesti and hops do (#83)
- **density**: Keep chains moving on densities that diverge at a face (#29)
- **density**: Keep truncated-normal chord draws inside the chord when the mode lies far away (#42)
- **warmup**: Keep the center inside the polytope when fixing thin directions, and report fixed coordinates (#31)
- **warmup**: Measure constraint tolerances in units of x and report an empty polytope when an equality violates a bound (#40)
- **warmup**: Find every direction of the polytope with extremes = true (#39)
- **warmup**: Fix bounds within atol at their midpoint, report solver failures by status and retry them once (#72)
- **sampler**: Start chains inside the polytope after rounding or with extreme points (#38)
- **sampler**: Run the vertex-start test and start barrier walks and CRHMC from boundary points (#41)
- **sampler**: Continue chains from a copy of initial_state, so one state can start several chains (#69)
- **volume**: Keep the current point in the ball chord and fail on a zero ratio (#43)
- **rounding**: Warn when ellipsoid rounding stops because the ellipsoid cannot be factored (#73)

### Documentation

- **sampler**: State that a HitAndRun step is one move and a CoordinateHitAndRun step is k (#56)

## [0.1.0] - 2026-11-01

### Breaking Changes

- Rewrite the sampler around Polytope, warmup and hr_sample
- Rename the package to PolytopeSamplers.jl
- Sample through the AbstractMCMC interface and add ESS and R̂ diagnostics
- Round polytopes by maximum-volume ellipsoids and add coordinate hit-and-run (CHRR)
- Default warmup to one linear program, with extreme points computed for ACHR on use
- **sampler**: Report per-step statistics with getstats and store them in Chains internals (#21)

### Bug Fixes

- Bundle samples into a matrix in linear time
- **sampler**: Start chains of sample at random points of the inscribed ball and accept initial_params (#13)

### Documentation

- Write the README and port the constrained food web problem as an example
- Add a parallel food web replicates example
- Show the barrier walks on an unrounded box
- Bring docstrings up to date with the samplers and the default warmup
- Add a Documenter manual built by just docs
- **examples**: Build the food web polytope with FoodWebs.jl (#12)
- **examples**: Keep the pyramid biomasses of the original study in the food web examples (#12)

### Features

- Add the accelerated billiard walk
- Add multiphase Monte Carlo sampling, one-LP warmup and sampling to convergence
- Add the Dikin and Vaidya walks
- Add constrained Riemannian Hamiltonian Monte Carlo (CRHMC)
- Sample non-uniform densities on polytopes with Target
- **sampler**: Return MCMCChains.Chains with chain_type = Chains and add getparams (#14)
- **density**: Add GaussianTarget and ExponentialTarget, drawn exactly on chords by hit-and-run (#20)
- **volume**: Estimate polytope volumes with a standard error by multiphase Monte Carlo with cooling balls (#19)

### Miscellaneous

- Add a justfile, Runic formatting, JETLS lint, CI and a git-cliff changelog
- Leave the pre-conventional history out of the changelog
- Relicense under GPL-3.0-or-later
- Add citation metadata and a TagBot workflow for registered releases (#18)

### Performance

- Update slacks in place and default to a cyclic coordinate scan

### Testing

- Check uniformity against exact and rejection-sampled moments, feasibility, degenerate polytopes, JET and AllocCheck
- Add sampling checks adapted from volesti, hopsy, hops and PolyRound
- Add a convergence stress suite run by just stress
- Check slack updates, the cyclic scan and that samples stay inside the polytope
- Run the Dikin and Vaidya walks through the stress suite
- Run CRHMC through the stress suite
- Check non-uniform targets against exact moments
