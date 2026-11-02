# PolytopeSamplers.jl

Uniform sampling of convex polytopes by MCMC (hit-and-run and artificial centering
hit-and-run so far), with warmup linear programs solved by HiGHS through JuMP. Called
HRMCMC.jl until 2026. The main
use is sampling the interaction strengths of food webs consistent with an equilibrium, as in
[constrained-foodwebs](https://github.com/gabrielgellner/constrained-foodwebs).

## Layout

- `src/polytope.jl`: `Polytope` (`lower .<= x .<= upper`, `A * x .<= b`, `C * x .== d`),
  `isfeasible`, `isempty`, and the JuMP model of a polytope.
- `src/warmup.jl`: `warmup`, which rescales the variables by geometric-mean scaling
  (`scaled_polytope` and `gmscale` in `polytope.jl`, from PolytopeSamplerMatlab) and works in
  the scaled variables, converting widths to the original units before comparing them with
  `atol`. Its numerical safeguards (checked linear programs with retries, dependent
  equality rows, ellipsoid start-point clamps, final checks) each follow a reference
  sampler and cite it; check the references before adding one. By default removes
  redundant bounds and inequalities (one
  program each, `prune`), solves one Chebyshev-center program and one
  boundedness program (or, with `extremes = true`, the `2n` coordinate programs), finds the
  equalities that hold over the polytope (explicit, fixed coordinates, tight inequalities),
  and writes the polytope as `x = center + transform * z`, `G z ≤ h` with unit-norm rows.
  Samplers move in `z`. `WarmupCache` holds values computed on first use and shared by
  chains under a lock: `G Gᵀ` for the billiard walk and the extreme points for `ranges`
  and ACHR. `ranges` gives each coordinate's minimum and maximum (flux variability
  analysis).
- `src/rounding.jl`: the Chebyshev center, the maximum-volume inscribed ellipsoid (adapted
  from volesti and PolyRound), the ellipsoid rounding loop, and rounding from billiard-walk
  samples (the MMCS phases, adapted from volesti).
- `src/density.jl`: `Target` (a warmup with a log density, LogDensityProblems or a
  function), its evaluation in `z`, slice sampling on a chord, and `GaussianTarget` and
  `ExponentialTarget`, whose quadratic log densities hit-and-run draws exactly on a chord
  (truncated normal and exponential samplers of Robert 1995). States carry the density
  as a type parameter, `nothing` for the uniform distribution, so the uniform steps keep no
  density code and stay allocation free.
- `src/sampler.jl`: `PolytopeSampler`, `HitAndRun`, `ACHR`, `CoordinateHitAndRun` (slack
  updates adapted from hops), the allocation-free steps and the AbstractMCMC
  interface (`step`, `bundle_samples`, `sample` for a `Polytope`). `Warmup` and `Polytope`
  are `AbstractMCMC.AbstractModel`s.
- `src/billiard.jl`: `BilliardWalk`, the accelerated billiard walk (adapted from volesti).
- `src/crhmc.jl`: `CRHMC`, constrained Riemannian HMC in the reduced coordinates (dense
  metric `Gᵀ S⁻² G`), adapted from PolytopeSamplerMatlab; the implicit midpoint rule refines
  its solve with the start-of-step factor and refactors every 5 iterations, as rejections
  from failed integrations bias the chain (see the comments).
- `src/barrier.jl`: `DikinWalk` and `VaidyaWalk`, Metropolis-filtered barrier walks (adapted
  from polytopewalk, whose proposal draws with the wrong Cholesky factor; see the comment
  above `step!`).
- `src/adaptive.jl`: `sample_until_converged` (chains doubled until ESS and R̂ targets)
  and `mmcs`.
- `src/volume.jl`: `volume`, multiphase Monte Carlo with cooling balls (schedule adapted from
  volesti), each ratio from coordinate hit-and-run chains on polytope ∩ ball run to a Monte
  Carlo standard error, giving a standard error of the log volume.
- `src/diagnostics.jl`: `diagnostics`, ESS, R̂, means, standard deviations and their Monte
  Carlo standard errors per coordinate via MCMCDiagnosticTools.
- `ext/PolytopeSamplersMCMCChainsExt.jl`: `bundle_samples` for `chain_type =
  MCMCChains.Chains`, loaded with MCMCChains, with each step's `getstats` in `internals`.
  A new sampler state implements `getparams` and `getstats`.
- `test/correctness.jl`: checks adapted from the test suites of volesti, hopsy, hops and
  PolyRound (cube, skinny cube, simplex, vertex starts, Chebyshev centers). Add the checks
  of a reference implementation here when porting a new method.
- `test/`: one test file per source file. Workspace project with the test-only dependencies.
- `docs/`: the Documenter.jl manual, a workspace project (`just docs`). Pages in
  `docs/src/` run their `@example` blocks at build time, so a stale example fails the
  build; `checkdocs = :exports` fails it when an exported name is missing from `api.md`.
  Update the benchmark tables in `samplers.md` when a sampler's performance changes.
- `examples/`: worked problems, as a workspace project (FoodWebs.jl from GitHub, pinned to
  a release tag): `foodweb_study.jl` holds the settings of the original study (constraints,
  pyramid biomasses), `constrained_foodweb.jl` compares the samplers on one web,
  `foodweb_replicates.jl` runs many webs in parallel threads. FoodWebs.jl builds the food
  web polytope (`FeasibleCommunities`). Food web code may live here; move it to FoodWebs.jl
  once it is general or useful beyond these examples.
- The 2018–2019 history of the original prototype is kept in git; the package was rewritten
  in 2026 for Julia 1.13.

## Conventions

- Samples are the columns of the returned matrix; several chains are a vector of matrices.
- Compare samplers by effective samples per second from `diagnostics` over several chains.
- Functions that draw random numbers take an optional leading `rng::AbstractRNG`.
- The step clamps constraint slacks at zero and ignores direction components below
  `ZERO_TOL`, so rounding never gives an interval that excludes the current point; it does
  not tighten bounds.
- `step!` is type stable and allocation free; `test/sampler.jl` enforces this with JET's
  `@test_opt` and AllocCheck. Keep it so when adding a method.
- Julia 1.13 only. Imports are explicit (`using X: a, b`), checked by ExplicitImports.

## Adapting code from other samplers

The package is GPL-3.0-or-later so that it can adapt code from volesti and dingo (LGPL-3),
the COBRA Toolbox and PolytopeSamplerMatlab (GPL-3), COBREXA.jl (Apache-2.0), cobrapy
(GPL-2+/LGPL-2+) and hopsy, PolyRound and polytopewalk (MIT). Port from the paper and read
the reference code for the details that make it robust (tolerances, degenerate cases,
step-size rules). When a function adapts code, say so in a comment above it, naming the
project, file, commit and license, and keep the original copyright line.

## Tests

Test sampling against exact moments of the uniform distribution (unit square, simplex) or
a large rejection sample (the pentagon of the original test), under several seeds, with
tolerances of about five standard errors. Prefer the Monte Carlo standard errors from
`diagnostics` over fixed tolerances: they test the error bars users report too. Every new
sampler joins the `methods` tuples in `test/sampler.jl`, `test/correctness.jl` and
`test/stress.jl`.

`test/stress.jl` (`just stress`, about 10 minutes, not part of `just ci`) checks that R̂
flags chains started near corners, that runs doubled until R̂ < 1.01 give correct moments
on hard geometry, and that mixing cost grows with dimension as published (CHRR moves per
effective sample as n²). Run it after changing a sampler, rounding or diagnostics. Check a new stochastic test under 50 or more
shifted seeds before trusting it.

## Commands

```sh
just setup      # install Runic and JETLS as Julia apps, instantiate environments
just fmt        # format with Runic
just ci         # formatting check, JETLS lint and tests; run before every push and merge
just stress     # convergence stress tests (about 10 minutes)
just docs       # build the documentation
just example constrained_foodweb
```

## Checks

GitHub Actions runs `.github/workflows/CI.yml` (tests on Linux and macOS, Runic, JETLS) on
pull requests and on `main`, and `Docs.yml` builds the manual on pull requests and deploys
it to the `gh-pages` branch from `main` and release tags. The repository is public, so the
runners are free. Run `just ci` before pushing, and merge only when CI is green.

## Commits

Conventional commits, `type(scope): subject`, with the issue as `(#N)` at the end when
there is one. git-cliff builds `CHANGELOG.md` from the subjects (`cliff.toml`): the type
picks the section an entry lands in, and git-cliff drops a subject it cannot parse.

- `feat` → Features, `fix` → Bug Fixes, `docs` → Documentation, `perf` → Performance,
  `refactor` → Refactoring, `test` → Testing, `chore` → Miscellaneous.
- `!` after the type or scope (`fix!:`, `feat(sampler)!:`) marks a breaking change: it heads
  the entry under Breaking Changes and bumps the minor version while we are pre-1.0.
- `ci`, `chore(release)` and merge commits are left out of the changelog.
- Write the subject as the changelog line a user reads: what changed, in the imperative
  ("Add the nested-hierarchy model").

Releases run through `just release-1-preview` … `just release-4-merge`: preview the
version, write the changelog on a `release/vX` branch (edit it by hand if needed), open the
release PR with the version bumped in `Project.toml`, then merge, tag the merge commit and
create the GitHub release from the edited changelog section.

## Writing style

These rules cover all prose written for this repo: comments, docstrings, documentation,
commit messages, issues and PRs.

Write in plain language. State claims directly, in the order the reader
needs them. Use subject-verb-object sentences, concrete nouns, and the
actual number, name, or mechanism instead of an allusion to it.

Follow these rules from Strunk & White:

- Put statements in positive form.
- Use definite, specific, concrete language.
- Omit needless words.

Avoid these constructions everywhere (sentences, headings, titles, lists):

- Contrast by negation: "not X, but Y", "it's not about X, it's about Y",
  "X isn't the problem; Y is". State Y. Mention X only if someone
  actually holds that view.
- Hook clauses that read like half a question: "what X doesn't tell you",
  "the part nobody mentions", "the evidence Y lacks", "why this matters".
  State the fact the clause points at.
- Colon reveals: "The real issue: ...", "Here's the thing: ...",
  "The short version: ...". Drop the setup and write the sentence.
- Punchline structure: ending a sentence or list on a twist, a dramatic
  final item, or a short fragment for emphasis.
- Rhetorical questions that the text then answers.
- Framing work or ideas by what is missing when you can name what is
  present or proposed.

Titles (milestones, issues, PRs, headings):

- Name the deliverable or scope as a plain noun phrase or imperative.
  A reader should know from the title alone what gets merged or decided.
- Use a colon only to separate a component name from the work item
  (e.g. "Query DSL: add DuckDB backend"), never to introduce a hook.

Examples:

```text
Bad:  Conformance evidence: what the green checks do not prove
Good: Conformance: add negative and edge-case test fixtures

Bad:  Query DSL: a second backend, and the evidence the first one lacks
Good: Query DSL: add second backend; add validation tests for SQL backend

Bad:  The green checks tell you less than you'd think.
Good: The green checks only cover valid inputs; there are no tests
      for malformed DSDs or constraint violations.

Bad:  The problem isn't performance — it's correctness.
Good: The cache returns stale codelists after a version bump.

Bad:  Here's what the benchmark doesn't show: cold-start latency.
Good: The benchmark measures warm queries only; cold-start latency
      is untested.
```

Before finishing, reread the output. Rewrite any sentence that delays its
point, contrasts with a claim nobody made, or alludes to a fact instead of
stating it.
