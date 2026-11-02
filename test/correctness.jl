# Sampling correctness checks adapted from the test suites of other polytope samplers:
#
# - volesti, test/sampling_test.cpp and test/max_ellipsoid_rounding_test.cpp (commit 064e18c,
#   LGPL-3.0): the 10-dimensional cube and the skinny cube `[-100, 100] × [-1, 1]^9`, with
#   split R̂ below 1.1 for every coordinate, and rounding of the skinny cube.
# - hopsy, tests/proposal.py (commit 54516d7, MIT): each sample mean within five standard
#   errors of the true mean, with the standard error taken from the chain (here the Monte
#   Carlo standard error of `diagnostics`), so that the test also checks that the error
#   estimate is honest.
# - hops, tests/MarkovChain/Proposal/CoordinateHitAndRunTestSuite.cpp (commit b62592e, MIT):
#   chains started on the boundary stay in the polytope.
# - PolyRound, PolyRound/unit_tests.py (commit 6d3aee7, MIT): the Chebyshev center of a
#   symmetric and a cut cube, and a polytope that the equalities reduce to a point.

using PolytopeSamplers: PolytopeSamplers, ACHR, BilliardWalk, CoordinateHitAndRun, CRHMC,
    DikinWalk,
    HitAndRun, VaidyaWalk,
    MCMCSerial,
    Polytope, diagnostics, sample, warmup
using LinearAlgebra: I, svdvals
using Random: Xoshiro

const all_methods = (
    HitAndRun(), ACHR(), CoordinateHitAndRun(), CoordinateHitAndRun(scan = :random), BilliardWalk(),
    DikinWalk(), VaidyaWalk(), CRHMC(),
)

# Four chains and their diagnostics. The Metropolis-filtered barrier walks need about 50
# times as many steps per effective sample as the others, so they run 40 times as long.
function run_chains(w, method; n = 2_000, thinning = 5, seed = 1)
    if method isa Union{DikinWalk, VaidyaWalk}
        thinning *= 40
    end
    chains = sample(Xoshiro(seed), w, method, MCMCSerial(), n, 4; discard_initial = 200, thinning)
    return diagnostics(chains)
end

# hopsy's check, with the Monte Carlo standard errors of `diagnostics`: every mean within
# five standard errors of the truth and every standard deviation within six. Over 40 seeds
# these z-scores have root mean square 0.9 to 1.1 for every method, and about 1% exceed 3;
# standard deviations have the heavier tail.
function moments_match(d, μ, σ)
    return all(@. abs(d.mean - μ) < 5 * d.mcse_mean) && all(@. abs(d.std - σ) < 6 * d.mcse_std)
end

skinny_cube(d) = Polytope([-100.0; fill(-1.0, d - 1)], [100.0; fill(1.0, d - 1)])

@testset "correctness" begin
    @testset "10-dimensional cube ($m)" for m in all_methods
        w = warmup(Polytope(fill(-1.0, 10), fill(1.0, 10)))
        d = run_chains(w, m)
        @test d.max_rhat < 1.1
        @test moments_match(d, zeros(10), fill(sqrt(1 / 3), 10))
    end

    @testset "skinny cube, rounded ($m)" for m in all_methods
        w = warmup(skinny_cube(10))
        d = run_chains(w, m)
        @test d.max_rhat < 1.1
        @test moments_match(d, zeros(10), sqrt.([200^2 / 12; fill(1 / 3, 9)]))
    end

    @testset "rounding makes the skinny cube a cube" begin
        # The inscribed ellipsoid of a box is the box's own axis-aligned ellipsoid, so the
        # rounded polytope is [-1, 1]^10: every facet at distance one from the center
        w = warmup(skinny_cube(10))
        @test w.center ≈ zeros(10) atol = 1.0e-3
        @test w.h ≈ ones(20) rtol = 1.0e-3
        @test sort(svdvals(w.transform)) ≈ [fill(1.0, 9); 100.0] rtol = 1.0e-3
    end

    @testset "rounding raises the ESS of hit-and-run on the skinny cube" begin
        rounded = run_chains(warmup(skinny_cube(10)), HitAndRun())
        unrounded = run_chains(warmup(skinny_cube(10); rounding = false), HitAndRun())
        # Unrounded directions rarely line up with the long axis, so x1 barely moves
        @test rounded.ess[1] > 10 * unrounded.ess[1]
        @test unrounded.max_rhat > rounded.max_rhat
    end

    @testset "10-dimensional simplex ($m)" for m in all_methods
        # x ≥ 0, sum(x) = 1: the marginals are Beta(1, 9), with mean 1/10 and variance 9/1100
        w = warmup(Polytope(zeros(10), ones(10); C = ones(1, 10), d = [1.0]))
        d = run_chains(w, m; thinning = 10)
        @test d.max_rhat < 1.1
        @test moments_match(d, fill(0.1, 10), fill(sqrt(9 / 1100), 10))
    end

    @testset "chains started at a vertex stay inside ($m)" for m in all_methods
        P = Polytope(fill(-1.0, 3), fill(1.0, 3))
        w = warmup(P; extremes = true)
        @test !isempty(w.zpoints) # the loop below ran on no points before #41
        for j in axes(w.zpoints, 2)
            state = PolytopeSamplers.chain_state(m, w)
            state.z .= w.zpoints[:, j]
            X = sample(Xoshiro(j), w, m, 100; initial_state = state)
            @test all(x -> x in P, eachcol(X))
            @test X[:, end] != w.points[:, j]
        end
    end

    @testset "Chebyshev center" begin
        # The cube [-1, 1]^3 cut by sum(x) ≤ 0, as rows of unit norm
        G = [I(3); -I(3); ones(1, 3) / sqrt(3)]
        h = [ones(6); 0.0]
        x, r0 = PolytopeSamplers.chebyshev_center(G, h, PolytopeSamplers.HiGHS.Optimizer)
        @test all(abs.(x .- x[1]) .< 1.0e-10)
        @test r0 > 0
        # Moving the facet x1 ≥ -1 to x1 ≥ 0 pushes the center up in x1
        h[4] = 0.0
        x, r1 = PolytopeSamplers.chebyshev_center(G, h, PolytopeSamplers.HiGHS.Optimizer)
        @test all(x[1] .> x[2:3])
        @test r0 > r1
    end

    @testset "equalities that leave a single point" begin
        P = Polytope(zeros(2), ones(2); C = [1.0 0.0; 0.0 1.0], d = [0.3, 0.6])
        w = warmup(P)
        @test size(w.transform, 2) == 0
        X = sample(Xoshiro(1), w, CoordinateHitAndRun(), 3)
        @test all(x -> x ≈ [0.3, 0.6], eachcol(X))
    end
end
