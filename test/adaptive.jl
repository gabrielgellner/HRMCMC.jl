using PolytopeSamplers: PolytopeSamplers, ACHR, BilliardWalk, CoordinateHitAndRun, HitAndRun,
    MCMCSerial, Polytope, mmcs, sample, sample_until_converged, warmup
using LinearAlgebra: svdvals
using Random: Xoshiro

# Means within five and standard deviations within six Monte Carlo standard errors
function moments_match(d, μ, σ)
    return all(@. abs(d.mean - μ) < 5 * d.mcse_mean) && all(@. abs(d.std - σ) < 6 * d.mcse_std)
end

skinny = Polytope([-100.0; fill(-1.0, 9)], [100.0; fill(1.0, 9)])
simplex20 = Polytope(zeros(20), ones(20); C = ones(1, 20), d = [1.0])

@testset "warmup without extreme points" begin
    @testset "Chebyshev center" begin
        # The triangle x ≥ 0, x1 + x2 ≤ 1 has inradius 1 / (2 + √2) at (r, r)
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
        w = warmup(P; extremes = false, rounding = :none)
        r = 1 / (2 + sqrt(2))
        @test w.center ≈ [r, r] atol = 1.0e-8
        @test w.radius ≈ r atol = 1.0e-8
        @test size(w.points, 2) == 0
        @test !w.rounded
    end

    @testset "equalities and fixed coordinates" begin
        w = warmup(simplex20; extremes = false, rounding = :none)
        @test size(w.transform, 2) == 19
        @test sum(w.center) ≈ 1
        w = warmup(Polytope([0.0, 0.5, 0.0], [1.0, 0.5, 1.0]); extremes = false)
        @test size(w.transform) == (3, 2)
        @test all(iszero, w.transform[2, :])
        @test w.center[2] == 0.5
        X = sample(Xoshiro(1), Polytope(zeros(2), ones(2); C = [1.0 0.0; 0.0 1.0], d = [0.3, 0.6]), BilliardWalk(), 3; extremes = false)
        @test all(x -> x ≈ [0.3, 0.6], eachcol(X))
    end

    @testset "flat polytopes find their equalities" begin
        # x1 + x2 ≤ 1 and x1 + x2 ≥ 1: the largest ball has radius zero, and one more program
        # finds the equality (#46), without the extreme points of extremes = true
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0; -1.0 -1.0], b = [1.0, -1.0])
        w = warmup(P; extremes = false)
        @test isempty(w.points)
        @test size(w.transform, 2) == 1
        X = sample(Xoshiro(1), w, BilliardWalk(), 100; progress = false)
        @test all(x -> abs(x[1] + x[2] - 1) < 1.0e-9, eachcol(X))
    end

    @testset "errors" begin
        @test_throws ArgumentError warmup(Polytope(zeros(2), ones(2); C = [1.0 1.0; 1.0 1.0], d = [1.0, 2.0]); extremes = false)
        @test_throws ArgumentError warmup(Polytope(zeros(2), ones(2); A = [-1.0 -1.0], b = [-3.0]); extremes = false)
        @test_throws ArgumentError warmup(Polytope(zeros(2), fill(Inf, 2)); extremes = false)
        @test_throws ArgumentError warmup(skinny; rounding = :cube)
    end

    @testset "ACHR computes extreme points on first use" begin
        w = warmup(skinny)
        @test size(w.points, 2) == 0
        @test w.cache.zpoints === nothing
        chains = sample(Xoshiro(1), w, ACHR(), MCMCSerial(), 500, 3)
        # The 20 coordinate programs reach 11 distinct vertices of the box; the optima of
        # random objectives bring the points to 2k = 20 or more (#50)
        zp = w.cache.zpoints
        we = warmup(skinny; extremes = true)
        @test size(zp, 2) >= 20
        @test sort(collect(eachcol(we.points))) ≈ sort(collect(eachcol(w.transform * zp[:, 1:size(we.points, 2)] .+ w.center)))
        # The chains share one set of points, in the slice of the warmup
        @test PolytopeSamplers.extreme_zpoints(w) === zp
        @test all(p -> p in skinny, eachcol(w.transform * zp .+ w.center))
        @test all(x -> x in skinny, eachcol(reduce(hcat, chains)))
    end

    @testset "rounding from samples" begin
        # The skinny cube rounds to near a cube: principal axes 100 and 1 in ratio
        w = warmup(Xoshiro(1), skinny; extremes = false, rounding = :samples)
        @test w.rounded
        sv = svdvals(w.transform)
        @test 50 < sv[1] / sv[end] < 200
        X = sample(Xoshiro(2), w, BilliardWalk(), 4_000)
        @test all(x -> x in skinny, eachcol(X))
    end
end

@testset "sample_until_converged" begin
    w = warmup(skinny)
    @testset "converges with correct moments ($m)" for m in (BilliardWalk(), CoordinateHitAndRun(), HitAndRun())
        r = sample_until_converged(Xoshiro(1), w, m; target_ess = 400)
        @test r.converged
        @test r.diagnostics.min_ess >= 400 && r.diagnostics.max_rhat < 1.01
        @test length(r.chains) == 4 && all(==((10, r.n)), size.(r.chains))
        @test moments_match(r.diagnostics, zeros(10), [200 / sqrt(12); fill(sqrt(1 / 3), 9)])
    end

    @testset "lengthening and reproducibility" begin
        r = sample_until_converged(Xoshiro(3), w, HitAndRun(); target_ess = 2_000, n = 100)
        @test r.n * r.thinning > 100
        @test all(==((10, r.n)), size.(r.chains))
        @test sample_until_converged(Xoshiro(3), w, HitAndRun(); target_ess = 2_000, n = 100).chains == r.chains
        @test all(x -> x in skinny, eachcol(reduce(hcat, r.chains)))
    end

    @testset "stops at max_n with a warning" begin
        @test_logs (:warn,) match_mode = :any begin
            r = sample_until_converged(Xoshiro(4), w, HitAndRun(); target_ess = 10^6, n = 100, max_n = 200)
            @test !r.converged
            @test r.n * r.thinning == 200
        end
    end

    @testset "thins slowly mixing chains (#53)" begin
        # Unthinned hit-and-run in 20 dimensions needs tens of steps per effective sample;
        # the stored points stay below 20 per effective sample after the last round's
        # growth (under 19 over 60 seeds)
        cube = warmup(Polytope(zeros(20), ones(20)))
        r = sample_until_converged(Xoshiro(1), cube, HitAndRun())
        @test r.converged
        @test r.thinning > 1
        @test r.n * 4 / r.diagnostics.min_ess < 40
        @test moments_match(r.diagnostics, fill(0.5, 20), fill(sqrt(1 / 12), 20))
    end

    @testset "multivariate R̂ (#53)" begin
        r = sample_until_converged(Xoshiro(1), w, CoordinateHitAndRun(); target_ess = 400, multivariate = true)
        @test r.converged
        @test r.diagnostics.multivariate_rhat < 1.01
        @test isnan(sample_until_converged(Xoshiro(1), w, CoordinateHitAndRun(); target_ess = 400).diagnostics.multivariate_rhat)
    end

    @testset "argument checks" begin
        @test_throws ArgumentError sample_until_converged(w; nchains = 1)
        @test_throws ArgumentError sample_until_converged(w; n = 2)
        @test_throws ArgumentError sample_until_converged(w; n = 9)
        @test_throws ArgumentError sample_until_converged(w; n = 100, max_n = 50)
        @test_throws ArgumentError sample_until_converged(w; thinning = 0)
    end
end

@testset "mmcs" begin
    @testset "$name" for (name, P, μ, σ) in (
            ("skinny cube", skinny, zeros(10), [200 / sqrt(12); fill(sqrt(1 / 3), 9)]),
            ("simplex", simplex20, fill(0.05, 20), fill(sqrt(19 / (400 * 21)), 20)),
        )
        r = mmcs(Xoshiro(5), P; target_ess = 1_000)
        @test r.converged
        @test size(r.warmup.points, 2) == 0
        @test r.diagnostics.min_ess >= 1_000 && r.diagnostics.max_rhat < 1.01
        @test moments_match(r.diagnostics, μ, σ)
        @test all(x -> x in P, eachcol(reduce(hcat, r.chains)))
    end
end
