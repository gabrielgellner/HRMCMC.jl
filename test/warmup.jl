using PolytopeSamplers: Polytope, Warmup, warmup
using LinearAlgebra: I, norm, rank, svdvals
using PolytopeSamplers: PolytopeSamplers, BilliardWalk, sample, sample_until_converged
using Random: Xoshiro
using Test: @test_logs
using JuMP: get_attribute

include(joinpath(@__DIR__, "fixtures", "thin_foodweb.jl"))

@testset "warmup" begin
    @testset "errors" begin
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, 1.0]; A = [-1.0 -1.0], b = [-3.0]))
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, Inf]))
        # The equality puts x₁ above its upper bound (#40)
        empty = Polytope([0.0, 0.0], [1.0, 1.0]; C = [1.0 0.0], d = [2.0])
        @test_throws ArgumentError warmup(empty)
        @test_throws ArgumentError warmup(empty; extremes = true)
        @test_throws ArgumentError warmup(Polytope([0.0, 0.0], [1.0, Inf]); extremes = true)
        # Unbounded with a bounded inscribed ball: the strip 0 ≤ x1 - x2 ≤ 1
        @test_throws ArgumentError warmup(Polytope(fill(-Inf, 2), fill(Inf, 2); A = [1.0 -1.0; -1.0 1.0], b = [1.0, 0.0]))
        @test PolytopeSamplers.isbounded([1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0], PolytopeSamplers.HiGHS.Optimizer)
        @test !PolytopeSamplers.isbounded([1.0 0.0; -1.0 0.0; 0.0 1.0], PolytopeSamplers.HiGHS.Optimizer)
        # Bounded in its normals' cone but not in their span: G d = 0 for d = (1, 1)
        @test !PolytopeSamplers.isbounded([1.0 -1.0; -1.0 1.0], PolytopeSamplers.HiGHS.Optimizer)
    end

    @testset "full-dimensional triangle" begin
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
        w = warmup(P; extremes = true, rounding = false)
        @test w isa Warmup
        @test !w.rounded
        # The three vertices, each extreme in some coordinate
        @test sort(collect(eachcol(w.points))) == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        @test w.center ≈ [1 / 3, 1 / 3]
        @test size(w.transform) == (2, 2)
        @test w.transform' * w.transform ≈ I
        # The reduced constraints are the four bounds and the diagonal edge, with unit-norm
        # rows
        @test size(w.G) == (5, 2)
        @test all(r -> norm(r) ≈ 1, eachrow(w.G))
        @test w.zpoints ≈ w.transform' * (w.points .- w.center)
        # The maximum-volume ellipse of a triangle is centered at its centroid
        @test warmup(P).center ≈ [1 / 3, 1 / 3] atol = 1.0e-4
    end

    @testset "rounding" begin
        # The box [0, 100] × [0, 1] rounds to a square: the inscribed ellipse has semi-axes
        # 50 and 1/2 and every constraint lies at distance one from the center
        w = warmup(Polytope([0.0, 0.0], [100.0, 1.0]))
        @test w.rounded
        @test w.center ≈ [50, 0.5] rtol = 1.0e-4
        @test sort(svdvals(w.transform)) ≈ [0.5, 50] rtol = 1.0e-3
        @test w.h ≈ ones(4) rtol = 1.0e-4
        # A long thin simplex: rounding leaves a polytope whose largest inscribed ball has
        # radius near that of the regular simplex in the same dimension
        P = Polytope(zeros(3), [1000.0, 1.0, 1.0]; A = [0.001 1.0 1.0], b = [1.0])
        w = warmup(P)
        @test w.rounded
        @test all(x -> x in P, eachcol(w.transform * w.zpoints .+ w.center))
        _, r = PolytopeSamplers.chebyshev_center(w.G, w.h, PolytopeSamplers.HiGHS.Optimizer)
        @test 0.9 < r < 1.1
    end

    @testset "maximum-volume ellipsoid of a box" begin
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = [3.0, 1.0, 1.0, 1.0] # [-1, 3] × [-1, 1]
        E2, c, converged = PolytopeSamplers.max_volume_ellipsoid(A, b, [0.5, 0.1]; reg = 1.0e-10)
        @test converged
        @test c ≈ [1, 0] atol = 1.0e-5
        # Semi-axes 2 and 1: E2 = diag(1/4, 1)
        @test E2 ≈ [0.25 0; 0 1] atol = 1.0e-4
        @test_throws ArgumentError PolytopeSamplers.max_volume_ellipsoid(A, b, [5.0, 0.0]; reg = 1.0e-3)
    end

    @testset "equalities restrict the directions" begin
        # The 3-simplex x ≥ 0, sum(x) = 1 in four variables
        P = Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0])
        w = warmup(P)
        @test size(w.transform) == (4, 3)
        @test norm(ones(1, 4) * w.transform) < 1.0e-12
        @test sum(w.center) ≈ 1 atol = 1.0e-12
        @test all(p -> p in P, eachcol(w.points))
    end

    @testset "fixed coordinates have zero direction rows" begin
        P = Polytope([0.0, 0.5, 0.0], [1.0, 0.5, 1.0])
        w = warmup(P)
        @test size(w.transform) == (3, 2)
        @test all(iszero, w.transform[2, :])
        @test w.center[2] == 0.5
    end

    @testset "inequalities that hold with equality" begin
        # x1 + x2 ≤ 1 and x1 + x2 ≥ 1 leave the segment x1 + x2 = 1
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0; -1.0 -1.0], b = [1.0, -1.0])
        w = warmup(P)
        @test size(w.transform, 2) == 1
        @test abs(sum(w.transform)) < 1.0e-12
        # A coordinate pinned by inequalities, x2 ≤ 0.3 and x2 ≥ 0.3
        Q = Polytope([0.0, 0.0], [1.0, 1.0]; A = [0.0 1.0; 0.0 -1.0], b = [0.3, -0.3])
        wq = warmup(Q)
        @test size(wq.transform, 2) == 1
        @test all(iszero, wq.transform[2, :])
        # The scale of a row does not decide whether it holds with equality (#40):
        # 1e-9 x₁ ≤ 0.5e-9 leaves x₁ a range of 0.5, and 1e6 (x₁ + x₂) = 1e6 as two
        # inequalities leaves a segment
        small = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0e-9 0.0], b = [0.5e-9])
        large = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0e6 1.0e6; -1.0e6 -1.0e6], b = [1.0e6, -1.0e6])
        for extremes in (false, true)
            @test size(warmup(small; extremes).transform, 2) == 2
            @test size(warmup(large; extremes).transform, 2) == 1
        end
    end

    @testset "extreme points that miss a direction" begin
        # The triangle with vertices (0, 0), (1, 1) and (0.5, 0.4) has every coordinate
        # extreme at the first two, where x₂ ≤ x₁ is tight; extremes = true sampled the
        # segment between them (#39)
        P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [-1.0 1.0; 0.8 -1.0; 1.2 -1.0], b = [0.0, 0.0, 0.2])
        w = warmup(P; extremes = true)
        @test size(w.transform, 2) == 2
        @test rank(w.points .- w.points[:, 1]) == 2
        @test all(p -> p in P, eachcol(w.points))
    end

    @testset "a single point" begin
        w = warmup(Polytope([0.2, 0.7], [0.2, 0.7]))
        @test size(w.transform) == (2, 0)
        @test w.center == [0.2, 0.7]
    end

    # Issue #31: with atol = 1e-6 the warmup fixed thin directions at a center 1.1e-6 below a
    # bound of 0, and every sample kept the violation
    @testset "thin directions" begin
        P = thin_foodweb_polytope()
        w = @test_logs (:warn, r"fixed directions") match_mode = :any warmup(Xoshiro(1), P; atol = 1.0e-6)
        @test PolytopeSamplers.constraint_violation(P, w.center) <= 1.0e-9
        X = sample(Xoshiro(2), w, BilliardWalk(), 500; progress = false)
        @test all(x -> PolytopeSamplers.constraint_violation(P, x) <= 1.0e-9, eachcol(X))
        @test length(PolytopeSamplers.fixed_coordinates(w)) == 2
        @test occursin("2 fixed coordinates", sprint(show, w))
        r = sample_until_converged(Xoshiro(3), w, BilliardWalk(); target_ess = 100)
        @test r.fixed == PolytopeSamplers.fixed_coordinates(w)
        # The default atol samples the thin directions
        w8 = warmup(Xoshiro(1), P)
        @test isempty(PolytopeSamplers.fixed_coordinates(w8))
        @test size(w8.transform, 2) == 83
        @test PolytopeSamplers.constraint_violation(P, w8.center) <= 1.0e-9
        # A box with a side of 1e-7 is sampled in 4 dimensions by default; with atol = 1e-6
        # the side is fixed inside the box, with a warning
        box = Polytope(zeros(4), [1.0, 1.0, 1.0, 1.0e-7])
        @test size(warmup(box).transform, 2) == 4
        wb = @test_logs (:warn, r"fixed directions") match_mode = :any warmup(box; atol = 1.0e-6)
        @test size(wb.transform, 2) == 3
        @test 0 <= wb.center[4] <= 1.0e-7
        # The warmup's linear programs tighten HiGHS's feasibility tolerances to 1e-9
        model = PolytopeSamplers.lp(PolytopeSamplers.HiGHS.Optimizer)
        @test get_attribute(model, "primal_feasibility_tolerance") == 1.0e-9
        @test get_attribute(model, "dual_feasibility_tolerance") == 1.0e-9
        # Implicit equalities stay fixed at the default: the simplex as two inequalities
        S = Polytope(zeros(6), ones(6); A = [ones(1, 6); -ones(1, 6)], b = [1.0, -1.0])
        @test size(warmup(S).transform, 2) == 5
    end
end
