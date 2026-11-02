using PolytopeSamplers: Polytope, Warmup, ranges, warmup
using LinearAlgebra: Diagonal, I, norm, rank, svdvals
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
        # The reduced constraints are the two lower bounds and the diagonal edge, with
        # unit-norm rows. The upper bounds, reached only at the vertices (1, 0) and (0, 1),
        # are redundant; without pruning all five remain
        @test size(w.G) == (3, 2)
        @test w.removed == 2
        @test all(r -> norm(r) ≈ 1, eachrow(w.G))
        @test size(warmup(P; extremes = true, rounding = false, prune = false).G) == (5, 2)
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
        # :samples rounding of a skinny box reaches a ratio near 3. The rule that stops after a
        # phase improving by less than 10% stopped 15 of 20 seeds at ratios up to 13108 (#45);
        # over 50 seeds the largest ratio is now 1.7
        widths = [1.0e6; ones(9)]
        for seed in 1:3
            w = warmup(Xoshiro(seed), Polytope(zeros(10), widths); rounding = :samples)
            sv = svdvals(w.transform \ Diagonal(widths))
            @test sv[1] / sv[end] < 4
        end
        # With 4 chains per phase, each continuing from its last point, the largest ratio
        # over 50 seeds is 1.5, and a seed gives the same rounding for the same chains
        function round_box(seed)
            G = [Matrix(1.0I, 10, 10); -Matrix(1.0I, 10, 10)]
            h = [widths; zeros(10)]
            center, T = zeros(10), Matrix(1.0I, 10, 10)
            PolytopeSamplers.round_by_samples!(
                Xoshiro(seed), Polytope(zeros(10), widths), center, T, G, h;
                optimizer = PolytopeSamplers.HiGHS.Optimizer, chains = 4,
            )
            return T
        end
        for seed in 1:3
            sv = svdvals(round_box(seed) \ Diagonal(widths))
            @test sv[1] / sv[end] < 4
        end
        @test round_box(1) == round_box(1)
    end

    @testset "redundant constraints" begin
        # The unit cube with x₁ + x₂ ≤ 5, which no point reaches, x₁ + x₂ + x₃ ≤ 3, reached
        # only at (1, 1, 1), and two more copies of x₁ ≤ 1. Pruning keeps one of the three
        # copies and the six facets of the cube
        A = [1.0 1.0 0.0; 1.0 1.0 1.0; 1.0 0.0 0.0; 1.0 0.0 0.0]
        P = Polytope(zeros(3), ones(3); A, b = [5.0, 3.0, 1.0, 1.0])
        Q, removed = PolytopeSamplers.remove_redundant(P, PolytopeSamplers.HiGHS.Optimizer)
        @test removed == 4
        @test Q.A == [1.0 0.0 0.0]
        @test Q.lower == zeros(3)
        @test Q.upper == [Inf, 1.0, 1.0]
        for extremes in (false, true)
            w = warmup(P; extremes)
            @test w.polytope === P
            @test w.removed == 4
            @test size(w.G, 1) == 6
            @test size(warmup(P; extremes, prune = false).G, 1) == 10
        end
        @test occursin("4 redundant constraints removed", sprint(show, warmup(P)))
        # The upper bounds of the simplex are reached only at its vertices
        simplex = Polytope(zeros(3), ones(3); A = ones(1, 3), b = [1.0])
        w = warmup(simplex)
        @test w.removed == 3
        @test size(w.G, 1) == 4
        X = sample(Xoshiro(1), w, BilliardWalk(), 2000)
        @test all(x -> x in simplex, eachcol(X))
        # On the triangle 0 ≤ x₁ ≤ x₂ ≤ 1 the bound x₂ ≥ 0 is reached only at the origin, and
        # an infinite bound has no program
        w = warmup(Polytope([0.0, 0.0], [Inf, 1.0]; A = [1.0 -1.0], b = [0.0]))
        @test w.removed == 1
        @test size(w.G, 1) == 3
    end

    @testset "ranges" begin
        box = Polytope([-1.0, 0.0, 2.0], [1.0, 3.0, 5.0])
        @test ranges(box) == ([-1.0, 0.0, 2.0], [1.0, 3.0, 5.0])
        simplex = Polytope(zeros(3), fill(Inf, 3); A = ones(1, 3), b = [1.0])
        lower, upper = ranges(simplex)
        @test lower ≈ zeros(3) atol = 1.0e-9
        @test upper ≈ ones(3) atol = 1.0e-9
        # x₃ is fixed by its bounds, and x₁ + x₂ ≤ 1 with x₁ + x₂ ≥ 1 fixes x₁ + x₂ but
        # neither coordinate; x₄ = x₁ by an equality
        P = Polytope(
            [0.0, 0.0, 0.5, 0.0], [1.0, 1.0, 0.5, 1.0];
            A = [1.0 1.0 0.0 0.0; -1.0 -1.0 0.0 0.0], b = [1.0, -1.0],
            C = [1.0 0.0 0.0 -1.0], d = [0.0],
        )
        lower, upper = ranges(P)
        @test lower ≈ [0, 0, 0.5, 0] atol = 1.0e-9
        @test upper ≈ [1, 1, 0.5, 1] atol = 1.0e-9
        # A warmup with extreme points uses them; one without solves the programs once and
        # shares them with ACHR. P has an equality that is not written as one, so its default
        # warmup falls back to extreme points
        for Q in (box, simplex, P)
            exact = ranges(Q)
            @test all(isapprox.(ranges(warmup(Q; extremes = true)), exact; atol = 1.0e-9))
            w = warmup(Q)
            @test all(isapprox.(ranges(w), exact; atol = 1.0e-9))
            @test (w.cache.points === nothing) == (Q === P)
            @test PolytopeSamplers.extreme_zpoints(w) ≈
                w.transform \ (PolytopeSamplers.extreme_points(w) .- w.center)
        end
        # Samples stay inside the ranges
        lower, upper = ranges(simplex)
        X = sample(Xoshiro(1), warmup(simplex), BilliardWalk(), 1000)
        @test all(lower .- 1.0e-9 .<= minimum(X; dims = 2))
        @test all(maximum(X; dims = 2) .<= upper .+ 1.0e-9)
    end

    @testset "failed ellipsoid rounding warns" begin
        # G has a zero column, so the ellipsoid's matrix is singular and cannot be factored;
        # rounding stopped without a message before #73. (A random polytope 1e9 times
        # narrower in one coordinate also fails, but only on some platforms.)
        G = [1.0 0.0; -1.0 0.0]
        h = [1.0, 1.0]
        @test_logs (:warn, r"ellipsoid rounding stopped in round 1") PolytopeSamplers.round_polytope!(
            zeros(2), [1.0 0.0; 0.0 1.0], G, h; optimizer = PolytopeSamplers.HiGHS.Optimizer,
        )
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
