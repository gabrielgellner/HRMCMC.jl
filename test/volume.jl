using PolytopeSamplers: PolytopeSamplers, Polytope, volume, warmup
using Random: Xoshiro

# The cross-polytope |x|₁ ≤ 1 in n dimensions, one inequality per sign pattern
cross_polytope(n) = Polytope(
    fill(-1.0, n), fill(1.0, n);
    A = reduce(vcat, [collect(s)' for s in Iterators.product(fill((-1.0, 1.0), n)...)]), b = ones(2^n),
)

@testset "volume" begin
    # Exact log volumes; the simplex slice x ≥ 0, Σx = 1 in n variables has (n-1)-dimensional
    # volume √n / (n-1)!
    cases = [
        ("cube", Polytope(zeros(10), ones(10)), 0.0),
        ("box", Polytope(zeros(30), collect(1.0:30.0)), sum(log, 1:30)),
        ("skinny box", Polytope(zeros(5), [1000.0, 1, 1, 1, 1]), log(1000)),
        ("simplex", Polytope(zeros(20), ones(20); A = ones(1, 20), b = [1.0]), -sum(log, 1:20)),
        ("cross-polytope", cross_polytope(8), 8 * log(2) - sum(log, 1:8)),
        ("simplex slice", Polytope(zeros(8), ones(8); C = ones(1, 8), d = [1.0]), log(8) / 2 - sum(log, 1:7)),
        ("fixed coordinate", Polytope([0.0, 0.0, 0.5], [2.0, 3.0, 0.5]), log(6)),
    ]
    # Over 50 seeds, the z-scores of the log volume had root mean square 0.9 to 1.13 and
    # stayed below 2.6
    @testset "exact volumes ($name)" for (name, P, exact) in cases
        w = warmup(P)
        for seed in 1:3
            v = volume(Xoshiro(seed), w)
            @test v.converged
            @test abs(v.log_volume - exact) < 4 * v.log_stderr
            @test v.log_stderr <= 0.1
            @test v.volume ≈ exp(v.log_volume)
            @test first(v.radii) == Inf
            @test issorted(v.radii; rev = true)
            @test length(v.ratios) == length(v.radii)
        end
    end

    @testset "error" begin
        v = volume(Xoshiro(1), Polytope(zeros(5), ones(5)); error = 0.02)
        @test v.log_stderr <= 0.02
        @test abs(v.log_volume) < 4 * v.log_stderr
    end

    @testset "ball chord keeps the current point" begin
        # With |z|² rounded above ρ², the ball's chord along z₁ is the point t = -z₁, which
        # lies outside the polytope's chord [-0.3, 0.5] when z₁ = 0.4 (#43)
        tmin, tmax = PolytopeSamplers.ball_chord(-0.3, 0.5, 0.4, 1.0, 1.0 - 0.16 - 1.0e-12)
        @test tmin <= 0 <= tmax
        @test -0.3 <= tmin && tmax <= 0.5
        @test PolytopeSamplers.ball_chord(-0.3, 0.5, 0.4, 1.0, Inf) == (-0.3, 0.5)
        # Inside the ball, the chord is the intersection
        @test all(PolytopeSamplers.ball_chord(-2.0, 2.0, 0.0, 0.0, 1.0) .≈ (-1.0, 1.0))
    end

    @testset "single point and errors" begin
        v = volume(Polytope([0.2, 0.7], [0.2, 0.7]))
        @test v.volume == 1
        @test v.log_stderr == 0
        P = Polytope(zeros(3), ones(3))
        @test volume(Xoshiro(2), P).log_volume == volume(Xoshiro(2), P).log_volume
        @test_throws ArgumentError volume(P; error = 0)
        @test_throws ArgumentError volume(P; nchains = 1)
        @test_throws ArgumentError volume(Polytope(zeros(2), fill(Inf, 2)))
    end
end
