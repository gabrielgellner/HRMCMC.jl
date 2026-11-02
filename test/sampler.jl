using PolytopeSamplers: PolytopeSamplers, ACHR, BilliardWalk, CoordinateHitAndRun, CRHMC,
    DikinWalk,
    HitAndRun, VaidyaWalk, MCMCSerial, MCMCThreads, Polytope, sample, warmup
using AllocCheck: check_allocs
using JET: @test_opt
using MCMCChains: Chains
using Random: Xoshiro
using Statistics: mean, var

# The statistics of one step from the pentagon's warmup
step_stats(m, w) = AbstractMCMC.getstats(last(AbstractMCMC.step(Xoshiro(1), w, m)))

# A pentagon from the original test of the package, M * x ≤ f in the box [0, 5.5]^2
const M = [3 / 2 -1; 1 1; -2 -1]
const f = [5.0, 10.0, -3.0]
const pentagon = Polytope([0.0, 0.0], [5.5, 5.5]; A = M, b = f)

# Mean and variance of the uniform distribution on the pentagon, by rejection sampling
function pentagon_moments(rng, n)
    X = 5.5 .* rand(rng, 2, n)
    keep = [all(M * x .<= f) for x in eachcol(X)]
    Y = X[:, keep]
    return vec(mean(Y; dims = 2)), vec(var(Y; dims = 2))
end

const methods = (
    HitAndRun(), ACHR(), CoordinateHitAndRun(), CoordinateHitAndRun(scan = :random), BilliardWalk(),
    DikinWalk(), VaidyaWalk(), CRHMC(),
)

# The Metropolis-filtered barrier walks need far longer chains for the fixed tolerances of
# the uniformity tests below; test/correctness.jl checks them against their own Monte Carlo
# standard errors
const fast_methods = filter(m -> !(m isa Union{DikinWalk, VaidyaWalk}), methods)

@testset "sample" begin
    @testset "output shape ($m)" for m in methods
        w = warmup(pentagon)
        @test size(sample(Xoshiro(1), w, m, 7)) == (2, 7)
        @test size(sample(w, m, 1)) == (2, 1)
        @test size(sample(pentagon, m, 3)) == (2, 3)
        chains = sample(Xoshiro(1), w, m, MCMCSerial(), 5, 3)
        @test chains isa Vector{Matrix{Float64}}
        @test size.(chains) == fill((2, 5), 3)
        @test_throws ArgumentError sample(Xoshiro(1), w, m, 10; discard_initial = -1)
    end

    @testset "reproducible from a seeded rng ($m)" for m in methods
        w = warmup(pentagon)
        X = sample(Xoshiro(3), w, m, 100)
        @test sample(Xoshiro(3), w, m, 100) == X
        @test sample(Xoshiro(3), pentagon, m, 100) == X
        @test sample(Xoshiro(3), w, m, MCMCThreads(), 50, 4) ==
            sample(Xoshiro(3), pentagon, m, MCMCSerial(), 50, 4)
        # The chain leaves the warmup center
        @test any(x -> x != w.center, eachcol(X))
    end

    @testset "chain starts ($m)" for m in methods
        w = warmup(pentagon)
        x0 = [3.0, 4.0]
        c = PolytopeSamplers.start_state(m, w, x0)
        @test c.x ≈ x0
        # The first point is one step from the given start
        X = sample(Xoshiro(1), w, m, 5; initial_params = x0)
        rng = Xoshiro(1)
        @test X[:, 1] == PolytopeSamplers.step!(rng, PolytopeSamplers.start_state(m, w, x0)).x
        @test all(x -> x in pentagon, eachcol(X))
        @test_throws ArgumentError sample(Xoshiro(1), w, m, 5; initial_params = [0.0, 0.0])
        starts = [[3.0, 4.0], [2.5, 5.0], [4.0, 3.0]]
        chains = sample(Xoshiro(1), w, m, MCMCSerial(), 5, 3; initial_params = starts, progress = false)
        @test all(X -> all(x -> x in pentagon, eachcol(X)), chains)
        @test_throws ArgumentError sample(
            Xoshiro(1), w, m, MCMCSerial(), 5, 3; initial_params = [[3.0, 4.0], [0.0, 0.0], [4.0, 3.0]], progress = false,
        )
        # Without initial_params, the chains of an ensemble start apart
        firsts = [X[:, 1] for X in sample(Xoshiro(2), w, m, MCMCSerial(), 4, 4; progress = false)]
        @test allunique(firsts)
    end

    @testset "random starts lie inside" begin
        # The center z = 0 of these warmups is not the center of the largest inscribed ball,
        # so starts drawn in a ball of the Chebyshev radius fell outside the triangle (#38)
        triangle = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])
        for extremes in (false, true), rounding in (:none, :ellipsoid, :samples)
            w = warmup(Xoshiro(1), triangle; extremes, rounding)
            starts = [PolytopeSamplers.start_state(HitAndRun(), w, Xoshiro(s)).z for s in 1:2000]
            @test all(z -> all(w.G * z .< w.h), starts)
        end
    end

    @testset "chain starts on a slice ($m)" for m in methods
        # x₁ + x₂ = 1 in the unit square, with x₃ fixed at 0.5
        P = Polytope([0.0, 0.0, 0.5], [1.0, 1.0, 0.5]; C = [1.0 1.0 0.0], d = [1.0])
        w = warmup(P)
        @test PolytopeSamplers.start_state(m, w, [0.3, 0.7, 0.5]).x ≈ [0.3, 0.7, 0.5]
        @test_throws ArgumentError sample(Xoshiro(1), w, m, 5; initial_params = [0.3, 0.6, 0.5])
        @test_throws ArgumentError sample(Xoshiro(1), w, m, 5; initial_params = [0.3, 0.7, 0.4])
    end

    @testset "MCMCChains output ($m)" for m in methods
        w = warmup(pentagon)
        X = sample(Xoshiro(1), w, m, 20; discard_initial = 3, thinning = 2)
        c = sample(Xoshiro(1), w, m, 20; discard_initial = 3, thinning = 2, chain_type = Chains)
        @test c isa Chains
        @test Array(c[:, c.name_map.parameters, :]) == X'
        @test c.name_map.parameters == [Symbol("x[1]"), Symbol("x[2]")]
        @test c.name_map.internals == collect(keys(step_stats(m, w)))
        @test range(c) == 4:2:42
        chains = sample(Xoshiro(1), w, m, MCMCSerial(), 20, 3)
        c = sample(Xoshiro(1), w, m, MCMCSerial(), 20, 3; chain_type = Chains, param_names = ["a", "b"])
        @test size(c, 3) == 3
        @test c.name_map.parameters == [:a, :b]
        @test all(j -> c.value[:, [:a, :b], j] == chains[j]', 1:3)
        @test_throws DimensionMismatch sample(Xoshiro(1), w, m, 5; chain_type = Chains, param_names = ["a"])
    end

    @testset "getstats ($m)" for m in methods
        w = warmup(pentagon)
        c = PolytopeSamplers.start_state(m, w, [3.0, 4.0])
        rng = Xoshiro(1)
        stats = [AbstractMCMC.getstats(PolytopeSamplers.step!(rng, c)) for _ in 1:500]
        if m isa Union{DikinWalk, VaidyaWalk, CRHMC}
            @test all(s -> s.is_accept isa Bool, stats)
            @test last(stats).acceptance_rate ≈ mean(s -> s.is_accept, stats)
            @test 0 < last(stats).acceptance_rate <= 1
        end
        m isa CRHMC && @test all(s -> s.step_size > 0, stats)
        if m isa BilliardWalk
            @test all(s -> s.n_reflections >= 0, stats)
            @test any(s -> s.n_reflections > 0, stats)
        end
        m isa Union{HitAndRun, ACHR, CoordinateHitAndRun} && @test isempty(last(stats))
    end

    @testset "getparams ($m)" for m in methods
        w = warmup(pentagon)
        t, c = AbstractMCMC.step(Xoshiro(1), w, m; initial_params = [3.0, 4.0])
        @test AbstractMCMC.getparams(c) == t.x
        @test AbstractMCMC.getparams(w, c) == t.x
        @test AbstractMCMC.getparams(c) !== c.x
        @test t.stats == AbstractMCMC.getstats(c)
    end

    @testset "discard_initial and thinning ($m)" for m in methods
        w = warmup(pentagon)
        long = sample(Xoshiro(5), w, m, 30)
        @test sample(Xoshiro(5), w, m, 10; discard_initial = 5) == long[:, 6:15]
        @test sample(Xoshiro(5), w, m, 10; thinning = 3) == long[:, 1:3:28]
    end

    @testset "every point is feasible ($m)" for m in methods
        # The pentagon lifted to three variables with x3 = 6 - x1, x3 ≥ 1 and x2 fixed in a
        # second copy, so every kind of constraint is active
        P = Polytope(
            [0.0, 0.0, 1.0, 2.0], [5.5, 5.5, Inf, 2.0];
            A = [M zeros(3, 2)], b = f, C = [1.0 0.0 1.0 0.0], d = [6.0],
        )
        X = sample(Xoshiro(4), P, m, 20_000; discard_initial = 100)
        @test all(x -> PolytopeSamplers.isfeasible(P, x; atol = 1.0e-10), eachcol(X))
        @test all(==(2.0), X[4, :])
        @test maximum(abs.(X[1, :] .+ X[3, :] .- 6)) < 1.0e-12
    end

    # Hit-and-run points are correlated, so these tolerances are several standard errors of
    # the thinned chains. Each check runs under several seeds.
    @testset "uniform on the unit square ($m)" for m in fast_methods
        w = warmup(Polytope(zeros(2), ones(2)))
        for seed in 1:5
            X = sample(Xoshiro(seed), w, m, 20_000; thinning = 5)
            @test all(abs.(mean(X; dims = 2) .- 1 / 2) .< 0.01)
            @test all(abs.(var(X; dims = 2) .- 1 / 12) .< 0.005)
        end
    end

    @testset "uniform on the simplex ($m)" for m in fast_methods
        # The marginals of the uniform distribution on the 3-simplex are Beta(1, 3)
        w = warmup(Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0]))
        for seed in 1:5
            X = sample(Xoshiro(seed), w, m, 20_000; thinning = 10)
            @test all(abs.(mean(X; dims = 2) .- 1 / 4) .< 0.01)
            @test all(abs.(var(X; dims = 2) .- 3 / 80) .< 0.004)
        end
    end

    @testset "uniform on the pentagon ($m)" for m in fast_methods
        μ, σ2 = pentagon_moments(Xoshiro(0), 2_000_000)
        w = warmup(pentagon)
        # The coordinates have standard deviation 1.4, so the means of 20_000 nearly
        # independent points have standard error 0.01
        for seed in 1:5
            X = sample(Xoshiro(seed), w, m, 20_000; thinning = 5)
            @test all(abs.(mean(X; dims = 2) .- μ) .< 0.06)
            @test all(abs.(var(X; dims = 2) ./ σ2 .- 1) .< 0.05)
        end
    end

    @testset "a single point ($m)" for m in methods
        X = sample(Xoshiro(1), Polytope([0.2, 0.7], [0.2, 0.7]), m, 5)
        @test all(==([0.2, 0.7]), eachcol(X))
    end

    @testset "unrounded warmup ($m)" for m in methods
        w = warmup(pentagon; rounding = false)
        X = sample(Xoshiro(1), w, m, 2000)
        @test all(x -> x in pentagon, eachcol(X))
    end

    @testset "CoordinateHitAndRun moves and scan" begin
        @test_throws ArgumentError CoordinateHitAndRun(moves = 0)
        @test_throws ArgumentError CoordinateHitAndRun(scan = :sweep)
        @test sprint(show, CoordinateHitAndRun()) == "CoordinateHitAndRun()"
        @test sprint(show, CoordinateHitAndRun(moves = 3)) == "CoordinateHitAndRun(moves = 3)"
        @test sprint(show, CoordinateHitAndRun(moves = 3, scan = :random)) ==
            "CoordinateHitAndRun(moves = 3, scan = :random)"
        # A cyclic scan with one move per step changes the coordinates of z in turn
        w5 = warmup(Polytope(zeros(5), ones(5)))
        c = PolytopeSamplers.chain_state(CoordinateHitAndRun(moves = 1), w5)
        rng = Xoshiro(3)
        changed = map(1:10) do _
            z0 = copy(c.z)
            PolytopeSamplers.step!(rng, c)
            findall(c.z .!= z0)
        end
        @test changed == [[j] for j in [1:5; 1:5]]
        w = warmup(pentagon)
        # Steps of one move, keeping every second, follow steps of two moves
        X2 = sample(Xoshiro(2), w, CoordinateHitAndRun(moves = 2), 2)
        X1 = sample(Xoshiro(2), w, CoordinateHitAndRun(moves = 1), 2; discard_initial = 1, thinning = 2)
        @test X2 ≈ X1
    end

    @testset "BilliardWalk options" begin
        @test_throws ArgumentError BilliardWalk(trajectory_length = 0.0)
        @test_throws ArgumentError BilliardWalk(max_reflections = 0)
        @test sprint(show, BilliardWalk()) == "BilliardWalk()"
        @test sprint(show, BilliardWalk(trajectory_length = 2.0, max_reflections = 5)) ==
            "BilliardWalk(trajectory_length = 2.0, max_reflections = 5)"
        w = warmup(pentagon)
        c = PolytopeSamplers.chain_state(BilliardWalk(), w)
        @test c.trajectory_length ≈ 2 * sqrt(2) * w.radius
        @test c.max_reflections == 2000
        # Long trajectories with one reflection allowed are mostly abandoned: the chain stays
        # put for those steps and every point stays in the polytope
        X = sample(Xoshiro(1), w, BilliardWalk(trajectory_length = 50.0, max_reflections = 1), 200)
        @test all(x -> x in pentagon, eachcol(X))
        @test count(j -> X[:, j] == X[:, j - 1], 2:200) > 100
    end

    # CRHMC keeps slacks with its cached metric, which may be at the last proposal
    @testset "slacks updated in place match recomputed slacks ($m)" for m in filter(m -> !(m isa CRHMC), methods)
        w = warmup(pentagon)
        c = PolytopeSamplers.chain_state(m, w)
        rng = Xoshiro(4)
        for _ in 1:1_000
            PolytopeSamplers.step!(rng, c)
            @test c.slack ≈ w.h .- w.G * c.z atol = 1.0e-10
        end
    end

    @testset "steps are type stable and allocation free ($m)" for m in methods
        c = PolytopeSamplers.chain_state(m, warmup(pentagon))
        rng = Xoshiro(1)
        @test_opt PolytopeSamplers.step!(rng, c)
        @test isempty(check_allocs(PolytopeSamplers.step!, (typeof(rng), typeof(c))))
    end
end
