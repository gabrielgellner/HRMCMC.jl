# Stress tests of convergence, run with `just stress` (several minutes; not part of
# `just ci`). Each block prints what it measures, then checks it.
#
# - Diagnostics flag chains that have not converged, and estimates are right whenever they
#   say a run has converged.
# - Mixing cost grows with dimension as the literature reports: coordinate moves per
#   effective sample as n² for CHRR (Haraldsdóttir et al. 2017, whose COBRA default
#   spacing is 8n² moves), and few steps per effective sample for the billiard walk
#   (Chalkis et al. 2021).
# - Hard geometry, each sampler run until max R̂ < 1.01 and then checked against the exact
#   moments: extreme aspect ratios, redundant and duplicate constraints, implicit
#   equalities, many facets (cross-polytope), equality polytopes (Birkhoff), 200 dimensions.
# - Two converged samplers agree on a random polytope.

using PolytopeSamplers: PolytopeSamplers, ACHR, BilliardWalk, CoordinateHitAndRun, CRHMC, volume,
    DikinWalk,
    HitAndRun, VaidyaWalk,
    MCMCSerial, Polytope, diagnostics, mmcs, sample, warmup
using AbstractMCMC: AbstractMCMC
using LinearAlgebra: I, dot, norm
using Printf: @printf
using Random: Xoshiro, randn
using Statistics: mean
using Test: @test, @testset

AbstractMCMC.setprogress!(false)

const all_methods = (
    HitAndRun(), ACHR(), CoordinateHitAndRun(), CoordinateHitAndRun(scan = :random), BilliardWalk(),
    DikinWalk(), VaidyaWalk(), CRHMC(),
)

# Every mean within five Monte Carlo standard errors of the truth, every standard deviation
# within six (see test/correctness.jl)
function moments_match(d, μ, σ)
    return all(@. abs(d.mean - μ) < 5 * d.mcse_mean) && all(@. abs(d.std - σ) < 6 * d.mcse_std)
end

function report(label, d, seconds)
    @printf(
        "  %-44s min ESS %7.0f  max R̂ %6.3f  %8.0f ESS/s\n", label, d.min_ess, d.max_rhat,
        d.min_ess / seconds,
    )
    return
end

function run_chains(w, m, n; seed = 1, kwargs...)
    t = @elapsed chains = sample(Xoshiro(seed), w, m, MCMCSerial(), n, 4; kwargs...)
    return diagnostics(chains), t
end

# Double the chain length until max R̂ < 1.01, the protocol users should follow, up to
# `cap` points per chain. Returns the diagnostics of the last run, its length and time.
function run_to_convergence(w, m; n = 1_000, cap = 64_000, thinning = 5, seed = 1)
    # The barrier walks need about 100 times the steps per effective sample of the others
    if m isa Union{DikinWalk, VaidyaWalk}
        thinning *= 5
        cap ÷= 2
    end
    while true
        d, t = run_chains(w, m, n; seed, discard_initial = n ÷ 4, thinning)
        (d.max_rhat < 1.01 || 2n > cap) && return d, n, t
        n *= 2
    end
    return
end

function report_converged(label, d, n, t)
    @printf(
        "  %-44s n %6d  min ESS %7.0f  max R̂ %6.3f  %8.0f ESS/s\n", label, n, d.min_ess,
        d.max_rhat, d.min_ess / t,
    )
    return
end

# Chain states starting at `fraction` of the way from the center to the extreme points of
# a warmup built with `extremes = true`
function starts(m, w, fraction)
    return map(1:4) do j
        s = PolytopeSamplers.chain_state(m, w)
        s.z .= fraction .* w.zpoints[:, j]
        s
    end
end

simplex(n) = Polytope(zeros(n), ones(n); C = ones(1, n), d = [1.0])
# Marginals of the uniform distribution on {x ≥ 0, sum(x) = 1} in n variables: Beta(1, n - 1)
simplex_std(n) = sqrt((n - 1) / (n^2 * (n + 1)))

@testset "stress" begin
    @testset "diagnostics flag chains started near the corners" begin
        # Chains start 90% of the way to four vertices of the 49-dimensional simplex. Short
        # runs must show R̂ > 1.05; any run with R̂ < 1.01 must have correct moments.
        w = warmup(simplex(50); extremes = true)
        println("Overdispersed starts on the 49-dimensional simplex")
        for m in all_methods
            reached = false
            for n in (100, 1_000, 10_000)
                d, t = run_chains(w, m, n; initial_state = starts(m, w, 0.9))
                report("$m, n = $n", d, t)
                n == 100 && @test d.max_rhat > 1.05
                if d.max_rhat < 1.01
                    reached = true
                    @test moments_match(d, fill(0.02, 50), fill(simplex_std(50), 50))
                end
            end
            m isa BilliardWalk && @test reached
        end
    end

    @testset "coordinate hit-and-run cannot leave a vertex" begin
        # At a vertex of the simplex no coordinate axis points into the polytope, so every
        # coordinate chord has length zero; R̂ shows the chains are stuck
        w = warmup(simplex(20); extremes = true)
        d, t = run_chains(w, CoordinateHitAndRun(), 1_000; initial_state = starts(CoordinateHitAndRun(), w, 1.0))
        println("CHRR started at vertices of the 19-dimensional simplex")
        report("CoordinateHitAndRun(), n = 1000", d, t)
        @test d.max_rhat > 1.5
        d, t = run_chains(w, BilliardWalk(), 1_000; initial_state = starts(BilliardWalk(), w, 1.0))
        report("BilliardWalk(), n = 1000", d, t)
        @test d.max_rhat < 1.01
    end

    @testset "mixing cost against dimension" begin
        println("Coordinate moves (CHRR) or steps (others) per effective sample, simplex")
        dims = [10, 20, 40, 80]
        slopes = Dict{String, Float64}()
        for m in (CoordinateHitAndRun(), HitAndRun(), BilliardWalk(), DikinWalk(), VaidyaWalk())
            cost = map(dims) do k
                w = warmup(simplex(k + 1))
                d, _ = run_chains(w, m, 4_000; seed = k, discard_initial = 2_000)
                per_step = 4 * 4_000 / d.min_ess
                m isa CoordinateHitAndRun ? per_step * k : per_step
            end
            x = log.(dims)
            y = log.(cost)
            slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
            slopes[string(m)] = slope
            @printf("  %-24s %s   slope %.2f\n", m, join(round.(cost; sigdigits = 3), "  "), slope)
        end
        @test 1.5 < slopes["CoordinateHitAndRun()"] < 2.6
        @test slopes["BilliardWalk()"] < 0.7
    end

    @testset "aspect ratio 10^6 ($m)" for m in all_methods
        sides = 10.0 .^ range(-3, 3; length = 20)
        w = warmup(Polytope(zeros(20), sides))
        d, n, t = run_to_convergence(w, m)
        report_converged("box with sides 1e-3 to 1e3, $m", d, n, t)
        @test d.max_rhat < 1.01
        @test moments_match(d, sides ./ 2, sides ./ sqrt(12))
    end

    @testset "redundant and duplicate constraints ($m)" for m in all_methods
        # The cube [-1, 1]^10 with 500 random halfspaces that do not cut it and each facet
        # repeated three times. The samplers see every row with `prune = false`; pruning
        # leaves one copy of each of the 20 facets
        rng = Xoshiro(7)
        R = randn(rng, 500, 10)
        A = [R; repeat([1.0I(10); -1.0I(10)], 3)]
        b = [vec(sum(abs, R; dims = 2)) .+ 0.5; ones(60)]
        P = Polytope(fill(-2.0, 10), fill(2.0, 10); A, b)
        w = warmup(P; prune = false)
        @test size(w.G, 1) == 580
        @test size(warmup(P).G, 1) == 20
        d, n, t = run_to_convergence(w, m)
        report_converged("cube with 560 extra rows, $m", d, n, t)
        @test d.max_rhat < 1.01
        @test moments_match(d, zeros(10), fill(sqrt(1 / 3), 10))
    end

    @testset "implicit equalities and fixed coordinates ($m)" for m in all_methods
        # 30 variables: x1..x10 free in [0, 1]; x11..x20 tied to x1..x10 by pairs of
        # inequalities; x21..x30 fixed by their bounds
        n = 30
        A = zeros(20, n)
        for i in 1:10
            A[2i - 1, i] = 1
            A[2i - 1, 10 + i] = -1
            A[2i, i] = -1
            A[2i, 10 + i] = 1
        end
        lower = [zeros(20); fill(0.25, 10)]
        upper = [ones(20); fill(0.25, 10)]
        P = Polytope(lower, upper; A, b = zeros(20))
        w = warmup(P)
        @test size(w.transform, 2) == 10
        d, n, t = run_to_convergence(w, m)
        report_converged("10 free of 30 variables, $m", d, n, t)
        free = 1:20
        @test d.max_rhat < 1.01
        @test moments_match(
            (; mean = d.mean[free], std = d.std[free], mcse_mean = d.mcse_mean[free], mcse_std = d.mcse_std[free]),
            fill(0.5, 20), fill(sqrt(1 / 12), 20),
        )
        @test all(isnan, d.ess[21:30])
    end

    @testset "cross-polytope ($m)" for m in all_methods
        # |x1| + ... + |x8| ≤ 1, 256 facets; E[x_i^2] = 2 / ((d + 1)(d + 2))
        dim = 8
        A = [((j >> (i - 1)) & 1 == 1 ? 1.0 : -1.0) for j in 0:(2^dim - 1), i in 1:dim]
        P = Polytope(fill(-1.0, dim), fill(1.0, dim); A, b = ones(2^dim))
        w = warmup(P)
        d, n, t = run_to_convergence(w, m)
        report_converged("cross-polytope, 256 facets, $m", d, n, t)
        @test d.max_rhat < 1.01
        @test moments_match(d, zeros(dim), fill(sqrt(2 / ((dim + 1) * (dim + 2))), dim))
    end

    @testset "Birkhoff polytope B_4 ($m)" for m in all_methods
        # Doubly stochastic 4 × 4 matrices: 16 variables, 8 equalities of rank 7. By
        # symmetry every entry has mean 1/4 and the same standard deviation.
        C = zeros(8, 16)
        for i in 1:4, j in 1:4
            C[i, 4(j - 1) + i] = 1
            C[4 + j, 4(j - 1) + i] = 1
        end
        w = warmup(Polytope(zeros(16), ones(16); C, d = ones(8)))
        @test size(w.transform, 2) == 9
        # With the redundant upper bounds pruned, ACHR's largest |z| of the 16 means exceeds
        # 5 on 1 of 90 seeds (seed 1, 5.15) and its median is 1.9, as without pruning
        d, n, t = run_to_convergence(w, m; seed = 2)
        report_converged("Birkhoff B_4, $m", d, n, t)
        @test d.max_rhat < 1.01
        σ = mean(d.std)
        @test moments_match(d, fill(0.25, 16), fill(σ, 16))
    end

    @testset "samples stay inside the polytope ($m)" for m in all_methods
        # 100 000 steps each on a simplex, a box of aspect ratio 10^6, the cross-polytope and
        # a pentagon: no inequality is violated by more than 1e-12 relative to its scale,
        # equalities hold to 1e-13, and no billiard trajectory is abandoned
        dim = 8
        cross = Polytope(
            fill(-1.0, dim), fill(1.0, dim);
            A = [((j >> (i - 1)) & 1 == 1 ? 1.0 : -1.0) for j in 0:(2^dim - 1), i in 1:dim],
            b = ones(2^dim),
        )
        polytopes = [
            simplex(50), Polytope(zeros(20), 10.0 .^ range(-3, 3; length = 20)), cross,
            Polytope([0.0, 0.0], [5.5, 5.5]; A = [3 / 2 -1; 1 1; -2 -1], b = [5.0, 10.0, -3.0]),
        ]
        worst_inequality = 0.0
        worst_equality = 0.0
        abandoned = 0
        for P in polytopes
            w = warmup(P)
            c = PolytopeSamplers.chain_state(m, w)
            rng = Xoshiro(1)
            for _ in 1:100_000
                z0 = copy(c.z)
                PolytopeSamplers.step!(rng, c)
                m isa BilliardWalk && c.z == z0 && (abandoned += 1)
                x = c.x
                scale = max(1, norm(x))
                for i in eachindex(x)
                    worst_inequality = max(worst_inequality, (x[i] - P.upper[i]) / max(1, abs(P.upper[i])))
                    worst_inequality = max(worst_inequality, (P.lower[i] - x[i]) / max(1, abs(P.lower[i])))
                end
                for i in eachindex(P.b)
                    a = view(P.A, i, :)
                    worst_inequality = max(worst_inequality, (dot(a, x) - P.b[i]) / (norm(a) * scale))
                end
                for i in eachindex(P.d)
                    a = view(P.C, i, :)
                    worst_equality = max(worst_equality, abs(dot(a, x) - P.d[i]) / (norm(a) * scale))
                end
            end
        end
        @printf("  %-44s worst inequality %8.1e  worst equality %8.1e\n", m, worst_inequality, worst_equality)
        @test worst_inequality <= 1.0e-12
        @test worst_equality <= 1.0e-13
        @test abandoned == 0
    end

    @testset "200-dimensional simplex" begin
        t = @elapsed w = warmup(simplex(201))
        @printf("200-dimensional simplex: warmup and rounding %.1f s\n", t)
        d, t = run_chains(w, BilliardWalk(), 2_000; discard_initial = 200)
        report("BilliardWalk()", d, t)
        @test d.max_rhat < 1.01
        @test moments_match(d, fill(1 / 201, 201), fill(simplex_std(201), 201))
    end

    @testset "mmcs on hard geometry" begin
        # One linear program and rounding from samples, then billiard chains to 1000 ESS
        dim = 8
        cross = Polytope(
            fill(-1.0, dim), fill(1.0, dim);
            A = [((j >> (i - 1)) & 1 == 1 ? 1.0 : -1.0) for j in 0:(2^dim - 1), i in 1:dim],
            b = ones(2^dim),
        )
        sides = 10.0 .^ range(-3, 3; length = 20)
        C = zeros(8, 16)
        for i in 1:4, j in 1:4
            C[i, 4(j - 1) + i] = 1
            C[4 + j, 4(j - 1) + i] = 1
        end
        cases = [
            ("box with sides 1e-3 to 1e3", Polytope(zeros(20), sides), sides ./ 2, sides ./ sqrt(12)),
            ("cross-polytope, 256 facets", cross, zeros(dim), fill(sqrt(2 / ((dim + 1) * (dim + 2))), dim)),
            ("Birkhoff B_4", Polytope(zeros(16), ones(16); C, d = ones(8)), fill(0.25, 16), nothing),
            ("200-dimensional simplex", simplex(201), fill(1 / 201, 201), fill(simplex_std(201), 201)),
        ]
        println("mmcs")
        # On the 200-dimensional simplex, seed 1 gives a largest |z| of the 201 standard
        # deviations near the threshold of 6 (5.96 before #89, 6.36 after); over 24 seeds the
        # medians are 3.7 and 4.0, and seed 2 gives 3.2 and 3.0
        for (name, P, μ, σ) in cases
            t = @elapsed r = mmcs(Xoshiro(2), P; target_ess = 1_000)
            d = r.diagnostics
            @printf("  %-44s n %6d  min ESS %7.0f  max R̂ %6.3f  %6.1f s\n", name, r.n, d.min_ess, d.max_rhat, t)
            @test r.converged
            @test moments_match(d, μ, something(σ, fill(mean(d.std), length(μ))))
        end
    end

    @testset "two samplers agree on a random polytope" begin
        # 15 dimensions, 60 random facets inside the box [-3, 3]^15: no exact answer, so
        # converged CHRR and billiard chains must agree within their standard errors
        rng = Xoshiro(11)
        A = randn(rng, 60, 15)
        P = Polytope(fill(-3.0, 15), fill(3.0, 15); A, b = ones(60) .+ rand(rng, 60))
        w = warmup(P)
        a, ta = run_chains(w, CoordinateHitAndRun(), 4_000; discard_initial = 500, thinning = 2)
        b, tb = run_chains(w, BilliardWalk(), 4_000; seed = 2, discard_initial = 500)
        println("Random polytope, 15 dimensions, 60 facets")
        report("CoordinateHitAndRun()", a, ta)
        report("BilliardWalk()", b, tb)
        @test a.max_rhat < 1.01 && b.max_rhat < 1.01
        @test all(@. abs(a.mean - b.mean) < 5 * sqrt(a.mcse_mean^2 + b.mcse_mean^2))
        @test all(@. abs(a.std - b.std) < 6 * sqrt(a.mcse_std^2 + b.mcse_std^2))
    end

    @testset "volume error bars cover the exact volume" begin
        # z-scores of the log volume over 50 seeds, on polytopes needing two to ten phases
        for (P, exact) in [
                (Polytope(zeros(40), ones(40)), 0.0),
                (Polytope(zeros(20), ones(20); A = ones(1, 20), b = [1.0]), -sum(log, 1:20)),
                (Polytope(zeros(15), ones(15); C = ones(1, 15), d = [1.0]), log(15) / 2 - sum(log, 1:14)),
                (Polytope(zeros(60), ones(60); A = ones(1, 60), b = [1.0]), -sum(log, 1:60)),
            ]
            w = warmup(P)
            z = [(v = volume(Xoshiro(100 + s), w); (v.log_volume - exact) / v.log_stderr) for s in 1:50]
            @printf "volume z-scores: rms %.2f, mean %.2f, %d of 50 within 2\n" sqrt(mean(abs2, z)) mean(z) count(<(2), abs.(z))
            @test 0.7 < sqrt(mean(abs2, z)) < 1.4
            @test count(<(2), abs.(z)) >= 42
        end
    end
end
