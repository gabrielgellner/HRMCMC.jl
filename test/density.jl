using PolytopeSamplers: PolytopeSamplers, ACHR, BilliardWalk, CRHMC, CoordinateHitAndRun,
    DikinWalk, ExponentialTarget, GaussianTarget, HitAndRun, MCMCSerial, Polytope, Target, VaidyaWalk,
    diagnostics, sample, sample_until_converged, warmup
using AllocCheck: check_allocs
using Distributions: Gamma, Normal, cdf, truncated
using JET: @test_opt
using LogDensityProblems: LogDensityProblems, LogDensityOrder
using Random: Xoshiro
using LinearAlgebra: I
using Statistics: cor, mean, std

const density_methods = (
    HitAndRun(), CoordinateHitAndRun(), CoordinateHitAndRun(scan = :random), DikinWalk(),
    VaidyaWalk(), CRHMC(),
)

# Means within five and standard deviations within six Monte Carlo standard errors
function moments_match(d, μ, σ)
    return all(@. abs(d.mean - μ) < 5 * d.mcse_mean) && all(@. abs(d.std - σ) < 6 * d.mcse_std)
end

# A log density implementing LogDensityProblems directly: an isotropic Gaussian
struct Gaussian
    μ::Vector{Float64}
    σ::Float64
end
LogDensityProblems.dimension(g::Gaussian) = length(g.μ)
LogDensityProblems.capabilities(::Type{Gaussian}) = LogDensityOrder{1}()
LogDensityProblems.logdensity(g::Gaussian, x) = -sum(abs2, x .- g.μ) / (2 * g.σ^2)
function LogDensityProblems.logdensity_and_gradient(g::Gaussian, x)
    return LogDensityProblems.logdensity(g, x), -(x .- g.μ) ./ g.σ^2
end

@testset "non-uniform targets" begin
    @testset "getstats with a density ($m)" for m in density_methods
        P = Polytope(zeros(2), ones(2))
        t = Target(P, Gaussian([0.3, 0.6], 0.5))
        c = PolytopeSamplers.start_state(m, t, [0.5, 0.5])
        rng = Xoshiro(1)
        for _ in 1:50
            PolytopeSamplers.step!(rng, c)
            s = AbstractMCMC.getstats(c)
            @test s.lp ≈ LogDensityProblems.logdensity(t.logdensity, c.x)
            m isa Union{HitAndRun, CoordinateHitAndRun} && @test s.n_evaluations >= 1
            m isa Union{HitAndRun, CoordinateHitAndRun} && @test s.n_exhausted == 0
        end
    end

    @testset "starts where the density is zero ($m)" for m in density_methods
        # The density is zero on x₁ < 0.5, which covers half of the ball of random starts.
        # Random starts are drawn again until the density is positive; a chain started
        # where it is zero throws, as it would accept every point of its first chord (#70)
        w = warmup(Polytope(zeros(2), ones(2)))
        ℓ(x) = x[1] >= 0.5 ? 0.0 : -Inf
        t = Target(w, ℓ; gradient = (g, x) -> (g .= 0; ℓ(x)))
        for seed in 1:20
            c = PolytopeSamplers.start_state(m, t, Xoshiro(seed))
            @test c.x[1] >= 0.5
        end
        @test_throws ArgumentError sample(Xoshiro(1), t, m, 10; initial_params = [0.2, 0.5], progress = false)
        # No random start in the ball has positive density
        far = Target(w, x -> x[1] > 0.99 ? 0.0 : -Inf; gradient = (g, x) -> (g .= 0; x[1] > 0.99 ? 0.0 : -Inf))
        @test_throws ArgumentError sample(Xoshiro(1), far, m, 10; progress = false)
    end

    @testset "exhausted slices are counted" begin
        # Shrinkage ends at the current point unless its density has changed since it was
        # evaluated, so this log density is finite on its first call only: every point of
        # the slice is rejected and the chain stays put
        calls = Ref(0)
        t = Target(Polytope(zeros(2), ones(2)), x -> (calls[] += 1) == 1 ? 0.0 : -Inf)
        c = PolytopeSamplers.start_state(HitAndRun(), t, [0.5, 0.5])
        PolytopeSamplers.step!(Xoshiro(1), c)
        @test AbstractMCMC.getstats(c).n_exhausted == 1
        @test c.x == [0.5, 0.5]
    end

    @testset "Target" begin
        P = Polytope(zeros(3), ones(3))
        t = Target(P, x -> -sum(x))
        @test t isa Target
        @test sprint(show, t) == "Target on a 3-dimensional slice of a 3-variable polytope"
        @test LogDensityProblems.capabilities(t.logdensity) == LogDensityOrder{0}()
        tg = Target(P, x -> -sum(x); gradient = (g, x) -> fill!(g, -1))
        @test LogDensityProblems.capabilities(tg.logdensity) == LogDensityOrder{1}()
        value, gradient = LogDensityProblems.logdensity_and_gradient(tg.logdensity, [0.1, 0.2, 0.3])
        @test value ≈ -0.6
        @test gradient == fill(-1.0, 3)
        @test Target(warmup(P), Gaussian(zeros(3), 1.0)) isa Target
        @test_throws DimensionMismatch Target(P, Gaussian(zeros(2), 1.0))
        @test_throws ArgumentError Target(P, Gaussian(zeros(3), 1.0); gradient = (g, x) -> g)
        @test_throws ArgumentError Target(P, 1.0)
    end

    @testset "unsupported samplers" begin
        t = Target(Polytope(zeros(3), ones(3)), x -> -sum(x))
        @test_throws ArgumentError sample(Xoshiro(1), t, BilliardWalk(), 10)
        @test_throws ArgumentError sample(Xoshiro(1), t, ACHR(), 10)
        # CRHMC needs the gradient
        @test_throws ArgumentError sample(Xoshiro(1), t, CRHMC(), 10)
    end

    @testset "reproducible ($m)" for m in density_methods
        t = Target(Polytope(zeros(3), ones(3)), x -> -sum(x); gradient = (g, x) -> fill!(g, -1))
        @test sample(Xoshiro(2), t, m, 50) == sample(Xoshiro(2), t, m, 50)
        @test all(x -> x in t.warmup.polytope, eachcol(sample(Xoshiro(2), t, m, 200)))
    end

    @testset "Dirichlet on the simplex ($m)" for m in density_methods
        # Marginals of Dirichlet(α) are Beta(αᵢ, Σα - αᵢ)
        α = [2.0, 3.0, 4.0, 5.0, 6.0]
        a = sum(α)
        P = Polytope(zeros(5), ones(5); C = ones(1, 5), d = [1.0])
        t = Target(P, x -> sum((α .- 1) .* log.(x)); gradient = (g, x) -> (g .= (α .- 1) ./ x))
        r = sample_until_converged(Xoshiro(3), t, m; target_ess = 1_000)
        @test r.converged
        @test moments_match(r.diagnostics, α ./ a, sqrt.(α .* (a .- α) ./ (a^2 * (a + 1))))
    end

    @testset "truncated exponentials on a box ($m)" for m in density_methods
        # exp(-λᵢ xᵢ) on [0, 1]: mean 1/λ - 1/(e^λ - 1), variance 1/λ² - e^λ/(e^λ - 1)², a
        # concentrated density for λ = 20
        λ = [1.0, 5.0, 20.0]
        t = Target(Polytope(zeros(3), ones(3)), x -> -sum(λ .* x); gradient = (g, x) -> (g .= -λ))
        μ = @. 1 / λ - 1 / (exp(λ) - 1)
        σ = @. sqrt(1 / λ^2 - exp(λ) / (exp(λ) - 1)^2)
        r = sample_until_converged(Xoshiro(4), t, m; target_ess = 1_000)
        @test r.converged
        @test moments_match(r.diagnostics, μ, σ)
    end

    @testset "correlated Gaussian ($m)" for m in density_methods
        # N(0, Σ) with correlation 0.8, inside a box 8 standard deviations wide
        Σ = [1.0 0.8; 0.8 1.0]
        Σinv = inv(Σ)
        t = Target(
            Polytope(fill(-8.0, 2), fill(8.0, 2)), x -> -(x' * Σinv * x) / 2;
            gradient = (g, x) -> (g .= -(Σinv * x)),
        )
        r = sample_until_converged(Xoshiro(5), t, m; target_ess = 1_000)
        @test r.converged
        @test moments_match(r.diagnostics, zeros(2), ones(2))
        # The correlation has standard error about (1 - ρ²)/√ESS = 0.011
        @test cor(reduce(hcat, r.chains)')[1, 2] ≈ 0.8 atol = 0.06
    end

    @testset "densities that diverge at a face ($m)" for m in density_methods
        # gamma(1/4, θ) on each coordinate of [0, 1]³ diverges at xᵢ = 0; rounding puts chains
        # on or just past that face, where the density is infinite or log(xᵢ) is undefined
        θ, k = [0.5, 1.0, 2.0], 0.25
        t = Target(
            Polytope(zeros(3), ones(3)), x -> sum(@. (k - 1) * log(x) - x / θ);
            gradient = (g, x) -> (g .= (k - 1) ./ x .- 1 ./ θ),
        )
        n = m isa Union{DikinWalk, VaidyaWalk} ? 4_000 : 20_000
        chains = sample(Xoshiro(1), t, m, MCMCSerial(), n, 4)
        # no chain stops moving
        @test all(X -> !all(==(X[:, end]), eachcol(X[:, (end - 99):end])), chains)
        if !(m isa Union{DikinWalk, VaidyaWalk})
            # truncated gamma moments, E[xʲ] = θʲ Γ(k + j)/Γ(k) F_{k+j}(1)/F_k(1)
            F(a, θi) = cdf(Gamma(a, θi), 1)
            μ = [k * θi * F(k + 1, θi) / F(k, θi) for θi in θ]
            m2 = [k * (k + 1) * θi^2 * F(k + 2, θi) / F(k, θi) for θi in θ]
            @test moments_match(diagnostics(chains), μ, sqrt.(m2 .- μ .^ 2))
        end
    end

    @testset "log densities at faces" begin
        P = Polytope(zeros(2), ones(2))
        w = warmup(P; rounding = false)
        d = PolytopeSamplers.ZDensity(Target(w, x -> sum(log, x)))
        # a z just outside the face x₁ = 0 maps to x₁ = 0, where log(0) = -Inf
        z = (w.transform \ ([-1.0e-17, 0.5] .- w.center))
        @test PolytopeSamplers.to_x!(d, z)[1] == 0
        @test PolytopeSamplers.logdensity_z(d, z) == -Inf
        # infinite and undefined log densities are outside the support
        @test PolytopeSamplers.logdensity_z(PolytopeSamplers.ZDensity(Target(w, x -> Inf)), zeros(2)) == -Inf
        @test PolytopeSamplers.logdensity_z(PolytopeSamplers.ZDensity(Target(w, x -> NaN)), zeros(2)) == -Inf
    end

    @testset "truncated normal draws" begin
        rng = Xoshiro(6)
        # Inside, in both tails, far in a tail, narrow and one-sided intervals
        for (α, β) in [(-1.0, 1.0), (-3.0, 3.0), (0.5, Inf), (-Inf, -0.5), (10.0, Inf), (5.0, 5.01), (2.0, 2.5), (-0.1, 100.0), (-Inf, Inf)]
            z = [PolytopeSamplers.truncated_randn(rng, α, β) for _ in 1:100_000]
            d = truncated(Normal(), α, β)
            @test all(x -> α <= x <= β, z)
            @test abs(mean(z) - mean(d)) < 5 * std(d) / sqrt(length(z))
            @test std(z) ≈ std(d) rtol = 0.02
        end
        # Exponential and uniform chords
        z = [PolytopeSamplers.draw_quadratic(rng, 0.0, 2.0, 0.0, 1.0) for _ in 1:100_000]
        @test abs(mean(z) - (1 / 2 - 1 / (exp(2) - 1))) < 0.003
        z = [PolytopeSamplers.draw_quadratic(rng, 0.0, -2.0, 0.0, 1.0) for _ in 1:100_000]
        @test abs(mean(z) - (1 / 2 + 1 / (exp(2) - 1))) < 0.003
        z = [PolytopeSamplers.draw_quadratic(rng, 0.0, 0.0, -1.0, 3.0) for _ in 1:100_000]
        @test abs(mean(z) - 1) < 0.02
        # Modes far outside the chord, where m + σ z cancelled to one value outside the chord
        # (#42): the density is close to an exponential with rate b near tmin, or near tmax
        # for negative b
        for (a, b) in [(2.0e-12, 1.0e3), (1.0e-6, 1.0e2), (2.0e-12, -1.0e3), (1.0e-6, -1.0e2)]
            z = [PolytopeSamplers.draw_quadratic(rng, a, b, 0.0, 1.0) for _ in 1:100_000]
            @test all(t -> 0 <= t <= 1, z)
            @test allunique(z[1:100])
            dist = b > 0 ? mean(z) : 1 - mean(z)
            @test abs(dist * abs(b) - 1) < 0.02
        end
    end

    @testset "GaussianTarget and ExponentialTarget" begin
        box = Polytope(fill(-1.0, 2), fill(1.0, 2))
        t = GaussianTarget(box, [0.5, 0.0], [1.0 0.8; 0.8 1.0])
        @test t isa Target
        x = [0.2, -0.3]
        r = x .- [0.5, 0.0]
        @test LogDensityProblems.logdensity(t.logdensity, x) ≈ -(r' * inv([1.0 0.8; 0.8 1.0]) * r) / 2
        value, g = LogDensityProblems.logdensity_and_gradient(t.logdensity, x)
        @test g ≈ -inv([1.0 0.8; 0.8 1.0]) * r
        @test_throws ArgumentError GaussianTarget(box, zeros(2), [1.0 2.0; 2.0 1.0])
        @test_throws DimensionMismatch GaussianTarget(box, zeros(3), [1.0 0.0; 0.0 1.0])
        @test_throws DimensionMismatch ExponentialTarget(box, zeros(3))
        e = ExponentialTarget(box, [1.0, -2.0])
        @test LogDensityProblems.logdensity(e.logdensity, x) ≈ -(0.2 - 2 * -0.3)
        @test LogDensityProblems.logdensity_and_gradient(e.logdensity, x)[2] == [-1.0, 2.0]
    end

    @testset "exact chord steps are type stable and allocation free ($m)" for m in (HitAndRun(), CoordinateHitAndRun())
        box = Polytope(fill(-1.0, 3), fill(1.0, 3))
        for t in (GaussianTarget(box, zeros(3), [1.0 0.5 0.0; 0.5 1.0 0.0; 0.0 0.0 2.0]), ExponentialTarget(box, ones(3)))
            c = PolytopeSamplers.chain_state(m, t)
            @test c.density isa PolytopeSamplers.ZQuadratic
            rng = Xoshiro(1)
            @test_opt PolytopeSamplers.step!(rng, c)
            @test isempty(check_allocs(PolytopeSamplers.step!, (typeof(rng), typeof(c))))
        end
    end

    @testset "density steps are allocation free ($m)" for m in density_methods
        # A function target with an in-place gradient, and a Gaussian given as a target, go
        # through the chain's buffers (#52); before, CRHMC allocated 96 to 576 bytes a step
        w = warmup(Xoshiro(1), Polytope(zeros(5), ones(5); A = ones(1, 5), b = [3.0]))
        ℓ(x) = -sum(abs2, x) / 2
        targets = (Target(w, ℓ; gradient = (g, x) -> (g .= .-x)), GaussianTarget(w, fill(0.5, 5), Matrix(0.1I(5)); whiten = false))
        for t in targets
            c = PolytopeSamplers.start_state(m, t, Xoshiro(1))
            rng = Xoshiro(2)
            for _ in 1:10
                PolytopeSamplers.step!(rng, c)
            end
            @test (@allocated PolytopeSamplers.step!(rng, c)) == 0
        end
    end

    @testset "x follows z on density chords" begin
        # Slice evaluations move x along T u instead of recomputing it (#52); after many
        # moves it still matches center + T z
        w = warmup(Xoshiro(1), Polytope(zeros(4), [1.0, 2.0, 3.0, 4.0]; A = ones(1, 4), b = [5.0]))
        t = Target(w, x -> -sum(abs2, x .- 1) / 2)
        for m in (HitAndRun(), CoordinateHitAndRun())
            c = PolytopeSamplers.start_state(m, t, Xoshiro(1))
            rng = Xoshiro(3)
            for _ in 1:250
                PolytopeSamplers.advance!(rng, c)
            end
            @test c.x ≈ w.center .+ w.transform * c.z atol = 1.0e-10
        end
    end

    @testset "whitened Gaussian targets" begin
        # Correlation 0.99 on [-10, 10]²: CHRR had a minimum ESS of 64 and R̂ 1.09 from 4 chains
        # of 2000 sweeps in the polytope's coordinates; whitened, as hops samples truncated
        # Gaussians (#52), the Gaussian is round in the chain's coordinates. Whitening is off
        # by default, as it slowed sampling on a food web (see the docstring)
        box = Polytope(fill(-10.0, 2), fill(10.0, 2))
        Σ = [1.0 0.99; 0.99 1.0]
        t = GaussianTarget(box, zeros(2), Σ; whiten = true)
        d = diagnostics(sample(Xoshiro(1), t, CoordinateHitAndRun(), MCMCSerial(), 2000, 4; progress = false))
        @test d.min_ess > 2000
        @test d.max_rhat < 1.01
        @test moments_match(d, zeros(2), ones(2))
        # Directions where the Gaussian is wider than the polytope are not stretched
        wide = GaussianTarget(Polytope(zeros(2), ones(2)), [0.5, 0.5], Matrix(100.0I(2)); whiten = true)
        @test wide.warmup.transform ≈ warmup(Polytope(zeros(2), ones(2))).transform
    end

    @testset "truncated Gaussian on a box ($m)" for m in density_methods
        # Independent coordinates: truncated normal marginals
        μ, σ = [0.5, 0.0, -2.0], [0.7, 0.3, 1.0]
        t = GaussianTarget(Polytope(fill(-1.0, 3), fill(1.0, 3)), μ, [σ[i]^2 * (i == j) for i in 1:3, j in 1:3])
        d = [truncated(Normal(μ[i], σ[i]), -1, 1) for i in 1:3]
        for seed in 1:3
            r = sample_until_converged(Xoshiro(seed), t, m; target_ess = 1_000)
            @test r.converged
            @test moments_match(r.diagnostics, mean.(d), std.(d))
        end
    end

    @testset "ExponentialTarget on a box ($m)" for m in density_methods
        # exp(-c x) on [-1, 1]: mean 1/c - coth(c), variance 1/c² - csch²(c)
        c = [1.0, -5.0, 20.0]
        t = ExponentialTarget(Polytope(fill(-1.0, 3), fill(1.0, 3)), c)
        for seed in 1:3
            r = sample_until_converged(Xoshiro(seed), t, m; target_ess = 1_000)
            @test r.converged
            @test moments_match(r.diagnostics, @.(1 / c - coth(c)), @.(sqrt(1 / c^2 - csch(c)^2)))
        end
    end

    @testset "exact and slice draws agree on a slice ($m)" for m in (HitAndRun(), CoordinateHitAndRun())
        # A correlated Gaussian on the simplex, where the warmup coordinates are rotated
        μ = [0.5, 0.1, 0.4, 0.0]
        Σ = [0.5^abs(i - j) for i in 1:4, j in 1:4] ./ 4
        w = warmup(Polytope(zeros(4), ones(4); C = ones(1, 4), d = [1.0]))
        Q = inv(Σ)
        exact = sample_until_converged(Xoshiro(7), GaussianTarget(w, μ, Σ), m; target_ess = 2_000)
        slice = sample_until_converged(Xoshiro(8), Target(w, x -> -((x - μ)' * Q * (x - μ)) / 2), m; target_ess = 2_000)
        de, ds = exact.diagnostics, slice.diagnostics
        @test all(@. abs(de.mean - ds.mean) < 5 * sqrt(de.mcse_mean^2 + ds.mcse_mean^2))
        @test all(@. abs(de.std - ds.std) < 6 * sqrt(de.mcse_std^2 + ds.mcse_std^2))
    end
end
