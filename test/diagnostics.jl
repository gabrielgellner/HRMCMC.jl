using PolytopeSamplers: PolytopeSamplers, HitAndRun, MCMCSerial, Polytope, diagnostics, sample, warmup
using Random: Xoshiro
using Statistics: mean

@testset "diagnostics" begin
    @testset "independent draws" begin
        # 4 chains of 1000 independent uniform points: ESS near 4000, R̂ near 1
        chains = [rand(Xoshiro(seed), 3, 1000) for seed in 1:4]
        d = diagnostics(chains)
        @test length(d.ess) == length(d.rhat) == 3
        @test all(3000 .< d.ess .< 5000)
        @test d.max_rhat < 1.01
        @test d.min_ess == minimum(d.ess)
        @test all(3000 .< diagnostics(chains; kind = :tail).ess .< 5000)
        # Standard errors of the mean and standard deviation of uniform draws
        @test d.mean ≈ vec(mean(reduce(hcat, chains); dims = 2))
        @test all(abs.(d.mcse_mean ./ sqrt(1 / 12 / 4000) .- 1) .< 0.2)
        @test all(abs.(d.mcse_std ./ (sqrt(1 / 12) * sqrt(0.8 / 4 / 4000)) .- 1) .< 0.2)
    end

    @testset "one chain" begin
        d = diagnostics(rand(Xoshiro(1), 2, 1000))
        @test all(700 .< d.ess .< 1300)
        @test d.max_rhat < 1.02
    end

    @testset "detects correlation" begin
        # Unthinned hit-and-run in 20 dimensions moves little per step
        w = warmup(Polytope(zeros(20), ones(20)))
        chains = sample(Xoshiro(1), w, HitAndRun(), MCMCSerial(), 1000, 4)
        @test diagnostics(chains).min_ess < 4000 / 5
    end

    @testset "detects chains that disagree" begin
        chains = [rand(Xoshiro(1), 2, 500), rand(Xoshiro(2), 2, 500) .+ 1]
        @test diagnostics(chains).max_rhat > 1.1
    end

    @testset "detects chains that differ in spread" begin
        # Two chains on [-1, 1] and two on [-3, 3]: same center, so the bulk R̂ is near 1,
        # and the tail R̂ flags them (#44)
        chains = [s .* (2 .* rand(Xoshiro(seed), 1, 1000) .- 1) for (seed, s) in zip(1:4, (1, 1, 3, 3))]
        @test diagnostics(chains).max_rhat > 1.05
    end

    @testset "constant coordinates" begin
        w = warmup(Polytope([0.0, 0.5, 0.0], [1.0, 0.5, 1.0]))
        chains = sample(Xoshiro(1), w, HitAndRun(), MCMCSerial(), 500, 2; thinning = 5)
        d = diagnostics(chains)
        @test isnan(d.ess[2]) && isnan(d.rhat[2])
        @test d.min_ess == minimum(d.ess[[1, 3]])
        @test d.max_rhat < 1.05
        @test isnan(diagnostics(fill(0.5, 2, 10)).min_ess)
    end

    @testset "multivariate R̂ (#53)" begin
        # Wide along (1, 1) and narrow along (1, -1), with two chains shifted by 5 standard
        # deviations of the narrow direction: every coordinate agrees, the combination
        # x₁ - x₂ does not
        chains = map(1:4) do seed
            rng = Xoshiro(seed)
            u, v = 10 .* randn(rng, 1000), 0.1 .* randn(rng, 1000) .+ (seed > 2 ? 0.5 : 0.0)
            permutedims(hcat(u .+ v, u .- v)) ./ sqrt(2)
        end
        d = diagnostics(chains; multivariate = true)
        @test d.max_rhat < 1.01
        @test d.multivariate_rhat > 5
        @test isnan(diagnostics(chains).multivariate_rhat)
        # Points on the slice x₁ + x₂ = 1 with a constant coordinate have a singular
        # covariance
        slice = map(1:4) do seed
            u = rand(Xoshiro(seed), 1, 1000)
            vcat(u, 1 .- u, fill(0.5, 1, 1000))
        end
        @test 1 <= diagnostics(slice; multivariate = true).multivariate_rhat < 1.01
        @test 1 <= diagnostics([rand(Xoshiro(seed), 3, 1000) for seed in 1:4]; multivariate = true).multivariate_rhat < 1.01
    end

    @testset "coordinates in blocks" begin
        # 1100 coordinates of 4 chains of 1000 points span two blocks of DIAGNOSTICS_BLOCK
        chains = [rand(Xoshiro(seed), 1100, 1000) for seed in 1:4]
        d = diagnostics(chains)
        @test 1100 * 4000 > PolytopeSamplers.DIAGNOSTICS_BLOCK
        for i in (1, 1048, 1049, 1100)
            di = diagnostics([X[i:i, :] for X in chains])
            @test d.ess[i] ≈ only(di.ess) && d.rhat[i] ≈ only(di.rhat) && d.mcse_std[i] ≈ only(di.mcse_std)
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError diagnostics(Matrix{Float64}[])
        @test_throws DimensionMismatch diagnostics([rand(2, 10), rand(2, 11)])
        @test_throws ArgumentError diagnostics(rand(2, 3))
    end
end
