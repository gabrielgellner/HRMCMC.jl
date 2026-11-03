# Sample the interaction strengths of one food web with each sampler and compare them. The
# polytope holds the Lotka–Volterra parameters with an equilibrium at the biomasses `x` (see
# `foodweb_study.jl`). Run with `just example constrained_foodweb`.

include("foodweb_study.jl")

using PolytopeSamplers: ACHR, BilliardWalk, CoordinateHitAndRun, HitAndRun, MCMCThreads,
    diagnostics, mmcs, sample, warmup
using Random: Xoshiro
using Statistics: mean

let rng = Xoshiro(2026), S = 15
    web = generalized_cascade_network(rng, S, 0.15)
    x = pyramid_biomass(rng, web)
    (; fw, P, draws) = feasible_food_web(rng, web, x)
    stable_fraction(chains) = mean(is_stable(community_matrix(fw, θ)) for X in chains for θ in eachcol(X))

    unrounded = warmup(P; rounding = false)
    t_round = @elapsed rounded = warmup(P)
    println("feasible growth rates after $draws draws; ", rounded)
    println("rounding took ", round(t_round; digits = 2), " s")

    # Four chains per sampler; compare effective samples per second
    runs = [
        ("HitAndRun, unrounded", unrounded, HitAndRun()),
        ("ACHR, unrounded", unrounded, ACHR()),
        ("HitAndRun", rounded, HitAndRun()),
        ("ACHR", rounded, ACHR()),
        ("CoordinateHitAndRun", rounded, CoordinateHitAndRun()),
        ("BilliardWalk", rounded, BilliardWalk()),
    ]
    for (name, w, method) in runs
        t = @elapsed chains = sample(
            rng, w, method, MCMCThreads(), 2_000, 4; discard_initial = 1_000, thinning = 10,
            progress = false,
        )
        d = diagnostics(chains)
        println(
            rpad(name, 22), " min ESS ", lpad(round(Int, d.min_ess), 5), ", max R̂ ",
            rpad(round(d.max_rhat; digits = 3), 5), ", ", lpad(round(Int, d.min_ess / t), 6),
            " ESS/s, stable fraction ", round(stable_fraction(chains); digits = 3),
        )
    end

    # Multiphase Monte Carlo sampling: one LP, rounding from samples, chains until
    # 1000 effective samples with R̂ < 1.01
    t = @elapsed r = mmcs(rng, P; target_ess = 1_000)
    println(
        rpad("mmcs", 22), " min ESS ", lpad(round(Int, r.diagnostics.min_ess), 5), ", max R̂ ",
        rpad(round(r.diagnostics.max_rhat; digits = 3), 5), ", ", round(t; digits = 2),
        " s in all, stable fraction ", round(stable_fraction(r.chains); digits = 3),
    )
end
