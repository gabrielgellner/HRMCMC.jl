# Rounding: an affine change of coordinates that makes the polytope `{z : G z ≤ h}` close to
# round, so that hit-and-run chains take long steps in every direction. Each round computes
# the maximum-volume inscribed ellipsoid and maps it to the unit ball, until the ellipsoid
# is nearly a ball (Haraldsdóttir et al. 2017, as in the COBRA Toolbox, PolyRound and
# volesti).

"""
    chebyshev_center(G, h, optimizer)

The center and radius of the largest ball in `{z : G z ≤ h}`, for unit-norm rows of `G`,
from one linear program.
"""
function chebyshev_center(G, h, optimizer)
    model = lp(optimizer)
    k = size(G, 2)
    z = @variable(model, [1:k])
    r = @variable(model, lower_bound = 0)
    @constraint(model, G * z .+ r .<= h)
    @objective(model, MAX_SENSE, r)
    solve!(model)
    if !is_solved_and_feasible(model)
        status = termination_status(model)
        status == INFEASIBLE && throw(ArgumentError("the polytope is empty"))
        status in (DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) &&
            throw(ArgumentError("the polytope is unbounded: it holds balls of any radius"))
        error("the Chebyshev center linear program stopped with status $status")
    end
    c, radius = value.(z), value(r)
    # volesti rejects a ball with NaN or Inf (hpolytope.h); `r ≥ 0` holds to the solver's
    # tolerance, which left radii of -1e-12 (#83)
    all(isfinite, c) && isfinite(radius) ||
        error("the Chebyshev center linear program returned a point that is not finite")
    return c, max(radius, 0.0)
end

# Adapted from volesti, include/preprocess/max_inscribed_ellipsoid.hpp (commit 064e18c,
# LGPL-3.0; Copyright (c) 2012-2024 Vissarion Fisikopoulos, 2018-2024 Apostolos Chalkis,
# 2021 Vaibhav Thakkar), which ports Yin Zhang's MATLAB mve_solver as used in the COBRA
# Toolbox, and from PolyRound, PolyRound/static_classes/rounding/maximum_volume_ellipsoid.py
# (commit 6d3aee7, MIT; Copyright (c) 2026 PolyRound contributors).
#
# Zhang, Y. & Gao, L. (2003). On numerical solution of the maximum volume ellipsoid problem.
# SIAM Journal on Optimization 14, 53–76. <https://doi.org/10.1137/S1052623401397230>
"""
    max_volume_ellipsoid(A, b, x0; reg, maxiter = 150, tol = 1e-6)

The maximum-volume ellipsoid inscribed in `{x : A x ≤ b}` by Zhang's primal-dual
interior-point method, started from the interior point `x0`. Returns
`(E2, c, converged, distortion, failed)` with the ellipsoid `{x : (x - c)' E2 (x - c) ≤ 1}`.
`reg` regularizes the Newton systems.

As in PolyRound (maximum_volume_ellipsoid.py, `solve_mve`), the slacks at `x0` are raised to
at least `1e-7` and capped at `1e6` times the smallest, so that a start near a facet or
slacks spanning many orders of magnitude do not make the scaled system singular;
`distortion` is the largest raise to the floor. A cap only moves far facets closer, so the
ellipsoid stays inscribed; it is inscribed in the polytope itself only when `distortion`
is 0.
`failed` is true when the ellipsoid's matrix could not be factored, which ends the
iteration with `E2` unusable.

As in PolyRound, a solve stops after `maxiter = 150` iterations, and after 10 while
`reg > 1e-6`: the early rounds of `round_polytope!` only need a rough ellipsoid, as the
next round starts from the rounded polytope. The change of the objective is tested every 10
iterations, so a single short step does not pass for convergence. The Newton system is
symmetric positive definite (a Schur product plus a positive diagonal) and is factored by
Cholesky, as in PolyRound, at half the cost of LU; `Q = A E2⁻¹ Aᵀ` comes from one
triangular solve `W = A / U` and the symmetric product `W Wᵀ`.
"""
function max_volume_ellipsoid(A, b, x0; reg::Real, maxiter::Integer = 150, tol::Real = 1.0e-6)
    m, n = size(A)
    minmu = 1.0e-8
    tau0 = 0.75
    bnrm = norm(b)
    # Scale each row by its slack at x0, so that x0 is the origin and every slack is one
    s0 = b .- A * x0
    all(isfinite, s0) || throw(ArgumentError("the slacks at x0 must be finite"))
    s = clamp.(s0, MVE_MIN_SLACK, max(minimum(s0), MVE_MIN_SLACK) * MVE_MAX_SLACK_RATIO)
    # Capping a slack moves a far facet closer, so the ellipsoid stays inscribed; raising
    # one to the floor moves a facet out, so only that changes the answer
    distortion = maximum(s .- s0; init = 0.0)
    A = A ./ s
    x = zeros(n)
    y = ones(m)
    z = zeros(m)
    bmAx = ones(m)
    last_r1 = last_r2 = prev_obj = -Inf
    converged = false
    failed = false
    E2 = zeros(n, n)
    for i in 1:maxiter
        Ay = sqrt.(y) .* A
        E2 = Ay' * Ay
        F = cholesky(Symmetric(E2); check = false)
        if !issuccess(F)
            failed = true
            break
        end
        objval = sum(log, diag(F.U)) # half the log-determinant of E2
        W = A / F.U
        Q = W * W'
        h = sqrt.(max.(diag(Q), 0.0))
        if i == 1
            t = minimum(bmAx ./ h)
            y ./= t^2
            h .*= t
            z .= max.(0.1, bmAx .- h)
            Q .*= t^2
        end
        yz = y .* z
        yh = y .* h
        gap = sum(yz) / m
        rmu = max(min(0.5, gap) * gap, minmu)
        R1 = -(A' * yh)
        R2 = bmAx .- h .- z
        R3 = rmu .- yz
        r1 = maximum(abs, R1)
        r2 = maximum(abs, R2)
        r3 = maximum(abs, R3)
        res = max(r1, r2, r3)

        if i % 10 == 0
            λ = eigvals(Symmetric(E2))
            # Stop when the residuals stall on a badly shaped ellipsoid; the next round of
            # rounding restarts from a better shaped polytope with less regularization
            stalled = abs((last_r1 - r1) / min(abs(last_r1), abs(r1))) < 0.01 &&
                abs((last_r2 - r2) / min(abs(last_r2), abs(r2))) < 0.01
            if stalled && λ[end] / λ[1] > 100 && reg > 1.0e-10
                break
            end
            last_r1, last_r2 = r1, r2
            # The objective changed by less than `tol` over the last 10 iterations
            if isfinite(prev_obj) && abs(objval - prev_obj) <= tol * max(min(abs(objval), abs(prev_obj)), 1)
                converged = true
                break
            end
            prev_obj = objval
        end
        if res < tol * (1 + bnrm) && rmu <= minmu
            converged = true
            break
        end

        # Newton step on the primal-dual system
        YQ = y .* Q
        Gm = YQ .* YQ'
        y2h = 2 .* yh
        YA = y .* A
        for j in 1:m
            Gm[j, j] += max(reg, y2h[j] * z[j])
        end
        FG = cholesky!(Symmetric(Gm); check = false)
        if !issuccess(FG)
            failed = true
            break
        end
        Tm = FG \ ((h .+ z) .* YA)
        ATP = (y2h .* Tm .- YA)'
        R3Dy = R3 ./ y
        R23 = R2 .- R3Dy
        ATP_A = ATP * A
        for j in 1:n
            ATP_A[j, j] += reg
        end
        dx = ATP_A \ (R1 .+ ATP * R23)
        Adx = A * dx
        dyDy = FG \ (y2h .* (Adx .- R23))
        dy = y .* dyDy
        dz = R3Dy .- z .* dyDy
        ax = -1 / min(-0.5, minimum(-Adx ./ bmAx))
        ay = -1 / min(-0.5, minimum(dyDy))
        az = -1 / min(-0.5, minimum(dz ./ z))
        tau = max(tau0, 1 - res)
        astep = tau * min(1.0, ax, ay, az)
        x .+= astep .* dx
        y .+= astep .* dy
        z .+= astep .* dz
        bmAx .-= astep .* Adx
        reg > 1.0e-6 && i >= 10 && break
    end
    return E2, x .+ x0, converged, distortion, failed
end

# The smallest slack of the last ellipsoid's center below which `round_polytope!` starts
# the next round from a new Chebyshev center
const MVE_RECENTER_SLACK = 1.0e-3

# The floor and largest ratio of the starting slacks of the ellipsoid solver, PolyRound's
# `default_0_width` and `default_max_ratio_bmAx0` (default_settings.py)
const MVE_MIN_SLACK = 1.0e-7
const MVE_MAX_SLACK_RATIO = 1.0e6

# Adapted from PolyRound, static_classes/rounding/nearest_symmetric_positive_definite.py
# (commit 6d3aee7, MIT): the nearest symmetric positive definite matrix to `M` (Higham 1988),
# shifted until its smallest eigenvalue is positive. Inverting a nearly singular ellipsoid
# matrix leaves `inv(E2)` slightly indefinite; a fixed jitter instead floors every axis.
function nearest_spd(M)
    B = (M .+ M') ./ 2
    F = svd(B)
    H = F.V * Diagonal(F.S) * F.V'
    S = Symmetric((B .+ H) ./ 2)
    k = 0
    while true
        λ = eigvals(S)[1]
        λ > 0 && return S
        k += 1
        S = Symmetric(S + (-λ * k^2 + eps(minimum(diag(S)))) * I)
    end
    return S
end

# Scale each row of `G z ≤ h` to unit norm
function normalize_rows!(G, h)
    for i in axes(G, 1)
        r = norm(view(G, i, :))
        G[i, :] ./= r
        h[i] /= r
    end
    return G, h
end

"""
    round_polytope!(center, T, G, h; optimizer, maxiter = 20, max_ratio = 6)

Round the polytope `x = center + T z, G z ≤ h` in place by repeated maximum-volume
ellipsoids. Each round maps the ellipsoid to the unit ball and starts from the center of the
last one, or from the Chebyshev center in the first round and when that center lies within
`1e-3` of a facet; rounding stops when the ellipsoid's squared axis ratio is at most `max_ratio` (the
criterion of volesti), its solver has converged and its starting slacks needed no raise
(the criterion of PolyRound), or after `maxiter` rounds. The regularization starts at
`1e-4` and falls tenfold per round. PolyRound also goes on until the regularization reaches
`1e-8`; that took web50 of #33 from 31 s to 98 s without changing its roundness.

Adapted from PolyRound, static_classes/rounding/maximum_volume_ellipsoid.py (commit 6d3aee7,
MIT; Copyright (c) 2026 PolyRound contributors), `iterative_solve`: the ellipsoid is repaired to the nearest positive definite matrix
before it is factored, and the result is moved back inside when its center ends on the
boundary. PolyRound also scales the columns of `G` by geometric means first; `warmup`
scales the variables before the slice instead, which leaves the columns of `G` within a
factor of 4 on the food webs of #33, and scaling them again made one round of web50 fail to
converge in 500 iterations (60 s against 11 s). PolyRound also starts every round from a new
Chebyshev center; a program of the dense rounded constraints took 5 to 7 s on web50, and
starting from the last ellipsoid's center instead rounded it in 14 s instead of 33 s.
"""
function round_polytope!(center, T, G, h; optimizer, maxiter::Integer = 20, max_ratio::Real = 6)
    k = size(G, 2)
    k == 0 && return center, T, G, h
    ratio = Inf # squared axis ratio of the last ellipsoid mapped to the ball
    distortion = 0.0
    for round in 1:maxiter
        # The center of the last ellipsoid lies deep inside the rounded polytope, whose
        # inscribed radius is near 1; a new Chebyshev center is found when it does not, as
        # PolyRound does every round. On web50 of #33 a Chebyshev program of the dense
        # rounded constraints took 5-7 s, against 1-4 s for an ellipsoid.
        if round == 1 || minimum(h) < MVE_RECENTER_SLACK
            c, r = chebyshev_center(G, h, optimizer)
            r > 0 || throw(ArgumentError("the polytope has no interior in its span"))
            shift!(center, T, G, h, c)
        end
        # Powers of 10 computed exactly, so that the test `reg > 1e-6` in the ellipsoid solver
        # does not see 1.0000000000000002e-6
        reg = exp10(-min(round + 3, 10))
        E2, c, converged, distortion, failed = max_volume_ellipsoid(G, h, zeros(k); reg)
        λ = eigvals(Symmetric(E2))
        # A singular E2 has no inverse to factor
        F = !failed && λ[1] > 0 ? cholesky(nearest_spd(inv(Symmetric(E2))); check = false) : nothing
        if F === nothing || !issuccess(F)
            @warn "warmup: ellipsoid rounding stopped in round $round, as the ellipsoid could not be factored; " *
                (round == 1 ? "the polytope is not rounded. " : "the last ellipsoid mapped to the ball had axis ratio $(sqrt(ratio)). ") *
                "Chains may mix slowly; try rounding = :samples."
            break
        end
        ratio = λ[end] / λ[1]
        shift!(center, T, G, h, c)
        L = F.L
        T .= T * L
        G .= G * L
        normalize_rows!(G, h)
        (ratio <= max_ratio && converged && distortion <= 1.0e-12) && break
    end
    # PolyRound raises here; the map is still valid, but the last ellipsoid was not inscribed
    distortion > 1.0e-12 && @warn "warmup: the last ellipsoid of rounding started on a facet (slack raised by $(distortion)), so the polytope may be less round than reported"
    # The center of the last ellipsoid can end on a facet to rounding error
    if minimum(h) <= 0
        c, _ = chebyshev_center(G, h, optimizer)
        shift!(center, T, G, h, c)
    end
    return center, T, G, h
end

# Move the origin of the `z` coordinates to `c`
function shift!(center, T, G, h, c)
    center .+= T * c
    h .-= G * c
    return center, T, G, h
end

# Adapted from volesti, include/sampling/mmcs.hpp (commit 064e18c, LGPL-3.0; Copyright (c)
# 2012-2024 Vissarion Fisikopoulos, 2018-2024 Apostolos Chalkis): the rounding phases of
# multiphase Monte Carlo sampling, which scale the principal axes of billiard-walk samples
# whose singular values are at least twice the smallest, with trajectories of mean length
# 6√k r, and stop when the largest ratio is at most 3 or after 20 phases. Unlike volesti:
#
# - Rounding also stops when a phase lowers a ratio below NOISE_RATIO by less than 10%: with
#   N correlated samples the ratios of a round body do not fall below a noise floor near
#   (1 + √(k/N_eff)) / (1 - √(k/N_eff)), which can exceed 3, and further phases would only
#   stretch noise. Above NOISE_RATIO a phase that improves little has underestimated the
#   long axes, as on a skinny box, and rounding goes on: on [0, 1e6] × [0, 1]⁹ the 10% rule
#   stopped 15 of 20 seeds at ratios up to 13108 (#45).
# - Phases draw a quarter of the samples while the last ratio exceeds NOISE_RATIO, as they
#   only need the long axes (#37).
# - Each phase runs one chain per thread, as dingo's parallel_mmcs does, and each chain
#   continues from its last point mapped into the new coordinates, as hopsy's
#   run_multiphase_sampling does (hopsy, src/hopsy/misc.py, commit 54516d7, MIT; Copyright
#   (c) 2023 The hopsy development team). The affine map keeps the chains uniform, so only
#   the first phase burns in (#36).
# - The principal axes come from the eigendecomposition of the k × k matrix Z Zᵀ instead
#   of the singular value decomposition of the k × N samples Z, which would also compute
#   their N × k right singular vectors.
"""
    round_by_samples!(rng, P, center, T, G, h; optimizer, samples = max(20k, 1000),
                      maxiter = 20, max_ratio = 3, chains = Threads.nthreads())

Round the polytope `x = center + T z, G z ≤ h` in place from samples: each phase draws
`samples` points by the billiard walk on `chains` chains run in parallel threads (a quarter
as many points while the polytope is far from round), takes the principal axes of the
centered samples, and stretches the coordinates so that every principal axis with at least
twice the length of the shortest gets that length. The chains start at the Chebyshev center
and continue from their last points in later phases. For a given `rng` the result depends
on `chains`.
"""
function round_by_samples!(
        rng::AbstractRNG, P, center, T, G, h; optimizer, samples::Union{Nothing, Integer} = nothing,
        maxiter::Integer = 20, max_ratio::Real = 3, chains::Integer = Threads.nthreads(),
    )
    k = size(G, 2)
    k == 0 && return center, T, G, h
    chains >= 1 || throw(ArgumentError("chains must be at least 1, got $chains"))
    N = something(samples, max(20k, 1000))
    rngs = [Xoshiro(rand(rng, UInt64)) for _ in 1:chains]
    # The last point of each chain, in the current coordinates
    ends = zeros(k, chains)
    previous = Inf
    saved = nothing # the body before the last phase's map, and its ratio
    for phase in 1:maxiter
        c, r = chebyshev_center(G, h, optimizer)
        # As hopsy raises for a Chebyshev ball of radius 0 (misc.py, commit 54516d7, MIT)
        r > 0 || error("warmup: :samples rounding found no ball inside the polytope in phase $phase")
        shift!(center, T, G, h, c)
        ends .-= c
        w = Warmup(P, zeros(length(center), 0), center, T, G, h, zeros(k, 0), r, false; optimizer)
        method = BilliardWalk(trajectory_length = 6 * sqrt(k) * r)
        n = previous > NOISE_RATIO ? max(N ÷ 4, min(N, 2k)) : N
        Z = Matrix{Float64}(undef, k, n)
        burnin = phase == 1 ? k : 0
        tasks = map(1:chains) do i
            columns = (div((i - 1) * n, chains) + 1):div(i * n, chains)
            Threads.@spawn billiard_chain!(rngs[i], Z, columns, view(ends, :, i), method, w, burnin)
        end
        foreach(wait, tasks)
        μ = vec(mean(Z; dims = 2))
        Z .-= μ
        E = eigen(Symmetric(Z * Z'))
        λ = reverse(E.values)
        λ[end] > eps() * λ[1] ||
            throw(ArgumentError("the polytope has no width in some direction of its slice"))
        s = sqrt.(λ ./ λ[end])
        # Adapted from volesti, include/preprocess/svd_rounding.hpp (commit 064e18c,
        # LGPL-3.0): a phase whose samples are four times less round than the last stretched
        # along noise, so its map is undone and the next phase draws twice the samples
        if saved !== nothing && s[1] >= 4 * previous
            center .= saved.center
            T .= saved.T
            G .= saved.G
            h .= saved.h
            ends .= saved.ends
            previous = saved.previous
            N *= 2
            saved = nothing
            continue
        end
        (s[1] < 2 || (s[1] <= NOISE_RATIO && s[1] > 0.9 * previous)) && break
        saved = (; center = copy(center), T = copy(T), G = copy(G), h = copy(h), ends = copy(ends), previous)
        previous = s[1]
        s[s .< 2] .= 1
        shift!(center, T, G, h, μ)
        U = reverse(E.vectors; dims = 2)
        L = U * Diagonal(s)
        T .= T * L
        G .= G * L
        normalize_rows!(G, h)
        ends .= Diagonal(1 ./ s) * (U' * (ends .- μ))
        s[1] <= max_ratio && break
    end
    return center, T, G, h
end

# Run a billiard walk on `w` from `start`, which it updates to the last point, for `burnin`
# steps and then one step per column of `Z` in `columns`, writing the points there
function billiard_chain!(rng, Z, columns, start, method, w, burnin)
    state = chain_state(method, w)
    state.z .= start
    for _ in 1:burnin
        step!(rng, state)
    end
    for j in columns
        step!(rng, state)
        Z[:, j] = state.z
    end
    start .= state.z
    return nothing
end

# Ratios of singular values up to this are within a few times the noise floor of
# `round_by_samples!` (about 3 on food webs with 250 to 1000 directions)
const NOISE_RATIO = 10.0
