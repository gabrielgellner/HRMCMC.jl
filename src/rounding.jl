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
    optimize!(model)
    if !is_solved_and_feasible(model)
        status = termination_status(model)
        status == INFEASIBLE && throw(ArgumentError("the polytope is empty"))
        status in (DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) &&
            throw(ArgumentError("the polytope is unbounded: it holds balls of any radius"))
        error("the Chebyshev center linear program stopped with status $status")
    end
    return value.(z), value(r)
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
    max_volume_ellipsoid(A, b, x0; reg, maxiter = 500, tol = 1e-6)

The maximum-volume ellipsoid inscribed in `{x : A x ≤ b}` by Zhang's primal-dual
interior-point method, started from the interior point `x0`. Returns `(E2, c, converged)`
with the ellipsoid `{x : (x - c)' E2 (x - c) ≤ 1}`. `reg` regularizes the Newton systems.
"""
function max_volume_ellipsoid(A, b, x0; reg::Real, maxiter::Integer = 500, tol::Real = 1.0e-6)
    m, n = size(A)
    minmu = 1.0e-8
    tau0 = 0.75
    bnrm = norm(b)
    # Scale each row by its slack at x0, so that x0 is the origin and every slack is one
    s0 = b .- A * x0
    all(>(0), s0) || throw(ArgumentError("x0 must be strictly inside the polytope"))
    A = A ./ s0
    x = zeros(n)
    y = ones(m)
    z = zeros(m)
    bmAx = ones(m)
    last_r1 = last_r2 = prev_obj = -Inf
    converged = false
    E2 = zeros(n, n)
    for i in 1:maxiter
        E2 = A' * (y .* A)
        F = cholesky(Symmetric(E2); check = false)
        issuccess(F) || break
        objval = sum(log, diag(F.U)) # half the log-determinant of E2
        Q = A * (F \ Matrix(A'))
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
        end
        if (res < tol * (1 + bnrm) && rmu <= minmu) ||
                (isfinite(prev_obj) && abs(objval - prev_obj) <= tol * max(min(abs(objval), abs(prev_obj)), 1))
            converged = true
            break
        end
        prev_obj = objval

        # Newton step on the primal-dual system
        YQ = y .* Q
        Gm = YQ .* YQ'
        y2h = 2 .* yh
        YA = y .* A
        for j in 1:m
            Gm[j, j] += max(reg, y2h[j] * z[j])
        end
        luG = lu!(Gm)
        Tm = luG \ ((h .+ z) .* YA)
        ATP = (y2h .* Tm .- YA)'
        R3Dy = R3 ./ y
        R23 = R2 .- R3Dy
        ATP_A = ATP * A
        for j in 1:n
            ATP_A[j, j] += reg
        end
        dx = ATP_A \ (R1 .+ ATP * R23)
        Adx = A * dx
        dyDy = luG \ (y2h .* (Adx .- R23))
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
    end
    return E2, x .+ x0, converged
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
ellipsoids, starting from the Chebyshev center. Each round maps the ellipsoid to the unit
ball; rounding stops when the ellipsoid's squared axis ratio is at most `max_ratio` (the
criterion of volesti) or after `maxiter` rounds. The regularization of the ellipsoid solver
starts at `1e-3` and falls tenfold per round (as in PolyRound and volesti).
"""
function round_polytope!(center, T, G, h; optimizer, maxiter::Integer = 20, max_ratio::Real = 6)
    k = size(G, 2)
    k == 0 && return center, T, G, h
    c, r = chebyshev_center(G, h, optimizer)
    r > 0 || throw(ArgumentError("the polytope has no interior in its span"))
    shift!(center, T, G, h, c)
    reg = 1.0e-3
    for _ in 1:maxiter
        E2, c, converged = max_volume_ellipsoid(G, h, zeros(k); reg)
        λ = eigvals(Symmetric(E2))
        λ[1] > 0 || break
        F = cholesky(Symmetric(inv(Symmetric(E2)) + 1.0e-8 * I); check = false)
        issuccess(F) || break
        shift!(center, T, G, h, c)
        L = F.L
        T .= T * L
        G .= G * L
        normalize_rows!(G, h)
        reg = max(reg / 10, 1.0e-10)
        (λ[end] / λ[1] <= max_ratio && converged) && break
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
"""
    round_by_samples!(rng, P, center, T, G, h; optimizer, samples = max(20k, 1000),
                      maxiter = 20, max_ratio = 3)

Round the polytope `x = center + T z, G z ≤ h` in place from samples: each phase draws
`samples` points by the billiard walk from the Chebyshev center (a quarter as many while
the polytope is far from round), takes the singular value decomposition of the centered
samples, and stretches the coordinates so that every principal axis with at least twice the
length of the shortest gets that length.
"""
function round_by_samples!(
        rng::AbstractRNG, P, center, T, G, h; optimizer, samples::Union{Nothing, Integer} = nothing,
        maxiter::Integer = 20, max_ratio::Real = 3,
    )
    k = size(G, 2)
    k == 0 && return center, T, G, h
    N = something(samples, max(20k, 1000))
    previous = Inf
    for _ in 1:maxiter
        c, r = chebyshev_center(G, h, optimizer)
        shift!(center, T, G, h, c)
        w = Warmup(P, zeros(length(center), 0), center, T, G, h, zeros(k, 0), r, false; optimizer)
        state = chain_state(BilliardWalk(trajectory_length = 6 * sqrt(k) * r), w)
        n = previous > NOISE_RATIO ? max(N ÷ 4, min(N, 2k)) : N
        Z = Matrix{Float64}(undef, k, n)
        for _ in 1:k
            step!(rng, state)
        end
        for j in 1:n
            step!(rng, state)
            Z[:, j] = state.z
        end
        μ = vec(mean(Z; dims = 2))
        F = svd(Z .- μ)
        F.S[end] > 0 || throw(ArgumentError("the polytope has no width in some direction of its slice"))
        s = F.S ./ F.S[end]
        (s[1] < 2 || (s[1] <= NOISE_RATIO && s[1] > 0.9 * previous)) && break
        previous = s[1]
        s[s .< 2] .= 1
        shift!(center, T, G, h, μ)
        L = F.U * Diagonal(s)
        T .= T * L
        G .= G * L
        normalize_rows!(G, h)
        s[1] <= max_ratio && break
    end
    return center, T, G, h
end

# Ratios of singular values up to this are within a few times the noise floor of
# `round_by_samples!` (about 3 on food webs with 250 to 1000 directions)
const NOISE_RATIO = 10.0
