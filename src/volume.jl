"""
    volume([rng], P::Polytope; error = 0.1, nchains = 4, n = 500, max_n = 1_000_000, kwargs...)
    volume([rng], w::Warmup; error = 0.1, nchains = 4, n = 500, max_n = 1_000_000)

Estimate the volume of a polytope by multiphase Monte Carlo with cooling balls (Chalkis,
Emiris & Fisikopoulos 2023). Given a [`Polytope`](@ref), the other keyword arguments go to
[`warmup`](@ref). For a polytope with equalities, this is the `k`-dimensional volume of the
slice it lies on, in the units of its variables; a single point has volume 1.

In the coordinates `z` of the warmup, the polytope `K` is cut by balls `B(ρ₁) ⊃ B(ρ₂) ⊃ …`
centered at `z = 0`, chosen so that each body `K ∩ B(ρᵢ₊₁)` holds about an eighth of the
volume of `K ∩ B(ρᵢ)`, down to a last ball of which 10–15% lies in `K`. The volume is the
product of the volume of the last ball, the share of it inside `K`, estimated from
independent uniform points of the ball, and the ratios `vol(K ∩ B(ρᵢ)) / vol(K ∩ B(ρᵢ₊₁))`,
each estimated from `nchains` coordinate hit-and-run chains on `K ∩ B(ρᵢ)`, continued from
the chains that chose the radius `ρᵢ₊₁`. A ratio is the share of the chains' states inside
the next ball after each coordinate move. The chains first run `2n` sweeps and keep the last
`n`, then lengthen, keeping the second half, until R̂ of the distance from the center is below 1.01 and the Monte Carlo
standard error of the ratio is small enough for `error`; each round extrapolates the length
from the last one's standard error, at most doubling it. The variance `error²` of the log
volume is split as in volesti: `error² / 4L` for the share of the last ball, whose
independent points are cheap, and the rest equally over the `L - 1` ratios from chains.

Returns a named tuple:

- `volume` and `log_volume`;
- `log_stderr`: the standard error of `log_volume`, about the relative standard error of
  `volume`, from the Monte Carlo standard errors of the phases, which are independent; at
  most `error` when every phase converges;
- `radii`: the radii of the balls, from `Inf` (the polytope itself) to the last ball;
- `ratios`: the estimated share of each body inside the next ball, then the share of the
  last ball inside the polytope;
- `converged`: whether every phase reached its error before keeping `max_n` sweeps per chain
  (of `2max_n` run);
- `warmup`.

This adapts the annealing schedule of volesti's `volume_cooling_balls` and differs in its
stopping rule: volesti stops each phase when a sliding window of running ratios varies
little, which gives no standard error; here each phase runs until its Monte Carlo standard
error, which accounts for the correlation of the chain, meets the target.

Chalkis, A., Emiris, I. Z. & Fisikopoulos, V. (2023). A practical algorithm for volume
estimation based on billiard trajectories and simulated annealing. ACM Journal of
Experimental Algorithmics 28. <https://doi.org/10.1145/3584182>

# Examples
```julia
cube = Polytope(zeros(10), ones(10))
v = volume(Xoshiro(1), cube)
v.volume, v.log_stderr  # about 1
```
"""
function volume(
        rng::AbstractRNG, w::Warmup; error::Real = 0.1, nchains::Integer = 4, n::Integer = 500,
        max_n::Integer = 1_000_000,
    )
    error > 0 || throw(ArgumentError("error must be positive, got $error"))
    nchains >= 2 || throw(ArgumentError("nchains must be at least 2 for R̂, got $nchains"))
    n >= 10 || throw(ArgumentError("n must be at least 10 for R̂, got $n"))
    max_n >= n || throw(ArgumentError("max_n must be at least n"))
    k = size(w.transform, 2)
    k == 0 && return (;
        volume = 1.0, log_volume = 0.0, log_stderr = 0.0, radii = Float64[], ratios = Float64[],
        converged = true, warmup = w,
    )
    r_in = minimum(w.h) # the largest ball centered at z = 0 inside the polytope
    r_in > 0 || throw(ArgumentError("the warmup center is not inside the polytope"))
    ρ_last = last_ball_radius(rng, w, r_in)
    radii, pilots = ball_schedule(rng, w, r_in, ρ_last, nchains)
    # Adapted from volesti, include/volume/volume_cooling_balls.hpp (commit 064e18c,
    # LGPL-3.0; Copyright (c) 2012-2020 Vissarion Fisikopoulos, 2018-2020 Apostolos
    # Chalkis): of the variance error² of the log volume, the share of the last ball, from
    # cheap independent points, gets error²/4L and the L - 1 chain phases split the rest
    L = length(radii)
    ball_target = error / (2 * sqrt(L))
    target = error * sqrt((4L - 1) / (4L * (L - 1)))
    log_volume = 0.5 * logdet(Symmetric(w.transform' * w.transform)) + log_ball_volume(k, ρ_last)
    variance = 0.0
    ratios = Float64[]
    converged = true
    for i in 1:(length(radii) - 1)
        r, se, ok = body_ratio(pilots[i], radii[i + 1]; target, n, max_n)
        r > 0 || throw(ErrorException("volume: no point of phase $i fell in the next ball after $max_n points per chain; raise max_n"))
        push!(ratios, r)
        log_volume -= log(r)
        variance += (se / r)^2
        converged &= ok
    end
    f, se = ball_share(rng, w, ρ_last, ball_target)
    push!(ratios, f)
    log_volume += log(f)
    variance += (se / f)^2
    log_stderr = sqrt(variance)
    converged || @warn "volume: some phases did not reach their error after $max_n points per chain" log_stderr
    return (; volume = exp(log_volume), log_volume, log_stderr, radii, ratios, converged, warmup = w)
end

function volume(rng::AbstractRNG, P::Polytope; error::Real = 0.1, nchains::Integer = 4, n::Integer = 500, max_n::Integer = 1_000_000, kwargs...)
    return volume(rng, warmup(rng, P; kwargs...); error, nchains, n, max_n)
end

volume(model::Union{Warmup, Polytope}; kwargs...) = volume(default_rng(), model; kwargs...)

# The share of each body that the next ball keeps, as volesti's bounds of 0.1 and 0.15
const BALL_SHARE_LOW = 0.1
const BALL_SHARE_HIGH = 0.15
const BALL_PILOT = 1200

# log of the volume of the k-dimensional ball of radius ρ, π^(k/2) ρ^k / Γ(k/2 + 1)
function log_ball_volume(k, ρ)
    # Γ(k/2 + 1) is (k/2)! for even k and (k/2)(k/2 - 1)⋯(1/2)√π for odd k
    log_gamma = iseven(k) ? sum(log, 1:(k ÷ 2); init = 0.0) :
        sum(j -> log(j - 0.5), 1:((k + 1) ÷ 2)) + log(π) / 2
    return k / 2 * log(π) + k * log(ρ) - log_gamma
end

# A uniform point of the ball of radius ρ centered at 0, in `z`
function ball_point!(rng, z, ρ)
    randn!(rng, z)
    z .*= ρ * rand(rng)^(1 / length(z)) / norm(z)
    return z
end

# The number of `points` uniform points of the ball of radius ρ that lie in the polytope,
# tested BALL_BATCH at a time with one product G Z, which reads G by columns
function ball_hits(rng, w, ρ, points)
    Z = Matrix{Float64}(undef, size(w.G, 2), BALL_BATCH)
    Y = Matrix{Float64}(undef, size(w.G, 1), BALL_BATCH)
    hits = 0
    for first in 1:BALL_BATCH:points
        b = min(BALL_BATCH, points - first + 1)
        foreach(z -> ball_point!(rng, z, ρ), eachcol(view(Z, :, 1:b)))
        mul!(view(Y, :, 1:b), w.G, view(Z, :, 1:b))
        hits += count(j -> all(i -> Y[i, j] <= w.h[i], axes(Y, 1)), 1:b)
    end
    return hits
end

const BALL_BATCH = 256

# The share of `points` uniform points of the ball of radius ρ that lie in the polytope
ball_share_sample(rng, w, ρ, points) = ball_hits(rng, w, ρ, points) / points

# Adapted from volesti, include/volume/volume_cooling_balls.hpp (commit 064e18c, LGPL-3.0;
# Copyright (c) 2012-2020 Vissarion Fisikopoulos, 2018-2020 Apostolos Chalkis): the share
# bounds of 0.1 and 0.15, the first try at 2√k times the inner radius, and the bisection.
# The radius of the last ball, of which a share between BALL_SHARE_LOW and BALL_SHARE_HIGH
# lies in the polytope, by bisection on the share of BALL_PILOT uniform points (volesti's
# `get_first_ball`). The share falls from 1 at `r_in` to 0 once the ball holds the polytope.
function last_ball_radius(rng, w, r_in)
    k = size(w.G, 2)
    lo, hi = r_in, 2 * sqrt(k) * r_in
    for _ in 1:100
        f = ball_share_sample(rng, w, hi, BALL_PILOT)
        f < BALL_SHARE_LOW && break
        BALL_SHARE_LOW <= f <= BALL_SHARE_HIGH && return hi
        lo, hi = hi, 2hi
    end
    mid = (lo + hi) / 2
    for _ in 1:100
        mid = (lo + hi) / 2
        f = ball_share_sample(rng, w, mid, BALL_PILOT)
        if f < BALL_SHARE_LOW
            hi = mid
        elseif f > BALL_SHARE_HIGH
            lo = mid
        else
            return mid
        end
    end
    return mid
end

# The share of the ball of radius ρ inside the polytope from independent uniform points,
# with relative standard error at most `target`, and its standard error
function ball_share(rng, w, ρ, target)
    hits, total, batch = 0, 0, BALL_PILOT
    while true
        hits += ball_hits(rng, w, ρ, batch)
        total += batch
        f = hits / total
        se = sqrt(f * (1 - f) / total)
        f > 0 && se <= target * f && return f, se
        # the binomial sample size for the target, or twice as many points without hits
        batch = f > 0 ? max(ceil(Int, (1 - f) / (f * target^2)) - total, BALL_PILOT) : total
    end
    return
end

# Adapted from volesti, include/volume/volume_cooling_balls.hpp (commit 064e18c, LGPL-3.0;
# Copyright (c) 2012-2020 Vissarion Fisikopoulos, 2018-2020 Apostolos Chalkis): the sequence
# of balls and its stopping rule.
# The radii of the balls, from Inf to `ρ_last`: from BALL_PILOT points of each body
# `K ∩ B(ρᵢ)`, the next radius keeps a share of 1/8 of them, unless at least BALL_SHARE_LOW of
# them already lie in the last ball (volesti's `get_sequence_of_polytopeballs`, with the
# radius from a quantile of the distances in place of its bisection). Also returns the
# pilot chains of each body `K ∩ B(ρᵢ)` before the last ball, which `body_ratio` continues,
# as volesti carries its walk's state from one phase to the next.
function ball_schedule(rng, w, r_in, ρ_last, nchains)
    radii = [Inf]
    pilots = Vector{BallChain}[]
    for _ in 1:1000
        chains, d = pilot_distances(rng, w, last(radii), r_in, nchains)
        push!(pilots, chains)
        if mean(<=(ρ_last), d) >= BALL_SHARE_LOW
            push!(radii, ρ_last)
            return radii, pilots
        end
        ρ = max(quantile(d, 1 / 8), ρ_last)
        push!(radii, ρ)
        ρ == ρ_last && return radii, pilots
    end
    throw(ErrorException("volume: no ball schedule after 1000 phases"))
end

# `nchains` chains on `K ∩ B(ρ)` and the distances from the center of BALL_PILOT of their
# points, after a burn-in of 100 sweeps
function pilot_distances(rng, w, ρ, r_in, nchains)
    per_chain = cld(BALL_PILOT, nchains)
    chains = [BallChain(Xoshiro(rand(rng, UInt64)), w, ρ, r_in) for _ in 1:nchains]
    d = zeros(per_chain, nchains)
    Threads.@threads for j in 1:nchains
        c = chains[j]
        for _ in 1:100
            sweep!(c)
        end
        for i in 1:per_chain
            sweep!(c)
            d[i, j] = sqrt(c.norm2)
        end
    end
    return chains, vec(d)
end

# The share of `K ∩ B(ρ)` inside `B(ρ_next)`, from `chains` on `K ∩ B(ρ)` lengthened until R̂
# of the distance from the center is below 1.01 and the relative Monte Carlo standard error of
# the share is at most `target`; returns the share, its standard error and whether it converged
function body_ratio(chains, ρ_next; target, n, max_n)
    nchains = length(chains)
    d = [Float64[] for _ in 1:nchains]
    f = [Float64[] for _ in 1:nchains]
    kept = n
    while true
        Threads.@threads for j in 1:nchains
            c = chains[j]
            for _ in (length(d[j]) + 1):(2kept)
                inside = sweep!(c, ρ_next^2)
                push!(d[j], sqrt(c.norm2))
                push!(f[j], inside / length(c.z))
            end
        end
        x = Array{Float64}(undef, kept, nchains, 1)
        inner = Array{Float64}(undef, kept, nchains, 1)
        for j in 1:nchains
            x[:, j, 1] = view(d[j], (kept + 1):(2kept))
            inner[:, j, 1] = view(f[j], (kept + 1):(2kept))
        end
        r = mean(inner)
        # every point inside the next ball gives a ratio of 1 with no error
        se = r == 1 ? 0.0 : r > 0 ? only(mcse(inner; kind = mean)) : Inf
        rhat = only(ess_rhat(x).rhat)
        ok = r > 0 && rhat < 1.01 && se <= target * r
        (ok || kept >= max_n) && return r, se, ok
        # The standard error falls as 1/√kept: 1.2 times the length that reaches the target,
        # at most twice this one, or twice while R̂ is too high, no point is inside or the
        # standard error is NaN (mcse of a window of constant fractions)
        grow = rhat < 1.01 && r > 0 && isfinite(se) ? 1.2 * (se / (target * r))^2 : 2.0
        kept = min(clamp(ceil(Int, grow * kept), kept + 1, 2kept), max_n)
    end
    return
end

# A coordinate hit-and-run chain on `K ∩ B(ρ)` in the coordinates `z` of a warmup, started at
# a uniform point of the ball of radius `r_in` inside `K`. Each sweep moves every coordinate
# in turn; along coordinate `j` the ball allows `|z|² + 2 t zⱼ + t² ≤ ρ²`.
mutable struct BallChain
    const rng::Xoshiro
    const warmup::Warmup
    const z::Vector{Float64}
    const slack::Vector{Float64} # h - G z
    const radius2::Float64 # ρ², Inf for the polytope itself
    norm2::Float64 # |z|²
    sweeps::Int
end

function BallChain(rng, w::Warmup, ρ, r_in)
    z = ball_point!(rng, zeros(size(w.G, 2)), r_in)
    return BallChain(rng, w, z, w.h .- w.G * z, ρ^2, sum(abs2, z), 0)
end

# The polytope's chord `[tmin, tmax]` along coordinate `j` cut by the ball `|z|² ≤ ρ²`. The
# interval keeps `t = 0`: rounding can put `|z|²` above `ρ²`, where the ball's chord shrinks
# to `t = -zⱼ`, which may lie outside the polytope's chord
function ball_chord(tmin, tmax, zj, norm2, radius2)
    isfinite(radius2) || return tmin, tmax
    s = sqrt(max(zj^2 - norm2 + radius2, 0.0))
    return min(max(tmin, -zj - s), 0.0), max(min(tmax, -zj + s), 0.0)
end

function sweep!(c::BallChain, inner2 = -Inf)
    w = c.warmup
    z = c.z
    if c.sweeps % SLACK_REFRESH == 0
        mul!(c.slack, w.G, z)
        c.slack .= w.h .- c.slack
    end
    c.sweeps += 1
    c.norm2 = sum(abs2, z)
    inside = 0
    for j in eachindex(z)
        g = view(w.G, :, j)
        tmin, tmax = ball_chord(chord(c.slack, g)..., z[j], c.norm2, c.radius2)
        t = uniform(c.rng, tmin, tmax)
        c.norm2 += t * (2 * z[j] + t)
        z[j] += t
        for i in eachindex(c.slack)
            c.slack[i] -= t * g[i]
        end
        inside += c.norm2 <= inner2
    end
    return inside
end
