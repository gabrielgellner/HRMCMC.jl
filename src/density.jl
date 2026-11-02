"""
    Target(w::Warmup, ℓ; gradient = nothing)
    Target(P::Polytope, ℓ; gradient = nothing, kwargs...)

The distribution on a polytope with density proportional to `exp(ℓ(x))`, a model for
[`sample`](@ref) and [`sample_until_converged`](@ref). Given a [`Polytope`](@ref), the
other keyword arguments go to [`warmup`](@ref).

`ℓ` is a function of the vector `x` of the polytope's variables, evaluated at points clamped
to the polytope's bounds, or an object implementing
the LogDensityProblems.jl interface (`logdensity`, `dimension`, `capabilities`, and
`logdensity_and_gradient` for gradients). For a function, pass `gradient = (g, x) -> ...`,
which fills `g` with `∇ℓ(x)`, to use [`CRHMC`](@ref). `ℓ` is called from every chain, on
separate threads when Julia has them.

A log density of `+Inf` or `NaN` counts as outside the support, as for a density that
diverges at a face. [`HitAndRun`](@ref) and [`CoordinateHitAndRun`](@ref) draw each point on the chord from
the density along it by slice sampling, so they leave the target invariant for any `ℓ`
([`GaussianTarget`](@ref) and [`ExponentialTarget`](@ref) are drawn exactly on chords);
[`DikinWalk`](@ref) and [`VaidyaWalk`](@ref) add `ℓ` to their Metropolis ratio;
[`CRHMC`](@ref) adds `-ℓ` to its potential and needs the gradient. [`BilliardWalk`](@ref)
and [`ACHR`](@ref) sample only the uniform distribution.

# Examples
```julia
P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])  # the simplex
# Dirichlet(2, 3, 4) on the simplex
target = Target(P, x -> sum((a - 1) * log(xi) for (a, xi) in zip((2, 3, 4), x));
                gradient = (g, x) -> (g .= ((2, 3, 4) .- 1) ./ x))
r = sample_until_converged(Xoshiro(1), target, CRHMC())
```
"""
struct Target{L} <: AbstractModel
    warmup::Warmup
    logdensity::L
    Target{L}(warmup::Warmup, logdensity::L) where {L} = new{L}(warmup, logdensity)
end

function Target(w::Warmup, ℓ; gradient = nothing)
    n = nvars(w.polytope)
    ℓ = ℓ isa Function ? FunctionLogDensity(ℓ, gradient, n) : ℓ
    gradient === nothing || ℓ isa FunctionLogDensity ||
        throw(ArgumentError("gradient applies only when ℓ is a function"))
    LogDensityProblems.capabilities(ℓ) === nothing &&
        throw(ArgumentError("ℓ must be a function or implement the LogDensityProblems interface"))
    LogDensityProblems.dimension(ℓ) == n ||
        throw(DimensionMismatch("ℓ has dimension $(LogDensityProblems.dimension(ℓ)), the polytope has $n variables"))
    return Target{typeof(ℓ)}(w, ℓ)
end

function Target(P::Polytope, ℓ; gradient = nothing, kwargs...)
    return Target(warmup(P; kwargs...), ℓ; gradient)
end

function Base.show(io::IO, t::Target)
    print(io, "Target on a ", size(t.warmup.transform, 2), "-dimensional slice of a ")
    print(io, nvars(t.warmup.polytope), "-variable polytope")
    return
end

# A log density given as a function, with an optional in-place gradient
struct FunctionLogDensity{F, G}
    f::F
    gradient!::G
    n::Int
end

LogDensityProblems.dimension(d::FunctionLogDensity) = d.n
LogDensityProblems.capabilities(::Type{<:FunctionLogDensity}) = LogDensityOrder{1}()
function LogDensityProblems.capabilities(::Type{<:FunctionLogDensity{F, Nothing}}) where {F}
    return LogDensityOrder{0}()
end
LogDensityProblems.logdensity(d::FunctionLogDensity, x) = d.f(x)
function LogDensityProblems.logdensity_and_gradient(d::FunctionLogDensity, x)
    g = similar(x, Float64)
    d.gradient!(g, x)
    return d.f(x), g
end

"""
    GaussianTarget(w::Warmup, μ, Σ)
    GaussianTarget(P::Polytope, μ, Σ; kwargs...)

The Gaussian distribution with mean `μ` and covariance `Σ` (a positive definite matrix)
restricted to the polytope: a [`Target`](@ref) with `ℓ(x) = -(x - μ)ᵀ Σ⁻¹ (x - μ) / 2`.
Given a [`Polytope`](@ref), the other keyword arguments go to [`warmup`](@ref).

On a chord the density is a truncated normal, which [`HitAndRun`](@ref) and
[`CoordinateHitAndRun`](@ref) draw exactly, without slice sampling. The other samplers that
take densities use `ℓ` and its gradient as for any target.

# Examples
```julia
box = Polytope(fill(-1.0, 2), fill(1.0, 2))
t = GaussianTarget(box, [0.5, 0.0], [1.0 0.8; 0.8 1.0])
r = sample_until_converged(Xoshiro(1), t, CoordinateHitAndRun())
```
"""
function GaussianTarget(w::Warmup, μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    n = nvars(w.polytope)
    length(μ) == n || throw(DimensionMismatch("μ has length $(length(μ)), the polytope has $n variables"))
    size(Σ) == (n, n) || throw(DimensionMismatch("Σ must be $n×$n, got $(size(Σ))"))
    F = cholesky(Symmetric(Matrix{Float64}(Σ)); check = false)
    issuccess(F) || throw(ArgumentError("Σ must be positive definite"))
    return Target(w, QuadraticLogDensity(Matrix(inv(F)), Vector{Float64}(μ), zeros(n)))
end

function GaussianTarget(P::Polytope, μ, Σ; kwargs...)
    return GaussianTarget(warmup(P; kwargs...), μ, Σ)
end

"""
    ExponentialTarget(w::Warmup, c)
    ExponentialTarget(P::Polytope, c; kwargs...)

The distribution with density proportional to `exp(-cᵀx)` on the polytope: a
[`Target`](@ref) with `ℓ(x) = -cᵀx`. Given a [`Polytope`](@ref), the other keyword arguments
go to [`warmup`](@ref).

On a chord the density is a truncated exponential, which [`HitAndRun`](@ref) and
[`CoordinateHitAndRun`](@ref) draw exactly, without slice sampling.

# Examples
```julia
P = Polytope(zeros(3), ones(3); C = ones(1, 3), d = [1.0])  # the simplex
r = sample_until_converged(Xoshiro(1), ExponentialTarget(P, [1.0, 5.0, 20.0]), HitAndRun())
```
"""
function ExponentialTarget(w::Warmup, c::AbstractVector{<:Real})
    n = nvars(w.polytope)
    length(c) == n || throw(DimensionMismatch("c has length $(length(c)), the polytope has $n variables"))
    return Target(w, QuadraticLogDensity(zeros(n, n), zeros(n), Vector{Float64}(c)))
end

ExponentialTarget(P::Polytope, c; kwargs...) = ExponentialTarget(warmup(P; kwargs...), c)

# The log density `-(x - μ)ᵀ Q (x - μ) / 2 - cᵀx`, which is quadratic along any line
struct QuadraticLogDensity
    Q::Matrix{Float64}
    μ::Vector{Float64}
    c::Vector{Float64}
end

LogDensityProblems.dimension(d::QuadraticLogDensity) = length(d.μ)
LogDensityProblems.capabilities(::Type{QuadraticLogDensity}) = LogDensityOrder{1}()
function LogDensityProblems.logdensity(d::QuadraticLogDensity, x)
    r = x .- d.μ
    return -dot(r, d.Q, r) / 2 - dot(d.c, x)
end
function LogDensityProblems.logdensity_and_gradient(d::QuadraticLogDensity, x)
    r = x .- d.μ
    Qr = d.Q * r
    return -dot(r, Qr) / 2 - dot(d.c, x), -Qr .- d.c
end

# The geometry and the per-chain density of a model
geometry(w::Warmup) = w
geometry(t::Target) = t.warmup
chain_density(::Warmup) = nothing
chain_density(t::Target) = ZDensity(t)

# A log density in the coordinates `z` of a warmup, with its own buffer for `x`, so each
# chain has one
struct ZDensity{L}
    ℓ::L
    warmup::Warmup
    x::Vector{Float64}
end

ZDensity(t::Target) = ZDensity(t.logdensity, t.warmup, copy(t.warmup.center))

has_gradient(d::ZDensity) = LogDensityProblems.capabilities(d.ℓ) != LogDensityOrder{0}()

# `center + T z`, clamped to the bounds of the polytope: on a face `x = center + T z` can fall
# outside a bound by rounding error, where a density such as `log(xᵢ)` is undefined
function to_x!(d::ZDensity, z)
    mul!(d.x, d.warmup.transform, z)
    P = d.warmup.polytope
    for i in eachindex(d.x)
        d.x[i] = clamp(d.x[i] + d.warmup.center[i], P.lower[i], P.upper[i])
    end
    return d.x
end

# The log density at `z`, with `+Inf` and `NaN` taken as `-Inf`. A density that diverges at a
# face, such as a gamma density with shape below 1 at `xᵢ = 0`, is infinite only on a set of
# measure zero, so excluding those points leaves the target unchanged; a chain that moved onto
# one could never accept another point.
function logdensity_z(d::ZDensity, z)
    value = Float64(LogDensityProblems.logdensity(d.ℓ, to_x!(d, z)))
    return value < Inf ? value : -Inf
end

# Add `scale * ∇_z ℓ(z) = scale * Tᵀ ∇_x ℓ(x)` to `out`; return ℓ(z)
function add_gradient_z!(out, d::ZDensity, z, scale)
    value, g = LogDensityProblems.logdensity_and_gradient(d.ℓ, to_x!(d, z))
    mul!(out, d.warmup.transform', g, scale, 1.0)
    return value < Inf ? Float64(value) : -Inf
end

# A quadratic log density in the coordinates `z` of a warmup, `-zᵀ Q z / 2 - gᵀ z` up to a
# constant, for exact draws on chords: `Q = Tᵀ Q_x T` and `g = Tᵀ (Q_x (center - μ) + c)` for
# `x = center + T z`. `density` evaluates the log density itself, for the statistics.
struct ZQuadratic
    density::ZDensity{QuadraticLogDensity}
    Q::Matrix{Float64}
    g::Vector{Float64}
    Qu::Vector{Float64} # Q u for a direction u
end

function ZQuadratic(t::Target{QuadraticLogDensity})
    w, d = t.warmup, t.logdensity
    T = w.transform
    Q = Symmetric(T' * d.Q * T)
    g = T' * (d.Q * (w.center .- d.μ) .+ d.c)
    return ZQuadratic(ZDensity(t), Matrix(Q), g, zeros(size(T, 2)))
end

# The density that hit-and-run draws from on a chord: exactly for a quadratic log density,
# by slice sampling otherwise
chord_density(t::Target) = chain_density(t)
chord_density(t::Target{QuadraticLogDensity}) = ZQuadratic(t)

# The coefficients `(a, b)` of the log density `-a t² / 2 - b t` along `z + t u` (or along
# coordinate `j`)
function chord_coefficients(d::ZQuadratic, z, u::AbstractVector)
    mul!(d.Qu, d.Q, u)
    return dot(u, d.Qu), dot(z, d.Qu) + dot(d.g, u)
end

function chord_coefficients(d::ZQuadratic, z, j::Integer)
    return d.Q[j, j], dot(view(d.Q, :, j), z) + d.g[j]
end

# Draw `t` on `[tmin, tmax]` from the density proportional to `exp(-a t² / 2 - b t)`, `a ≥ 0`:
# a truncated normal, a truncated exponential when the quadratic term is negligible on the
# chord, and uniform when both are. When the mode `m = -b / a` lies far outside the chord,
# `m + σ z` would lose the draw to cancellation, so a tail draw is measured from the nearer
# end of the chord; hops instead falls back to a uniform draw when its draw leaves the chord
function draw_quadratic(rng::AbstractRNG, a, b, tmin, tmax)
    L = tmax - tmin
    if a * L^2 > 1.0e-12
        σ = 1 / sqrt(a)
        # the ends in standard units, (t - m) / σ, without forming m
        α, β = (a * tmin + b) * σ, (a * tmax + b) * σ
        t = if α > 0
            tmin + σ * tail_offset(rng, α, L / σ)
        elseif β < 0
            tmax - σ * tail_offset(rng, -β, L / σ)
        else
            -b / a + σ * truncated_randn(rng, α, β)
        end
        return clamp(t, tmin, tmax)
    elseif abs(b) * L > 1.0e-12
        # inverse distribution function of the exponential with rate |b| from the end with
        # the larger density
        e = -log1p(rand(rng) * expm1(-abs(b) * L)) / abs(b)
        return b > 0 ? tmin + e : tmax - e
    end
    return tmin + L * rand(rng)
end

# A standard normal draw restricted to `[α, β]`, by rejection from the proposals of Robert
# (1995): a normal when the interval holds 0 and is wide, a uniform when it is narrow, and in
# a tail a translated exponential, or a uniform when the interval is shorter than Robert's
# threshold. Every case accepts at least about a third of its proposals, so far tails cost no
# more than the center.
#
# Robert, C. P. (1995). Simulation of truncated normal variables. Statistics and Computing 5,
# 121–125. <https://doi.org/10.1007/BF00143942>
function truncated_randn(rng::AbstractRNG, α, β)
    α > 0 && return α + tail_offset(rng, α, β - α)
    β < 0 && return β - tail_offset(rng, -β, β - α)
    if β - α > sqrt(2π)
        while true
            z = randn(rng)
            α <= z <= β && return z
        end
    end
    while true
        z = α + (β - α) * rand(rng)
        randexp(rng) > z^2 / 2 && return z
    end
    return
end

# `z - α` for a standard normal draw `z` restricted to `[α, α + w]` with `α > 0`, `w ≤ Inf`.
# Working with the offset keeps its precision when `α` is large.
function tail_offset(rng::AbstractRNG, α, w)
    root = sqrt(α^2 + 4)
    λ = (α + root) / 2
    if w > 2 * sqrt(ℯ) / (α + root) * exp((α^2 - α * root) / 4)
        δ = 2 / (α + root) # λ - α
        while true
            u = randexp(rng) / λ
            u <= w && randexp(rng) > (u - δ)^2 / 2 && return u
        end
    end
    while true
        u = w * rand(rng)
        randexp(rng) > u * (2α + u) / 2 && return u
    end
    return
end

# The longest run of shrinkage steps before a slice update gives up and stays put; each
# step at least halves the expected interval, so this is reached only for degenerate
# densities
const SLICE_MAX = 200

"""
    slice_on_chord(rng, logp, logp0, tmin, tmax)

Draw `t` from the density proportional to `exp(logp(t))` on `[tmin, tmax]`, a chord through
the current point `t = 0` whose log density is `logp0`, by slice sampling with shrinkage
(Neal 2003, Section 4.2.2, with the whole chord as the initial interval). Returns `t` and
`logp(t)`.

Neal, R. M. (2003). Slice sampling. Annals of Statistics 31, 705–767.
<https://doi.org/10.1214/aos/1056562461>
"""
function slice_on_chord(rng::AbstractRNG, logp, logp0, tmin, tmax)
    logy = logp0 - randexp(rng)
    for _ in 1:SLICE_MAX
        t = tmin + (tmax - tmin) * rand(rng)
        lp = logp(t)
        lp > logy && return t, lp
        if t < 0
            tmin = t
        else
            tmax = t
        end
    end
    return 0.0, logp0
end

# The log density at `z`, zero for the uniform distribution
logdensity_or_zero(::Nothing, z) = 0.0
logdensity_or_zero(d::ZDensity, z) = logdensity_z(d, z)
