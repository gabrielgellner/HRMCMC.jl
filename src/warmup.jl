# Values computed on first use and shared by the chains of a warmup, under the lock: Gᵀ, G Gᵀ
# and, without rounding, the trajectory length for the billiard walk, and the extreme points
# in `x` and `z` for `ranges` and ACHR when the warmup was built without them
mutable struct WarmupCache
    reflections::Union{Nothing, Tuple{Matrix{Float64}, Matrix{Float64}}}
    trajectory_length::Union{Nothing, Float64}
    points::Union{Nothing, Matrix{Float64}}
    zpoints::Union{Nothing, Matrix{Float64}}
    const optimizer::Any
    const lock::ReentrantLock
end

"""
    Warmup

A polytope in the coordinates that samplers move in, built by [`warmup`](@ref). Points are
`x = center + transform * z` for `z` in `{z : G z ≤ h}`, where `z` has one coordinate per
direction in which the polytope has width and the rows of `G` have unit norm.

- `polytope`: the [`Polytope`](@ref) sampled.
- `points`: the distinct extreme points of each coordinate, one per column.
- `center`: the point `z = 0`, the center of the last inscribed ellipsoid when rounded,
  otherwise the center of the largest inscribed ball, or the mean of `points` with
  `extremes = true`.
- `transform`: the `n × k` matrix taking `z` to `x - center`. Its columns are orthonormal
  when not rounded.
- `G`, `h`: the constraints of `P` in `z`, leaving out those that are constant on the slice.
- `zpoints`: `points` in `z` coordinates.
- `radius`: the radius of the largest ball in `{z : G z ≤ h}`, near 1 when rounded.
- `rounded`: whether `transform` rounds the polytope.
- `removed`: the number of redundant bounds and inequalities left out of `G` and `h`.

With `warmup(P; extremes = false)` (the default), `points` and `zpoints` have no columns;
[`ACHR`](@ref) computes the extreme points when a chain first needs them.
"""
struct Warmup <: AbstractModel
    polytope::Polytope
    points::Matrix{Float64}
    center::Vector{Float64}
    transform::Matrix{Float64}
    G::Matrix{Float64}
    h::Vector{Float64}
    zpoints::Matrix{Float64}
    radius::Float64
    rounded::Bool
    removed::Int
    cache::WarmupCache
end

function Warmup(
        P, points, center, transform, G, h, zpoints, radius, rounded;
        removed::Integer = 0, optimizer = HiGHS.Optimizer,
    )
    return Warmup(
        P, points, center, transform, G, h, zpoints, radius, rounded, removed,
        WarmupCache(nothing, nothing, nothing, nothing, optimizer, ReentrantLock()),
    )
end

# Gᵀ and G Gᵀ, computed once per warmup, so that chains on separate threads share one copy
# instead of each holding an m × m matrix
function reflection_matrices(w::Warmup)
    cache = w.cache
    return lock(cache.lock) do
        if cache.reflections === nothing
            cache.reflections = (Matrix(w.G'), w.G * w.G')
        end
        cache.reflections
    end
end

# The extreme points of each coordinate: `w.points`, or, for a warmup built without them,
# the `2n` linear programs solved once on first use
function extreme_points(w::Warmup)
    isempty(w.points) || return w.points
    cache = w.cache
    return lock(cache.lock) do
        if cache.points === nothing
            cache.points = coordinate_extremes(w.polytope, cache.optimizer)
        end
        cache.points
    end
end

# The extreme points of each coordinate in `z`. A warmup built from the largest inscribed
# ball has no equalities beyond those of its slice, so the points lie in it.
function extreme_zpoints(w::Warmup)
    isempty(w.zpoints) || return w.zpoints
    size(w.transform, 2) == 0 && return w.zpoints
    points = extreme_points(w)
    cache = w.cache
    return lock(cache.lock) do
        if cache.zpoints === nothing
            cache.zpoints = w.transform \ (points .- w.center)
        end
        cache.zpoints
    end
end

"""
    ranges(P::Polytope; optimizer = HiGHS.Optimizer)
    ranges(w::Warmup)

The smallest and largest value of each coordinate over the polytope, as the vectors
`(lower, upper)`, from the `2n` linear programs that minimize and maximize each coordinate.
Metabolic modeling calls this flux variability analysis. A coordinate with
`lower[i] == upper[i]` is fixed by the constraints, and a narrow range means that the
constraints nearly determine it.

For a warmup the extreme points of `warmup(P; extremes = true)` give the ranges without
further programs. Otherwise the programs are solved on first use and kept in the warmup for
later calls and for [`ACHR`](@ref) chains.

The ranges of samples lie inside these and can fall well short of them in thin corners of
the polytope.

# Examples
```jldoctest
julia> lower, upper = ranges(Polytope([0.0, 0.0], [1.0, 1.0]; A = [2.0 1.0], b = [1.0]));

julia> upper
2-element Vector{Float64}:
 0.5
 1.0
```
"""
ranges(P::Polytope; optimizer = HiGHS.Optimizer) = point_ranges(coordinate_extremes(P, optimizer))
ranges(w::Warmup) = point_ranges(extreme_points(w))

point_ranges(points) = (vec(minimum(points; dims = 2)), vec(maximum(points; dims = 2)))

# The coordinates that do not vary on the slice of a warmup: explicit equalities, equal bounds,
# and the equalities that `warmup` finds to hold over the polytope, within `atol`
fixed_coordinates(w::Warmup) = findall(i -> all(iszero, view(w.transform, i, :)), axes(w.transform, 1))

function Base.show(io::IO, w::Warmup)
    print(io, "Warmup on a ", size(w.transform, 2), "-dimensional slice of a ")
    print(io, nvars(w.polytope), "-variable polytope")
    w.rounded && print(io, ", rounded")
    fixed = length(fixed_coordinates(w))
    fixed > 0 && print(io, ", ", fixed, fixed == 1 ? " fixed coordinate" : " fixed coordinates")
    isempty(w.points) || print(io, ", with ", size(w.points, 2), " extreme points")
    w.removed > 0 && print(io, ", ", w.removed, w.removed == 1 ? " redundant constraint" : " redundant constraints", " removed")
    return
end

"""
    warmup([rng], P::Polytope; extremes = false, rounding = :ellipsoid, prune = true,
           optimizer = HiGHS.Optimizer, atol = 1e-8)

Prepare `P` for sampling and return a [`Warmup`](@ref) to [`sample`](@ref) from.

With `extremes = false` (the default), solve one linear program for the largest ball inside
`P` in the slice of `C * x .== d` and the fixed coordinates, as in multiphase Monte Carlo
sampling ([`mmcs`](@ref)). A polytope with room for a ball of positive radius has no other
equalities. If the ball has radius below `atol`, `P` has equalities beyond the explicit
ones, and `warmup` falls back to `extremes = true`.

With `extremes = true`, solve `2n` linear programs, minimizing and maximizing each of the
`n` coordinates. The extreme points locate the constraints that hold with equality over all
of `P`: a coordinate whose range is within `atol` of zero, or an inequality whose slack is
within `atol` of zero at every extreme point. Chains move only in the directions that keep
these equalities and `C * x .== d`, from a center that satisfies every constraint: the mean
of the extreme points, with the equalities held at their values there.

`atol` is in the units of the variables. A polytope only `atol` wide in some direction is
treated as flat in it: its samples are held at the center's value there, an error of at
most `atol`. A coordinate whose bounds are within `atol` of each other is fixed at their
midpoint. The default of `1e-8`, as in PolytopeSamplerMatlab, is ten times the
feasibility tolerance that `warmup` sets for HiGHS (`1e-9`), so only equalities that hold
over the polytope are fixed; directions as thin as `1e-7` are sampled. With a larger `atol`,
`warmup` warns when it fixes a direction wider than `1e-8`. The
coordinates that do not vary are listed by `fixed` in the result of
[`sample_until_converged`](@ref). The `2n` programs cost far more for large `n`: 58 s for
`n = 200` with 1000 constraints, against 1.3 s for the whole warmup with
`extremes = false`. [`ACHR`](@ref) uses the extreme points, and solves the `2n` programs on
first use when the warmup was built without them.

`rounding` changes coordinates so that the polytope is close to round, which makes every
sampler take long steps in thin directions:

- `:ellipsoid` (or `true`): map the maximum-volume inscribed ellipsoid to the unit ball,
  repeatedly, until it is nearly a ball. Each round factors dense matrices of size the
  number of constraints.
- `:samples`: draw billiard-walk samples and map their principal axes to equal lengths,
  repeatedly, as in multiphase Monte Carlo sampling. Runs one chain per thread and uses
  `rng`, so a seeded result depends on `Threads.nthreads()`.
- `:none` (or `false`).

With `prune = true` (the default), solve one linear program per finite bound and inequality,
and leave out of `G` and `h` each one whose removal leaves the polytope unchanged. These
include constraints that are tight only on a lower-dimensional face, such as a bound that
the other constraints reach at a single vertex. Each sampler step costs time in proportion
to the number of constraints, and `:ellipsoid` rounding factors matrices of that size. On a
food web with 282 variables, pruning removes 502 of 796 constraints and the warmup takes
6.4 s instead of 16.2 s. The programs are slow when `A` is dense: with 200 variables and
1000 dense inequalities they took 90 times as long as the rest of the warmup, so pass
`prune = false` for such polytopes. `w.polytope` keeps every constraint, and `w.removed`
counts those left out.

`warmup` first rescales the variables by geometric-mean scaling, as PolytopeSamplerMatlab
does, so that the constraint coefficients of each variable are near 1, and solves its linear
programs and rounding in the scaled variables. Without it, polytopes whose coefficients span
many orders of magnitude, such as food webs with biomasses from `1e-3` to `1e3`, gave the
solver inaccurate points. The result is mapped back to `P`, and `atol` stays in the units of
the variables.

Throws an `ArgumentError` when `P` is empty or unbounded; with `extremes = false` one more
linear program checks boundedness. A linear program that stops for another reason raises
an error naming its status. `optimizer` is any JuMP optimizer for linear programs;
each call builds its own model, so separate tasks may call `warmup` at the same time.
"""
function warmup(
        rng::AbstractRNG, P::Polytope; extremes::Bool = false,
        rounding::Union{Bool, Symbol} = :ellipsoid, prune::Bool = true,
        optimizer = HiGHS.Optimizer, atol::Real = 1.0e-8,
    )
    rounding = rounding === true ? :ellipsoid : rounding === false ? :none : rounding
    rounding in (:ellipsoid, :samples, :none) ||
        throw(ArgumentError("rounding must be :ellipsoid, :samples or :none, got :$rounding"))
    n = nvars(P)
    F, gap = fix_close_bounds(P, atol)
    gap > THIN_TOL && thin_warning(gap, atol)
    # Everything below works in the scaled variables `x̄ = c .* x`, and widths and slacks are
    # converted to units of `x` before they are compared with `atol`
    S, c = scaled_polytope(F)
    Q, removed = prune ? remove_redundant(S, optimizer) : (S, 0)
    # Without room for a ball of radius `atol` the polytope may have further equalities, which
    # only the extreme points find. A ball of radius `r` in `x̄` is at least `r / maximum(c)`
    # wide in `x` in every direction.
    slice = extremes ? nothing : chebyshev_slice(Q, optimizer, atol, atol * maximum(c))
    if slice === nothing
        points, center, T = extreme_slice(Q, optimizer, atol, c)
        G, h = reduced_constraints(Q, center, T)
    else
        center, T, G, h = slice
        points = zeros(n, 0)
    end
    if rounding === :ellipsoid
        round_polytope!(center, T, G, h; optimizer)
    elseif rounding === :samples
        round_by_samples!(rng, S, center, T, G, h; optimizer)
    end
    # Every sample keeps the center's values in the fixed directions, so a center outside the
    # polytope would put every sample outside; PolyRound and PolytopeSamplerMatlab also stop here
    violation = constraint_violation(S, center)
    if violation > THIN_TOL * max(1.0, norm(center, Inf))
        throw(ErrorException("warmup: the center violates a constraint of the polytope by $(violation); try a smaller atol or a different optimizer"))
    end
    zpoints = isempty(T) ? zeros(0, size(points, 2)) : T \ (points .- center)
    radius = isempty(T) ? 0.0 : last(chebyshev_center(G, h, optimizer))
    return Warmup(
        P, points ./ c, center ./ c, T ./ c, G, h, zpoints, radius, rounding !== :none;
        removed, optimizer,
    )
end

warmup(P::Polytope; kwargs...) = warmup(default_rng(), P; kwargs...)

# The distinct extreme points of each coordinate, one per column, from `2n` linear programs.
# A program that stops with a status other than optimal or unbounded is solved once more on
# a new model, which starts without the previous basis, as PolyRound resets the basis and
# solves again (PolyRound/static_classes/lp_interfacing.py, commit 6d3aee7, MIT). The
# programs are solved on the scaled polytope and the points mapped back.
function coordinate_extremes(P::Polytope, optimizer)
    S, c = scaled_polytope(P)
    return scaled_coordinate_extremes(S, optimizer) ./ c
end

function scaled_coordinate_extremes(P::Polytope, optimizer)
    model, x = lp_model(P, optimizer)
    optimize!(model)
    check_feasible(model)

    n = nvars(P)
    points = Matrix{Float64}(undef, n, 2n)
    for i in 1:n, (k, sense) in enumerate((MIN_SENSE, MAX_SENSE))
        @objective(model, sense, x[i])
        optimize!(model)
        if !is_solved_and_feasible(model) && !unbounded(model)
            model, x = lp_model(P, optimizer)
            @objective(model, sense, x[i])
            optimize!(model)
        end
        if !is_solved_and_feasible(model)
            unbounded(model) && throw(ArgumentError("the polytope is unbounded in coordinate $i"))
            extreme = sense == MIN_SENSE ? "minimum" : "maximum"
            error("the linear program for the $extreme of coordinate $i stopped with status $(termination_status(model))")
        end
        points[:, 2(i - 1) + k] = value.(x)
    end
    return unique(points; dims = 2)
end

function thin_warning(width, atol)
    @warn "warmup: fixed directions in which the polytope has width up to $(width), below atol = $(atol); its samples do not vary in them. Pass a smaller atol, such as 1e-9, to sample them." maxlog = 1
    return
end

# Whether a linear program stopped because its objective is unbounded
unbounded(model) = termination_status(model) in (DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED)

# Throw unless a linear program over the polytope found a point: an `ArgumentError` when the
# polytope is empty, an error naming the status when the solver stopped for another reason
function check_feasible(model)
    is_solved_and_feasible(model) && return model
    status = termination_status(model)
    status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) && throw(ArgumentError("the polytope is empty"))
    return error("the feasibility linear program stopped with status $status")
end

# `P` with each pair of bounds closer than `atol` fixed at its midpoint, and the largest
# such gap. A coordinate whose bounds differ by rounding error would otherwise leave the
# polytope no room for a ball, and `warmup` would fall back to the `2n` programs.
# PolytopeSamplerMatlab compares bounds with a relative tolerance (Polytope.m, dblcmp).
function fix_close_bounds(P::Polytope, atol)
    close = findall(@. 0 < P.upper - P.lower <= atol)
    isempty(close) && return P, 0.0
    lower, upper = copy(P.lower), copy(P.upper)
    mid = (lower[close] .+ upper[close]) ./ 2
    lower[close] .= mid
    upper[close] .= mid
    gap = maximum(P.upper[close] .- P.lower[close])
    return Polytope(lower, upper, P.A, P.b, P.C, P.d), gap
end

# Adapted from PolyRound, PolyRound/static_classes/constraint_removal_reduction.py (commit
# 6d3aee7, MIT; Copyright (c) 2026 PolyRound contributors): `P` without its redundant
# constraints, and the number removed. A bound or inequality is redundant when, with its
# right-hand side `r` raised by `max(1, |r|)`, the maximum of its left-hand side over the
# polytope stays within `THIN_TOL * max(1, |r|)` of `r`. PolyRound raises by 1, which the
# tolerance hides for `|r| ≥ 1e8`, so a bound such as `x ≤ 1e8` that is the only upper
# bound of `x` would be removed. This also removes weakly
# redundant constraints, which are tight only on a lower-dimensional face, such as a bound
# that the other constraints reach at a single vertex. Each redundant constraint leaves
# the model before the next is tested, so of two identical rows one is kept. Unlike
# PolyRound, which first solves the program without the raise to find the rows that hold
# with equality, this solves one program per constraint; `warmup` finds the equalities.
function remove_redundant(P::Polytope, optimizer)
    model, x, rows = lp_model(P, optimizer)
    optimize!(model)
    check_feasible(model)
    lower, upper = copy(P.lower), copy(P.upper)
    A, b = unit_rows(P.A, P.b)
    keep = trues(length(b))
    removed = 0
    raise(r) = max(1.0, abs(r))
    # A program that is not solved keeps its constraint, as in hops
    # (LinearProgramClpImpl.cpp, commit b62592e, MIT). A solution that violates `P` by more
    # than 100 times the solver's feasibility tolerance is inaccurate and keeps it too, the
    # check of PolyRound's `check_lps` (lp_interfacing.py, commit 6d3aee7, MIT): a removed
    # constraint holds at every point of the relaxed program, so such a point shows the
    # solve went wrong. Inaccurate optimal points removed binding bounds before #83.
    function redundant(f, rhs)
        @objective(model, MAX_SENSE, f)
        optimize!(model)
        is_solved_and_feasible(model) || return false
        value(f) <= rhs + THIN_TOL * raise(rhs) || return false
        return constraint_violation(P, value.(x)) <= LP_CHECK_TOL
    end
    for i in eachindex(upper)
        lower[i] == upper[i] && continue
        if isfinite(upper[i])
            set_upper_bound(x[i], upper[i] + raise(upper[i]))
            if redundant(x[i], upper[i])
                delete_upper_bound(x[i])
                upper[i] = Inf
                removed += 1
            else
                set_upper_bound(x[i], upper[i])
            end
        end
        if isfinite(lower[i])
            set_lower_bound(x[i], lower[i] - raise(lower[i]))
            if redundant(-x[i], -lower[i])
                delete_lower_bound(x[i])
                lower[i] = -Inf
                removed += 1
            else
                set_lower_bound(x[i], lower[i])
            end
        end
    end
    for i in eachindex(b)
        set_normalized_rhs(rows[i], b[i] + raise(b[i]))
        if redundant(sum(A[i, j] * x[j] for j in axes(A, 2) if A[i, j] != 0; init = 0.0), b[i])
            delete(model, rows[i])
            keep[i] = false
            removed += 1
        else
            set_normalized_rhs(rows[i], b[i])
        end
    end
    Q = Polytope(lower, upper, P.A[keep, :], P.b[keep], P.C, P.d)
    return Q, removed
end

# The extreme points of each coordinate, a center inside the polytope, and a basis of the
# directions in which the polytope has width. The center is the mean of the extreme points, a
# convex combination of points of the polytope, and the coordinates and inequalities found to
# hold with equality are fixed at their values there, so the center satisfies every constraint
# to the solver's tolerance; a center moved onto the midpoints of their ranges could leave the
# polytope by up to `atol`.
#
# The coordinate extremes need not span the polytope: on the triangle with vertices (0, 0),
# (1, 1) and (0.5, 0.4) they are the first two, at which `x₂ ≤ x₁` is tight (#39). So every
# inequality tight at the extreme points is checked by a linear program that maximizes its
# slack, whose solution joins the points when the slack exceeds `atol`; and every direction
# in which the points do not spread is measured by two programs. The equalities that hold
# over a polytope are among its constraint rows, so the checked rows give its directions.
#
# In `warmup`, `P` is scaled to `x̄ = c .* x` and every width and slack is converted to units
# of `x` before it is compared with `atol`: a range of coordinate `i` divided by `cᵢ`, and a
# width or slack along a unit vector `u` of `x̄` divided by `‖c .* u‖`, as `uᵀx̄ = (c .* u)ᵀx`.
function extreme_slice(P::Polytope, optimizer, atol, c = ones(nvars(P)))
    points = scaled_coordinate_extremes(P, optimizer)
    model, x = lp_model(P, optimizer)
    points = [points slack_points(model, x, P, points, atol, c)]
    center = vec(mean(points; dims = 2))
    Ceq, deq, width = active_equalities(P, points, center, atol, c)
    # Move the center onto `C * x .== d`, which the linear programs satisfy only to the
    # solver's tolerance; the other rows hold at the center already
    if !isempty(deq)
        center .-= pinv(Ceq) * (Ceq * center .- deq)
    end
    # A direction spread less than `atol * maximum(c)` in `x̄` may be thinner than `atol` in
    # `x`, so it is measured by `keep_wide_directions`
    T, D = direction_basis(Ceq, points, center, atol * maximum(c))
    T, wide, dropped = keep_wide_directions(model, x, T, D, atol, c)
    points = [points wide]
    width = max(width, dropped)
    width > THIN_TOL && thin_warning(width, atol)
    return points, center, T
end

# The slice of `C * x .== d` and the fixed coordinates, centered on the largest ball inside
# the polytope in that slice, from one linear program. Returns `nothing` when the ball has
# radius below `ratol` (default `atol`), as the polytope may then have further equalities.
function chebyshev_slice(P::Polytope, optimizer, atol, ratol = atol)
    n = nvars(P)
    fixed = findall(P.lower .== P.upper)
    E = Matrix{Float64}(I, n, n)
    C, d = unit_rows(P.C, P.d)
    Ceq = [C; E[fixed, :]]
    deq = [d; P.lower[fixed]]
    x0 = isempty(deq) ? zeros(n) : pinv(Ceq) * deq
    if !isempty(deq) && norm(Ceq * x0 .- deq) > atol * max(1, norm(deq))
        throw(ArgumentError("the polytope is empty: its equalities have no solution"))
    end
    T = isempty(deq) ? E : nullspace(Ceq)
    T[abs.(T) .< ZERO_TOL] .= 0
    G, h = reduced_constraints(P, x0, T; clamp = false)
    if size(T, 2) == 0
        isfeasible(P, x0; atol) || throw(ArgumentError("the polytope is empty"))
        return x0, T, G, max.(h, 0.0)
    end
    c, r = chebyshev_center(G, h, optimizer)
    r > ratol || return nothing
    isbounded(G, optimizer) || throw(ArgumentError("the polytope is unbounded"))
    return shift!(x0, T, G, h, c)
end

# Whether `{z : G z ≤ h}` (nonempty) is bounded: no direction `d ≠ 0` has `G d ≤ 0`. That
# holds exactly when `G` has full column rank, ruling out `G d = 0`, and some `y > 0` has
# `Gᵀ y = 0`, ruling out `G d ≤ 0` with `G d ≠ 0` (Stiemke's theorem of the alternative);
# the second is a linear feasibility problem in `y ≥ 1`. Only an infeasible program means
# unbounded: a solver that stops for another reason raises an error, as the reference
# samplers report failed programs, and before #83 a bounded polytope was called unbounded.
function isbounded(G, optimizer)
    m, k = size(G)
    m == 0 && return k == 0
    rank(G) == k || return false
    model = lp(optimizer)
    y = @variable(model, [1:m], lower_bound = 1)
    @constraint(model, G' * y .== 0)
    optimize!(model)
    is_solved_and_feasible(model) && return true
    status = termination_status(model)
    status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) && return false
    return error("the boundedness linear program stopped with status $status")
end

# The bounds and inequalities of `P` as `G z ≤ h` for `x = center + T z`, with unit-norm rows.
# Rows that are zero on the slice (fixed coordinates, inequalities that hold with equality)
# are left out, and slacks are clamped at zero against rounding error in `center` unless
# `clamp = false` (when `center` need not be inside).
function reduced_constraints(P::Polytope, center, T; clamp::Bool = true)
    k = size(T, 2)
    rows = Vector{Float64}[]
    rhs = Float64[]
    # a row constant on the slice is dropped, but one violated there leaves the slice empty
    tol = THIN_TOL * max(1.0, norm(center, Inf))
    function add!(g, s, scale)
        r = norm(g)
        if r > ROW_TOL * scale
            push!(rows, g ./ r)
            push!(rhs, (clamp ? max(s, 0.0) : s) / r)
        elseif !clamp && s < -tol * scale
            throw(ArgumentError("the polytope is empty: its equalities violate a bound or inequality"))
        end
        return
    end
    for i in 1:nvars(P)
        g = T[i, :]
        isfinite(P.upper[i]) && add!(g, P.upper[i] - center[i], 1.0)
        isfinite(P.lower[i]) && add!(-g, center[i] - P.lower[i], 1.0)
    end
    for i in eachindex(P.b)
        a = P.A[i, :]
        add!(T' * a, P.b[i] - dot(a, center), norm(a))
    end
    G = isempty(rows) ? zeros(0, k) : permutedims(reduce(hcat, rows))
    return G, rhs
end

# The equalities `C * x .== d` with the coordinates whose range is within `atol` and the
# inequalities whose slack is within `atol` at every extreme point added as rows, held at their
# values at `center`; and the largest such range or slack, which is the solver's tolerance for
# equalities that hold over the polytope and more for directions that are only thin. Ranges
# and slacks are in units of `x = x̄ ./ c` (see `extreme_slice`).
function active_equalities(P::Polytope, points, center, atol, c = ones(nvars(P)))
    n = nvars(P)
    C, d = unit_rows(P.C, P.d)
    rows = [C]
    rhs = [d]
    width = 0.0
    for i in 1:n
        lo, hi = extrema(view(points, i, :))
        if (hi - lo) / c[i] <= atol && P.lower[i] != P.upper[i]
            e = zeros(1, n)
            e[i] = 1
            push!(rows, e)
            push!(rhs, [center[i]])
            width = max(width, (hi - lo) / c[i])
        end
    end
    # slacks in units of x, so that atol is a distance whatever the scale of the row
    A, b = unit_rows(P.A, P.b)
    slack = b .- A * points
    for i in eachindex(b)
        largest = maximum(view(slack, i, :)) / norm(c .* view(A, i, :))
        if largest <= atol
            push!(rows, A[i:i, :])
            push!(rhs, [dot(view(A, i, :), center)])
            width = max(width, largest)
        end
    end
    fixed = findall(P.lower .== P.upper)
    if !isempty(fixed)
        push!(rows, Matrix{Float64}(I, n, n)[fixed, :])
        push!(rhs, P.lower[fixed])
    end
    return reduce(vcat, rows), reduce(vcat, rhs), width
end

# Ranges and slacks up to this are the solver's tolerance on a constraint that holds with
# equality over the polytope; above it, and up to `atol`, the polytope is thin but not flat
const THIN_TOL = 1.0e-8

# The largest violation of the polytope accepted from a linear program's solution: 100 times
# the feasibility tolerance of 1e-9 that `lp` sets for HiGHS, as PolyRound accepts 100 times
const LP_CHECK_TOL = 1.0e-7

# A constraint row whose norm on the slice is below this fraction of its norm in `x` is
# constant on the slice
const ROW_TOL = 1.0e-9

# Orthonormal directions that keep `Ceq * x` fixed and along which the extreme points spread,
# and the other directions that keep `Ceq * x` fixed, in which the polytope may be thinner
# than `atol` or the points may miss its width; `keep_wide_directions` checks those. Entries
# below `ZERO_TOL` are set to zero, so that a direction moves a fixed coordinate by exactly
# zero. A singular value counts as spread when it exceeds the `sqrt(m) * atol` that `m`
# points each within `atol` of a hyperplane can reach.
function direction_basis(Ceq, points, center, atol)
    n = length(center)
    Z = isempty(Ceq) ? Matrix{Float64}(I, n, n) : nullspace(Ceq)
    F = svd(Z' * (points .- center); full = true)
    m = size(points, 2)
    k = count(>(sqrt(m) * atol), F.S)
    basis = Z * F.U[:, 1:k]
    basis[abs.(basis) .< ZERO_TOL] .= 0
    return basis, Z * F.U[:, (k + 1):end]
end

# For each inequality whose slack is within `atol` at every point, the point of the
# polytope that maximizes its slack, when that slack exceeds `atol`, with slacks in units of
# `x = x̄ ./ c` (see `extreme_slice`)
function slack_points(model, x, P::Polytope, points, atol, c = ones(nvars(P)))
    A, b = unit_rows(P.A, P.b)
    found = Vector{Float64}[]
    for i in eachindex(b)
        tol = atol * norm(c .* view(A, i, :))
        maximum(b[i] .- view(A, i, :)' * points) <= tol || continue
        any(p -> b[i] - dot(view(A, i, :), p) > tol, found) && continue
        @objective(model, MIN_SENSE, sum(A[i, j] * x[j] for j in axes(A, 2) if A[i, j] != 0; init = 0.0))
        optimize!(model)
        is_solved_and_feasible(model) || continue
        p = value.(x)
        b[i] - dot(view(A, i, :), p) > tol && push!(found, p)
    end
    return isempty(found) ? zeros(size(points, 1), 0) : reduce(hcat, found)
end

# The directions `D` in which the polytope has width above `atol`, by minimizing and
# maximizing along each, added to the basis `T`; returns the basis, the points that reach
# the widths, and the largest width of the directions left out, in units of `x = x̄ ./ c`
# (see `extreme_slice`)
function keep_wide_directions(model, x, T, D, atol, c = ones(size(T, 1)))
    wide = Vector{Float64}[]
    kept = Vector{Float64}[]
    dropped = 0.0
    for v in eachcol(D)
        ends = map((MIN_SENSE, MAX_SENSE)) do sense
            @objective(model, sense, sum(v[j] * x[j] for j in eachindex(v) if v[j] != 0; init = 0.0))
            optimize!(model)
            value.(x)
        end
        w = (dot(v, ends[2]) - dot(v, ends[1])) / norm(c .* v)
        if w > atol
            push!(kept, v)
            append!(wide, ends)
        else
            dropped = max(dropped, w)
        end
    end
    isempty(kept) && return T, zeros(size(T, 1), 0), dropped
    V = reduce(hcat, kept)
    V[abs.(V) .< ZERO_TOL] .= 0
    return [T V], reduce(hcat, wide), dropped
end
