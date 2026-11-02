"""
    Polytope(lower, upper; A, b, C, d)

The convex polytope of points `x` with `lower .<= x .<= upper`, `A * x .<= b` and
`C * x .== d`. Bounds may be infinite as long as the constraints bound the polytope. A
coordinate with `lower[i] == upper[i]` is fixed at that value.

`A`, `b`, `C` and `d` default to no constraints and must be finite. Constraints of the form
`L * x .>= f` are `A = -L, b = -f`. Bounds of magnitude `1e20` or more are infinite, as for
HiGHS.

`isempty(P)` solves a linear program to check whether `P` has any points, and `x in P`
checks a point (see [`isfeasible`](@ref)).

# Examples
```jldoctest
julia> P = Polytope([0.0, 0.0], [1.0, 1.0]; A = [1.0 1.0], b = [1.0])  # a triangle
Polytope in 2 variables with 1 inequalities and 0 equalities

julia> [0.2, 0.3] in P
true

julia> isempty(P)
false
```
"""
struct Polytope <: AbstractModel
    lower::Vector{Float64}
    upper::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    C::Matrix{Float64}
    d::Vector{Float64}

    function Polytope(lower, upper, A, b, C, d)
        n = length(lower)
        length(upper) == n ||
            throw(DimensionMismatch("lower has length $n but upper has length $(length(upper))"))
        # PolyRound rejects infinite or NaN constraints (api.py:119-123): an infinite `b`
        # makes every slack infinite, so the rounding stops at once on a wrong ellipsoid
        (any(isnan, lower) || any(isnan, upper)) && throw(ArgumentError("bounds must not be NaN"))
        all(isfinite, A) && all(isfinite, b) && all(isfinite, C) && all(isfinite, d) ||
            throw(ArgumentError("A, b, C and d must be finite; leave out a constraint instead of making it infinite"))
        # Bounds beyond 1e20 in magnitude are infinite, as HiGHS treats them; a finite bound of
        # 1e30 would otherwise be a row whose slack swamps the others in rounding
        if any(x -> isfinite(x) && abs(x) >= INFINITE_BOUND, lower) || any(x -> isfinite(x) && abs(x) >= INFINITE_BOUND, upper)
            lower = [abs(x) >= INFINITE_BOUND ? -Inf : x for x in lower]
            upper = [abs(x) >= INFINITE_BOUND ? Inf : x for x in upper]
        end
        all(lower .<= upper) || throw(ArgumentError("lower bounds must not exceed upper bounds"))
        size(A) == (length(b), n) ||
            throw(DimensionMismatch("A must be $(length(b))×$n to match b and the bounds, got $(size(A))"))
        size(C) == (length(d), n) ||
            throw(DimensionMismatch("C must be $(length(d))×$n to match d and the bounds, got $(size(C))"))
        return new(lower, upper, A, b, C, d)
    end
end

function Polytope(
        lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real};
        A::AbstractMatrix{<:Real} = zeros(0, length(lower)), b::AbstractVector{<:Real} = zeros(0),
        C::AbstractMatrix{<:Real} = zeros(0, length(lower)), d::AbstractVector{<:Real} = zeros(0),
    )
    return Polytope(lower, upper, A, b, C, d)
end

nvars(P::Polytope) = length(P.lower)

function Base.show(io::IO, P::Polytope)
    print(io, "Polytope in ", nvars(P), " variables with ", length(P.b), " inequalities and ")
    print(io, length(P.d), " equalities")
    return
end

"""
    isfeasible(P::Polytope, x; atol = 1e-8)

Whether `x` satisfies every constraint of `P` to within `atol`, a distance in units of `x`:
an inequality or equality row is divided by its norm first. `x in P` uses the default
tolerance.
"""
function isfeasible(P::Polytope, x::AbstractVector{<:Real}; atol::Real = 1.0e-8)
    length(x) == nvars(P) || throw(DimensionMismatch("x has length $(length(x)), P has $(nvars(P)) variables"))
    all(P.lower .- atol .<= x .<= P.upper .+ atol) || return false
    A, b = unit_rows(P.A, P.b)
    all(A * x .<= b .+ atol) || return false
    C, d = unit_rows(P.C, P.d)
    return all(abs.(C * x .- d) .<= atol)
end

# The rows of `M` and the entries of `v` divided by the norms of the rows, so that residuals
# and solver tolerances are distances in units of `x`; zero rows are kept
function unit_rows(M, v)
    r = [norm(view(M, i, :)) for i in axes(M, 1)]
    r[r .== 0] .= 1
    return M ./ r, v ./ r
end

Base.in(x::AbstractVector{<:Real}, P::Polytope) = isfeasible(P, x)

# The largest amount by which `x` violates a bound, an inequality or an equality of `P`, with
# inequalities and equalities measured in units of `x` (divided by the norm of their row)
function constraint_violation(P::Polytope, x)
    v = 0.0
    for i in eachindex(x)
        v = max(v, P.lower[i] - x[i], x[i] - P.upper[i])
    end
    for i in eachindex(P.b)
        a = view(P.A, i, :)
        r = norm(a)
        v = max(v, r > 0 ? (dot(a, x) - P.b[i]) / r : -P.b[i])
    end
    for i in eachindex(P.d)
        c = view(P.C, i, :)
        r = norm(c)
        v = max(v, r > 0 ? abs(dot(c, x) - P.d[i]) / r : abs(P.d[i]))
    end
    return v
end

# Bounds at or beyond this magnitude are infinite, as for HiGHS
const INFINITE_BOUND = 1.0e20

# A silent JuMP model. For HiGHS, the primal and dual feasibility tolerances are tightened from
# 1e-7 to 1e-9, as PolyRound does for Gurobi, so that solver error does not pass for a direction
# in which the polytope has width above the `atol` of `warmup`, and each solve stops after
# 600 s, PolyRound's limit (default_settings.py), instead of hanging the warmup.
function lp(optimizer)
    model = Model(optimizer)
    set_silent(model)
    if solver_name(model) == "HiGHS"
        set_attribute(model, "primal_feasibility_tolerance", 1.0e-9)
        set_attribute(model, "dual_feasibility_tolerance", 1.0e-9)
        isinf(get_attribute(model, "time_limit")) && set_attribute(model, "time_limit", 600.0)
    end
    return model
end

# Solve `model`. Presolve can stop with INFEASIBLE_OR_UNBOUNDED on a degenerate program
# that is neither; hops solves again without presolve
# (LinearProgramGurobiImpl.cpp:96-100, commit b62592e, MIT), as here for HiGHS.
function solve!(model)
    optimize!(model)
    if termination_status(model) == INFEASIBLE_OR_UNBOUNDED && solver_name(model) == "HiGHS"
        set_attribute(model, "presolve", "off")
        optimize!(model)
        set_attribute(model, "presolve", "choose")
    end
    return model
end

# A JuMP model with the constraints of `P` and no objective, its variables and the rows of
# `A * x .<= b`. Constraints are built from the nonzero coefficients, as the matrices of food
# web problems are mostly zeros, and from rows of unit norm, so that the solver's feasibility
# tolerance is a distance in `x` (PolyRound normalizes its rows too).
function lp_model(P::Polytope, optimizer)
    model = lp(optimizer)
    n = nvars(P)
    A, b = unit_rows(P.A, P.b)
    C, d = unit_rows(P.C, P.d)
    x = @variable(model, [1:n])
    for i in 1:n
        isfinite(P.lower[i]) && set_lower_bound(x[i], P.lower[i])
        isfinite(P.upper[i]) && set_upper_bound(x[i], P.upper[i])
    end
    rows = map(eachindex(b)) do i
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n if A[i, j] != 0; init = 0.0) <= b[i])
    end
    for i in eachindex(d)
        @constraint(model, sum(C[i, j] * x[j] for j in 1:n if C[i, j] != 0; init = 0.0) == d[i])
    end
    return model, x, rows
end

"""
    isempty(P::Polytope; optimizer = HiGHS.Optimizer)

Whether `P` has no points, decided by a linear program.
"""
function Base.isempty(P::Polytope; optimizer = HiGHS.Optimizer)
    model, _ = lp_model(first(scaled_polytope(P)), optimizer)
    solve!(model)
    is_solved_and_feasible(model) && return false
    # With no objective the problem cannot be unbounded, so either status means infeasible
    status = termination_status(model)
    status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) && return true
    return error("the feasibility linear program stopped with status $status")
end

# Adapted from PolytopeSamplerMatlab, code/prepare/gmscale.m (commit f85505c, GPL-3.0;
# Michael Saunders, Systems Optimization Laboratory, Stanford University, after a routine
# of Robert Fourer, 1979): geometric-mean scaling. Returns column scales `c` and row scales
# `r`, powers of 2, such that the entries of `R⁻¹ A C⁻¹` are near 1. Each pass divides every
# row and then every column by the geometric mean of its largest and smallest nonzero
# magnitudes, with the smallest raised to at least `damp` times the largest, until the
# largest ratio within a column improves by less than the factor `scltol`; a final pass makes
# the largest entry of each column 1.
function gmscale(A::AbstractMatrix; scltol::Real = 0.9, maxpass::Integer = 10, damp::Real = 1.0e-4)
    m, n = size(A)
    A = abs.(A)
    pow2(x) = exp2(ceil(log2(x)))
    # the largest and smallest nonzero magnitude of each column (dims = 1) or row (dims = 2)
    extremes(S, dims) = (
        vec(maximum(S; dims, init = 0.0)),
        vec(minimum(s -> s == 0 ? Inf : s, S; dims, init = Inf)),
    )
    aratio = 1.0e50
    rscale = ones(m)
    cscale = ones(n)
    for npass in 0:maxpass
        cmax, cmin = extremes(A ./ rscale, 1)
        cmin[isinf.(cmin)] .= 0
        sratio = maximum((cmax[j] / (cmin[j] + eps()) for j in 1:n if cmax[j] > 0); init = 1.0)
        npass > 0 && (cscale = sqrt.(max.(cmin, damp .* cmax) .* cmax))
        (npass >= 2 && sratio >= aratio * scltol) && break
        npass == maxpass && break
        aratio = sratio
        cscale[cscale .== 0] .= 1
        cscale = pow2.(cscale)
        rmax, rmin = extremes(A ./ cscale', 2)
        rmin[isinf.(rmin)] .= 0
        rscale = sqrt.(max.(rmin, damp .* rmax) .* rmax)
        rscale[rscale .== 0] .= 1
        rscale = pow2.(rscale)
    end
    cscale = vec(maximum(A ./ rscale; dims = 1, init = 0.0))
    cscale[cscale .== 0] .= 1
    return pow2.(cscale), rscale
end

# `P` in the variables `x̄ = c .* x`, with `c` from the geometric-mean scaling of its
# constraint matrix `[A; C]`, and `c`. Linear programs and rounding are solved on the scaled
# polytope, as PolytopeSamplerMatlab rescales before its preprocessing (Polytope.m, rescale):
# on food web polytopes whose coefficients span six orders of magnitude, HiGHS reported
# optimal points that violated bounds by 0.59 and maximum-volume ellipsoids failed without
# it (#83). The scales are powers of 2, so mapping back is exact.
function scaled_polytope(P::Polytope)
    M = [P.A; P.C]
    c = isempty(M) ? ones(nvars(P)) : first(gmscale(M))
    return Polytope(P.lower .* c, P.upper .* c, P.A ./ c', P.b, P.C ./ c', P.d), c
end
