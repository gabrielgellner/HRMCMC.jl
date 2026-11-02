"""
    Polytope(lower, upper; A, b, C, d)

The convex polytope of points `x` with `lower .<= x .<= upper`, `A * x .<= b` and
`C * x .== d`. Bounds may be infinite as long as the constraints bound the polytope. A
coordinate with `lower[i] == upper[i]` is fixed at that value.

`A`, `b`, `C` and `d` default to no constraints. Constraints of the form `L * x .>= f` are
`A = -L, b = -f`.

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

# A silent JuMP model. For HiGHS, the primal and dual feasibility tolerances are tightened from
# 1e-7 to 1e-9, as PolyRound does for Gurobi, so that solver error does not pass for a direction
# in which the polytope has width above the `atol` of `warmup`.
function lp(optimizer)
    model = Model(optimizer)
    set_silent(model)
    if solver_name(model) == "HiGHS"
        set_attribute(model, "primal_feasibility_tolerance", 1.0e-9)
        set_attribute(model, "dual_feasibility_tolerance", 1.0e-9)
    end
    return model
end

# A JuMP model with the constraints of `P` and no objective. Constraints are built from the
# nonzero coefficients, as the matrices of food web problems are mostly zeros, and from rows
# of unit norm, so that the solver's feasibility tolerance is a distance in `x` (PolyRound
# normalizes its rows too).
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
    for i in eachindex(b)
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n if A[i, j] != 0; init = 0.0) <= b[i])
    end
    for i in eachindex(d)
        @constraint(model, sum(C[i, j] * x[j] for j in 1:n if C[i, j] != 0; init = 0.0) == d[i])
    end
    return model, x
end

"""
    isempty(P::Polytope; optimizer = HiGHS.Optimizer)

Whether `P` has no points, decided by a linear program.
"""
function Base.isempty(P::Polytope; optimizer = HiGHS.Optimizer)
    model, _ = lp_model(P, optimizer)
    optimize!(model)
    is_solved_and_feasible(model) && return false
    # With no objective the problem cannot be unbounded, so either status means infeasible
    status = termination_status(model)
    status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) && return true
    return error("the feasibility linear program stopped with status $status")
end
