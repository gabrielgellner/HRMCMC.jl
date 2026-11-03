# The constrained food webs of Gellner, McCann & Hastings
# (https://github.com/gabrielgellner/constrained-foodwebs): random webs, their biomasses and
# growth rates, and the polytope of interaction strengths consistent with the equilibrium.
# Shared by `constrained_foodweb.jl` and `foodweb_replicates.jl`.
#
# Adapted from FoodWebs.jl (https://github.com/gabrielgellner/FoodWebs.jl), src/topology.jl,
# src/utils.jl, src/feasible.jl and ext/FoodWebsPolytopeSamplersExt.jl at v0.9.2 (0216f35),
# GPL-3.0-or-later, keeping only the settings of the original study.
# Copyright (c) 2026 Gabriel Gellner

using LinearAlgebra: Diagonal, I, eigvals
using PolytopeSamplers: Polytope
using Random: AbstractRNG

# A web is an S×S Bool matrix with `web[i, j]` when species `j` eats species `i`.

# The generalized cascade model (Stouffer et al. 2005, Ecology 86, 1301): species `j` eats
# each lower-ranked species with probability `p_j ~ Beta(1, β)`, `E[p] = 2CS / (S - 1)`. Webs
# are redrawn until they are connected and have `C S²` links within 3%, or the nearest whole
# number. Lower-ranked prey give every species a feeding path from a basal species.
function generalized_cascade_network(rng::AbstractRNG, S, C; tol = 0.03, maxiter = 10_000)
    β = (S - 1) / (2C * S) - 1
    target = C * S^2
    for _ in 1:maxiter
        # Beta(1, β) by inversion
        p = 1 .- rand(rng, S) .^ (1 / β)
        web = [i < j && rand(rng) < p[j] for i in 1:S, j in 1:S]
        abs(count(web) - target) <= max(tol * target, 0.5) && is_connected(web) && return web
    end
    return error("no connected web with $(round(Int, target)) links in $maxiter draws")
end

# Whether every species is linked to every other, ignoring the direction of feeding
function is_connected(web)
    S = size(web, 1)
    reached = falses(S)
    reached[1] = true
    frontier = [1]
    while !isempty(frontier)
        i = pop!(frontier)
        for j in 1:S
            if !reached[j] && (web[i, j] || web[j, i])
                reached[j] = true
                push!(frontier, j)
            end
        end
    end
    return all(reached)
end

# Prey-averaged trophic levels: 1 for basal species, and one more than the mean of the prey
function trophic_levels(web)
    D = Matrix{Float64}(transpose(web))
    D ./= max.(sum(D; dims = 2), 1)
    return (I - D) \ ones(size(web, 1))
end

# Biomass falling by `δ` per trophic level from `top`, fuzzed by up to 20% (an Eltonian
# pyramid for δ > 0), as in the original study
function pyramid_biomass(rng::AbstractRNG, web; top = 10.0, δ = 2.0)
    tl = trophic_levels(web) .- 1
    return top .+ δ .* (maximum(tl) .- tl) .* (0.8 .+ 0.2 .* rand(rng, length(tl)))
end

# Growth rates `rᵢ ~ U(0, 5)` for basal species and `rᵢ ~ -U(0, 1)` for consumers
growth_rates(rng::AbstractRNG, web) =
    [any(web[:, j]) ? -rand(rng) : 5 * rand(rng) for j in axes(web, 2)]

# The Lotka–Volterra parameters on `web` with the equilibrium biomasses `x` and growth rates
# `r`, that is `r + A x = 0`. The coordinates are, for each feeding link (prey `i`, predator
# `j`), the prey's loss `λ` at `A[i, j] = -λ` and the predator's gain `γ` at `A[j, i] = γ`, and
# for each species eaten by another its self-regulation `σ` at `A[i, i] = -σ`. A predator gains
# at most what its prey loses, `γ ≤ λ`, and every coordinate lies in `[0, 5]`.
struct FoodWebPolytope
    x::Vector{Float64}
    r::Vector{Float64}
    # One (kind, i, j) per coordinate, with the coordinate at A[i, j]
    coordinates::Vector{Tuple{Symbol, Int, Int}}
end

function FoodWebPolytope(web, x, r)
    S = size(web, 1)
    coordinates = Tuple{Symbol, Int, Int}[]
    for i in 1:S
        any(web[i, :]) && push!(coordinates, (:self, i, i))
        for j in 1:S
            web[i, j] && push!(coordinates, (:loss, i, j))
            web[j, i] && push!(coordinates, (:gain, i, j))
        end
    end
    return FoodWebPolytope(x, r, coordinates)
end

sign_in_A(kind) = kind === :gain ? 1.0 : -1.0

# The polytope of `fw` in the coordinates of `fw.coordinates`
function polytope(fw::FoodWebPolytope)
    n, S = length(fw.coordinates), length(fw.x)
    index = Dict(c => k for (k, c) in enumerate(fw.coordinates))
    # Row i of r + A x = 0
    C = zeros(S, n)
    for (k, (kind, i, j)) in enumerate(fw.coordinates)
        C[i, k] = sign_in_A(kind) * fw.x[j]
    end
    # γ - λ ≤ 0 for each link
    links = [(i, j) for (kind, i, j) in fw.coordinates if kind === :loss]
    A = zeros(length(links), n)
    for (row, (i, j)) in enumerate(links)
        A[row, index[(:gain, j, i)]] = 1
        A[row, index[(:loss, i, j)]] = -1
    end
    return Polytope(zeros(n), fill(5.0, n); A, b = zeros(length(links)), C, d = -fw.r)
end

# The community matrix `J = diag(x) A`, the Jacobian at the equilibrium, at the point `θ`
function community_matrix(fw::FoodWebPolytope, θ)
    S = length(fw.x)
    A = zeros(S, S)
    for (k, (kind, i, j)) in enumerate(fw.coordinates)
        A[i, j] += sign_in_A(kind) * θ[k]
    end
    return Diagonal(fw.x) * A
end

# The equilibrium is locally stable when every eigenvalue of J has a negative real part
is_stable(J) = maximum(real, eigvals(J)) < 0

# Growth rates redrawn until the polytope is nonempty, as in the original study
function feasible_food_web(rng::AbstractRNG, web, x; maxiter = 1000)
    for draws in 1:maxiter
        fw = FoodWebPolytope(web, x, growth_rates(rng, web))
        P = polytope(fw)
        isempty(P) || return (; fw, P, draws)
    end
    return error("no feasible growth rates in $maxiter draws")
end
