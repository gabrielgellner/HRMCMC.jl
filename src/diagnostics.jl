"""
    diagnostics(chains; kind = :bulk, multivariate = false)

Effective sample size and split-R̂ of every coordinate, from one chain (a matrix with one
point per column) or several chains of equal length (a vector of such matrices), as
returned by [`sample`](@ref).

Returns a named tuple:

- `ess`, `rhat`: one value per coordinate, `NaN` for coordinates that are constant over
  every chain, such as fixed coordinates. `rhat` is the rank R̂, the larger of the R̂ of
  the rank-normalized points (bulk) and of their distances from the median (tail), which
  also flags chains that agree in location but differ in spread.
- `mean`, `std`: the mean and standard deviation of each coordinate over all chains.
- `mcse_mean`, `mcse_std`: their Monte Carlo standard errors, which account for the
  correlation between points; report estimates with these as error bars.
- `min_ess`, `max_rhat`: the smallest ESS and the largest R̂ over the varying coordinates.
- `multivariate_rhat`: with `multivariate = true`, the multivariate R̂ of Brooks & Gelman
  (1998) of the split chains, the largest R̂ of any linear combination of the coordinates;
  `NaN` otherwise. It flags chains that disagree along a direction such as `x₁ - x₂` while
  every coordinate agrees. It costs a covariance matrix of the coordinates.
- `multivariate_rhat_iid`: with `multivariate = true`, the mean multivariate R̂ of
  independent draws with the same dimension, number of chains, chain length and ESS (the
  harmonic mean of the bulk ESS); `NaN` otherwise. The multivariate R̂ grows with the dimension `p` even
  for chains that agree, as the `2m` half-chain means estimate a `p`-dimensional covariance:
  with an ESS near 4000 per coordinate it is about 1.007 for `p = 10`, 1.025 for 50 and 1.09
  for 200. Compare `multivariate_rhat - multivariate_rhat_iid` with the cutoff of the
  coordinates' R̂, as [`sample_until_converged`](@ref) does.

`kind` sets the ESS: `:bulk` measures the center of each marginal and `:tail` its 5% and
95% quantiles (Vehtari et al. 2021). A common rule is to run until `max_rhat < 1.01` and
`min_ess` is at least 100 per chain. Computed with MCMCDiagnosticTools.jl.

Brooks, S. P. & Gelman, A. (1998). General methods for monitoring convergence of iterative
simulations. Journal of Computational and Graphical Statistics 7, 434–455.
<https://doi.org/10.1080/10618600.1998.10474787>

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of
MCMC. Bayesian Analysis 16, 667–718. <https://doi.org/10.1214/20-BA1221>
"""
function diagnostics(
        chains::AbstractVector{<:AbstractMatrix{<:Real}}; kind::Symbol = :bulk,
        multivariate::Bool = false,
    )
    isempty(chains) && throw(ArgumentError("no chains given"))
    allequal(size, chains) || throw(DimensionMismatch("chains must all have the same size"))
    nvars, ndraws = size(first(chains))
    # MCMCDiagnosticTools gives NaN for split chains of 4 points or fewer, which would read
    # as constant coordinates
    ndraws >= 10 || throw(ArgumentError("chains need at least 10 points, got $ndraws"))
    ess, bulk, rhat, means, stds, mcse_mean, mcse_std = (Vector{Float64}(undef, nvars) for _ in 1:7)
    # MCMCDiagnosticTools takes draws × chains × parameters; a block of coordinates at a
    # time keeps the copy small
    width = clamp(DIAGNOSTICS_BLOCK ÷ (ndraws * length(chains)), 1, nvars)
    x = Array{Float64}(undef, ndraws, length(chains), width)
    for first in 1:width:nvars
        cols = first:min(first + width - 1, nvars)
        xb = view(x, :, :, 1:length(cols))
        for (j, X) in enumerate(chains)
            transpose!(view(xb, :, j, :), view(X, cols, :))
        end
        # kind = :rank gives the bulk ESS and the larger of the bulk and tail R̂
        rank = ess_rhat(xb; kind = :rank)
        bulk[cols] = rank.ess
        ess[cols] = kind === :bulk ? rank.ess : ess_rhat(xb; kind).ess
        rhat[cols] = rank.rhat
        means[cols] = vec(mean(xb; dims = (1, 2)))
        stds[cols] = vec(std(xb; dims = (1, 2)))
        mcse_mean[cols] = mcse(xb; kind = mean)
        mcse_std[cols] = mcse(xb; kind = std)
    end
    varying = .!isnan.(ess)
    min_ess = any(varying) ? minimum(ess[varying]) : NaN
    max_rhat = any(varying) ? maximum(rhat[varying]) : NaN
    multivariate_rhat, multivariate_rhat_iid = if multivariate && any(varying)
        rhat_mv, dim = brooks_gelman_rhat(chains, findall(varying), stds)
        # the harmonic mean, as the noise of B/n along a direction goes as 1 / ESS
        ess_hm = 1 / mean(1 ./ bulk[varying])
        rhat_mv, iid_brooks_gelman_rhat(dim, 2length(chains), ndraws ÷ 2, ess_hm)
    else
        NaN, NaN
    end
    return (;
        ess, rhat, mean = means, std = stds, mcse_mean, mcse_std, min_ess, max_rhat,
        multivariate_rhat, multivariate_rhat_iid,
    )
end

diagnostics(X::AbstractMatrix{<:Real}; kwargs...) = diagnostics([X]; kwargs...)

# Draws × chains × coordinates copied at a time by `diagnostics`, 32 MB
const DIAGNOSTICS_BLOCK = 2^22

# Adapted from volesti, include/diagnostics/multivariate_psrf.hpp (commit 064e18c, LGPL-3.0;
# Copyright (c) 2012-2020 Vissarion Fisikopoulos, 2018-2020 Apostolos Chalkis), which splits
# one chain in halves; here every chain is split, as for split-R̂.
# The multivariate potential scale reduction factor of Brooks & Gelman (1998) over the
# coordinates `vars`, with standard deviations `stds`, from the 2m half chains of length n:
# (n - 1)/n + (m + 1)/m λ, with λ the largest eigenvalue of W⁻¹ B/n, W the mean covariance
# within half chains and B/n the covariance of their means. Points on a slice have a
# singular W, so λ is computed in the span of the eigenvectors of the correlation matrix
# with eigenvalues above MULTIVARIATE_RANK_TOL times the largest. Returns the R̂ and the
# dimension of that span.
function brooks_gelman_rhat(chains, vars, stds)
    n = size(first(chains), 2) ÷ 2
    halves = [view(X, vars, (k * n + 1):((k + 1) * n)) ./ view(stds, vars) for X in chains for k in 0:1]
    m = length(halves)
    p = length(vars)
    W = zeros(p, p)
    centers = zeros(p, m)
    for (j, Y) in enumerate(halves)
        μ = vec(mean(Y; dims = 2))
        centers[:, j] = μ
        Y .-= μ
        mul!(W, Y, Y', 1 / ((n - 1) * m), 1.0)
    end
    centers .-= mean(centers; dims = 2)
    Bn = centers * centers' / (m - 1)
    λ, U = eigen(Symmetric(W))
    keep = λ .> MULTIVARIATE_RANK_TOL * maximum(λ)
    S = U[:, keep] ./ sqrt.(λ[keep])'
    λmax = maximum(eigvals(Symmetric(S' * Bn * S)))
    return (n - 1) / n + (m + 1) / m * λmax, count(keep)
end

const MULTIVARIATE_RANK_TOL = 1.0e-10

# The mean of `brooks_gelman_rhat` for `m` half chains of `n` independent draws each in `p`
# dimensions, with `ess` effective samples in all. Each half-chain mean then has covariance
# W m / ess, so (m - 1) W^(-1/2) B/n W^(-1/2) is a Wishart matrix with m - 1 degrees of
# freedom scaled by m / ess, and λ is m / ess times the largest eigenvalue of the covariance
# of m standard normal vectors, simulated with a fixed seed. Estimating W from m(n - 1)
# degrees of freedom inflates it by ν / (ν - p - 1), the mean of an inverse Wishart matrix.
# Measured against 50 seeds of 4 chains of independent normal draws in 1 to 282 dimensions,
# this is within 0.002 of the mean R̂ for 1000 points per chain, and within 0.001 for 4000.
function iid_brooks_gelman_rhat(p, m, n, ess)
    ν = m * (n - 1)
    ν > p + 1 || return Inf
    rng = Xoshiro(0)
    Z = Matrix{Float64}(undef, m, p)
    λ = 0.0
    for _ in 1:IID_RHAT_DRAWS
        randn!(rng, Z)
        Z .-= mean(Z; dims = 1)
        λ += maximum(eigvals(Symmetric(Z * Z'))) / (m - 1)
    end
    λ /= IID_RHAT_DRAWS
    return (n - 1) / n + (m + 1) / m * λ * m / ess * ν / (ν - p - 1)
end

const IID_RHAT_DRAWS = 1000
