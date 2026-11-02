module PolytopeSamplers

using AbstractMCMC: AbstractMCMC, AbstractMCMCEnsemble, AbstractModel, AbstractSampler,
    MCMCDistributed, MCMCSerial, MCMCThreads, sample
using HiGHS: HiGHS
using LogDensityProblems: LogDensityProblems, LogDensityOrder
using JuMP: @constraint, @objective, @variable, DUAL_INFEASIBLE, INFEASIBLE,
    INFEASIBLE_OR_UNBOUNDED,
    MAX_SENSE, MIN_SENSE, Model,
    delete, delete_lower_bound, delete_upper_bound, get_attribute, is_solved_and_feasible, optimize!,
    set_attribute, set_lower_bound, set_normalized_rhs, set_silent, set_upper_bound,
    solver_name,
    termination_status, value
using LinearAlgebra: Diagonal, Hermitian, I, Symmetric, UpperTriangular, cholesky, cholesky!, diag, dot, eigen, eigvals, issuccess, ldiv!, lmul!, logdet, lu!, mul!, norm,
    ColumnNorm, normalize!, nullspace, pinv, qr, rank, svd, transpose!
using Random: AbstractRNG, Xoshiro, default_rng, randexp, randn!
using MCMCDiagnosticTools: ess_rhat, mcse
using Statistics: mean, quantile, std

include("polytope.jl")
export Polytope, isfeasible

include("rounding.jl")
include("warmup.jl")
export ranges, warmup
public Warmup

include("density.jl")
export ExponentialTarget, GaussianTarget, Target

include("sampler.jl")
include("billiard.jl")
include("barrier.jl")
include("crhmc.jl")
export ACHR, BilliardWalk, CoordinateHitAndRun, CRHMC, DikinWalk, HitAndRun, VaidyaWalk
export MCMCDistributed, MCMCSerial, MCMCThreads, sample
public AbstractHitAndRun, PolytopeSampler

include("diagnostics.jl")
export diagnostics

include("adaptive.jl")
export mmcs, sample_until_converged

include("volume.jl")
export volume

end
