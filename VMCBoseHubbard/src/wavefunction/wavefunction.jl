# wavefunction.jl

# Define Wavefunction abstract type for multiple trial states
abstract type Wavefunction end

# ── Gutzwiller ────────────────────────────────────────────────────────────────

# Pre-generate list of factorial values
const LOGFACTORIAL_TABLE = [loggamma(m + 1) for m in 0:100]

# Store Gutzwiller variational parameter (κ) and log of Gutzwiller coefficients
struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    κ::T
    log_f::Vector{T}
end

#=
Purpose: construct and store Gutzwiller coefficients from Gutzwiller variational parameter (κ)
Input: κ (Gutzwiller variational parameter), n_max (maximum site occupancy), logfact (pre-generated factorial values)
Output: GutzwillerWavefunction struct
Author: Will Mallah
Last Updated: 06/09/2026
=#
function GutzwillerWavefunction(κ::Real, n_max::Int; logfact = LOGFACTORIAL_TABLE)
    log_f = [-0.5 * κ * n^2 - 0.5 * logfact[n + 1] for n in 0:n_max]
    log_Z = logsumexp(2 .* log_f)
    log_f .-= 0.5 * log_Z
    return GutzwillerWavefunction(κ, log_f)
end

# ── Jastrow ───────────────────────────────────────────────────────────────────

# Store real-space Jastrow variational parameters/Jastrow potentials (v_r).
# One coefficient per lattice displacement shell (see GeneralLattice.pair_shell):
# length == lattice.n_shells, ordered as GeneralLattice.shell_reps. On a 1D chain
# this reduces to the ⌊L/2⌋+1 ring-distance coefficients.
struct JastrowWavefunction{T <: Real} <: Wavefunction
    vr::Vector{T}
end