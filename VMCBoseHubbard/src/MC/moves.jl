# moves.jl

#=
Purpose: calculate the log of the acceptance probability for the Gutzwiller trial state
Input: n (system configuration), from (hop source site), to (hop target site), ψ (wavefunction)
Output: log of the acceptance ratio between two nearest neighbor hop-connected system configurations for the Gutzwiller trial state
Author: Will Mallah
Last Updated: 02/12/26

The Metropolis acceptance for sampling p(n) = |⟨n|ψ_G⟩|² with a symmetric hop
proposal is |ψ_G(n')/ψ_G(n)|². The Gutzwiller amplitude f(n) = e^{-κn²/2}/√(n!)
(see GutzwillerWavefunction) already carries the 1/√(n!) Fock weight, so the two
log_f differences below ARE log|ψ_G(n')/ψ_G(n)|² in full — including the √(n!)
multiplicity ratio. The earlier `+ log(n_from) - log(n_to+1)` term added that
multiplicity a second time (correct only for hardcore n≤1, where it vanishes),
biasing soft-core sampling; it has been removed.
=#
function log_acceptance_ratio_gutzwiller(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::GutzwillerWavefunction
)
    log_ratio =
        2 * (ψ.log_f[n[from_site]]     - ψ.log_f[n[from_site] + 1]) +
        2 * (ψ.log_f[n[to_site] + 2]   - ψ.log_f[n[to_site] + 1])

    return log_ratio
end

#=
Purpose: calculate the log of the acceptance probability for the Jastrow trial state
Input: n (system configuration), from (hop source site), to (hop target site), ψ (wavefunction)
Output: log of the acceptance ratio between two nearest neighbor hop-connected system configurations for the Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function log_acceptance_ratio_realspace_jastrow(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::Wavefunction,
    lattice
)

    Δlogpsi = compute_delta_logpsi_realspace(n, from_site, to_site, ψ, lattice)
    # 2·Δlogpsi = log|ψ_J'/ψ_J|² for the Jastrow factor; the log(n) term is the
    # k=0 condensate-reference multiplicity |Φ₀'/Φ₀|² = n_from/(n_to+1).
    log_ratio = 2.0 * Δlogpsi + log(n[from_site]) - log(n[to_site] + 1)

    return log_ratio
end