# measurements.jl

#=
Purpose: calculate local potential energy (trial state independent)
Input: n (vector of integers describing the system configuration, U (interaction parameter)
Output: local potential energy for given configuration
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_potential_energy(n::Vector{Int}, U::Float64)
    Epot = 0.0
    for ni in n
        Epot += 0.5 * U * ni * (ni - 1)
    end
    return Epot
end


#=
Purpose: calculate particle correlation function for a given configuration
Input: n (vector of integers describing the system configuration
Output: particle correlation function for given configuration
Author: Will Mallah
Last Updated: 07/01/2026
=#
function local_density_density_correlation(n::Vector{Int}, lattice)
    ps = lattice.pair_shell
    L  = length(n)
    C  = zeros(Float64, lattice.n_shells)
    @inbounds for i in 1:L, j in 1:L
        C[ps[i, j]] += n[i] * n[j]
    end
    return C ./ L      # per ordered-pair displacement shell (shell 1 = on-site Σn_i²)
end


# ── Gutzwiller ────────────────────────────────────────────────────────────────
#=
Purpose: calculate local kinetic energy for the Gutzwiller trial state
Input: n (vector of integers describing the system configuration, t (hopping parameter), n_max (maximum site occupancy), ψ (wavefunction struct), lattice (lattice struct)
Output: local kinetic energy of given configuration for Gutzwiller trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_kinetic_energy_gutzwiller(
    n::Vector{Int},
    t,
    n_max::Int,
    ψ::GutzwillerWavefunction,
    lattice
)
    log_f = ψ.log_f
    L = length(n)
    E_kin = 0.0

    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j → i.  The amplitude is a property of the BOND, so both
                # directions read t[i,j]; a real Bose-Hubbard t is symmetric.
                if n[j] > 0 && n[i] < n_max
                    log_R = (log_f[n[i] + 2] + log_f[n[j]]) - (log_f[n[i] + 1] + log_f[n[j] + 1])
                    E_kin -= hop_amplitude(t, i, j) * sqrt((n[i] + 1) * n[j]) * exp(log_R)
                end

                # hop i → j
                if n[i] > 0 && n[j] < n_max
                    log_R = (log_f[n[j] + 2] + log_f[n[i]]) - (log_f[n[j] + 1] + log_f[n[i] + 1])
                    E_kin -= hop_amplitude(t, i, j) * sqrt((n[j] + 1) * n[i]) * exp(log_R)
                end
            end
        end
    end

    return E_kin
end


#=
Purpose: calculate total local energy for the Gutzwiller trial state
Input: n (vector of integers describing the system configuration, ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy of given configuration for Gutzwiller trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_energy_gutzwiller(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System, n_max::Int64)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    E_pot = local_potential_energy(n, U)
    E_kin = local_kinetic_energy_gutzwiller(n, t, n_max, ψ, lattice)
    return E_kin + E_pot, E_kin, E_pot
end


# ── Jastrow ───────────────────────────────────────────────────────────────────
#=
Purpose: calculate local kinetic energy for the Jastrow trial state
Input: n (vector of integers describing the system configuration, t (hopping parameter), n_max (maximum site occupancy), ψ (wavefunction struct), lattice (lattice struct)
Output: local kinetic energy of given configuration for Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_kinetic_energy_jastrow(
    n::Vector{Int},
    t,
    n_max::Int,
    ψ::JastrowWavefunction,
    lattice
)
    L = length(n)
    Ekin = 0.0

    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j -> i gives a_i^† a_j
                if n[j] > 0 && n[i] < n_max
                    Δlogpsi = compute_delta_logpsi_realspace(n, j, i, ψ, lattice)
                    Ekin -= hop_amplitude(t, i, j) * n[j] * exp(Δlogpsi)
                end

                # hop i -> j gives a_j^† a_i
                if n[i] > 0 && n[j] < n_max
                    Δlogpsi = compute_delta_logpsi_realspace(n, i, j, ψ, lattice)
                    Ekin -= hop_amplitude(t, i, j) * n[i] * exp(Δlogpsi)
                end
            end
        end
    end

    return Ekin
end


#=
Purpose: calculate total local energy for the Jastrow trial state
Input: n (vector of integers describing the system configuration, ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy of given configuration for Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_energy_jastrow(
    n::Vector{Int},
    sys::System,
    n_max::Int,
    ψ::JastrowWavefunction
)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    Epot = local_potential_energy(n, U)
    Ekin = local_kinetic_energy_jastrow(n, t, n_max, ψ, lattice)
    return Ekin + Epot, Ekin, Epot
end


#=
Purpose: construct and store Gutzwiller coefficients from Gutzwiller variational parameter (κ)
Input: κ (Gutzwiller variational parameter), n_max (maximum site occupancy), logfact (pre-generated factorial values)
Output: GutzwillerWavefunction struct
Author: Will Mallah
Last Updated: 06/09/2026
=#
#=
Anisotropic real-space Jastrow convention (general lattice):

    logψ(n) = -½ ∑_{ordered (i,j)} v[shell(i,j)] n_i n_j ,

summed over all ordered pairs including i=j (shell 1 = on-site r=0). The uniform
½-over-ordered-pairs form reproduces the old 1D per-R prefactors exactly: an
off-diagonal unordered pair is counted twice (as (i,j) and (j,i), same shell by
the ±-fold), and the self / opposite-site terms fall out with the right weight.

O[s] = ∂ logψ / ∂ v[s] = -½ ∑_{(i,j): shell=s} n_i n_j  (used by the SR gradient/metric).
=#
function logpsi_derivatives_realspace(n::Vector{Int}, lattice)
    ps = lattice.pair_shell
    L  = length(n)
    O  = zeros(Float64, lattice.n_shells)
    @inbounds for i in 1:L, j in 1:L
        O[ps[i, j]] += n[i] * n[j]
    end
    O .*= -0.5
    return O
end


#=
Purpose: compute the change in the log of the Jastrow exponetial piece of the wavefunction
Input: n (system configuration), from_site (hop source site), to_site (hop target site), ψ (wavefunction)
Output: change in the log of the Jastrow exponetial piece of the wavefunction
Author: Will Mallah
Last Updated: 06/09/2026
=#
#=
Change in logψ for a single-particle hop `from_site → to_site` (one boson leaves
`from`, arrives at `to`) under the anisotropic Jastrow. With logψ = -½ ∑ v[shell] n_i n_j
and δ_a=-1 (from), δ_b=+1 (to), only sites a,b change occupation, giving the O(L) result

    Δlogψ = (h_a − h_b) − v[self] + v[shell(a,b)] ,     h_i = ∑_j v[shell(i,j)] n_j ,

where `self` is shell 1 (the on-site r=0 term). Reduces to the old 1D expression on a chain.
=#
function compute_delta_logpsi_realspace(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::JastrowWavefunction,
    lattice
)
    vr = ψ.vr
    ps = lattice.pair_shell
    L  = length(n)

    a = from_site
    b = to_site

    h_a = 0.0
    h_b = 0.0
    @inbounds for j in 1:L
        h_a += vr[ps[a, j]] * n[j]
        h_b += vr[ps[b, j]] * n[j]
    end
    v_self = vr[ps[a, a]]        # shell 1 (on-site / r = 0)

    return (h_a - h_b) - v_self + vr[ps[a, b]]
end