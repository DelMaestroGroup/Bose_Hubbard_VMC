# MC_integration.jl

# ── Results struct ─────────────────────────────────────────────────────────────
#=
Stores all outputs from a VMC run.
  energies: vector of per-block mean energies, used for autocorrelation diagnostics.
  gradient/metric: used directly by optimize_SR.
=#
struct VMCResults
    mean_energy                 ::Float64
    sem_energy                  ::Float64
    mean_kinetic                ::Float64
    sem_kinetic                 ::Float64
    mean_potential              ::Float64
    sem_potential               ::Float64
    mean_density_density_corr   ::Vector{Float64}
    sem_density_density_corr    ::Vector{Float64}
    gradient                    ::Vector{Float64}
    gradient_standard_error     ::Vector{Float64}
    metric                      ::Matrix{Float64}
    num_samples                 ::Int64
    acceptance_ratio            ::Float64
    energies                    ::Vector{Float64}    # block means, for tau estimation
    num_failed_moves            ::Int
end


# ── Walker validation ──────────────────────────────────────────────────────────
function check_and_warn_walker(n::Vector{Int}, n_max::Int)
    any(isnan, n)                    && (@warn "Walker contains NaN: $n";                   return false)
    any(isinf, n)                    && (@warn "Walker contains Inf: $n";                   return false)
    any(x -> x < 0 || x > n_max, n) && (@warn "Walker has out-of-bounds occupation: $n";   return false)
    return true
end


# ── Walker struct (Jastrow only) ───────────────────────────────────────────────
mutable struct Walker
    n      ::Vector{Int}
end

function initialize_walker(n::Vector{Int}, wf::JastrowWavefunction)
    return Walker(copy(n))
end


# ── Shared helper ──────────────────────────────────────────────────────────────
#=
Returns a near-uniform configuration with N particles on L sites,
with any remainder particles placed on the first sites.
=#
function ground_state_configuration(N::Int, L::Int, n_max::Int)
    N > L * n_max && error("Impossible: N > L * n_max")
    n = fill(div(N, L), L)
    for i in 1:(N % L)
        n[i] += 1
    end
    return n
end


# ── MC integration: Gutzwiller ─────────────────────────────────────────────────
#=
Performs canonical-ensemble MC for a Gutzwiller wavefunction.
Measurements begin after num_equil_steps to allow the walkers to thermalize.
Blocking is used throughout to obtain SEMs that account for autocorrelation.
=#
function MC_integration(sys::System,
                        wf::GutzwillerWavefunction,
                        n_max::Int;
                        num_walkers     ::Int = 200,
                        num_MC_steps    ::Int = 30000,
                        num_equil_steps ::Int = 5000,
                        block_size      ::Int = 200)

    L = length(sys.lattice.neighbors)
    N = sys.N
    n_corr = sys.lattice.n_shells        # density-density correlation is per displacement shell

    walkers = [ground_state_configuration(N, L, n_max) for _ in 1:num_walkers]

    # ── Block accumulators ────────────────────────────────────────────────────
    block_sum_E = 0.0;  block_sum_T = 0.0;  block_sum_V = 0.0
    block_sum_density_density_corr = zeros(Float64, n_corr)
    block_sum_g = 0.0;  block_count = 0

    block_means_E  = Float64[];  block_means_T = Float64[]
    block_means_V  = Float64[];  block_gradients = Float64[]
    block_means_density_density_corr = Vector{Vector{Float64}}()

    # Global sums for the SR metric (computed over all post-equilibration samples)
    sum_O = 0.0;  sum_OO = 0.0;  sum_EO = 0.0
    num_samples = 0;  num_accepted_moves = 0;  num_failed_moves = 0

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    @showprogress enabled=true "Running Gutzwiller VMC..." for step in 1:num_MC_steps
        for i in 1:num_walkers
            n = walkers[i]

            # Determine the source site by randomly selecting number 1 to L
            from = rand(1:L)

            # Determine reciever site by randomly selecting from neighbors of source site
            to = rand(sys.lattice.neighbors[from])

            # Ensure move is physical
            move_valid = (n[from] > 0) && (n[to] < n_max)

            # If move is physical, calculate acceptance ratio
            if move_valid && from != to
                log_ratio = log_acceptance_ratio_gutzwiller(n, from, to, wf)

                if isfinite(log_ratio) && log(rand()) < log_ratio
                    n[from] -= 1;  n[to] += 1
                    num_accepted_moves += 1
                else
                    num_failed_moves += 1
                end
            else
                num_failed_moves += 1
            end

            # ── Measurements ─────────────────────────────────────────────────
            if step >= num_equil_steps
                E, T, V = local_energy_gutzwiller(n, wf, sys, n_max)
                density_density_corr = local_density_density_correlation(n, sys.lattice)
                if isfinite(E)
                    block_sum_E += E;  block_sum_T += T;  block_sum_V += V
                    block_sum_density_density_corr .+= density_density_corr
                    block_count += 1

                    O = -0.5 * sum(n .^ 2)       # ∂(log|ψ|)/∂κ for Gutzwiller
                    block_sum_g += 2 * E * O
                    sum_O  += O;  sum_OO += O * O;  sum_EO += E * O

                    if block_count == block_size
                        push!(block_means_E,   block_sum_E / block_size)
                        push!(block_means_T,   block_sum_T / block_size)
                        push!(block_means_V,   block_sum_V / block_size)
                        push!(block_gradients, block_sum_g / block_size)
                        push!(block_means_density_density_corr, block_sum_density_density_corr / block_size)

                        block_sum_E = block_sum_T = block_sum_V = block_sum_g = 0.0
                        block_sum_density_density_corr .= 0.0
                        block_count = 0
                    end
                    num_samples += 1
                end
            end
        end
    end

    # ── Final statistics ──────────────────────────────────────────────────────
    acceptance_ratio = num_accepted_moves / (num_accepted_moves + num_failed_moves)

    if isempty(block_means_E)
        @warn "No valid energy samples collected!"
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, Float64[], Float64[],
                          Float64[], Float64[], zeros(1, 1),
                          0, acceptance_ratio, Float64[], num_failed_moves)
    end

    n_blocks = length(block_means_E)
    E_mean   = mean(block_means_E)
    E_error  = std(block_means_E) / sqrt(n_blocks)

    O_mean  = sum_O  / num_samples
    OO_mean = sum_OO / num_samples
    EO_mean = sum_EO / num_samples

    g = [2.0 * (EO_mean - E_mean * O_mean)]
    S = [OO_mean - O_mean^2;;]

    g_blocks  = reshape(block_gradients, :, 1) .- mean(block_gradients)
    SE_g      = vec(std(g_blocks, dims=1)) ./ sqrt(n_blocks)

    return VMCResults(
        E_mean, E_error,
        mean(block_means_T), std(block_means_T) / sqrt(n_blocks),
        mean(block_means_V), std(block_means_V) / sqrt(n_blocks),
        mean(block_means_density_density_corr), std(block_means_density_density_corr) / sqrt(n_blocks),
        g, SE_g, S,
        num_samples, acceptance_ratio, block_means_E, num_failed_moves
    )
end


# ── MC integration: Jastrow ────────────────────────────────────────────────────
#=
Performs canonical-ensemble MC for a real-space Jastrow wavefunction.
Gradient and metric are Nv-dimensional (one component per Jastrow coefficient).
Only completed blocks contribute to the gradient and metric estimates.
=#
function MC_integration(sys::System,
                        wf::JastrowWavefunction,
                        n_max::Int;
                        final_run::Bool = false,
                        num_walkers     ::Int = 200,
                        num_MC_steps    ::Int = 30000,
                        num_equil_steps ::Int = 5000,
                        block_size      ::Int = 200)

    L  = length(sys.lattice.neighbors)
    N  = sys.N
    Nv = length(wf.vr)
    n_corr = sys.lattice.n_shells
    Nv == n_corr || error("Jastrow vr has $Nv entries but the lattice has $n_corr displacement shells.")

    walkers = [initialize_walker(ground_state_configuration(N, L, n_max), wf)
               for _ in 1:num_walkers]

    # ── Block accumulators ────────────────────────────────────────────────────
    block_sum_E  = 0.0;  block_sum_T  = 0.0;  block_sum_V  = 0.0
    block_sum_density_density_corr = zeros(Float64, n_corr)
    block_sum_O  = zeros(Float64, Nv)
    block_sum_OO = zeros(Float64, Nv, Nv)
    block_sum_EO = zeros(Float64, Nv)
    block_count  = 0

    block_means_E  = Float64[];  block_means_T  = Float64[];  block_means_V  = Float64[]
    block_means_O  = Vector{Vector{Float64}}()
    block_means_EO = Vector{Vector{Float64}}()
    block_means_density_density_corr = Vector{Vector{Float64}}()

    # Running totals over completed blocks only (used for point estimates)
    used_sum_E  = 0.0;  used_sum_T  = 0.0;  used_sum_V  = 0.0
    used_sum_O  = zeros(Float64, Nv)
    used_sum_OO = zeros(Float64, Nv, Nv)
    used_sum_EO = zeros(Float64, Nv)
    used_sum_density_density_corr = zeros(Float64, n_corr)
    num_samples_used = 0;  num_accepted_moves = 0;  num_failed_moves = 0

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    @showprogress enabled=true "Running Jastrow VMC..." for step in 1:num_MC_steps
        for w in walkers
            n    = w.n

            # Determine the source site by randomly selecting number 1 to L
            from = rand(1:L)

            # Determine reciever site by randomly selecting from neighbors of source site
            to = rand(sys.lattice.neighbors[from])

            # Ensure move is physical
            move_valid = (n[from] > 0) && (n[to] < n_max)

            # If move is physical, calculate acceptance ratio
            if move_valid && from != to
                Δlogpsi = compute_delta_logpsi_realspace(n, from, to, wf, sys.lattice)
                log_ratio = log_acceptance_ratio_realspace_jastrow(n, from, to, wf, sys.lattice)

                if isfinite(log_ratio) && log(rand()) < log_ratio
                    n[from]  -= 1;  n[to] += 1
                    num_accepted_moves += 1
                else
                    num_failed_moves += 1
                end
            else
                num_failed_moves += 1
            end

            # ── Measurements ─────────────────────────────────────────────────
            if step > num_equil_steps
                E, T, V = local_energy_jastrow(w.n, sys, n_max, wf)
                if final_run
                    density_density_corr = local_density_density_correlation(w.n, sys.lattice)
                else
                    density_density_corr = zeros(Float64, n_corr)
                end
                if isfinite(E)
                    O = logpsi_derivatives_realspace(w.n, sys.lattice)

                    block_sum_E  += E;  block_sum_T  += T;  block_sum_V  += V
                    block_sum_density_density_corr  .+= density_density_corr
                    block_sum_O  .+= O
                    block_sum_OO .+= O * O'
                    block_sum_EO .+= E .* O
                    block_count  += 1

                    if block_count == block_size
                        push!(block_means_E,  block_sum_E  / block_size)
                        push!(block_means_T,  block_sum_T  / block_size)
                        push!(block_means_V,  block_sum_V  / block_size)
                        push!(block_means_density_density_corr,  block_sum_density_density_corr  / block_size)
                        push!(block_means_O,  copy(block_sum_O  ./ block_size))
                        push!(block_means_EO, copy(block_sum_EO ./ block_size))

                        used_sum_E   += block_sum_E;   used_sum_T   += block_sum_T
                        used_sum_V   += block_sum_V;   used_sum_O   .+= block_sum_O
                        used_sum_OO  .+= block_sum_OO; used_sum_EO  .+= block_sum_EO
                        used_sum_density_density_corr .+= block_sum_density_density_corr
                        num_samples_used += block_size

                        block_sum_E  = block_sum_T = block_sum_V = 0.0
                        block_sum_O .= 0.0;  block_sum_OO .= 0.0;  block_sum_EO .= 0.0
                        block_sum_density_density_corr .= 0.0
                        block_count  = 0
                    end
                end
            end
        end
    end

    # ── Final statistics ──────────────────────────────────────────────────────
    total_attempts   = num_accepted_moves + num_failed_moves
    acceptance_ratio = total_attempts > 0 ? num_accepted_moves / total_attempts : 0.0
    n_blocks         = length(block_means_E)

    if n_blocks == 0 || num_samples_used == 0
        @warn "No valid completed blocks collected!"
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, Float64[], Float64[],
                          Float64[], Float64[], zeros(Float64, 0, 0),
                          0, acceptance_ratio, Float64[], num_failed_moves)
    end

    block_count > 0 && @warn "Discarding incomplete final block of $block_count samples."
    n_blocks < 5    && @warn "Only $n_blocks completed blocks; gradient error bars may be unreliable."

    E_mean  = used_sum_E  / num_samples_used
    T_mean  = used_sum_T  / num_samples_used
    V_mean  = used_sum_V  / num_samples_used
    density_density_corr_mean = used_sum_density_density_corr / num_samples_used
    O_mean  = used_sum_O  / num_samples_used
    OO_mean = used_sum_OO / num_samples_used
    EO_mean = used_sum_EO / num_samples_used

    if n_blocks > 1
        E_error = std(block_means_E) / sqrt(n_blocks)
        T_error = std(block_means_T) / sqrt(n_blocks)
        V_error = std(block_means_V) / sqrt(n_blocks)
        density_density_corr_error = std(block_means_density_density_corr) / sqrt(n_blocks)
    else
        E_error = T_error = V_error = density_density_corr_error = Inf
    end

    g     = 2.0 .* (EO_mean .- E_mean .* O_mean)
    S_raw = OO_mean .- O_mean * O_mean'
    S     = 0.5 .* (S_raw + S_raw')        # enforce exact symmetry

    SE_g = if n_blocks > 1
        g_blocks = Matrix{Float64}(undef, n_blocks, Nv)
        for b in 1:n_blocks
            g_blocks[b, :] .= 2.0 .* (
                block_means_EO[b] .- E_mean .* block_means_O[b] .-
                block_means_E[b]  .* O_mean .+ E_mean .* O_mean
            )
        end
        vec(std(g_blocks, dims=1)) ./ sqrt(n_blocks)
    else
        fill(Inf, Nv)
    end

    return VMCResults(
        E_mean, E_error,
        T_mean, T_error,
        V_mean, V_error,
        density_density_corr_mean, density_density_corr_error,
        g, SE_g, S,
        num_samples_used, acceptance_ratio, block_means_E, num_failed_moves
    )
end