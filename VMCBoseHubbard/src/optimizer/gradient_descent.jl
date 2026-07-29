# gradient_descent.jl

# ── Parameter helpers ──────────────────────────────────────────────────────────
#=
flatten_params:   extract the variational parameters as a plain Vector.
unflatten_params: reconstruct a wavefunction from an updated parameter vector.
                  Dispatch on the old wavefunction's type determines which
                  constructor to call. Clamping is applied here for Gutzwiller
                  so the optimizer loop itself stays type-agnostic.
=#

flatten_params(wf::GutzwillerWavefunction) = [wf.κ]
flatten_params(wf::JastrowWavefunction)    = copy(wf.vr)

function unflatten_params(v::Vector{<:Real}, ::GutzwillerWavefunction, n_max::Int)
    κ = clamp(v[1], 1e-12, 10.0)
    return GutzwillerWavefunction(κ, n_max)
end

function unflatten_params(v::Vector{<:Real}, ::JastrowWavefunction, ::Int)
    return JastrowWavefunction(copy(v))
end


# ── SNR helper ─────────────────────────────────────────────────────────────────
#=
Returns the signal-to-noise ratio |g_i| / σ(g_i) for each gradient component.
A component is considered statistically resolved if snr[i] > z_grad.
=#
function snr_vector(g::Vector{Float64}, SE_g::Vector{Float64};
                    zero_tol::Float64 = 1e-16)
    snr = similar(g)
    for i in eachindex(g)
        if SE_g[i] > zero_tol
            snr[i] = abs(g[i]) / SE_g[i]
        elseif abs(g[i]) <= zero_tol
            snr[i] = 0.0
        else
            snr[i] = Inf
        end
    end
    return snr
end


# ── Stochastic Reconfiguration optimizer ──────────────────────────────────────
#=
Minimizes the variational energy using the Stochastic Reconfiguration (SR)
update rule:
    Δv = η · (S + λ_eff·I)⁻¹ · g ,   λ_eff = λ · tr(S)/dim

where g is the energy gradient and S is the quantum geometric tensor (metric).
Works for any Wavefunction subtype via flatten_params / unflatten_params dispatch.

Two properties make the DEFAULT settings converge across arbitrary geometries,
sizes, and fillings without per-system retuning (see below for why the naive
absolute-`λ`/absolute-`step_tol` form did not):

  * `λ` is applied RELATIVE to the metric scale, `λ_eff = λ·tr(S)/dim` (dim = #
    parameters). The scale of S grows with system size and filling (its entries
    are variances of O ~ Σ nᵢnⱼ), so a bare absolute `λ` under-regularizes large
    systems; the relative form fixes the same fractional shift everywhere. This
    also tames the EXACT null mode every canonical Jastrow has: because the
    displacement shells tile all ordered pairs incl. on-site, Σₛ Oₛ = −½N² is
    constant at fixed N, so "shift every vₛ by a constant" is a zero-curvature
    normalization gauge (an exact 0-eigenvector of S). It does not change the
    normalized state, but with too small a shift its gradient noise is amplified
    ∝ 1/λ_eff and the parameters wander forever.

  * convergence is governed by the SCALE-FREE energy-resolution test: stop once
    the SR-predicted energy drop `predicted_drop = η_ref·gᵀ(S+λ_eff·I)⁻¹g` falls
    below `z_energy·(energy SEM)`. Both sides carry the same MC-noise scaling, so
    one `z_energy` transfers across systems. This is exactly a state-space "we've
    stopped moving" test too, since the Fubini–Study step length satisfies
    `sr_step = √(η_ref·predicted_drop)` (S annihilates the gauge mode, so sr_step
    also ignores harmless wandering along it). The two are redundant, so the old
    ABSOLUTE step threshold — the only non-transferable knob — is off by default.

Convergence is assessed using a patience counter: the optimizer must satisfy
one of the convergence criteria for `patience` consecutive iterations before
stopping. Criteria:
  1. All gradient components are statistically indistinguishable from zero.
  2. The predicted energy drop is smaller than the energy resolution (z_energy·SEM),
     AND — only if the optional `step_tol` is set finite — the Fubini–Study step
     length is below it. With the default `step_tol = Inf` criterion 2 is purely
     the energy-resolution test.

The reference learning rate η_ref is used only for convergence diagnostics
(predicted drop and FS step), decoupling the stopping condition from the
choice of η.

kwargs
------
η              : learning rate
λ              : RELATIVE diagonal regularization; the shift is λ·tr(S)/dim
max_step       : hard clip on ||Δv||; useful for Gutzwiller to avoid large
                 steps early in optimization (set to Inf to disable)
z_grad         : SNR threshold for a gradient component to be "resolved"
z_energy       : multiplier on energy SEM for the predicted-drop convergence test
step_tol       : OPTIONAL Fubini–Study step-length cap; default Inf leaves the
                 scale-free energy test in charge. Set finite (dimensionless,
                 size-independent) to also require the state to have stopped moving.
min_iters      : minimum iterations before convergence is checked
patience       : consecutive convergence hits required to stop
max_iters      : hard iteration cap
=#
function optimize_SR(sys::System,
                     wf::Wavefunction,
                     n_max::Int;
                     η               ::Float64 = 0.05,
                     λ               ::Float64 = 2e-2,   # RELATIVE: shift = λ·tr(S)/dim
                     max_step        ::Float64 = Inf,
                     num_walkers     ::Int     = 200,
                     num_MC_steps    ::Int     = 30000,
                     num_equil_steps ::Int     = 5000,
                     block_size      ::Int     = 200,
                     z_grad          ::Float64 = 1.0,
                     z_energy        ::Float64 = 2.0,
                     step_tol        ::Float64 = Inf,    # optional FS step-length cap; Inf ⇒ energy-governed
                     min_iters       ::Int     = 5,
                     patience        ::Int     = 2,
                     max_iters       ::Int     = 200)

    history          = NamedTuple[]
    prev_E           = nothing
    prev_err         = nothing
    convergence_hits = 0
    η_ref            = 0.05     # fixed reference η for convergence diagnostics only

    for iter in 1:max_iters

        # ── Monte Carlo ───────────────────────────────────────────────────────
        result = MC_integration(sys, wf, n_max;
                                num_walkers     = num_walkers,
                                num_MC_steps    = num_MC_steps,
                                num_equil_steps = num_equil_steps,
                                block_size      = block_size)

        E    = result.mean_energy
        err  = result.sem_energy
        g    = result.gradient
        SE_g = result.gradient_standard_error
        S    = result.metric

        if any(!isfinite, g) || any(!isfinite, SE_g) || any(!isfinite, S)
            @warn "Stopping at iteration $iter: non-finite gradient or metric."
            break
        end

        # ── Adaptive step control (overshoot backtracking) ────────────────────
        # A fixed η that suits a small lattice can be too large for a larger /
        # higher-filling one, where it overshoots the minimum and the energy
        # oscillates (period-2) instead of settling. If the LAST step raised the
        # energy by a GROSS amount, halve η so the optimizer converges. The trigger
        # requires the ascent to be both statistically significant AND a sizeable
        # fraction of the energy scale, so ordinary flat-region MC wiggles (a few σ,
        # ≪1% of |E|) leave η — and the fast small-system convergence — untouched.
        if prev_E !== nothing
            ascent = E - prev_E
            if ascent > z_energy * sqrt(err^2 + prev_err^2) && ascent > 0.01 * abs(E)
                η = max(0.5 * η, 1e-4)
            end
        end

        # ── Natural gradient step ─────────────────────────────────────────────
        # Relative diagonal shift: λ_eff scales with the metric so one λ works
        # across system sizes/fillings and regularizes the fixed-N gauge null mode
        # (see the header). Floored to keep the linear solve non-singular.
        λ_eff     = max(λ * tr(S) / length(g), 1e-10)
        direction = (S + λ_eff * I) \ g
        Δv        = η     .* direction
        Δv_ref    = η_ref .* direction

        # Optional hard clip on step size
        if isfinite(max_step) && norm(Δv) > max_step
            Δv .*= max_step / norm(Δv)
        end

        # ── Convergence diagnostics ───────────────────────────────────────────
        snr            = snr_vector(g, SE_g; zero_tol = 1e-16)
        num_resolved   = count(x -> x > z_grad, snr)
        predicted_drop = dot(g, Δv_ref)
        rms_step       = norm(Δv_ref) / sqrt(length(Δv_ref))   # raw step (diagnostic only)
        # Fubini–Study step length: distance moved in state space. S annihilates
        # the gauge null mode, so this ignores harmless wandering along it and is
        # dimensionless/size-independent — the quantity the stop test uses.
        sr_step        = sqrt(max(dot(Δv_ref, S * Δv_ref), 0.0))

        if predicted_drop < 0.0
            @warn "SR step is not a descent direction at iteration $iter." predicted_drop
            predicted_drop = 0.0
        end

        # ── Progress output ───────────────────────────────────────────────────
        println("── Iteration $iter ─────────────────────────────")
        println("  Energy           = $(round(E, digits=8)) ± $(round(err, digits=8))")
        println("  Gradient norm    = $(round(norm(g), digits=6))")
        println("  Max SNR          = $(round(maximum(snr), digits=4))")
        println("  Resolved comps   = $num_resolved / $(length(g))")
        println("  FS step          = $(round(sr_step, digits=6))  (λ_eff=$(round(λ_eff, sigdigits=3)))")

        if prev_E !== nothing
            ΔE     = abs(E - prev_E)
            ΔE_err = sqrt(err^2 + prev_err^2)
            println("  |ΔE|             = $(round(ΔE, digits=8))")
            println("  σ(ΔE)           = $(round(ΔE_err, digits=8))")
        end

        push!(history, (
            wavefunction   = flatten_params(wf),
            energy         = E,
            sem_energy     = err,
            gradient       = copy(g),
            snr            = copy(snr),
            predicted_drop = predicted_drop,
            rms_step       = rms_step,
            sr_step        = sr_step
        ))

        # ── Convergence test ──────────────────────────────────────────────────
        gradient_zero     = num_resolved == 0
        energy_unresolved = predicted_drop <= z_energy * err
        step_small        = sr_step <= step_tol

        converged_now = iter >= min_iters &&
                        (gradient_zero || (energy_unresolved && step_small))

        if converged_now
            convergence_hits += 1
            println("  Convergence candidate: $convergence_hits / $patience")
        else
            convergence_hits = 0
        end

        if convergence_hits >= patience
            println("Converged at iteration $iter.")
            println("  gradient_zero     = ", gradient_zero)
            println("  energy_unresolved = ", energy_unresolved)
            println("  step_small        = ", step_small)
            wf = unflatten_params(flatten_params(wf) .- Δv, wf, n_max)
            break
        end

        # ── Parameter update ──────────────────────────────────────────────────
        wf       = unflatten_params(flatten_params(wf) .- Δv, wf, n_max)
        prev_E   = E
        prev_err = err
        η       *= 0.998
    end

    return wf, history
end