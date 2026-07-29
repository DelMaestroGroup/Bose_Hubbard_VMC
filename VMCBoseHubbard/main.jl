push!(LOAD_PATH, joinpath(dirname(@__FILE__), "src"))

using VMCBoseHubbard
using ArgParse
using Random
using TimerOutputs
using Printf

# ============================================================
# Command line parsing
# ============================================================

function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--dimension", "-D"
            help = "Lattice dimension (1 = chain, 2 = square; other geometries via Pigsfli)"
            arg_type = Int
            required = true
            default = 1
            dest_name = "D"

        "--length", "-L"
            help = "Sites along axis 1 (chain length, or square Lx)"
            arg_type = Int
            required = true
            dest_name = "L"

        "--Ly"
            help = "Sites along axis 2 for a 2D square lattice (defaults to L)"
            arg_type = Int
            required = false
            default = nothing
            dest_name = "Ly"

        "--particle-number", "-N"
            help = "Total number of particles"
            arg_type = Int
            required = true
            dest_name = "N"

        "--interaction", "-U"
            help = "Interaction strength"
            arg_type = Float64
            required = true
            dest_name = "U"

        "--t", "-t"
            help = "Hopping parameter"
            arg_type = Float64
            default = 1.0

        "--trial-state"
            help = "Trial state type (gutzwiller, jastrow)"
            arg_type = String
            required = true

        "--kappa"
            help = "Initial Gutzwiller parameter"
            arg_type = Float64
            default = 1.0

        "--jastrow-potentials"
            help = "Initial Jastrow parameters"
            arg_type = String
            default = ""
            dest_name = "vr_init"

        "--n-max"
            help = "Maximum site occupancy"
            arg_type = Int
            default = -1
            dest_name = "n_max"

        "--eta"
            help = "SR learning rate"
            arg_type = Float64
            default = 0.01

        "--seed"
            help = "Random number generator seed"
            arg_type = Int
            default = 1234

        "--no-optimization"
            help = "Do not run optimization"
            action = :store_false

        # ====================================================
        # Optimization parameters
        # ====================================================

        "--max-iters"
            help = "Maximum number of MC runs in optimization"
            arg_type = Int
            default = 100
            dest_name = "max_iters"
            
        "--opt-num-walkers"
            help = "Optimization walkers"
            arg_type = Int
            default = 100
            dest_name = "opt_num_walkers"

        "--opt-num-MC-steps"
            help = "Optimization MC steps"
            arg_type = Int
            default = 5_000
            dest_name = "opt_num_MC_steps"

        "--opt-num-equil-steps"
            help = "Optimization equilibration steps"
            arg_type = Int
            default = 1_000
            dest_name = "opt_num_equil_steps"

        "--opt-block-size"
            help = "Optimization block size"
            arg_type = Int
            default = 500
            dest_name = "opt_block_size"

        # ====================================================
        # Final MC parameters
        # ====================================================

        "--final-num-walkers"
            help = "Final MC walkers"
            arg_type = Int
            default = 100
            dest_name = "final_num_walkers"

        "--final-num-MC-steps"
            help = "Final MC steps"
            arg_type = Int
            default = 5_000
            dest_name = "final_num_MC_steps"

        "--final-num-equil-steps"
            help = "Final MC equilibration steps"
            arg_type = Int
            default = 1_000
            dest_name = "final_num_equil_steps"

        "--final-block-size"
            help = "Final MC block size"
            arg_type = Int
            default = 500
            dest_name = "final_block_size"

        # ====================================================
        # Output control
        # ====================================================

        "--output-dir"
            help = "Base output directory"
            arg_type = String
            default = "data/VMC"

        "--save-history"
            help = "Save SR optimization history"
            action = :store_true

        "--skip-timing"
            help = "Disable timing output"
            action = :store_true
    end

    args = parse_args(s)

    Random.seed!(args["seed"])

    if args["L"] <= 1
        error("Please input a system size larger than one")
    end

    if args["N"] == 0
        error("Please input a non-zero particle number")
    end

    # Empty vr_init ⇒ let main() default it to zeros(lattice.n_shells) once the
    # lattice (and hence the shell count) is known; a supplied string is parsed here.
    if args["vr_init"] == ""
        args["vr_init"] = Float64[]
    else
        args["vr_init"] = [parse(Float64, strip(x)) for x in split(args["vr_init"], ',')]
    end

    if args["Ly"] === nothing
        args["Ly"] = args["L"]
    end

    if args["n_max"] == -1
        args["n_max"] = args["N"]
    end

    return args
end


# ============================================================
# Helper formatting
# ============================================================

function format_param(x::Real)
    return string(round(x; digits = 6))
end


# ============================================================
# Parameter file
# ============================================================

function write_parameters_file(
    filepath::String,
    args
)

    open(filepath, "w") do io

        println(io, "# ==================================================")
        println(io, "# System Parameters")
        println(io, "# ==================================================")

        println(io, "L $(args["L"])")
        println(io, "N $(args["N"])")
        println(io, "U $(args["U"])")
        println(io, "t $(args["t"])")
        println(io, "U_over_t $(args["U"] / args["t"])")
        println(io, "n_max $(args["n_max"])")
        println(io, "trial_state $(args["trial-state"])")
        println(io, "seed $(args["seed"])")

        println(io)

        println(io, "# ==================================================")
        println(io, "# Optimization Parameters")
        println(io, "# ==================================================")

        println(io, "eta $(args["eta"])")
        println(io, "opt_num_walkers $(args["opt_num_walkers"])")
        println(io, "opt_num_MC_steps $(args["opt_num_MC_steps"])")
        println(io, "opt_num_equil_steps $(args["opt_num_equil_steps"])")
        println(io, "opt_block_size $(args["opt_block_size"])")

        println(io)

        println(io, "# ==================================================")
        println(io, "# Final MC Parameters")
        println(io, "# ==================================================")

        println(io, "final_num_walkers $(args["final_num_walkers"])")
        println(io, "final_num_MC_steps $(args["final_num_MC_steps"])")
        println(io, "final_num_equil_steps $(args["final_num_equil_steps"])")
        println(io, "final_block_size $(args["final_block_size"])")
    end
end


# ============================================================
# Output helpers
# ============================================================

function write_energy_results(
    filepath::String,
    U_over_t::Float64,
    result
)

    open(filepath, "w") do io

        println(io, "# U_over_t   energy   sem")

        println(
            io,
            "$(U_over_t) " *
            "$(result.mean_energy) " *
            "$(result.sem_energy)"
        )
    end
end


function write_energy_parts(
    filepath::String,
    U_over_t::Float64,
    result
)

    open(filepath, "w") do io

        println(
            io,
            "# U_over_t   E_kin   E_kin_sem   E_pot   E_pot_sem"
        )

        println(
            io,
            "$(U_over_t) " *
            "$(result.mean_kinetic) " *
            "$(result.sem_kinetic) " *
            "$(result.mean_potential) " *
            "$(result.sem_potential)"
        )
    end
end

function write_density_density_correlation(
    filepath::String,
    result
)

    open(filepath, "w") do io

        println(io, "# r   density_density_corr   sem")

        for r in 1:length(result.mean_density_density_corr)
            println(
                io,
                "$(r) " *
                "$(result.mean_density_density_corr[r]) " *
                "$(result.sem_density_density_corr[r])"
            )
        end
    end
end


function write_gutzwiller_parameter(
    filepath::String,
    κ::Float64
)

    open(filepath, "w") do io

        println(io, "# kappa")
        println(io, κ)
    end
end


# Write the optimized Jastrow potentials, one row per displacement shell in the
# canonical shell order. Columns: shell_index  <displacement key…>  v. Pigsfli's
# loader reads the LAST column (v), in this order, and pairs it with its own
# pair_shell (built identically), so the ordering can't drift between the sides.
function write_jastrow_potentials(
    filepath::String,
    vr::Vector{Float64},
    lattice
)
    open(filepath, "w") do io
        println(io, "# shell   displacement_key...   v")
        for s in 1:length(vr)
            key = lattice.shell_reps[s]
            println(io, "$(s) $(join(key, " ")) $(vr[s])")
        end
    end
end


function write_sr_history(
    filepath::String,
    history
)

    isempty(history) && return

    num_params = length(history[1].gradient)

    open(filepath, "w") do io

        println(
            io,
            "# iter energy " *
            join(["g_$i" for i in 1:num_params], " ")
        )

        for (iter, h) in enumerate(history)

            println(
                io,
                "$(iter) $(h.energy) " *
                join(string.(h.gradient), " ")
            )
        end
    end
end


# ============================================================
# Main
# ============================================================

function main()

    to = TimerOutput()

    args = parse_commandline()

    println("Run with parameters:")

    for (arg, val) in args
        println(@sprintf "  %35s  =>  %15s" arg val)
    end

    if args["skip-timing"]
        disable_timer!(to)
    end

    # ========================================================
    # System parameters
    # ========================================================

    D = args["D"]
    L = args["L"]
    N = args["N"]
    U = args["U"]
    t = args["t"]

    U_over_t = U / t

    trial_state = args["trial-state"]

    # ========================================================
    # Output directory structure
    # ========================================================

    base_dir = args["output-dir"]

    trial_state_dir = joinpath(
        base_dir,
        trial_state
    )

    system_dir = joinpath(
        trial_state_dir,
        "$(D)D/L$(L)_N$(N)"
    )

    interaction_dir = joinpath(
        system_dir,
        "U$(format_param(U))_t$(format_param(t))"
    )

    mkpath(interaction_dir)

    # ========================================================
    # Output file paths
    # ========================================================

    parameters_file = joinpath(
        interaction_dir,
        "parameters.dat"
    )

    energy_file = joinpath(
        interaction_dir,
        "energy.dat"
    )

    energy_parts_file = joinpath(
        interaction_dir,
        "energy_parts.dat"
    )

    density_density_corr_file = joinpath(
        interaction_dir,
        "density_density_corr.dat"
    )

    history_file = joinpath(
        interaction_dir,
        "sr_history.dat"
    )

    # ========================================================
    # Construct system
    # ========================================================

    # Standalone CLI supports the hypercubic geometries (1D chain / 2D square).
    # Non-cubic lattices (triangular/honeycomb/kagome/custom) are optimized through
    # Pigsfli, which builds the GeneralLattice from its LatticeModels geometry.
    lattice = D == 1 ? Lattice1D(L) :
              D == 2 ? Lattice2D(L, args["Ly"]) :
              error("Standalone CLI supports D = 1 or 2 (got $D); use Pigsfli for other lattices.")

    sys = System(t, U, N, lattice)

    # ========================================================
    # Construct trial state
    # ========================================================

    if trial_state == "gutzwiller"

        wavefunction_init = GutzwillerWavefunction(
            args["kappa"],
            args["n_max"]
        )

    elseif trial_state == "jastrow"

        # One coefficient per displacement shell (default all-zeros unless overridden).
        vr0 = isempty(args["vr_init"]) ? zeros(lattice.n_shells) : copy(args["vr_init"])
        length(vr0) == lattice.n_shells ||
            error("--jastrow-potentials has $(length(vr0)) values but the lattice has $(lattice.n_shells) shells.")
        wavefunction_init = JastrowWavefunction(vr0)

    else

        error(
            "Unknown --trial-state \"$trial_state\". " *
            "Valid options: \"gutzwiller\", \"jastrow\"."
        )
    end

    # ========================================================
    # Optimization
    # ========================================================

    wavefunction_opt = nothing
    history = nothing

    if args["no-optimization"]
        @timeit to "Gradient Descent" begin

            wavefunction_opt, history = optimize_SR(
                sys,
                wavefunction_init,
                args["n_max"];

                η               = args["eta"],
                num_walkers     = args["opt_num_walkers"],
                num_MC_steps    = args["opt_num_MC_steps"],
                num_equil_steps = args["opt_num_equil_steps"],
                block_size      = args["opt_block_size"],
                max_iters       = args["max_iters"]
            )
        end
    else
        wavefunction_opt = wavefunction_init
        history = []
    end

    # ========================================================
    # Final MC run
    # ========================================================

    final_result = nothing

    @timeit to "High Statistics MC" begin

        final_result = MC_integration(
            sys,
            wavefunction_opt,
            args["n_max"];

            final_run       = true,
            num_walkers     = args["final_num_walkers"],
            num_MC_steps    = args["final_num_MC_steps"],
            num_equil_steps = args["final_num_equil_steps"],
            block_size      = args["final_block_size"]
        )
    end

    # ========================================================
    # Write outputs
    # ========================================================

    @timeit to "Write Results to Disk" begin

        write_parameters_file(
            parameters_file,
            args
        )

        write_energy_results(
            energy_file,
            U_over_t,
            final_result
        )

        write_energy_parts(
            energy_parts_file,
            U_over_t,
            final_result
        )

        write_density_density_correlation(
            density_density_corr_file,
            final_result
        )

        if wavefunction_opt isa JastrowWavefunction

            vr_file = joinpath(
                interaction_dir,
                "vr.dat"
            )

            write_jastrow_potentials(
                vr_file,
                wavefunction_opt.vr,
                lattice
            )

        elseif wavefunction_opt isa GutzwillerWavefunction

            kappa_file = joinpath(
                interaction_dir,
                "kappa.dat"
            )

            write_gutzwiller_parameter(
                kappa_file,
                wavefunction_opt.κ
            )
        end

        if args["save-history"]

            write_sr_history(
                history_file,
                history
            )
        end
    end

    # ========================================================
    # Timing summary
    # ========================================================

    show(to)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end