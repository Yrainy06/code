using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
println("Julia threads: ", Threads.nthreads())

###############################################################################
# semidiscretization of the compressible Euler equations

const gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

# -------------------------------------------------------------------
# Build the path to Mach.txt (same folder as this script)
mach_file = joinpath(@__DIR__, "Mach.txt")

Mach = parse(Float64, strip(read(mach_file, String)))
println("Free-stream Mach number read from file: ", Mach)
# -------------------------------------------------------------------

const rho_inf = 1.4
const p_inf   = 1.0
# compute speed of sound and free-stream velocity
const a_inf = sqrt(gamma * p_inf / rho_inf)
const v_inf = Mach * a_inf

"""
    initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
"""
@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    prim = SVector(rho_inf, v_inf, 0.0, p_inf)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach3_flow

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_mach3_flow)

# Outflow boundary condition
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)

    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

boundary_conditions = (; Bottom = boundary_condition_slip_wall,
                       Top = boundary_condition_slip_wall,
                       Right = boundary_condition_outflow,
                       Left = boundary_condition_inflow)

volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs

polydeg = 1
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# boundary symbols
boundary_symbols = [:Bottom, :Top, :Right, :Left]
mesh_file = joinpath(@__DIR__, "Bump.inp")
# mesh = P4estMesh{2}(mesh_file, p4est_partition_allow_for_coarsening = false, boundary_symbols = boundary_symbols)
mesh = P4estMesh{2}(mesh_file;
                   initial_refinement_level = 1,
                   p4est_partition_allow_for_coarsening = false,
                   boundary_symbols = boundary_symbols)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.025,
                                     output_directory = joinpath(@__DIR__, "out_ref_l1_t6_once"),
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

###############################################################################
# run the simulation
# sol = solve(ode, SSPRK43(stage_limiter! = stage_limiter!);
#             abstol = 1.0e-6, reltol = 1.0e-3,
#             ode_default_options()..., callback = callbacks);
# sol = solve(
#     ode,
#     SSPRK43(stage_limiter! = stage_limiter!);
#     ode_default_options()...,
#     adaptive = true,
#     abstol = 1.0e-8,
#     reltol = 1.0e-6,
#     callback = callbacks
# )

elapsed_seconds = @elapsed begin
    sol = solve(
        ode,
        SSPRK43(stage_limiter! = stage_limiter!);
        ode_default_options()...,
        adaptive = true,
        abstol = 1.0e-8,
        reltol = 1.0e-6,
        callback = callbacks
    )
end

println("Final simulation time: ", sol.t[end])
println("Elapsed time: ", elapsed_seconds, " s")

timing_file = joinpath(
    @__DIR__,
    "timing_l1_t$(Threads.nthreads())_once.txt"
)

open(timing_file, "w") do io
    println(io, "refinement_level = 1")
    println(io, "Julia_threads = ", Threads.nthreads())
    println(io, "final_time = ", sol.t[end])
    println(io, "elapsed_seconds = ", elapsed_seconds)
end