module IPOMDPToolbox

using POMDPs
using IPOMDPs
using SARSOP
using POMDPTools
using POMDPPolicies
using BeliefUpdaters
using POMCPOW
import Base: (==)
export
    pomdpModel,
    ipomdpModel,

    printPOMDP,

    DiscreteInteractiveBelief,
    DiscreteInteractiveUpdater,

    ReductionSolver,
    ReductionPolicy,

    IBPISolver,
    IBPIPolicy,

    #temporary
    BPIPolicy,
    solve_fresh!,
    continue_solving,
    print_solver_stats,
    load_policy,
    IBPIsimulate

    include("interactivebelief.jl")
    include("gpomdp.jl")
    include("reductionsolver.jl")
    include("ibpisolver.jl")
    include("ipomdpstoolbox.jl")
    include("functions.jl")
    include("simulator.jl")
end
