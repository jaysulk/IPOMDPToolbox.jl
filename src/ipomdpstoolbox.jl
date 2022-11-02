"""
    Wrapper for representing the model of a POMDP frame
    Uses SARSOP in order to solve the POMDP
"""
struct pomdpModel{S,A,W,P} <: IPOMDPs.Model{A,W}
    history::DiscreteBelief

    # Immutable part of the structure! This is common to all the models of the same frame!
    frame::POMDP{S,A,W}

    # Data
    updater::DiscreteUpdater
    policy::P
    depth::Int64
end

"""
    Wrapper for reresenting the model of a IPOMDP frame
    uses ReductionSolver in order to solve the IPOMDP
"""
struct ipomdpModel{S,A,W,P} <: IPOMDPs.Model{A,W}
	history::DiscreteInteractiveBelief
	frame::IPOMDP{S,A,W}
    updater::DiscreteInteractiveUpdater
    policy::P
    depth::Int64
end

"""
    Constructs a Model given the relative problem.
    In case the model uses an offline solver (e.g. pomdpModel uses SARSOP), the problem is solved an all the necessary data in order to retreive the best action is stored
    Model(pomdp::POMDP)
    Model(ipomdp::IPOMDP)
"""
function IPOMDPs.Model(model;depth,solver,force)
    return IPOMDPs.Model(model)
end

function IPOMDPs.Model(pomdp::POMDP;depth=0)

    solver = POMCPOWSolver(criterion=MaxUCB(20.0))
    policy = POMCPOW.solve(solver, pomdp)
    updater = POMCPOW.updater(policy)
    belief = POMCPOW.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))

    return pomdpModel(belief, pomdp, updater, policy, depth)
end
function IPOMDPs.Model(ipomdp::IPOMDP;depth=0)
    t = 10.0
    for i = 1:depth
        t = t/10
    end
    solver = ReductionSolver(t)
    updater = DiscreteInteractiveUpdater(ipomdp)
    policy = IPOMDPs.solve(solver, ipomdp)
    belief = IPOMDPs.initialize_belief(updater; depth=depth)

    return ipomdpModel(belief, ipomdp, updater, policy, depth)
end

"""
    Returns the best action for the model. If the model uses an online solver (e.g. ipomdpModel uses ReductionSolver), the problem is solved and the optimal action is returned
    action(model::pomdpModel)
    action(model::ipomdpModel)
"""
function IPOMDPs.action(model::pomdpModel)
    return SARSOP.action(model.policy, model.history)
end
function IPOMDPs.action(model::ipomdpModel)
    return IPOMDPs.action(model.policy, model.history)
end

"""
    Updates the belief and returns the updated model
    tau(model::pomdpModel{S,A,W}, a::A, o::W)
    tau(model::ipomdpModel{S,A,W}, a::A, o::W)
"""
function IPOMDPs.tau(model::pomdpModel{S,A,W}, a::A, o::W) where {S,A,W}
    belief = BeliefUpdaters.update(model.updater, model.history, a, o)
    return pomdpModel(belief, model.frame, model.updater, model.policy, model.depth)
end

function IPOMDPs.tau(model::ipomdpModel{S,A,W}, a::A, o::W) where {S,A,W}
    belief = IPOMDPs.update(model.updater, model.history, a, o)
    return ipomdpModel(belief, model.frame, model.updater, model.policy, model.depth)
end

#FIXME: Maybe should return a distribution instead of just a probability?
function IPOMDPs.actionP(model::ipomdpModel{S,A,W}, a::A) where {S,A,W}
    if a == IPOMDPs.action(model)
        return 1.0
    else
        return 0.0
    end
end

function IPOMDPs.actionP(model::pomdpModel{S,A,W}, a::A) where {S,A,W}
    if a == IPOMDPs.action(model)
        return 1.0
    else
        return 0.0
    end
end



