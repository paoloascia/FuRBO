# FuRBO Stopping criterion
# 
# March 2024
##########
# Imports

def max_iterations(state, n_iteration):
    '''Function to evaluate if the maximum number of allowed iterations is reached.'''
    if state.it_counter < n_iteration:
        return False
    return True

def max_evaluations(state, n_evaluation):
    '''Function to evaluate if the maximum number of allowed evaluations is reached.'''
    samples_evaluated = state.n_init + state.it_counter * state.batch_size
    if samples_evaluated < n_evaluation:
        return False
    return True

def failed_GP(state):
    '''Function to evaluate if a GPR failed during the optimization.'''
    if state.failed_GP:
        print("GPR failed")
        return True
    return False

def min_radius(state, radius_min):
    '''Function to evaluate if MND radius is smaller than the minimum allowed radius'''
    if state.radius < radius_min:
        return True
    return False