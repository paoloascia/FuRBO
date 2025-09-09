# FuRBO Stopping criterion
# 
# March 2024
##########
# Imports

def max_iterations(state):
    if state.it_counter < state.n_iteration:
        return False
    return True

def max_evaluations(state):
    if state.samples_evaluated < state.n_iteration:
        return False
    return True