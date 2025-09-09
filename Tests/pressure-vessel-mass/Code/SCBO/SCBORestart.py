# FuRBO Stopping criterion
# 
# March 2024
##########
# Imports

def tr_size(state):
    if state.length < state.length_min:
        return True
    if state.failed_GP:
        return True
    return False

def tr_size_budget_constrained(state):
    if state.length < state.length_min:
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    if state.failed_GP:
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    return False