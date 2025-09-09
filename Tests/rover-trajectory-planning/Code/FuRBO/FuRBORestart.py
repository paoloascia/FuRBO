# FuRBO Stopping criterion
# 
# March 2024
##########
# Imports

def failed_GP(state):
    if state.failed_GP:
        # print('Restart')
        return True
    return False

def min_max_percentage(state):
    
    if state.percentage < state.percentage_min or state.percentage > state.percentage_max:
        # print('Restart')
        return True
    
    elif state.failed_GP:
        # print('Restart')
        return True
    
    else:
        return False
    
def min_percentage(state):
    
    if state.percentage < state.percentage_min:
        # print('Restart')
        return True
    
    elif state.failed_GP:
        # print('Restart')
        return True
    
    else:
        return False
    
def min_radius(state):
    
    if state.radius < state.radius_min:
        # print('Restart')
        return True
    
    elif state.failed_GP:
        # print('Restart')
        return True
    
    else:
        return False
    
def failed_GP_budget_constrained(state):
    if state.failed_GP:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    return False

def min_max_percentage_budget_constrained(state):
    
    if state.percentage < state.percentage_min or state.percentage > state.percentage_max:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    elif state.failed_GP:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True    
    else:
        return False
    
def min_percentage_budget_constrained(state):
    
    if state.percentage < state.percentage_min:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True    
    elif state.failed_GP:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True    
    else:
        return False
    
def min_radius_budget_constrained(state):
    
    if state.radius < state.radius_min:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    elif state.failed_GP:
        # print('Restart')
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True    
    else:
        return False

def tr_size_budget_constrained(state):
    if state.length < state.length_min:
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    if state.failed_GP:
        if state.n_init > (state.it_counter - state.samples_evaluated):
            return True
    return False