# FuRBO state updates for different loops
# 
# March 2024
##########
# Imports
import torch
###
# Custom imports
from utilities import get_best_index_for_batch


# possible FuRBO updates
def update_state(state,
                 X_next,
                 Y_next,
                 C_next,
                 constraint_model,
                 sobol,
                 **kwargs):
    
    # Merge together current best choices with new points
    if state.it_counter == 0:
        X_candidate = X_next
        Y_candidate = Y_next
        C_candidate = C_next
    else:
        X_candidate = torch.cat((state.best_batch_X, X_next), dim=0)
        Y_candidate = torch.cat((state.best_batch_Y, Y_next), dim=0)
        C_candidate = torch.cat((state.best_batch_C, C_next), dim=0)
    
    # print(Y_candidate)  # Debugging
    
    # Normalize C_candidate
    C_candidate_norm = C_candidate / torch.abs(C_candidate).max(dim=0).values

    if torch.any(torch.all(C_candidate_norm <= 0, dim=1)):
        # At least one new candidate is feasible: count how many feasible points are available
        mask = torch.prod(C_candidate_norm <= 0, dim=1)
        num_feasible = torch.sum(mask)
        
        if num_feasible <= state.tr_number:
            # If there the number of feasible points is equal or lower than the number of the trust regions, include all and the least violating
            top_values, top_indices = torch.topk(C_candidate_norm.min(dim=1).values, state.tr_number, largest=False)
        
        else:
            # If there are more feasible point, include the optimum in the best batch, even if not the most feasible point
            top_values, top_indices = torch.topk(C_candidate_norm.min(dim=1).values, state.tr_number, largest=False)
            if not torch.max(Y_candidate[top_indices]) == torch.max(Y_candidate):
                top_indices[-1] = get_best_index_for_batch(Y=Y_candidate, C=C_candidate)
    
    else:
        # No new candidate is feasible: find the tr_number least violating points
        top_values, top_indices = torch.topk(C_candidate_norm.min(dim=1).values, state.tr_number, largest=False)
        
    # Update best candidates
    state.best_batch_X = X_candidate[top_indices]
    state.best_batch_Y = Y_candidate[top_indices]
    state.best_batch_C = C_candidate[top_indices]
        
    # Update best value
        # Find the best value among the candidates
    best_ind = get_best_index_for_batch(Y=Y_candidate, C=C_candidate)
    x_next, y_next, c_next = X_candidate[best_ind], Y_candidate[best_ind], C_candidate[best_ind]
        # Update
    state.best_value = y_next.item()
    state.best_constraint_values = c_next
    state.best_parameters = x_next
    
    # Updare stopping criteria
    state.it_counter += 1
    if state.it_counter == state.n_iteration:
        state.finish_trigger = True

    return state
