# FuRBO trust region updates for different loops
# 
# March 2024
##########
# Imports
import torch

###
# Custom imports
from fcn.utilities import multivariate_circular

# Function to update the trust region with a smaller radius circle
def multinormal_radius(state,                # FuRBO state
                       percentage = 0.1,     # Percentage to define trust region (default 10%)
                       **tkwargs):
    '''Function to sample Multinormal Distribution of GPRs and define trust region'''
    # Update the trust regions based on the feasible region
    n_samples = 1000 * state.dim
    lb = torch.zeros(state.dim, **tkwargs)
    ub = torch.ones(state.dim, **tkwargs)
    
    # Update radius dimension
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.radius = min(2.0 * state.radius, 1.0)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.radius /= 2.0
        state.failure_counter = 0
    
    for ind, x_candidate in enumerate(state.best_X):
        # Generate the samples to evaluathe the feasible area on
        radius = state.radius
        samples = multivariate_circular(x_candidate, radius, n_samples, lb=lb, ub=ub, **tkwargs)
    
        # Evaluate samples on the models of the objective -> yy Tensor
        state.Y_model.eval()
        with torch.no_grad():
            posterior = state.Y_model.posterior(samples)
            samples_yy = posterior.mean.squeeze()
        
        # Evaluate samples on the models of the constraints -> yy Tensor
        state.C_model.eval()
        with torch.no_grad():
            posterior = state.C_model.posterior(samples)
            samples_cc = posterior.mean
        
        # Combine the constraints values
            # Normalize
        samples_cc /= torch.abs(samples_cc).max(dim=0).values
        samples_cc = torch.max(samples_cc, dim=1).values
        
        # Take the best X% of the drawn samples to define the trust region
        n_samples_tr = int(n_samples * percentage)
        
        # Order the samples for feasibility and for best objective
        if torch.any(samples_cc < 0):
            
            feasible_samples_id = torch.where(samples_cc <= 0)[0]
            infeasible_samples_id = torch.where(samples_cc > 0)[0]
            
            feasible_cc = -1 * samples_yy[feasible_samples_id]
            infeasible_cc = samples_cc[infeasible_samples_id]
            
            feasible_sorted, feasible_sorted_id = torch.sort(feasible_cc)
            infeasible_sorted, infeasible_sorted_id = torch.sort(infeasible_cc)
            
            original_feasible_sorted_indices = feasible_samples_id[feasible_sorted_id]
            original_infeasible_sorted_indices = infeasible_samples_id[infeasible_sorted_id]
            
            top_indices = torch.cat((original_feasible_sorted_indices, original_infeasible_sorted_indices))[:n_samples_tr]
        
        else:
            
            if n_samples_tr > len(samples_cc):
                n_samples_tr = len(samples_cc)
                
            if n_samples_tr < 4:
                n_samples_tr = 4
                
            top_values, top_indices = torch.topk(samples_cc, n_samples_tr, largest=False)
        
        # Set the box around the selected samples
        state.tr_lb[ind] = torch.min(samples[top_indices], dim=0).values
        state.tr_ub[ind] = torch.max(samples[top_indices], dim=0).values
        
        # Update volume of trust region
        state.tr_vol[ind] = torch.prod(state.tr_ub[ind] - state.tr_lb[ind])
        
    # return updated status with new trust regions
    return state
