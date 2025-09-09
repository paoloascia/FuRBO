# FuRBO trust region updates for different loops
# 
# March 2024
##########
# Imports
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.generation.sampling import MaxPosteriorSampling

import torch

###
# Custom imports
from utilities import multivariate_circular_two
from utilities import multivariate_circular
from utilities import get_fitted_model
# from plotting import constraints_2d_samples as plot_samples

def fixed_percentage(state,              # FuRBO state
                     percentage,         # Percentage to take (value 0 - 1)
                     **tkwargs
                     ):
    
    # Update the trust regions based on the feasible region
    n_samples = 1000 * state.dim
    lb = torch.zeros(state.dim, **tkwargs)
    ub = torch.ones(state.dim, **tkwargs)
    
    for ind, x_candidate in enumerate(state.best_X):
        # Generate the samples to evaluathe the feasible area on
        samples = multivariate_circular_two(x_candidate, n_samples, lb=lb, ub=ub, **tkwargs)
        
        # plot_samples(samples, state, no_save=False, **tkwargs)
    
        # Evaluate samples on the models of the objective -> yy Tensor
        state.Y_model.eval()
        with torch.no_grad():
            posterior = state.Y_model.posterior(samples)
            samples_yy = posterior.mean.squeeze()
            # samples_variance = posterior.variance     # For debugging
        
        # Evaluate samples on the models of the constraints -> yy Tensor
        state.C_model.eval()
        with torch.no_grad():
            posterior = state.C_model.posterior(samples)
            samples_cc = posterior.mean
            # samples_variance = posterior.variance     # For debugging
        
        # Combine the constraints values
            # Normalize
        samples_cc /= torch.abs(samples_cc).max(dim=0).values
        samples_cc = torch.max(samples_cc, dim=1).values
        
        # Take the best 10% of the drawn samples to define the trust region
        n_samples_tr = int(n_samples * percentage)
        
        # Order the samples for feasibility and for best objective
        if torch.any(samples_cc < 0):
            
            feasible_samples_id = torch.where(samples_cc <= 0)[0]
            infeasible_samples_id = torch.where(samples_cc > 0)[0]
            
            feasible_cc = samples_yy[feasible_samples_id]
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

def changing_percentage(state,
                        **tkwargs):
    
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.percentage = min(4.0 * state.percentage, 1.0)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.percentage /= 4.0
        state.failure_counter = 0
        
    return fixed_percentage(state,
                            state.percentage,
                            **tkwargs)

def generate_batch_one(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol,
    **tkwargs):
    
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Initialize tensor with samples to evaluate
    X_next = torch.ones((state.batch_size*state.tr_number, state.dim), **tkwargs)
    
    # Iterate over the several trust regions
    for i in range(state.tr_number):
        tr_lb = state.tr_lb[i]
        tr_ub = state.tr_ub[i]

        # Thompson Sampling w/ Constraints (like SCBO)
        dim = X.shape[-1]
        pert = sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), **tkwargs)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = state.best_batch_X[i].expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]
        
        # If a feasible point has been identified:
        if torch.any(torch.max(C, dim=1).values <= 0):
            # Sample on the candidate points using Constrained Max Posterior Sampling
            constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
                model=model, constraint_model=constraint_model, replacement=False
                )
            with torch.no_grad():
                X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constrained_thompson_sampling(X_cand, num_samples=batch_size)
        
        else:
            # Sample to minimize violation
            
            # First combine the constraints surrogates
            constraint_model.eval()
            with torch.no_grad():
                posterior = constraint_model.posterior(X_cand)
                C_cand = posterior.mean
                
            # Normalize
            C_cand /= torch.abs(C_cand).max(dim=0).values
            
            # Combine into one tensor
            C_cand = -1 * C_cand.max(dim=1).values
            
            # Reshape to (-1, 1)
            C_cand = C_cand.view(-1, 1)
            
            # Train one model on the combination
            constraint_model_united = get_fitted_model(X_cand, C_cand)
            
            # Sample the candidate points
            constraint_sampling = MaxPosteriorSampling(
                model=constraint_model_united, replacement=False)
            with torch.no_grad():
                X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constraint_sampling(X_cand, num_samples=batch_size)

    return X_next

# Function to update the trust region with a smaller radius circle
def multinormal_radius(state,              # FuRBO state
                       **tkwargs
                       ):
    
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
        
        # plot_samples(samples, state, no_save=False, **tkwargs)
    
        # Evaluate samples on the models of the objective -> yy Tensor
        state.Y_model.eval()
        with torch.no_grad():
            posterior = state.Y_model.posterior(samples)
            samples_yy = posterior.mean.squeeze()
            # samples_variance = posterior.variance     # For debugging
        
        # Evaluate samples on the models of the constraints -> yy Tensor
        state.C_model.eval()
        with torch.no_grad():
            posterior = state.C_model.posterior(samples)
            samples_cc = posterior.mean
            # samples_variance = posterior.variance     # For debugging
        
        # Combine the constraints values
            # Normalize
        samples_cc /= torch.abs(samples_cc).max(dim=0).values
        samples_cc = torch.max(samples_cc, dim=1).values
        
        # Take the best 10% of the drawn samples to define the trust region
        percentage = 0.1
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