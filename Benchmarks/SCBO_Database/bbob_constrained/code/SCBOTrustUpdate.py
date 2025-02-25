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
from utilities import get_fitted_model
# from plotting import constraints_2d_samples as plot_samples

def update_tr_length(state,              # FuRBO state
                     **tkwargs
                     ):
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0
        
    # print(state.length)
        
    state.tr_lb = torch.reshape(torch.clamp(state.best_X - state.length / 2.0, 0.0, 1.0), (state.tr_number, state.dim))
    state.tr_ub = torch.reshape(torch.clamp(state.best_X + state.length / 2.0, 0.0, 1.0), (state.tr_number, state.dim))
    return state


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