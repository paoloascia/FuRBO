# FuRBO sampling strategies
# 
##########
# Imports
import torch

from botorch.generation.sampling import ConstrainedMaxPosteriorSampling

# Utility functions
def get_initial_points_sobol(state,
                             **tkwargs):
    '''Function to generate the initial experimental design'''
    X_init = state.sobol.draw(n=state.n_init).to(**tkwargs)
    return X_init

def generate_batch_thompson_sampling(state,
                                     n_candidates,
                                     **tkwargs):
    '''Function to find net candidate optimum'''
    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))

    # Initialize tensor with samples to evaluate
    X_next = torch.ones((state.batch_size*state.tr_number, state.dim), **tkwargs)
    
    # Iterate over the several trust regions
    for i in range(state.tr_number):
        tr_lb = state.tr_lb[i]
        tr_ub = state.tr_ub[i]

        # Thompson Sampling w/ Constraints (like SCBO)
        pert = state.sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / state.dim, 1.0)
        mask = torch.rand(n_candidates, state.dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, state.dim - 1, size=(len(ind),), device=tkwargs['device'])] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = state.best_X[i].expand(n_candidates, state.dim).clone()
        X_cand[mask] = pert[mask]
        
        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=state.Y_model, constraint_model=state.C_model, replacement=False
            )
        with torch.no_grad():
            X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constrained_thompson_sampling(X_cand, num_samples=state.batch_size)
        
    return X_next

