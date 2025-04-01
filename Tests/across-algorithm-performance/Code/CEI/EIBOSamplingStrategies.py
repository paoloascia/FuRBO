# FuRBO sampling strategies
# 
# March 2024
##########
# Imports

from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective

import torch

def get_initial_points(status, **tkwargs):
    X_init = status.sobol.draw(n=status.n_init).to(**tkwargs)
    return X_init

def generate_batch(
    state,
    n_candidates,
    **tkwargs):
    
    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))
    
    # Merge objective and constraints models
    model = ModelListGP(state.Y_model, state.C_model)
    
    # Define acquisition function
    acq_func = ConstrainedExpectedImprovement(model=model,
                                                 best_f=state.best_Y[0],
                                                 objective_index=0,
                                                 constraints={1: [-1e30, 0],
                                                              2: [-1e30, 0],
                                                              3: [-1e30, 0],
                                                              4: [-1e30, 0],
                                                              5: [-1e30, 0],
                                                              6: [-1e30, 0],
                                                              7: [-1e30, 0],
                                                              8: [-1e30, 0],
                                                              9: [-1e30, 0],
                                                              10:[-1e30, 0],
                                                              11:[-1e30, 0],
                                                              12:[-1e30, 0],
                                                              13:[-1e30, 0],
                                                              14:[-1e30, 0],
                                                              15:[-1e30, 0],
                                                              16:[-1e30, 0],
                                                              17:[-1e30, 0],
                                                              18:[-1e30, 0],
                                                              19:[-1e30, 0],
                                                              20:[-1e30, 0],
                                                              21:[-1e30, 0],
                                                              22:[-1e30, 0],
                                                              23:[-1e30, 0],
                                                              24:[-1e30, 0]},
                                                  maximize = True)  #
    
    # Maximize CEI
    bounds = torch.tensor([[0.0] * state.dim, [1.0] * state.dim], device=tkwargs['device'], dtype=tkwargs['dtype'])
    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=bounds,
                                  q=1,
                                  num_restarts = 10,
                                  raw_samples = 256,
                                  sequential=True)
    # observe new values
    X_next = candidates.detach()

    return X_next

def objective(Z, X):
    return Z[..., 0]

def con_1(Z):
    return Z[..., 1]

def con_2(Z):
    return Z[..., 2]

def con_3(Z):
    return Z[..., 3]

def con_4(Z):
    return Z[..., 4]

def con_5(Z):
    return Z[..., 5]

def con_6(Z):
    return Z[..., 6]

def con_7(Z):
    return Z[..., 7]

def con_8(Z):
    return Z[..., 8]

def con_9(Z):
    return Z[..., 9]

def con_10(Z):
    return Z[..., 10]

def con_11(Z):
    return Z[..., 11]

def con_12(Z):
    return Z[..., 12]

def con_13(Z):
    return Z[..., 13]

def con_14(Z):
    return Z[..., 14]

def con_15(Z):
    return Z[..., 15]

def con_16(Z):
    return Z[..., 16]

def con_17(Z):
    return Z[..., 17]

def con_18(Z):
    return Z[..., 18]

def con_19(Z):
    return Z[..., 19]

def con_20(Z):
    return Z[..., 20]

def con_21(Z):
    return Z[..., 21]

def con_22(Z):
    return Z[..., 22]

def con_23(Z):
    return Z[..., 23]

def con_24(Z):
    return Z[..., 24]

def generate_batch_q(
    state,
    n_candidates,
    **tkwargs):
    
    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))
    
    # Merge objective and constraints models
    model = ModelListGP(state.Y_model, state.C_model)
    
    # Define acquisition function
    acq_func = qExpectedImprovement(model = model,
                                    best_f = state.best_Y,
                                    objective = GenericMCObjective(objective=objective),
                                    constraints = [con_1, con_2, con_3,
                                                   con_4, con_5, con_6,
                                                   con_7, con_8, con_9,
                                                   con_10, con_11, con_12,
                                                   con_13, con_14, con_15,
                                                   con_16, con_17, con_18,
                                                   con_19, con_20, con_21,
                                                   con_22, con_23, con_24])  #
    
    # Maximize CEI
    bounds = torch.tensor([[0.0] * state.dim, [1.0] * state.dim], device=tkwargs['device'], dtype=tkwargs['dtype']).view(2, -1)
    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=bounds,
                                  q=state.batch_size,
                                  num_restarts = 10,
                                  raw_samples = 256,
                                  sequential=True)
    # observe new values
    X_next = candidates.detach()

    return X_next

