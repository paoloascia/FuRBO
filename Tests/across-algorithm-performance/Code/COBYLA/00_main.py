# Script to evaluate BBOB on COBYLA

##########
# Imports
import cocoex
import numpy as np
import os
import time
import torch

from scipy.optimize import fmin_cobyla as fmin
from scipy.optimize import minimize
from scipy.optimize import Bounds

from torch.quasirandom import SobolEngine
from torch import Tensor
from botorch.utils.transforms import unnormalize

##########
# Helper function for tracking

def get_best_index_for_batch(n_tr, Y: Tensor, C: Tensor):
    """Return the index for the best point. One for each trust region."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return torch.topk(score.reshape(-1), k=n_tr).indices
    return torch.topk(C.clamp(min=0).sum(dim=-1), k=n_tr, largest=False).indices # Return smallest violation

def get_constraints(): 
    """Return a properly formatted constraint for minimize()"""
    def constraint(x): 
        """COBYLA constraints: Must return >= 0 for feasible solutions"""
        x_track.append(x)
        y_track.append(p(x))
        c_track.append(p.constraint(x))
        return -p.constraint(x) 
    
    return [{'type': 'ineq', 'fun': constraint}]  # Wrap in a dictionary inside a list 

def c_eval(x):
    
        x_track.append(x)
        y_track.append(p(x))
        c_track.append(p.constraint(x))
        return -p.constraint(x)

##########
# Main code    

# Define COCO input
suite_name = "bbob-constrained"
suite = cocoex.Suite(suite_name, "", "")

# Load random seeds
seeds = np.load('random_seeds.npy')

# Base directory for saving files
cwd_base = os.path.join(os.getcwd(), 'results')

# General log file
f_gen = open(os.path.join(cwd_base, '00_GeneralLog.txt'), 'w')

for p in suite:
    
    if not ('i01' in p.id or
            'i02' in p.id or
            'i03' in p.id):
        continue
    if not ('d02' in p.id or
            'd10' in p.id or
            'd40' in p.id ):
        continue
    # if not ('f053' in p.id):
        # print('Skipped')
        # continue
    
    x_track = []
    y_track = []
    c_track = []
  
    print(p.id)
    print(f"{p.index}) {p.id}:", file=f_gen)
    print("\t Started", file=f_gen)
    
    # Create directory for problem
    cwd_current = os.path.join(cwd_base, p.id)
    if not p.id in os.listdir(cwd_base):
        os.mkdir(cwd_current)
        
    # Log file
    f = open(os.path.join(cwd_current, '00_Log.txt'), 'w')
    print(f"{p.id}", file=f)
    print(f"{time.strftime('%x - %X')}: Start", file=f)
    
    # Start time evaluation
    tic = time.time()
    
    # Perform 30 repetitions
    for i, seed in enumerate(seeds):
        # Optimization start        
        # seed iterative for restarts
        seed_j = 0
    
        # Setting random seed
        np.random.seed(int(seed[seed_j]))
        
        # Defining initial guess for COBYLA
        sobol = SobolEngine(dimension=p.dimension, scramble=True, seed=int(seed[seed_j]))
        x0 = unnormalize(sobol.draw(n=int(3 * p.dimension)).cpu().numpy(), [p.lower_bounds, p.upper_bounds])
        Y0 = Tensor([p(x_) for x_ in x0]).unsqueeze(-1)
        C0 = Tensor([np.amin(c_eval(x_)) for x_ in x0]).unsqueeze(-1)
        x0 = x0[get_best_index_for_batch(1, Y0, C0)]
    
        res = minimize(p, 
                       x0, 
                       constraints = get_constraints(), 
                       method = "COBYLA", 
                       bounds = Bounds(p.lower_bounds, p.upper_bounds),
                       options = {
                                   "maxiter": 30 * p.dimension,
                                   "rhobeg": 2,
                                   "tol": 1e-30})
    
    
    # Elaborate results
    X_batch = [x_track[i:i + 30 * p.dimension + 3 * p.dimension + 1] for i in range(0, len(x_track), 30 * p.dimension + 3 * p.dimension + 1)]
    Y_batch = [y_track[i:i + 30 * p.dimension + 3 * p.dimension + 1] for i in range(0, len(y_track), 30 * p.dimension + 3 * p.dimension + 1)]
    C_batch = [np.amax(c_track[i:i + 30 * p.dimension + 3 * p.dimension + 1], axis=1) for i in range(0, len(c_track), 30 * p.dimension + 3 * p.dimension + 1)]
   
# Evaluate objective and constraints for all evaluated samples
# Y_batch = []
# C_batch = []
    
# for X_ in X_batch:
#     yy = []
#     cc = []
#     for xx in X_:
#         yy.append(p(xx))
#         cc.append(np.amax(p.constraint(xx)))
#     Y_batch.append(yy[1:])
#     C_batch.append(cc[1:])
        
    Y_batch = np.array(Y_batch)
    C_batch = np.array(C_batch)
    
    # Retrieve monotonic curves to save
    Y_f = np.copy(Y_batch)
    C_f = np.copy(C_batch)
    Y_f[np.where(C_f > 0)[0], np.where(C_f > 0)[1]] = 0
    Y_f[np.where(C_f > 0)[0], np.where(C_f > 0)[1]] = np.amax(Y_f)
    
    Y_f_monotonic = []
    for YY in Y_f:
        y_mono = []
        for yy in YY:
            if len(y_mono) == 0:
                y_mono = [yy]
            else:
                if yy < y_mono[-1]:
                    y_mono.append(yy)
                else:
                    y_mono.append(y_mono[-1])

        Y_f_monotonic.append(y_mono)
    
    Y_f_monotonic = np.array(Y_f_monotonic)
    
    # Save interesting curves
    np.save(os.path.join(cwd_current, '01_Y_mono.npy'), Y_f_monotonic)
    np.save(os.path.join(cwd_current,'02_Y_best.npy'), Y_batch)
    np.save(os.path.join(cwd_current,'02_C_best.npy'), C_batch)
    
    
    
        


