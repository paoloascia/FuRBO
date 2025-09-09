# Full ode for SCBO
#
# March 2024
##########
# Imports
import cocoex  # experimentation module
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle as pkl
import random
import warnings
import time

from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

from SCBOSamplingStrategies import generate_batch
from SCBOStates import variant_one
from SCBOStopping import max_evaluations as stopping_criterion
from SCBORestart import tr_size as restart_criterion
from SCBOTrustUpdate import update_tr_length as update_tr
from utilities import get_initial_points

from objectives import speed_reducer
from constraints import speed_reducer_cons

##########
# Main code    

# Load random seeds
seeds = np.load('random_seeds.npy')

# Base directory for saving files
cwd_base = os.path.join(os.getcwd(), 'results')

# General log file
f = open(os.path.join(cwd_base, '00_GeneralLog.txt'), 'w')
print("\t Started", file=f)
    
# Create directory for problem
cwd_current = os.path.join(cwd_base, 'speed_reducer')
    
# Check if directory already exists
if any([True if dir_ == 'speed_reducer' else False for dir_ in os.listdir(cwd_base)]):
    # Check if problem was already solved
    if any([True if dir_ == 'complete' else False for dir_ in os.listdir(os.path.join(cwd_base, 'speed_reducer'))]):
        raise TypeError('Already evaluated!')
        
else:
    os.mkdir(cwd_current)
    
# Log file
print(f"{time.strftime('%x - %X')}: Start", file=f)
    
# Start time evaluation
tic = time.time()
    
# Perform 10 repetitions
for i, seed in enumerate(seeds):
    # Optimization start
        
    # Check if seed is already evaluated
    if 'speed_reducer' + '_it_' + str(i) + '.torch' in os.listdir(os.path.join(cwd_base, 'speed_reducer')):
        continue
    
    # seed iterative for restarts
    seed_j = 0
    
    # Setting random seed
    SCBO_seed = int(seed[seed_j])
    torch.manual_seed(SCBO_seed)
    
    ###
    # Start PyTorch and warnings
    warnings.filterwarnings("ignore")
        
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    
    # Define objective
    obj = speed_reducer()
    
    # Define constraint
    cons = speed_reducer_cons()
            
    ###
    # Initialize FuRBO
    history = []
    batch_size = int(3 * obj.dimension)
    n_init = int(3 * obj.dimension)
    n_iteration = int(30 * obj.dimension)
    tr_number = 1
    iteration = 0
    n_samples = 0
    N_CANDIDATES = 2000
        
    # SCBO state initialization
    SCBO_status = variant_one(obj = obj,                        # Objective function
                              cons = cons,                      # Constraints function
                              batch_size = batch_size,          # Batch size of each iteration
                              n_init = n_init,                  # Number of initial points to evaluate
                              n_iteration = n_iteration,        # Number of iterations
                              tr_number = tr_number,            # number of Trust regions
                              seed = SCBO_seed,                 # Seed for Sobol sampling
                              history = history,                # saved history to make all plots
                              iteration = iteration,            # Numnber of iteration if restart
                              samples_evaluated = n_samples,    # Numnber of evaluations if restart
                              **tkwargs)
    
    while not SCBO_status.finish_trigger:
        
        if SCBO_status.restart_trigger:
            seed_j += 1
            SCBO_seed = int(seed[seed_j])
            torch.manual_seed(SCBO_seed)
    
        # SCBO state initialization
        SCBO_status = variant_one(obj = obj,                        # Objective function
                                  cons = cons,                      # Constraints function
                                  batch_size = batch_size,          # Batch size of each iteration
                                  n_init = n_init,                  # Number of initial points to evaluate
                                  n_iteration = n_iteration,        # Number of iterations
                                  tr_number = tr_number,            # number of Trust regions
                                  seed = SCBO_seed,                 # Seed for Sobol sampling
                                  history = history,                # saved history to make all plots
                                  iteration = iteration,            # Numnber of iteration if restart
                                  samples_evaluated = n_samples,    # Numnber of evaluations if restart
                                  **tkwargs)
    
        # generate intial batch of X
        X_next = get_initial_points(SCBO_status, **tkwargs)
        
        ###
        # Optimization loop
        while not SCBO_status.restart_trigger and not SCBO_status.finish_trigger:
               
            # Evaluate batch
            Y_next = []
            C_next = []
            for x in X_next:
                # Evaluate batch on obj ...
                Y_next.append(-1 * SCBO_status.obj(x))
                # ... and constraints
                C_next.append(SCBO_status.cons(x))
                
            Y_next = torch.tensor(Y_next).unsqueeze(-1)
            C_next = torch.tensor(C_next)
            
            # Update SCBO status with newly evaluated batch
            SCBO_status.update(X_next, Y_next, C_next, **tkwargs)   
            
            # Printing
            # Print best value so far and violation
            if (SCBO_status.best_C <= 0).all():
                print(f"{SCBO_status.it_counter-1}) Best value: {SCBO_status.best_Y[0]:.2e},"
                      f" Smallest TR volume: {torch.min(SCBO_status.tr_vol[0]):.2e}", file=f)
                print(f"{SCBO_status.it_counter-1}) Best value: {SCBO_status.best_Y[0]:.2e},"
                      f" Smallest TR volume: {torch.min(SCBO_status.tr_vol[0]):.2e}")
                print(f"Y: {Y_next}")
            else:
                violation = SCBO_status.best_C.clamp(min=0).sum()
                print(f"{SCBO_status.it_counter-1}) No feasible point yet! Smallest total violation: "
                      f"{violation:.2e}, Smallest TR volume: {torch.min(SCBO_status.tr_vol):.2e}", file=f)
                print(f"{SCBO_status.it_counter-1}) No feasible point yet! Smallest total violation: "
                      f"{violation:.2e}, Smallest TR volume: {torch.min(SCBO_status.tr_vol):.2e}")
                print(f"Y: {Y_next}")
                    
            # Update Trust regions
            SCBO_status = update_tr(SCBO_status, **tkwargs)
                
            # Generate new batch
            X_next = generate_batch(SCBO_status, N_CANDIDATES, **tkwargs)
                
            # Update stopping criterion
            SCBO_status.finish_trigger = stopping_criterion(SCBO_status)
            SCBO_status.restart_trigger = restart_criterion(SCBO_status)
                
        # extracting history
        history = SCBO_status.history
        iteration = SCBO_status.it_counter
        n_samples = SCBO_status.samples_evaluated
                
    filename_torch = 'speed_reducer' + '_it_' + str(i) + '.torch'
    torch.save(SCBO_status.history, os.path.join(cwd_current, filename_torch))
        
    t = (time.time() - tic) % 60
    print(f"{time.strftime('%x - %X')}: Finish", file=f)
    print(f"Computation time: {t:.2f} seconds", file=f)

    del SCBO_status
    
# Post-processing
# Read all repetitions
states = []
for torch_file in os.listdir(cwd_current):
    if 'torch' in torch_file:
        states.append(torch.load(os.path.join(cwd_current, torch_file), map_location=torch.device('cpu')))
      
# Extract best at each iteration
Y_batch = []
C_batch = []
for state in states:
    Y_batch.append(np.concatenate([-1 * event['batch']['Y'].cpu().numpy() for event in state]).reshape(-1)[:n_iteration])
    C_batch.append(np.concatenate([np.max(event['batch']['C'].cpu().numpy(), axis=1) for event in state])[:n_iteration])
        
# Create a monotonic curve
Y_best = np.array(Y_batch)
Y_best = Y_best.reshape(Y_best.shape[0], Y_best.shape[1])
C_best = np.array(C_batch)
C_best = C_best.reshape(C_best.shape[0], C_best.shape[1])

Y_f = np.copy(Y_best)
C_f = np.copy(C_best)
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
np.save(os.path.join(cwd_current,'02_Y_best.npy'), Y_best)
np.save(os.path.join(cwd_current,'02_C_best.npy'), C_best)
    
# Flag as complete
open(os.path.join(cwd_current, 'complete'), 'w').close()
