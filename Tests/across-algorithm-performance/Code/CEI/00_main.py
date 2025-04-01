##############################################################################
# Author: Paolo Ascia                                                        #
# Date: April 2023                                                           #
##############################################################################
# File: main.py                                                              #
###########                                                                  #
# Run the optimization                                                       #
##############################################################################

##############################################################################
# Imports
##########
import cocoex
import numpy as np
import os
import time
import torch
import warnings

from botorch.utils.transforms import unnormalize

##############################################################################
# Custom Imports
##########
from EIBORestart import failed_GP as restart_criterion
from EIBOSamplingStrategies import get_initial_points
from EIBOSamplingStrategies import generate_batch
from EIBOStates import variant_one
from EIBOStopping import max_evaluations as stopping_criterion

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
    
    if not ('i01' in p.id or 'i02' in p.id or 'i03' in p.id):
        continue
    if not 'd10' in p.id:
        continue
    if not ('f005' in p.id or 'f035' in p.id or 'f053' in p.id):
        # print('Skipped')
        continue
    
    print(p.id)
    print(f"{p.index}) {p.id}:", file=f_gen)
    print("\t Started", file=f_gen)
    
    # Create directory for problem
    cwd_current = os.path.join(cwd_base, p.id)
    
    # Check if directory already exists
    if any([True if dir_ == p.id else False for dir_ in os.listdir(cwd_base)]):
        # Check if problem was already solved
        if any([True if dir_ == 'complete' else False for dir_ in os.listdir(os.path.join(cwd_base, p.id))]):
            print("\t Completed - Previous evaluation", file=f_gen)
            continue
        
    else:
        os.mkdir(cwd_current)
    
    # Log file
    f = open(os.path.join(cwd_current, '00_Log.txt'), 'w')
    print(f"{p.id}", file=f)
    print(f"{time.strftime('%x - %X')}: Start", file=f)
    
    # Start time evaluation
    tic = time.time()
    
    for i, seed in enumerate(seeds):
        # Optimization start
        
        # Check if seed is already evaluated
        if p.id + '_it_' + str(i) + '.torch' in os.listdir(os.path.join(cwd_base, p.id)):
            continue
        
        # seed iterative for restarts
        print(f"Iteration {i}.")
        seed_j = 0
        
        # Setting random seed
        EIBO_seed = int(seed[seed_j])
        torch.manual_seed(EIBO_seed)
        
        ###
        # Start PyTorch and warnings
        warnings.filterwarnings("ignore")
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        dtype = torch.double
        tkwargs = {"device": device, "dtype": dtype}
            
        ###
        # Initialize FuRBO
        history = []
        batch_size = int(1) #int(3 * p.dimension)
        n_init = int(3 * p.dimension)
        n_iteration = int(30 * p.dimension)
        tr_number = 1
        iteration = 0
        n_samples = 0
        N_CANDIDATES = 2000
        
        EIBO_status = variant_one(obj = p,                      # Objective function
                                  cons = p.constraint,          # Constraints function
                                  batch_size = batch_size,      # Batch size of each iteration
                                  n_init = n_init,              # Number of initial points to evaluate
                                  n_iteration = n_iteration,    # Number of iterations
                                  seed = EIBO_seed,             # Seed for Sobol sampling
                                  history = history,            # saved history to make all plots
                                  iteration = iteration,        # Numnber of iteration if restart
                                  samples_evaluated = n_samples,   # Numnber of evaluations if restart
                                  **tkwargs)
    
        while not EIBO_status.finish_trigger:
        
            # Check for restart trigger
            if EIBO_status.restart_trigger:
                seed_j += 1
                EIBO_seed = int(seed[seed_j])
                torch.manual_seed(EIBO_seed)
        
            # EIBO state initialization
            EIBO_status = variant_one(obj = p,                      # Objective function
                                      cons = p.constraint,          # Constraints function
                                      batch_size = batch_size,      # Batch size of each iteration
                                      n_init = n_init,              # Number of initial points to evaluate
                                      n_iteration = n_iteration,    # Number of iterations
                                      seed = EIBO_seed,             # Seed for Sobol sampling
                                      history = history,            # saved history to make all plots
                                      iteration = iteration,        # Numnber of iteration if restart
                                      samples_evaluated = n_samples,   # Numnber of evaluations if restart
                                      **tkwargs)
    
            # Generate intial batch of X
            X_next = get_initial_points(EIBO_status, **tkwargs)
            # evaluate initial batch X on the objective function
            print("Initial DoE started")
            print("Initial DoE started", file = f)
        
            ###
            # Optimization loop
            while not EIBO_status.restart_trigger and not EIBO_status.finish_trigger:
                
                # Evaluate batch
                Y_next = []
                C_next = []
                for x in X_next:
                    # Evaluate batch on obj ...
                    Y_next.append(-1*EIBO_status.obj(unnormalize(x, [p.lower_bounds, p.upper_bounds])))
                    # ... and constraints
                    C_next.append(EIBO_status.cons(unnormalize(x, [p.lower_bounds, p.upper_bounds])))
                    
                Y_next = torch.tensor(Y_next).unsqueeze(-1)
                C_next = torch.tensor(C_next)

                # Update EIBO status with newly evaluated batch
                EIBO_status.update(X_next, Y_next, C_next, **tkwargs)   
        
                # Printing
                # Print best value so far and violation
                if (EIBO_status.best_C <= 0).all():
                    # print(f"{EIBO_status.it_counter-1}) Best value: {EIBO_status.best_Y[0]:.2e}", file=f)
                    print(f"{EIBO_status.it_counter-1}) Best value: {EIBO_status.best_Y[0]:.2e}")
                    print(f"{EIBO_status.it_counter-1}) Best value: {EIBO_status.best_Y[0]:.2e}", file = f)
                    # print(f"Y: {C_next}")
                else:
                    violation = EIBO_status.best_C.clamp(min=0).sum()
                    print(f"{EIBO_status.it_counter-1}) No feasible point yet! Smallest total violation: {violation:.2e}")
                    print(f"{EIBO_status.it_counter-1}) No feasible point yet! Smallest total violation: {violation:.2e}", file = f)
                    # print(f"Y: {C_next}")
                
                # Update stopping criterion
                EIBO_status.finish_trigger = stopping_criterion(EIBO_status)
                EIBO_status.restart_trigger = restart_criterion(EIBO_status)
            
                # Generate new batch of X
                X_next = generate_batch(EIBO_status, N_CANDIDATES, **tkwargs)
        
        
            # extracting history
            history = EIBO_status.history
            iteration = EIBO_status.it_counter
            n_samples = EIBO_status.samples_evaluated
        
    
        filename_torch = p.id + '_it_' + str(i) + '.torch'
        torch.save(EIBO_status.history, os.path.join(cwd_current, filename_torch))
    
        t = (time.time() - tic) % 60
        print(f"{time.strftime('%x - %X')}: Finish", file=f)
        print(f"Computation time: {t:.2f} seconds", file=f)
        print(f"\t Completed - time: {t:.2f} seconds", file=f_gen)
        
        del EIBO_status
    
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
        