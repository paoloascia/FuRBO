# Script to evaluate BBOB on COBYLA

##########
# Imports
import cocoex
import numpy as np
import os
import time

##########
# Helper function for tracking
x_track = []
y_track = []
c_track = []

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
        
        # Defining sampler
        sampler = np.random.default_rng(seed = int(seed[seed_j]))
        
        for _ in range(30 * p.dimension):
            # Generate random sample and evaluate objective and constraints
            x_track.append(sampler.uniform(low = p.lower_bounds, high = p.upper_bounds, size = p.dimension))
            y_track.append(p(x_track[-1]))
            c_track.append(p.constraint(x_track[-1]))
    
    
    # Elaborate results
    X_batch = [x_track[i:i + 30 * p.dimension] for i in range(0, len(x_track), 30 * p.dimension)]
    Y_batch = [y_track[i:i + 30 * p.dimension] for i in range(0, len(y_track), 30 * p.dimension)]
    C_batch = [np.amax(c_track[i:i + 30 * p.dimension], axis=1) for i in range(0, len(c_track), 30 * p.dimension)]
       
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
    
    
    
        


