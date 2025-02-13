# Full ode for FuRBO
#
# March 2024
##########
# Imports
import cocoex
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import os

# Batch size to choose
q = 'q1'

# Quantiles to plot
lower_quantiles = 0.1
centre_quantiles = 0.5
upper_quantiles = 0.9

##########
# Initiate directories
base_cwd = os.getcwd()
cwd_SCBO = os.path.join(base_cwd, 'SCBO_Database', 'bbob_constrained', 'batch_size_' + q)
cwd_FuRBO = os.path.join(base_cwd, 'FuRBO_Database', 'bbob_constrained', 'batch_size_' + q)
cwd_save = os.path.join(base_cwd, 'SCBO_Database', 'bbob_constrained', 'plots', 'batch_size_' + q)

##########
# Initiate COCO
suite_name = "bbob-constrained"
suite = cocoex.Suite(suite_name, "", "")

##########
# Load solutions
fmin = np.load('bbob-constrained-targets.npy', allow_pickle = True)
fmin = fmin.item(0)

###
# Iterate through the COCO instances and plot the curves
for p in suite:
    
    q=1 *p.dimension
    
    # Skip if not part of the study
    if not ('i01' in p.id):
        continue
    if not ('d02' in p.id or
            'd10' in p.id or
            'd40' in p.id):
        continue
    
    # Initiate plot
    print(p.id)
    fig = plt.figure()
    ax = plt.gca()
    patchList = []
    
    # Add solution line
    ax.hlines(0, 0, 30*q+3*p.dimension, color='black', ls='dashed', lw=2)
    
    # Check results for SCBO and extract the maximum feasible sample between FuRBO and SCBO
    max_Y = 0
    if any([True if p.id == cwd_ else False for cwd_ in os.listdir(cwd_SCBO)]):
        # Load information from SCBO
        filename = os.path.join(cwd_SCBO, p.id, '01_Y_mono.npy')
        Y_f_monotonic_S = np.load(filename) - fmin[p.id]
            
        if np.amax(Y_f_monotonic_S) > max_Y:
            max_Y = np.amax(Y_f_monotonic_S)
            
    # Check results for SCBO and extract the maximum feasible sample between FuRBO and SCBO
    if any([True if p.id == cwd_ else False for cwd_ in os.listdir(cwd_FuRBO)]):
        # Load information from FuRBO
        filename = os.path.join(cwd_FuRBO, p.id, '01_Y_mono.npy')
        Y_f_monotonic_F = np.load(filename) - fmin[p.id]
            
        if np.amax(Y_f_monotonic_F) > max_Y:
            max_Y = np.amax(Y_f_monotonic_F)
                
    if any([True if p.id == cwd_ else False for cwd_ in os.listdir(cwd_SCBO)]):
            
        # Substitute infeasible with maximum feasible evaluated
        Y_f_monotonic_S[np.where(Y_f_monotonic_S == np.amax(Y_f_monotonic_S))] = max_Y
            
        # Process data for SCBO
        mean = np.quantile(Y_f_monotonic_S, centre_quantiles, axis = 0)
        lb = np.quantile(Y_f_monotonic_S, lower_quantiles, axis = 0)
        ub = np.quantile(Y_f_monotonic_S, upper_quantiles, axis = 0)
        x = np.linspace(1, len(mean), len(mean))
        
        # Plot convergence of SCBO
        if lb[0] == lb[-1]:
            patchList.append(patches.Patch(color='darkgreen', label='SCBO-Failed'))
        else:
            ax.plot(x, mean, color = 'darkgreen', lw=2)
            ax.fill_between(x, lb, ub, alpha = 0.2, color='darkgreen', lw=2)
            patchList.append(patches.Patch(color='darkgreen', label='SCBO'))
    else:
        patchList.append(patches.Patch(color='darkgreen', label='SCBO-Not available'))
        
    # Check results for FuRBO
    if any([True if p.id == cwd_ else False for cwd_ in os.listdir(cwd_FuRBO)]):
            
        # Substitute infeasible with maximum feasible evaluated
        Y_f_monotonic_F[np.where(Y_f_monotonic_F == np.amax(Y_f_monotonic_F))] = max_Y
        
        # Process data for SCBO
        mean = np.quantile(Y_f_monotonic_F, centre_quantiles, axis = 0)
        lb = np.quantile(Y_f_monotonic_F, lower_quantiles, axis = 0)
        ub = np.quantile(Y_f_monotonic_F, upper_quantiles, axis = 0)
        x = np.linspace(1, len(mean), len(mean))
        
        # Plot convergence of SCBO
        if lb[0] == lb[-1]:
            patchList.append(patches.Patch(color='darkorange', label='FuRBO-Failed'))
        else:
            ax.plot(x, mean, color = 'darkorange', lw=2)
            ax.fill_between(x, lb, ub, alpha = 0.2, color='darkorange', lw=2)
            patchList.append(patches.Patch(color='darkorange', label='FuRBO'))
    else:
        patchList.append(patches.Patch(color='darkorange', label='FuRBO-Not available'))   
    
    # Add plot description
    ax.set_title(p.id) 
    ax.legend(handles=patchList, loc='upper right')
    ax.set_xlabel('Evaluations')
    ax.set_ylabel('f(obj)-f(sol)')
    
    # Save figure
    fig.savefig(os.path.join(cwd_save, p.id + '.png'))
    
    # Close figure
    plt.close(fig)
    
    