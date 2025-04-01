# Script to evaluate BBOB on COBYLA

##########
# Imports
import matplotlib
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

def load_data_(crv, cwd):
    
    crv = crv[:-5] + '1' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        y = np.load(file_name) - fmin[crv]
        y_max = np.amax(y)
        y[y==np.amax(y)] = y_max
        
    crv = crv[:-5] + '2' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_max < np.amax(tmp):
            y_max = np.amax(tmp)
            y[y==np.amax(y)] = y_max
        else:
            tmp[tmp==np.amax(tmp)] = y_max
        y = np.vstack([y, tmp])
        
    crv = crv[:-5] + '3' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_max < np.amax(tmp):
            y_max = np.amax(tmp)
            y[y==np.amax(y)] = y_max
        else:
            tmp[tmp==np.amax(tmp)] = y_max
        y = np.vstack([y, tmp])
        
    return y, y_max
    
def plot_curve(ax, y, color):
    
    # Elaborate data
    mean = np.mean(y, axis = 0)
    lb = mean - np.std(y, axis = 0)/np.sqrt(y.shape[0])
    ub = mean + np.std(y, axis = 0)/np.sqrt(y.shape[0])
    x = np.linspace(1, len(mean), len(mean))
    
    # Plot
    ax.plot(x, mean, color = color, lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color=color, lw=2)
    
    # axis information
    top = np.amax(ub) + 0.1 * np.amax(ub)
    middle = np.amax(mean)/2
    
    return top, middle

def plot_convergence(crv, ax):
    
    # Load data FuRBO
    cwd = os.path.join(os.getcwd(), 'Experiments', 'q1')
    y_1, y_1_max = load_data_(crv, cwd)
    
    # Load data SCBO
    cwd = os.path.join(os.getcwd(), 'Experiments', 'q1d')
    y_1d, y_1d_max = load_data_(crv, cwd)
    
    # Load data CEI
    cwd = os.path.join(os.getcwd(), 'Experiments', 'q2d')
    y_2d, y_2d_max = load_data_(crv, cwd)
    
    # Load data COBYLA
    cwd = os.path.join(os.getcwd(), 'Experiments', 'q3d')
    y_3d, y_3d_max = load_data_(crv, cwd)
    
    # Load data CMAES
    cwd = os.path.join(os.getcwd(), 'Experiments', 'q4d')
    y_4d, y_4d_max = load_data_(crv, cwd)
    
    # Load data Random Sampling
    cwd = os.path.join(os.getcwd(), 'Experiments', 'q5d')
    y_5d, y_5d_max = load_data_(crv, cwd)
        
    # Find worst feasible
    y_max = np.amax([y_1_max, y_1d_max, y_2d_max, y_3d_max, y_4d_max, y_5d_max])
        
    # Plot FuRBO data
    y_1[y_1==np.amax(y_1)] = y_max
    top_1, middle_1 = plot_curve(ax, y_1, 'darkorange')
    
    # Plot SCBO data
    y_1d[y_1d==np.amax(y_1d)] = y_max
    top_1d, middle_1d = plot_curve(ax, y_1d, 'darkgreen')
    
    # Plot CEI data
    y_2d[y_2d==np.amax(y_2d)] = y_max
    top_2d, middle_2d = plot_curve(ax, y_2d, 'darkred')

    # Plot COBYLA data
    y_3d[y_3d==np.amax(y_3d)] = y_max
    top_3d, middle_3d = plot_curve(ax, y_3d, 'darkblue')

    # Plot CMAES data
    y_4d[y_4d==np.amax(y_4d)] = y_max
    top_4d, middle_4d = plot_curve(ax, y_4d, 'cyan')

    # Plot Random Sampling data
    y_5d[y_5d==np.amax(y_5d)] = y_max
    top_5d, middle_5d = plot_curve(ax, y_5d, 'magenta')
    
    top = np.amax([top_1, top_1d, top_2d, top_3d, top_4d, top_5d])
    middle = np.amax([middle_1, middle_1d, middle_2d, middle_3d, middle_4d, middle_5d])
        
    ax.set_ylim(bottom = 0,
                top = top)
    
    ax.set_yticks([top,
                   middle,
                   0])
    
    ax.set_yticklabels([f"{(top):.1E}",
                        f"{(middle):.1E}",
                        "0.0"], rotation=45)
        
    return

##########
# Initialize plot
matplotlib.use('Agg')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['font.size'] = 18

fig = plt.figure(figsize=(9,8))
ax = plt.gca()

patchList = []

fmin = np.load('bbob-constrained-targets.npy', allow_pickle = True)
fmin = fmin.item(0)

##########
# Iterate through the curves and plot them
# Line 1
crv = "bbob-constrained_f035_i01_d10"
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300'])
ax.set_ylabel("Loss")
ax.set_title("$f_{bent\_cigar}$ \n"
             "Constraints: 24")

# Add legend
patchList = [patches.Patch(color='darkorange', label='Batch = 1'),
             patches.Patch(color='darkgreen', label='Batch = 1d'),
             patches.Patch(color='darkred', label='Batch = 2d'),
             patches.Patch(color='darkblue', label='Batch = 3d'),
             patches.Patch(color='cyan', label='Batch = 4d'),
             patches.Patch(color='magenta', label='Batch = 5d')]
ax.legend(handles=patchList, loc='upper right')   
    
# Add plot description
# ax.set_title(str(dir_)) 
# ax.legend(handles=patchList, loc='upper right')
# ax.set_xlabel('Evaluations')
# ax.set_ylabel('Loss')
# ax.set_ylim(bottom = 0)
    
# Save figure
fig.savefig(os.path.join(os.getcwd(), 'FuRBObatch' + '.png'), bbox_inches='tight', dpi=600)
    
# Close figure
plt.close(fig)

