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
        y = (np.load(file_name) - fmin[crv])[:,:300]
        y_max = np.amax(y)
        y[y==np.amax(y)] = y_max
        
    crv = crv[:-5] + '2' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = (np.load(file_name) - fmin[crv])[:,:300]
        if y_max < np.amax(tmp):
            y_max = np.amax(tmp)
            y[y==np.amax(y)] = y_max
        else:
            tmp[tmp==np.amax(tmp)] = y_max
        y = np.vstack([y, tmp])
        
    crv = crv[:-5] + '3' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = (np.load(file_name) - fmin[crv])[:,:300]
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
    cwd = os.path.join(os.getcwd(), 'Experiments', 'FuRBO')
    y_F, y_f_max = load_data_(crv, cwd)
    
    # Load data SCBO
    cwd = os.path.join(os.getcwd(), 'Experiments', 'SCBO')
    y_S, y_s_max = load_data_(crv, cwd)
    
    # Load data CEI
    cwd = os.path.join(os.getcwd(), 'Experiments', 'CEI')
    y_C, y_c_max = load_data_(crv, cwd)
    
    # Load data COBYLA
    cwd = os.path.join(os.getcwd(), 'Experiments', 'COBYLA')
    y_CB, y_cb_max = load_data_(crv, cwd)
    
    # Load data CMAES
    cwd = os.path.join(os.getcwd(), 'Experiments', 'CMAES')
    y_CM, y_cm_max = load_data_(crv, cwd)
    
    # Load data Random Sampling
    cwd = os.path.join(os.getcwd(), 'Experiments', 'RandomSampling')
    y_RS, y_rs_max = load_data_(crv, cwd)
        
    # Find worst feasible
    y_max = np.amax([y_f_max, y_s_max, y_c_max, y_cb_max, y_cm_max, y_rs_max])
        
    # Plot FuRBO data
    y_F[y_F==np.amax(y_F)] = y_max
    top_F, middle_F = plot_curve(ax, y_F, 'darkorange')
    
    # Plot SCBO data
    y_S[y_S==np.amax(y_S)] = y_max
    top_S, middle_S = plot_curve(ax, y_S, 'darkgreen')
    
    # Plot CEI data
    y_C[y_C==np.amax(y_C)] = y_max
    top_C, middle_C = plot_curve(ax, y_C, 'darkred')

    # Plot COBYLA data
    y_CB[y_CB==np.amax(y_CB)] = y_max
    top_CB, middle_CB = plot_curve(ax, y_CB, 'darkblue')

    # Plot CMAES data
    y_CM[y_CM==np.amax(y_CM)] = y_max
    top_CM, middle_CM = plot_curve(ax, y_CM, 'cyan')

    # Plot Random Sampling data
    y_RS[y_RS==np.amax(y_RS)] = y_max
    top_RS, middle_RS = plot_curve(ax, y_RS, 'magenta')
    
    top = np.amax([top_F, top_S, top_C, top_CB, top_CM, top_RS])
    middle = np.amax([middle_F, middle_S, middle_C, middle_CB, middle_CM, middle_RS])
        
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

fig = plt.figure(figsize=(16,5.5))
gs = gridspec.GridSpec(nrows=2,
                       ncols=3,
                       wspace=.35,
                       hspace=.25, 
                       height_ratios=[1, 8],
                       figure = fig)

patchList = []

fmin = np.load('bbob-constrained-targets.npy', allow_pickle = True)
fmin = fmin.item(0)

##########
# Iterate through the curves and plot them
# Line 1
crv = "bbob-constrained_f005_i01_d10"
ax = plt.subplot(gs[1, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300'])
ax.set_ylabel("Loss")
ax.set_title("$f_{sphere}$")

crv = "bbob-constrained_f035_i01_d10"
ax = plt.subplot(gs[1, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300'])
ax.set_xlabel('Evaluations')
ax.set_title("$f_{bent\_cigar}$")

crv = "bbob-constrained_f053_i01_d10"
ax = plt.subplot(gs[1, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300'])
ax.set_title("$f_{rat\_rot}$")

# Add legend
ax = plt.subplot(gs[0, :])
patchList = [patches.Patch(color='darkorange', label='FuRBO'),
             patches.Patch(color='darkgreen', label='SCBO'),
             patches.Patch(color='darkred', label='C-EI'),
             patches.Patch(color='darkblue', label='COBYLA'),
             patches.Patch(color='cyan', label='CMA-ES'),
             patches.Patch(color='magenta', label='Random sampling')]
ax.legend(ncols = 6,
          handles=patchList, loc='lower center')
ax.spines[['right', 'bottom', 'left', 'top']].set_visible(False) 
ax.set_xticks([])
ax.set_yticks([])   
    
# Add plot description
# ax.set_title(str(dir_)) 
# ax.legend(handles=patchList, loc='upper right')
# ax.set_xlabel('Evaluations')
# ax.set_ylabel('Loss')
# ax.set_ylim(bottom = 0)
    
# Save figure
fig.savefig(os.path.join(os.getcwd(), 'FuRBOcompAlg' + '.png'), dpi=600)
    
# Close figure
plt.close(fig)

