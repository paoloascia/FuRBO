# Script to evaluate BBOB on COBYLA

##########
# Imports
import matplotlib
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

def plot_convergence(crv, ax):
    
    cwd = os.path.join(os.getcwd(), 'Experiments', 'FuRBO')
    
    # Load data
    crv = crv[:-5] + '1' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        y_F = np.load(file_name) - fmin[crv]
        y_f_max = np.amax(y_F)
        y_F[y_F==np.amax(y_F)] = y_f_max
        
    crv = crv[:-5] + '2' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_f_max < np.amax(tmp):
            y_f_max = np.amax(tmp)
            y_F[y_F==np.amax(y_F)] = y_f_max
        else:
            tmp[tmp==np.amax(tmp)] = y_f_max
        y_F = np.vstack([y_F, tmp])
        
    crv = crv[:-5] + '3' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_f_max < np.amax(tmp):
            y_f_max = np.amax(tmp)
            y_F[y_F==np.amax(y_F)] = y_f_max
        else:
            tmp[tmp==np.amax(tmp)] = y_f_max
        y_F = np.vstack([y_F, tmp])
        
        
    cwd = os.path.join(os.getcwd(), 'Experiments', 'SCBO')
    
    # Load data
    crv = crv[:-5] + '1' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        y_S = np.load(file_name) - fmin[crv]
        y_s_max = np.amax(y_S)
        y_S[y_S==np.amax(y_S)] = y_s_max
        
    crv = crv[:-5] + '2' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_s_max < np.amax(tmp):
            y_s_max = np.amax(tmp)
            y_S[y_S==np.amax(y_S)] = y_s_max
        else:
            tmp[tmp==np.amax(tmp)] = y_s_max
        y_S = np.vstack([y_S, tmp])
        
    crv = crv[:-5] + '3' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_s_max < np.amax(tmp):
            y_s_max = np.amax(tmp)
            y_S[y_S==np.amax(y_S)] = y_s_max
        else:
            tmp[tmp==np.amax(tmp)] = y_s_max
        y_S = np.vstack([y_S, tmp])
    
    # Find worst feasible
    y_max = np.amax([y_f_max, y_s_max])
    
    # Exchange worst feasible
    y_F[y_F==np.amax(y_F)] = y_max
    y_S[y_S==np.amax(y_S)] = y_max
        
    # Elaborate FuRBO data
    mean = np.mean(y_F, axis = 0)
    lb = mean - np.std(y_F, axis = 0)/np.sqrt(y_F.shape[0])
    ub = mean + np.std(y_F, axis = 0)/np.sqrt(y_F.shape[0])
    x = np.linspace(1, len(mean), len(mean))
    
    # Plot convergence of FuRBO
    ax.plot(x, mean, color = 'darkorange', lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color='darkorange', lw=2)
    top = np.amax(ub) + 0.1 * np.amax(ub)
    middle = np.amax(mean)/2
        
    # Elaborate SCBO data
    mean = np.mean(y_S, axis = 0)
    lb = mean - np.std(y_S, axis = 0)/np.sqrt(y_S.shape[0])
    ub = mean + np.std(y_S, axis = 0)/np.sqrt(y_S.shape[0])
    x = np.linspace(1, len(mean), len(mean))
    
    # Plot convergence of SCBO
    ax.plot(x, mean, color = 'darkgreen', lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color='darkgreen', lw=2)
    if np.amax(ub) + 0.1 * np.amax(ub) > top:
        top = np.amax(ub) + 0.1 * np.amax(ub)
    if np.amax(mean)/2 > middle:
        middle = np.amax(mean)/2
        
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

fig = plt.figure(figsize=(16,16))
gs = gridspec.GridSpec(nrows=4,
                       ncols=3,
                       wspace=.35,
                       hspace=.35, 
                       height_ratios=[1, 8, 8, 8],
                       figure = fig)

patchList = []

fmin = np.load('bbob-constrained-targets.npy', allow_pickle = True)
fmin = fmin.item(0)

##########
# Iterate through the curves and plot them
# Line 1
crv = "bbob-constrained_f003_i01_d02"
ax = plt.subplot(gs[1, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel("$f_{sphere}$")
ax.set_title("Dimension 2")

crv = "bbob-constrained_f003_i01_d10"
ax = plt.subplot(gs[1, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels([])
ax.set_title("Dimension 10")

crv = "bbob-constrained_f003_i01_d40"
ax = plt.subplot(gs[1, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
ax.set_xticklabels([])
ax.set_title("Dimension 40")

# Line 2
crv = "bbob-constrained_f033_i01_d02"
ax = plt.subplot(gs[2, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel("Loss\n"
              "$f_{bent\_cigar}$")

crv = "bbob-constrained_f033_i01_d10"
ax = plt.subplot(gs[2, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels([])

crv = "bbob-constrained_f033_i01_d40"
ax = plt.subplot(gs[2, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
ax.set_xticklabels([])

# Line 3
crv = "bbob-constrained_f051_i01_d02"
ax = plt.subplot(gs[3, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation = 45)
# ax.set_xlabel('Evaluations')
ax.set_ylabel("$f_{rast\_rot}$")

crv = "bbob-constrained_f051_i01_d10"
ax = plt.subplot(gs[3, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300'], rotation = 45)
ax.set_xlabel('Evaluations')

crv = "bbob-constrained_f051_i01_d40"
ax = plt.subplot(gs[3, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
ax.set_xticklabels(['0', '200', '400', '600', '800', '1000', '1200'], rotation = 45)
# ax.set_xlabel('Evaluations')

# Add legend
ax = plt.subplot(gs[0, :])
patchList = [patches.Patch(color='darkorange', label='FuRBO'),
             patches.Patch(color='darkgreen', label='SCBO')]
ax.legend(ncols = 2,
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
fig.savefig(os.path.join(os.getcwd(), 'FuRBOdimStudy' + '.png'))
    
# Close figure
plt.close(fig)

