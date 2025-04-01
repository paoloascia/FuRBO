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

fig = plt.figure(figsize=(16,16))
gs = gridspec.GridSpec(nrows=11,
                       ncols=6,
                       wspace=.35,
                       hspace=.5,
                       height_ratios=[1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1],
                       figure = fig)

patchList = []

fmin = np.load('bbob-constrained-targets.npy', allow_pickle = True)
fmin = fmin.item(0)

##########
# Iterate through the curves and plot them
# Line 1
crv = "bbob-constrained_f001_i01_d02"
ax = plt.subplot(gs[1, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{sphere}$')
ax.set_title("Constraints: 1\n"
             "Active: 1")

crv = "bbob-constrained_f002_i01_d02"
ax = plt.subplot(gs[1, 1])
plot_convergence(crv, ax)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_title("Constraints: 3\n"
             "Active: 2")

crv = "bbob-constrained_f003_i01_d02"
ax = plt.subplot(gs[1, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_title("Constraints: 9\n"
             "Active: 6")

crv = "bbob-constrained_f004_i01_d02"
ax = plt.subplot(gs[1, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_title("Constraints: 17\n"
             "Active: 11")

crv = "bbob-constrained_f005_i01_d02"
ax = plt.subplot(gs[1, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_title("Constraints: 24\n"
             "Active: 16")

crv = "bbob-constrained_f006_i01_d02"
ax = plt.subplot(gs[1, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_title("Constraints: 54\n"
             "Active: 36")

# Line 2
crv = "bbob-constrained_f007_i01_d02"
ax = plt.subplot(gs[2, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{ellipsoid}$')

crv = "bbob-constrained_f008_i01_d02"
ax = plt.subplot(gs[2, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f009_i01_d02"
ax = plt.subplot(gs[2, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f010_i01_d02"
ax = plt.subplot(gs[2, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f011_i01_d02"
ax = plt.subplot(gs[2, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f012_i01_d02"
ax = plt.subplot(gs[2, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 3
crv = "bbob-constrained_f013_i01_d02"
ax = plt.subplot(gs[3, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{linear}$')

crv = "bbob-constrained_f014_i01_d02"
ax = plt.subplot(gs[3, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f015_i01_d02"
ax = plt.subplot(gs[3, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f016_i01_d02"
ax = plt.subplot(gs[3, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f017_i01_d02"
ax = plt.subplot(gs[3, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f018_i01_d02"
ax = plt.subplot(gs[3, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 4
crv = "bbob-constrained_f019_i01_d02"
ax = plt.subplot(gs[4, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{elli\_rot}$')

crv = "bbob-constrained_f020_i01_d02"
ax = plt.subplot(gs[4, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f021_i01_d02"
ax = plt.subplot(gs[4, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f022_i01_d02"
ax = plt.subplot(gs[4, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f023_i01_d02"
ax = plt.subplot(gs[4, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f024_i01_d02"
ax = plt.subplot(gs[4, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 5
crv = "bbob-constrained_f025_i01_d02"
ax = plt.subplot(gs[5, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel("Loss \n"
              "$f_{discus}$")

crv = "bbob-constrained_f026_i01_d02"
ax = plt.subplot(gs[5, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f027_i01_d02"
ax = plt.subplot(gs[5, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f028_i01_d02"
ax = plt.subplot(gs[5, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f029_i01_d02"
ax = plt.subplot(gs[5, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f030_i01_d02"
ax = plt.subplot(gs[5, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 6
crv = "bbob-constrained_f031_i01_d02"
ax = plt.subplot(gs[6, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{bent\_cigar}$')

crv = "bbob-constrained_f032_i01_d02"
ax = plt.subplot(gs[6, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f033_i01_d02"
ax = plt.subplot(gs[6, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f034_i01_d02"
ax = plt.subplot(gs[6, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f035_i01_d02"
ax = plt.subplot(gs[6, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f036_i01_d02"
ax = plt.subplot(gs[6, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 7
crv = "bbob-constrained_f037_i01_d02"
ax = plt.subplot(gs[7, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{diff\_power}$')

crv = "bbob-constrained_f038_i01_d02"
ax = plt.subplot(gs[7, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f039_i01_d02"
ax = plt.subplot(gs[7, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f040_i01_d02"
ax = plt.subplot(gs[7, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f041_i01_d02"
ax = plt.subplot(gs[7, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f042_i01_d02"
ax = plt.subplot(gs[7, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 8
crv = "bbob-constrained_f043_i01_d02"
ax = plt.subplot(gs[8, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])
ax.set_ylabel('$f_{rastrigin}$')

crv = "bbob-constrained_f044_i01_d02"
ax = plt.subplot(gs[8, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f045_i01_d02"
ax = plt.subplot(gs[8, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f046_i01_d02"
ax = plt.subplot(gs[8, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f047_i01_d02"
ax = plt.subplot(gs[8, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

crv = "bbob-constrained_f048_i01_d02"
ax = plt.subplot(gs[8, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels([])

# Line 9
crv = "bbob-constrained_f049_i01_d02"
ax = plt.subplot(gs[9, 0])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation=45)
# ax.set_xlabel('Evaluations')
ax.set_ylabel('$f_{rast\_rot}$')

crv = "bbob-constrained_f050_i01_d02"
ax = plt.subplot(gs[9, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation=45)
# ax.set_xlabel('Evaluations')

crv = "bbob-constrained_f051_i01_d02"
ax = plt.subplot(gs[9, 2])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation=45)
# ax.set_xlabel('Eval', loc = 'right')

crv = "bbob-constrained_f052_i01_d02"
ax = plt.subplot(gs[9, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation=45)
# ax.set_xlabel('uations', loc = 'left')

crv = "bbob-constrained_f053_i01_d02"
ax = plt.subplot(gs[9, 4])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation=45)
# ax.set_xlabel('Evaluations')

crv = "bbob-constrained_f054_i01_d02"
ax = plt.subplot(gs[9, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'], rotation=45)
# ax.set_xlabel('Evaluations')

# Add legend
ax = plt.subplot(gs[0, :])
patchList = [patches.Patch(color='darkorange', label='FuRBO'),
             patches.Patch(color='darkgreen', label='SCBO')]
ax.legend(ncols = 2,
          handles=patchList, 
          loc='lower center')
ax.spines[['right', 'bottom', 'left', 'top']].set_visible(False) 
ax.set_xticks([])
ax.set_yticks([])   

ax = plt.subplot(gs[10, :])
ax.set_xlabel('Evaluations')
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
fig.savefig(os.path.join(os.getcwd(), 'FuRBOtwoDim' + '.png'), dpi=600)
    
# Close figure
plt.close(fig)

