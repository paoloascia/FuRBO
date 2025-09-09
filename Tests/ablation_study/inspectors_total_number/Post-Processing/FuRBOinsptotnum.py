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
    
    # Load data with 10d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '10d')
    y_1, y_1_max = load_data_(crv, cwd)
    
    # Load data with 50d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '50d')
    y_2, y_2_max = load_data_(crv, cwd)
    
    # Load data with 100d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '100d')
    y_3, y_3_max = load_data_(crv, cwd)
    
    # Load data with 500d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '500d')
    y_4, y_4_max = load_data_(crv, cwd)
    
    # Load data with 1000d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '1000d')
    y_5, y_5_max = load_data_(crv, cwd)
    
    # Load data with 5000d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '5000d')
    y_6, y_6_max = load_data_(crv, cwd)
    
    # Load data with 10000d inspectors
    cwd = os.path.join(os.getcwd(), 'Experiments', '10000d')
    y_7, y_7_max = load_data_(crv, cwd)
        
    # Find worst feasible
    y_max = np.amax([y_1_max, y_2_max, y_3_max, y_4_max, y_5_max, y_6_max, y_7_max])
        
    # Plot 10d data
    y_1[y_1==np.amax(y_1)] = y_max
    top_1, middle_1 = plot_curve(ax, y_1, 'darkorange')

    # Plot 50d data
    y_2[y_2==np.amax(y_2)] = y_max
    top_2, middle_2 = plot_curve(ax, y_2, 'darkgreen')

    # Plot 100d data
    y_3[y_3==np.amax(y_3)] = y_max
    top_3, middle_3 = plot_curve(ax, y_3, 'darkred')
    
    # Plot 500d data
    y_4[y_4==np.amax(y_4)] = y_max
    top_4, middle_4 = plot_curve(ax, y_4, 'darkblue')
    
    # Plot 1000d data
    y_5[y_5==np.amax(y_5)] = y_max
    top_5, middle_5 = plot_curve(ax, y_5, 'cyan')
    
    # Plot 5000d data
    y_6[y_6==np.amax(y_6)] = y_max
    top_6, middle_6 = plot_curve(ax, y_6, 'magenta')
    
    # Plot 10000d data
    y_7[y_7==np.amax(y_7)] = y_max
    top_7, middle_7 = plot_curve(ax, y_7, 'black')
    
    top = np.amax([top_1, top_2, top_3, top_4, top_5, top_6, top_7])
    middle = np.amax([middle_1, middle_2, middle_3, middle_4, middle_5, middle_6, middle_7])
        
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
patchList = [patches.Patch(color='darkorange', label='10d'),
             patches.Patch(color='darkgreen', label='50d'),
             patches.Patch(color='darkred', label='100d'),
             patches.Patch(color='darkblue', label='500d'),
             patches.Patch(color='cyan', label='1000d'),
             patches.Patch(color='magenta', label='5000d'),
             patches.Patch(color='black', label='10000d')]
ax.legend(handles=patchList, loc='upper right')   
    
# Add plot description
# ax.set_title(str(dir_)) 
# ax.legend(handles=patchList, loc='upper right')
# ax.set_xlabel('Evaluations')
# ax.set_ylabel('Loss')
# ax.set_ylim(bottom = 0)
    
# Save figure
fig.savefig(os.path.join(os.getcwd(), 'FuRBOinsptotnum' + '.png'), bbox_inches='tight', dpi=600)
    
# Close figure
plt.close(fig)

