# Script to evaluate BBOB on COBYLA

# Script to evaluate BBOB on COBYLA

##########
# Imports
import cocoex
import matplotlib
import numpy as np
import os
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

##########
# Initialize plot
matplotlib.use('Agg')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['font.size'] = 18

fig = plt.figure(figsize=(10,8))
ax = plt.gca()

# Base directory
cwd = os.path.join(os.getcwd())

# Load FuRBO results
file_name = os.path.join(cwd, 'Experiments', 'FuRBO', '01_Y_mono.npy')
y_F = np.load(file_name)

# Load SCBO results
file_name = os.path.join(cwd, 'Experiments', 'SCBO', '01_Y_mono.npy')
y_S = np.load(file_name)

# Find worst feasible
y_max = np.amax([y_S, y_F])

# Substitute
y_F[y_F == np.amax(y_F)] = y_max
y_S[y_S == np.amax(y_S)] = y_max

# Elaborate FuRBO data
mean = np.mean(y_F, axis = 0)
lb = mean - np.std(y_F, axis = 0)/np.sqrt(y_F.shape[0])
ub = mean + np.std(y_F, axis = 0)/np.sqrt(y_F.shape[0])
x = np.linspace(1, len(mean), len(mean))
# Plot convergence of FuRBO
if not lb[0] == lb[-1]:
    ax.plot(x, mean, color = 'darkorange', lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color='darkorange', lw=2)
top = np.amax(ub) + 0.1 * np.amax(ub)
# middle = np.amax(mean)/2
bot = np.amin(lb) - 0.1 * np.amin(lb)
        
# Elaborate SCBO data
mean = np.mean(y_S, axis = 0)
lb = mean - np.std(y_S, axis = 0)/np.sqrt(y_S.shape[0])
ub = mean + np.std(y_S, axis = 0)/np.sqrt(y_S.shape[0])
x = np.linspace(1, len(mean), len(mean))
# Plot convergence of SCBO
if not lb[0] == lb[-1]:
    ax.plot(x, mean, color = 'darkgreen', lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color='darkgreen', lw=2)
if np.amax(ub) + 0.1 * np.amax(ub) > top:
    top = np.amax(ub) + 0.1 * np.amax(ub)
# if np.amax(mean)/2 > middle:
#     middle = np.amax(mean)/2
if np.amin(lb) - 0.1 * np.amin(lb) < bot:
    bot = np.amin(lb) - 0.1 * np.amin(lb)

middle = (top + bot)/2

# Add description
ax.set_title('Speed Reducer (7D)')

ax.set_xticks([0, 70, 140, 210])
ax.set_xlabel('Evaluations')

ax.set_ylim(bottom = bot,
            top = top)
ax.set_yticks([top,
               middle,
               bot]) 
ax.set_yticklabels([f"{(top):.1E}",
                    f"{(middle):.1E}",
                    f"{(bot):.1E}"], rotation=45)
ax.set_ylabel('Volume')

# Add legend
patchList = [patches.Patch(color='darkorange', label='FuRBO'),
             patches.Patch(color='darkgreen', label='SCBO')]

ax.legend(handles=patchList, loc='upper right')
    
# Save figure
fig.savefig(os.path.join(os.getcwd(), 'speedReducer.png'), dpi=300)
    
# Close figure
plt.close(fig)
