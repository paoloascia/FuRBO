# Full ode for FuRBO
#
# March 2024
##########
# Imports
import cocoex  # experimentation module
import math
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from botorch.utils.transforms import unnormalize

from torch import Tensor
from torch.quasirandom import SobolEngine

##########
# Custom imports
from utilities import get_fitted_model
from utilities import get_best_index_for_batch

##########
# Setting general MatPlotLib parameters 
cwd_save = os.path.join(os.getcwd())
matplotlib.use('Agg')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

#########
# Setting up PyTorch
device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

##########
# Opening optimization data

##########
# Selecting bbob function
# Define COCO input
suite_name = "bbob-constrained"
suite = cocoex.Suite(suite_name, "", "")
# Select p.id = bbob-constrained_f035_i01_d02
p = suite[510]

##########
# Plot of cons{tilde} + DoE + Best
# Initiate plot
fig = plt.figure(figsize = (6,6), 
                 dpi = 600)
ax = plt.gca()
# ax.set_aspect("equal")
    
# Plot contour plot of the function
resolution = 50
    
# Create a meshgrid from x and y
X, Y = torch.meshgrid(torch.linspace(0, 1, resolution), torch.linspace(0, 1, resolution), indexing="ij")
grid_x = torch.stack([X.flatten(), Y.flatten()], dim=-1)

# Train surrogate
seed = 24
sobolSampler = SobolEngine(dimension=p.dimension, scramble=True, seed=seed)
X_train = sobolSampler.draw(n=3 * p.dimension)
Y_train = Tensor([p(unnormalize(x_, [p.lower_bounds[0], p.upper_bounds[0]])) for x_ in X_train]).unsqueeze(-1)
C_train = Tensor([torch.amax(Tensor(p.constraint(unnormalize(x_, [p.lower_bounds[0], p.upper_bounds[0]])))) for x_ in X_train]).unsqueeze(-1)
C_model = get_fitted_model(X_train, C_train, p.dimension, max_cholesky_size = float("inf"))

# Sample surrogate
C_model.eval()
with torch.no_grad():
    C = C_model.posterior(grid_x)
    C = C.mean.view(resolution, resolution)

# Unnormalize xx
X = unnormalize(X, [p.lower_bounds[0], p.upper_bounds[0]])
Y = unnormalize(Y, [p.lower_bounds[1], p.upper_bounds[1]])
          
X, Y, C = X.cpu().numpy(), Y.cpu().numpy(), C.cpu().numpy()
            
# Create a contour plot
contour = ax.contourf(X, Y, C, levels=10, cmap='viridis')  # Use plt.contourf for filled contours
        
# Add samples
ax.scatter(unnormalize(X_train.cpu().numpy()[:,0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(X_train.cpu().numpy()[:,1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color = 'k')

# Add best sample
X_best = X_train[get_best_index_for_batch(1, Y_train, C_train)]
ax.scatter(unnormalize(X_best.cpu().numpy()[0][0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(X_best.cpu().numpy()[0][1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color = 'r')

# Add labels and title
# ax.set_xlabel('X-axis')
ax.set_xticks([])
# ax.set_ylabel('Y-axis')
ax.set_yticks([])
# ax.set_title("bbob-constrained_f035_i01_d02\n"
             # "Constraints GPR and samples")
        
# Add colorbar
# cbar = plt.colorbar(contour, ax=ax)

# Save figure
fig.savefig(os.path.join(cwd_save, '2DconTdoe' + '.png'), dpi = 600)

# Close figure
plt.close(fig)

