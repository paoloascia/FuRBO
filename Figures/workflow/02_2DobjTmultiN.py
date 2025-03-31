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
from utilities import multivariate_circular

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
Y_model = get_fitted_model(X_train, Y_train, p.dimension, max_cholesky_size = float("inf"))

# Sample surrogate
Y_model.eval()
with torch.no_grad():
    Z = Y_model.posterior(grid_x)
    Z = Z.mean.view(resolution, resolution)

# Unnormalize xx
X = unnormalize(X, [p.lower_bounds[0], p.upper_bounds[0]])
Y = unnormalize(Y, [p.lower_bounds[1], p.upper_bounds[1]])
          
X, Y, Z = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()
            
# Create a contour plot
contour = ax.contourf(X, Y, Z, levels=10, cmap='viridis')  # Use plt.contourf for filled contours
        
# Add multinormal distribution sampling
lb = torch.zeros(p.dimension, **tkwargs)
ub = torch.ones(p.dimension, **tkwargs)

X_best = X_train[get_best_index_for_batch(1, Y_train, C_train)]
samples = multivariate_circular(X_best[0], 0.5, 100 * p.dimension, lb=lb, ub=ub, **tkwargs)

# Plot multonormal samples
ax.scatter(unnormalize(samples.cpu().numpy()[:,0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(samples.cpu().numpy()[:,1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color='w', marker = 'x')

# Add best sample
ax.scatter(unnormalize(X_best.cpu().numpy()[0][0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(X_best.cpu().numpy()[0][1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color = 'r')

# Add labels and title
# ax.set_xlabel('X-axis')
ax.set_xticks([])
# ax.set_ylabel('Y-axis')
ax.set_yticks([])
# ax.set_title("bbob-constrained_f035_i01_d02\n"
#              "Constraints GPR and samples")
        
# Add colorbar
# cbar = plt.colorbar(contour, ax=ax)

# Save figure
fig.savefig(os.path.join(cwd_save, '2DobjTmultiN' + '.png'))

# Close figure
plt.close(fig)

