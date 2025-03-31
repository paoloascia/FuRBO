# Full ode for FuRBO
#
# March 2024
##########
# Imports
import cocoex  # experimentation module
import math
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
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
Y_model = get_fitted_model(X_train, Y_train, p.dimension, max_cholesky_size = float("inf"))
C_model = get_fitted_model(X_train, C_train, p.dimension, max_cholesky_size = float("inf"))

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
torch.manual_seed(1)
samples = multivariate_circular(X_best[0], 0.5, 100 * p.dimension, lb=lb, ub=ub, **tkwargs).to(torch.float32)

# Plot multonormal samples
ax.scatter(unnormalize(samples.cpu().numpy()[:,0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(samples.cpu().numpy()[:,1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color='w', marker = 'x')

# Identify top 10% of the samples
# Evaluate samples on the models of the objective -> yy Tensor
Y_model.eval()
with torch.no_grad():
    posterior = Y_model.posterior(samples)
    samples_yy = posterior.mean.squeeze()
        
# Evaluate samples on the models of the constraints -> yy Tensor
C_model.eval()
with torch.no_grad():
    posterior = C_model.posterior(samples)
    samples_cc = posterior.mean
        
# Combine the constraints values
# Normalize
samples_cc /= torch.abs(samples_cc).max(dim=0).values
samples_cc = torch.max(samples_cc, dim=1).values
        
# Take the best 10% of the drawn samples to define the trust region
n_samples = 100 * p.dimension
n_samples_tr = int(n_samples * 0.2)

# Order the samples for feasibility and for best objective
if torch.any(samples_cc < 0):
    
    feasible_samples_id = torch.where(samples_cc <= 0)[0]
    infeasible_samples_id = torch.where(samples_cc > 0)[0]
    
    feasible_cc = samples_yy[feasible_samples_id]
    infeasible_cc = samples_cc[infeasible_samples_id]
    
    feasible_sorted, feasible_sorted_id = torch.sort(feasible_cc)
    infeasible_sorted, infeasible_sorted_id = torch.sort(infeasible_cc)
    
    original_feasible_sorted_indices = feasible_samples_id[feasible_sorted_id]
    original_infeasible_sorted_indices = infeasible_samples_id[infeasible_sorted_id]
    
    top_indices = torch.cat((original_feasible_sorted_indices, original_infeasible_sorted_indices))[:n_samples_tr]
    
else:
    if n_samples_tr > len(samples_cc):
        n_samples_tr = len(samples_cc)
        
    if n_samples_tr < 4:
        n_samples_tr = 4
                
    top_values, top_indices = torch.topk(samples_cc, n_samples_tr, largest=False)
   
# Saving best samples
best_samples = samples[top_indices]

# Plotting
ax.scatter(unnormalize(best_samples.cpu().numpy()[:,0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(best_samples.cpu().numpy()[:,1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color = 'orange', marker = 'x')

# Add best sample
ax.scatter(unnormalize(X_best.cpu().numpy()[0][0], [p.lower_bounds[0], p.upper_bounds[0]]),
           unnormalize(X_best.cpu().numpy()[0][1], [p.lower_bounds[0], p.upper_bounds[0]]),
           color = 'r')

# Add trust region
lower_bound = unnormalize(torch.min(samples[top_indices], dim=0).values.cpu().numpy(), [p.lower_bounds[0], p.upper_bounds[0]])
upper_bound = unnormalize(torch.max(samples[top_indices], dim=0).values.cpu().numpy(), [p.lower_bounds[0], p.upper_bounds[0]])
ax.plot([lower_bound[0], upper_bound[0]], [lower_bound[1], lower_bound[1]], color = 'r')
ax.plot([lower_bound[0], upper_bound[0]], [upper_bound[1], upper_bound[1]], color = 'r')
ax.plot([lower_bound[0], lower_bound[0]], [lower_bound[1], upper_bound[1]], color = 'r')
ax.plot([upper_bound[0], upper_bound[0]], [lower_bound[1], upper_bound[1]], color = 'r')
        
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
fig.savefig(os.path.join(cwd_save, '2DobjTmultiNtr' + '.png'))

# Close figure
plt.close(fig)

