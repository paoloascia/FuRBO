# Full ode for FuRBO
#
# March 2024
##########
# Imports
import cocoex  # experimentation module
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from botorch.utils.transforms import unnormalize

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
# Selecting bbob function
# Define COCO input
suite_name = "bbob-constrained"
suite = cocoex.Suite(suite_name, "", "")
# Select p.id = bbob-constrained_f035_i01_d02
p = suite[510]

##########
# Base plot for bbob (f_obj + cons)
# Initiate plot
fig = plt.figure(figsize = (6,6), 
                 dpi = 600)
ax = plt.gca()
    
# Plot contour plot of the function
resolution = 200
    
# Create a meshgrid from x and y
x = torch.linspace(0, 1, resolution, **tkwargs)
y = torch.linspace(0, 1, resolution, **tkwargs)
X, Y = torch.meshgrid(x, y)
    
# Reshape samples for function evaliation
X = X.reshape(-1)
Y = Y.reshape(-1)
xx = torch.transpose(torch.stack([X, Y]), 0, 1)

        
# Evaluate the objective function at each combination of x and y
Z = torch.tensor([p(unnormalize(x_, [p.lower_bounds, p.upper_bounds])) for x_ in xx], **tkwargs).unsqueeze(-1)
        
# Evaluate constraint functions at each combination of x and y
C = torch.tensor([p.constraint(unnormalize(x_, [p.lower_bounds, p.upper_bounds])) for x_ in xx], **tkwargs).max(dim=1).values.unsqueeze(-1)
    
# Unnormalize xx
X = unnormalize(X, [p.lower_bounds[0], p.upper_bounds[0]])
Y = unnormalize(Y, [p.lower_bounds[1], p.upper_bounds[1]])
        
# Reshape for plotting
X, Y, Z, C = X.reshape((resolution, resolution)), Y.reshape((resolution, resolution)), Z.reshape((resolution, resolution)), C.reshape((resolution, resolution))
  
X, Y, Z, C = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), C.cpu().numpy()
            
# Plot constraints
ax.contourf(X, Y, C, levels=[0.0001, np.inf], colors = 'blue', alpha = 0.2)
    
# Create a contour plot
contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')  # Use plt.contourf for filled contours
        
# Add labels and title
# ax.set_xlabel('X-axis')
ax.set_xticks([])
# ax.set_ylabel('Y-axis')
ax.set_yticks([])
# ax.set_title("bbob-constrained_f035_i01_d02\n"
#              "Objective and constraints")
        
# Add colorbar
# cbar = plt.colorbar(contour, ax=ax)

# Save figure
fig.savefig(os.path.join(cwd_save, '2DobjCon' + '.png'))

# Close figure
plt.close(fig)

