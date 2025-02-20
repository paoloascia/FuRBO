# Main FuRBO optimization loop without restarts
#
# Author: Paolo Ascia
##########
# Imports
import time
import torch
import warnings

from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

###
# Custom imports
from fcn.samplingStrategies import get_initial_points_sobol as get_initial_points
from fcn.samplingStrategies import generate_batch_thompson_sampling as generate_batch
from fcn.states import Furbo_state_restart
from fcn.stoppingNrestartCriterion import failed_GP as GP_restart_criterion
from fcn.stoppingNrestartCriterion import max_iterations as stopping_criterion
from fcn.stoppingNrestartCriterion import min_radius as restart_criterion
from fcn.trustRegionUpdate import multinormal_radius as update_tr

##########
# Objective class
class ack():
    
    def __init__(self, dim, negate, **tkwargs):
        
        self.fun = Ackley(dim = dim, negate = negate).to(**tkwargs)
        self.fun.bounds[0, :].fill_(-5)
        self.fun.bounds[1, :].fill_(10)
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        
    def eval_(self, x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return self.fun(unnormalize(x, [self.lb, self.ub]))
    
##########
# Constraints class
class sum_():
    # enforcing that sum(x) <= threshold
    def __init__(self, threshold, lb, ub):
        
        self.lb = lb
        self.ub = ub
        self.threshold = threshold
        return 
    
    def c(self, x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return x.sum() - self.threshold
    
    def eval_(self, x):
        return self.c(unnormalize(x, [self.lb, self.ub]))
###
class norm_():
    # enforcing that ||x||_2 <= threshold
    def __init__(self, threshold, lb, ub):
        
        self.lb = lb
        self.ub = ub
        self.threshold = threshold
        return 
    
    def c(self, x):
        return torch.norm(x, p=2) - self.threshold
    
    def eval_(self, x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return self.c(unnormalize(x, [self.lb, self.ub]))

##########
# Start PyTorch and warnings
warnings.filterwarnings("ignore")
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

##########
# Performance measurements
tic = time.time()   # Save starting time

##########
# Initialize FuRBO
obj = ack(dim = 2,
          negate=True,
          **tkwargs)
cons = list([sum_(threshold = 0,
                  lb = obj.lb,
                  ub = obj.ub), 
             norm_(threshold = 0.5, 
                   lb = obj.lb, 
                   ub = obj.ub)])
batch_size = int(1)#3 * obj.dim)
n_init = int(10)# * obj.dim)
n_iteration = int(150)# * obj.dim)
tr_number = 1
N_CANDIDATES = 2000

# FuRBO state initialization
FuRBO_status = Furbo_state_restart(obj = obj,                        # Objective function
                                  cons = cons,                      # Constraints function
                                  batch_size = batch_size,          # Batch size of each iteration
                                  n_init = n_init,                  # Number of initial points to evaluate
                                  n_iteration = n_iteration,        # Number of iterations
                                  tr_number = tr_number,            # number of Trust regions
                                  **tkwargs)

##########
# Main optimization loop

# Initiate lists to save over the restarts
X_best, Y_best, C_best = [], [], []
X_all, Y_all, C_all = [], [], []

# Continue optimization the stopping criterions isn't triggered
while not FuRBO_status.finish_trigger: 
    
    # Reset status for restarting
    FuRBO_status.reset_status(**tkwargs)
    
    # generate intial batch of X
    X_next = get_initial_points(FuRBO_status, **tkwargs)
    
    # Reset and restart optimization
    while not FuRBO_status.restart_trigger and not FuRBO_status.finish_trigger:
                
        # Evaluate current batch (samples in X_next)
        Y_next = []
        C_next = []
        for x in X_next:
            # Evaluate batch on obj ...
            Y_next.append(FuRBO_status.obj.eval_(x))
            # ... and constraints
            C_next.append([c.eval_(x) for c in FuRBO_status.cons])
               
        # process vector for PyTorch
        Y_next = torch.tensor(Y_next).unsqueeze(-1).to(**tkwargs)
        C_next = torch.tensor(C_next).to(**tkwargs)
                
        # Update FuRBO status with newly evaluated batch
        FuRBO_status.update(X_next, Y_next, C_next, **tkwargs)   
                
        # Printing current best
        # If a feasible has been evaluated -> print current optimum (feasible sample with best objective value)
        if (FuRBO_status.best_C <= 0).all():
            best = FuRBO_status.best_Y.amax()
            print(f"{FuRBO_status.it_counter-1}) Best value: {best:.2e},"
                  f" MND radius: {FuRBO_status.radius}")
        
        # Else, if no feasible has been evaluated -> print smallest violation (the sample that violatest the least all constraints)
        else:
            violation = FuRBO_status.best_C.clamp(min=0).sum()
            print(f"{FuRBO_status.it_counter-1}) No feasible point yet! Smallest total violation: "
                  f"{violation:.2e}, MND radius: {FuRBO_status.radius}")
            
        # Update Trust regions
        FuRBO_status = update_tr(FuRBO_status,
                                 **tkwargs)
                
        # generate next batch to evaluate 
        X_next = generate_batch(FuRBO_status, N_CANDIDATES, **tkwargs)
        
        # Check if stopping criterion is met (budget exhausted and if GP failed)
        FuRBO_status.finish_trigger = stopping_criterion(FuRBO_status, n_iteration) 
        
        # Check if restart criterion is met
        FuRBO_status.restart_trigger = (restart_criterion(FuRBO_status, FuRBO_status.radius_min)
                                        or GP_restart_criterion(FuRBO_status))
        
    # Save samples evaluated before resetting the status
    X_all.append(FuRBO_status.X)
    Y_all.append(FuRBO_status.Y)
    C_all.append(FuRBO_status.C)

    # Save best sample of this run
    X_best.append(FuRBO_status.best_X)
    Y_best.append(FuRBO_status.best_Y)
    C_best.append(FuRBO_status.best_C)
    
# Print best value found so far
# Ri-elaborate for processing
X_best = torch.stack(X_best).to(**tkwargs)
Y_best = torch.stack(Y_best).to(**tkwargs)
C_best = torch.stack(C_best).to(**tkwargs)

# If a feasible has been evaluated -> print current optimum sample and yielded value
if (C_best <= 0).any():
    best = Y_best.amax()
    bext = X_best[Y_best.argmax()]
    print("Optimization finished \n"
         f"\t Optimum: {best:.2e}, \n"
         f"\t X: {bext}")
    
# Else, if no feasible has been evaluated -> print sample with smallest violation and the violation value
else:
    violation = C_best.sum(dim=2).amin()
    violaxion = X_best[C_best.sum(dim=2).argmin()]
   
    print("Optimization failed \n"
         f"\t Smallest violation: {violation:.2e}, \n"
         f"\t X: {violaxion}")
            
# Print performance measurement
tac = time.time()       # Save finish time
total_time = (tac - tic) % 60
print(f"Computation time: {total_time:.2f} seconds")

# Plotting  
import numpy as np
import matplotlib.pyplot as plt

# Transform vectors with all samples to elaborate for plots
X_all = torch.concatenate(X_all).to(**tkwargs)
Y_all = torch.concatenate(Y_all).to(**tkwargs)
C_all = torch.concatenate(C_all).to(**tkwargs)

# Transform values and constraints to numpy
Y_f = Y_all.cpu().numpy()
C_f = np.amax(C_all.cpu().numpy(), axis=1)

# Get infeasible values to worst value evaluated
Y_f[np.where(C_f > 0)[0]] = np.amin(Y_f)

# Extract a monotonic curve
Y_f_monotonic = []
for yy in Y_f:
    if len(Y_f_monotonic) == 0:
        Y_f_monotonic.append(yy)
    else:
        if yy > Y_f_monotonic[-1]:
            Y_f_monotonic.append(yy)
        else:
            Y_f_monotonic.append(Y_f_monotonic[-1])

# Exclude initial DoE and generate x-y values for plot
y = np.array(Y_f_monotonic).reshape(-1)
x = np.linspace(1, len(y), len(y))

# Plotting convergence
plt.plot(x, y, lw=3)

# Plot optimum line
plt.plot([0, np.amax(x)], [0, 0], '--k', lw=3)
plt.ylabel("Function value", fontsize=18)
plt.xlabel("Number of evaluations", fontsize=18)
plt.title("10D Ackley with 2 outcome constraints", fontsize=20),
plt.xlim([0, len(y)])
plt.grid(True)
    

    



    

