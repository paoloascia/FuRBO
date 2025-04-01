# FuRBO state intiate for different loops
# 
# March 2024
##########
# Imports
import math
from botorch.models.model_list_gp_regression import ModelListGP

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

# Custom imports 
from utilities import get_best_index_for_batch
from utilities import get_fitted_model

# possible EIBO states
class variant_one():
    
    # Initialization of the status
    def __init__(self,                  #
                 obj,                   # Objective function
                 cons,                  # Constraints function
                 batch_size,            # Batch size of each iteration
                 n_init,                # Number of initial points to evaluate
                 n_iteration,           # Number of total iterations
                 seed,                  # Seed for Sobol sampling
                 history,               # History if this is a restart
                 iteration,             # Number of iteration to keep counting when restarting
                 samples_evaluated,     # Numnber of evaluations if restart
                 **tkwargs):
        
        # Objective function handle
        self.obj = obj
        
        # Constraints function handle
        self.cons = cons
        
        # Problem dimensions
        self.batch_size: int = batch_size   # Dimension of the batch at each iteration
        self.n_init: int = n_init           # Number of initial samples
        self.dim: int = obj.dimension       # Dimension of the problem
        
        # Batch information
        self.batch_X: Tensor        # Current batch to evaluate: X values
        self.batch_Y: Tensor        # Current batch to evaluate: Y value
        self.batch_C: Tensor        # Current batch to evaluate: C values
        
        # Information on best sample
        self.best_X: Tensor                    # Current best: X values
        self.best_Y: float = -float("inf")     # Current best: Y value
        self.best_C: Tensor                    # Current best: C values
    
        self.it_counter: int = iteration  # Counter of iterations for stopping
        
        # Stopping criteria
        self.n_iteration: int = n_iteration                 # Maximum number of iterations allowed
        self.finish_trigger: bool = False                   # Trigger to stop optimization
        self.failed_GP : bool = False                       # Flag to pass to failed_GP in SCBORestart
        self.samples_evaluated: int = samples_evaluated     # Current number of evaluations
        
        # Restart criteria
        self.restart_trigger: bool = False       # Trigger to restart optimization
        
        # History
        self.history = history       # List where to store all relevant information
        self.seed = seed             # Save seed 
        
        # Utilities
        self.sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        
    # Update the status
    def update(self,
               X_next,
               Y_next,
               C_next,
               **tkwargs):
        
        # Save all samples evaluated
        if not hasattr(self, 'X'):
            self.X = X_next
            self.Y = Y_next
            self.C = C_next
        else:
            self.X = torch.cat((self.X, X_next), dim=0)
            self.Y = torch.cat((self.Y, Y_next), dim=0)
            self.C = torch.cat((self.C, C_next), dim=0)
            
        # update surrogates
        try:
            self.Y_model = get_fitted_model(self.X, self.Y, self.dim, max_cholesky_size = float("inf"))
            self.C_model = ModelListGP(*[get_fitted_model(self.X, C.reshape([C.shape[0],1]), self.dim, max_cholesky_size = float("inf")) for C in self.C.t()])

        except:
            self.failed_GP = True

        # Update batch information 
        self.batch_X = X_next
        self.batch_Y = Y_next
        self.batch_C = C_next
        
        # Update best value   
        best_id = get_best_index_for_batch(Y=self.Y, C=self.C)
        
        self.best_X = self.X[best_id]
        self.best_Y = self.Y[best_id]
        self.best_C = self.C[best_id]
        
        event = {'iteration': self.it_counter,
                 'batch': {'X': self.batch_X,
                           'Y': self.batch_Y,
                           'C': self.batch_C},
                 'best': {'X': self.best_X,
                          'Y': self.best_Y,
                          'C': self.best_C},
                 'seed': self.seed}
        
        self.history.append(event)
        
        # Update iteration counter
        self.it_counter += 1
        
        # Update samples counter
        self.samples_evaluated += len(Y_next)

        return
