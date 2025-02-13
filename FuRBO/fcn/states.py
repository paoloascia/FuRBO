# FuRBO state intiate for different loops
# 
# March 2024
##########
# Imports
from botorch.models.model_list_gp_regression import ModelListGP

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

##########
# Custom imports 
from fcn.utilities import get_best_index_for_batch
from fcn.utilities import get_fitted_model

# possible FuRBO states
class Furbo_state_single():
    '''Class to track optimization status without restart'''
    # Initialization of the status
    def __init__(self,              #
                 obj,               # Objective function
                 cons,              # Constraints function
                 batch_size,        # Batch size of each iteration
                 n_init,            # Number of initial points to evaluate
                 n_iteration,       # Number of total iterations
                 tr_number,         # number of Trust regions
                 **tkwargs):
        
        # Objective function handle
        self.obj = obj
        
        # Constraints function handle
        self.cons = cons
        
        # Domain bounds
        self.lb = obj.lb
        self.ub = obj.ub
        
        # Problem dimensions
        self.batch_size: int = batch_size      # Dimension of the batch at each iteration
        self.n_init: int = n_init              # Number of initial samples
        self.dim: int = obj.dim                # Dimension of the problem
        
        # Trust regions information
        self.tr_number: int = tr_number                                            # Number of trust regions to use during evolution
        self.tr_ub: float = torch.ones((self.tr_number, self.dim), **tkwargs)      # Upper bounds of trust region
        self.tr_lb: float = torch.zeros((self.tr_number, self.dim), **tkwargs)     # Lower bounds of trust region
        self.tr_vol: float = torch.prod(self.tr_ub - self.tr_lb, dim=1)            # Volume of trust region
        self.radius: float = 1.0                                                   # Percentage around which the trust region is built
        self.radius_min: float = 0.5**7                                            # Minimum percentage for trust region

        # Trust region updating 
        self.failure_counter: int = 0       # Counter of failure points to asses how algorithm is going
        self.success_counter: int = 0       # Counter of success points to asses how algorithm is going
        self.success_tolerance: int = 2     # Success tolerance for 
        self.failure_tolerance: int = 3     # Failure tolerance for
        
        # Tensor to save current batch information
        self.batch_X: Tensor        # Current batch to evaluate: X values
        self.batch_Y: Tensor        # Current batch to evaluate: Y value
        self.batch_C: Tensor        # Current batch to evaluate: C values
            
        # Stopping criteria information
        self.n_iteration: int = n_iteration     # Maximum number of iterations allowed
        self.it_counter: int = 0  # Counter of iterations for stopping
        self.finish_trigger: bool = False       # Trigger to stop optimization
        self.failed_GP : bool = False           # Flag to pass to failed_GP in FuRBORestart
        
        # Sobol sampler engine
        self.sobol = SobolEngine(dimension=self.dim, scramble=True)
        
    # Update the status
    def update(self,
               X_next,          # Samples X (input values) to update the status
               Y_next,          # Samples Y (objective value) to update the status
               C_next,          # Samples C (constraints values) to update the status
               **tkwargs):
        '''Function to update optimization status'''
        
        # Merge current batch with previously evaluated samples
        if not hasattr(self, 'X'):
            # If there are no previous samples, declare the Tensors
            self.X = X_next
            self.Y = Y_next
            self.C = C_next
        else:
            # Else, concatenate the new batch to the previous samples
            self.X = torch.cat((self.X, X_next), dim=0)
            self.Y = torch.cat((self.Y, Y_next), dim=0)
            self.C = torch.cat((self.C, C_next), dim=0)

        # update GPR surrogates
        try:
            self.Y_model = get_fitted_model(self.X, self.Y, self.dim, max_cholesky_size = float("inf"))
            self.C_model = ModelListGP(*[get_fitted_model(self.X, C.reshape([C.shape[0],1]), self.dim, max_cholesky_size = float("inf")) for C in self.C.t()])
        except:
            # If update fail, flag to stop entire optimization
            self.failed_GP = True
        
        # Update batch information 
        self.batch_X = X_next
        self.batch_Y = Y_next
        self.batch_C = C_next
            
        # Update best value
        # Find the best value among the candidates
        best_id = get_best_index_for_batch(n_tr=self.tr_number, Y=self.Y, C=self.C)
            
        # Update success and failure counters for trust region update
        # If attribute 'best_X' does not exist, DoE was just evaluated -> no update on counters
        if hasattr(self, 'best_X'):
            if (self.C[best_id] <= 0).all():
                # At least one new candidate is feasible
                if (self.Y[best_id] > self.best_Y).any() or (self.best_C > 0).any():
                    self.success_counter += 1
                    self.failure_counter = 0                
                else:
                    self.success_counter = 0
                    self.failure_counter += 1
            else:
                # No new candidate is feasible
                total_violation_next = self.C[best_id].clamp(min=0).sum(dim=-1)
                total_violation_center = self.best_C.clamp(min=0).sum(dim=-1)
                if total_violation_next < total_violation_center:
                    self.success_counter += 1
                    self.failure_counter = 0
                else:
                    self.success_counter = 0
                    self.failure_counter += 1
        
        # Update best values
        self.best_X = self.X[best_id]
        self.best_Y = self.Y[best_id]
        self.best_C = self.C[best_id]
        
        # Update iteration counter
        self.it_counter += 1
    
class Furbo_state_restart():
    '''Class to track optimization status with restart'''
    # Initialization of the status
    def __init__(self,              #
                 obj,               # Objective function
                 cons,              # Constraints function
                 batch_size,        # Batch size of each iteration
                 n_init,            # Number of initial points to evaluate
                 n_iteration,       # Number of total iterations
                 tr_number,         # number of Trust regions
                 **tkwargs):
        
        # Objective function handle
        self.obj = obj
        
        # Constraints function handle
        self.cons = cons
        
        # Domain bounds
        self.lb = obj.lb
        self.ub = obj.ub
        
        # Problem dimensions
        self.batch_size: int = batch_size      # Dimension of the batch at each iteration
        self.n_init: int = n_init              # Number of initial samples
        self.dim: int = obj.dim                # Dimension of the problem
        
        # Trust regions information
        self.tr_number: int = tr_number                                            # Number of trust regions to use during evolution
        self.tr_ub: float = torch.ones((self.tr_number, self.dim), **tkwargs)      # Upper bounds of trust region
        self.tr_lb: float = torch.zeros((self.tr_number, self.dim), **tkwargs)     # Lower bounds of trust region
        self.tr_vol: float = torch.prod(self.tr_ub - self.tr_lb, dim=1)            # Volume of trust region
        self.radius: float = 1.0                                                   # Percentage around which the trust region is built
        self.radius_min: float = 0.5**7                                            # Minimum percentage for trust region

        # Trust region updating 
        self.failure_counter: int = 0       # Counter of failure points to asses how algorithm is going
        self.success_counter: int = 0       # Counter of success points to asses how algorithm is going
        self.success_tolerance: int = 2     # Success tolerance for 
        self.failure_tolerance: int = 3     # Failure tolerance for
        
        # Tensor to save current batch information
        self.batch_X: Tensor        # Current batch to evaluate: X values
        self.batch_Y: Tensor        # Current batch to evaluate: Y value
        self.batch_C: Tensor        # Current batch to evaluate: C values
            
        # Stopping criteria information
        self.n_iteration: int = n_iteration     # Maximum number of iterations allowed
        self.it_counter: int = 0  # Counter of iterations for stopping
        self.finish_trigger: bool = False       # Trigger to stop optimization
        self.failed_GP : bool = False           # Flag to pass to failed_GP in FuRBORestart
        
        # Restart criteria information
        self.restart_trigger: bool = False
        
        # Sobol sampler engine
        self.sobol = SobolEngine(dimension=self.dim, scramble=True)
        
    # Update the status
    def update(self,
               X_next,          # Samples X (input values) to update the status
               Y_next,          # Samples Y (objective value) to update the status
               C_next,          # Samples C (constraints values) to update the status
               **tkwargs):
        '''Function to update optimization status'''
        
        # Merge current batch with previously evaluated samples
        if not hasattr(self, 'X'):
            # If there are no previous samples, declare the Tensors
            self.X = X_next
            self.Y = Y_next
            self.C = C_next
        else:
            # Else, concatenate the new batch to the previous samples
            self.X = torch.cat((self.X, X_next), dim=0)
            self.Y = torch.cat((self.Y, Y_next), dim=0)
            self.C = torch.cat((self.C, C_next), dim=0)

        # update GPR surrogates
        try:
            self.Y_model = get_fitted_model(self.X, self.Y, self.dim, max_cholesky_size = float("inf"))
            self.C_model = ModelListGP(*[get_fitted_model(self.X, C.reshape([C.shape[0],1]), self.dim, max_cholesky_size = float("inf")) for C in self.C.t()])
        except:
            # If update fail, flag to stop entire optimization
            self.failed_GP = True
        
        # Update batch information 
        self.batch_X = X_next
        self.batch_Y = Y_next
        self.batch_C = C_next
            
        # Update best value
        # Find the best value among the candidates
        best_id = get_best_index_for_batch(n_tr=self.tr_number, Y=self.Y, C=self.C)
            
        # Update success and failure counters for trust region update
        # If attribute 'best_X' does not exist, DoE was just evaluated -> no update on counters
        if hasattr(self, 'best_X'):
            if (self.C[best_id] <= 0).all():
                # At least one new candidate is feasible
                if (self.Y[best_id] > self.best_Y).any() or (self.best_C > 0).any():
                    self.success_counter += 1
                    self.failure_counter = 0                
                else:
                    self.success_counter = 0
                    self.failure_counter += 1
            else:
                # No new candidate is feasible
                total_violation_next = self.C[best_id].clamp(min=0).sum(dim=-1)
                total_violation_center = self.best_C.clamp(min=0).sum(dim=-1)
                if total_violation_next < total_violation_center:
                    self.success_counter += 1
                    self.failure_counter = 0
                else:
                    self.success_counter = 0
                    self.failure_counter += 1
        
        # Update best values
        self.best_X = self.X[best_id]
        self.best_Y = self.Y[best_id]
        self.best_C = self.C[best_id]
        
        # Update iteration counter
        self.it_counter += 1
        
    def reset_status(self,
                     **tkwargs):
        '''Function to reset the status for the restart'''
        
        # Reset trust regions size
        self.tr_ub: float = torch.ones((self.tr_number, self.dim), **tkwargs)      # Upper bounds of trust region
        self.tr_lb: float = torch.zeros((self.tr_number, self.dim), **tkwargs)     # Lower bounds of trust region
        self.tr_vol: float = torch.prod(self.tr_ub - self.tr_lb, dim=1)            # Volume of trust region
        self.radius: float = 1.0                                                   # Percentage around which the trust region is built
        self.radius_min: float = 0.5**7                                            # Minimum percentage for trust region

        # Reset counters to change trust region size 
        self.failure_counter: int = 0    # Counter of failure points to asses how algorithm is going
        self.success_counter: int = 0    # Counter of success points to asses how algorithm is going
        
        # Reset restart criteria trigger
        self.restart_trigger: bool = False      # Trigger to restart optimization
        self.failed_GP: bool = False            # Reset GPR failure trigger
        
        # Delete tensors with samples for training GPRs
        if hasattr(self, 'X'):
            del self.X
            del self.Y
            del self.C
        
        # Delete tensors with best value so far
        if hasattr(self, 'best_X'):
            del self.best_X
            del self.best_Y
            del self.best_C
        
        # Clear GPU memory
        if tkwargs["device"] == "cuda":
            torch.cuda.empty_cache()  