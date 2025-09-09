# FuRBO state intiate for different loops
# 
# March 2024
##########
# Imports
from botorch.models.model_list_gp_regression import ModelListGP

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

# Custom imports 
from utilities import get_best_index_for_batch
from utilities import get_fitted_model
from utilities import gaussian_copula as obj_scaling
from utilities import no_scaling as cons_scaling

# possible FuRBO states
class variant_one():
    
    # Initialization of the status
    def __init__(self,              #
                 obj,               # Objective function
                 cons,              # Constraints function
                 batch_size,        # Batch size of each iteration
                 n_init,            # Number of initial points to evaluate
                 n_iteration,       # Number of total iterations
                 tr_number,         # number of Trust regions
                 seed,              # Seed for Sobol sampling
                 history,           # saved history to make all plots
                 iteration,         # Numnber of iteration if restart
                 samples_evaluated, # Number of samples already evaluated
                 **tkwargs):
        
        # Objective function handle
        self.obj = obj
        self.lb, self.ub = obj.lower_bounds, obj.upper_bounds
        
        # Constraints function handle
        self.cons = cons
        
        # Problem dimensions
        self.batch_size: int = batch_size   # Dimension of the batch at each iteration
        self.n_init: int = n_init           # Number of initial samples
        self.dim: int = obj.dimension             # Dimension of the problem
        
        # Trust regions information
        self.tr_number: int = tr_number                                            # Number of trust regions to use during evolution
        self.tr_ub: float = torch.ones((self.tr_number, self.dim), **tkwargs)      # Upper bounds of trust region
        self.tr_lb: float = torch.zeros((self.tr_number, self.dim), **tkwargs)     # Lower bounds of trust region
        self.tr_vol: float = torch.prod(self.tr_ub - self.tr_lb, dim=1)            # Volume of trust region
        self.radius: float = 0.2                                                   # Percentage around which the trust region is built
        self.radius_min: float = 0.5**7                                            # Minimum percentage for trust region

        # Performance tracking
        self.failure_counter: int = 0    # Counter of failure points to asses how algorithm is going
        self.success_counter: int = 0    # Counter of success points to asses how algorithm is going
        
        # Thresholds to change trust region size
        self.success_tolerance: int = 2
        self.failure_tolerance: int = 3
        
        self.batch_X: Tensor        # Current batch to evaluate: X values
        self.batch_Y: Tensor        # Current batch to evaluate: Y value
        self.batch_C: Tensor        # Current batch to evaluate: C values
            
        self.it_counter: int = iteration  # Counter of iterations for stopping
        
        # Stopping criteria
        self.n_iteration: int = n_iteration     # Maximum number of iterations allowed
        self.finish_trigger: bool = False       # Trigger to stop optimization
        self.restart_trigger: bool = False      # Trigger to restart optimization
        self.failed_GP : bool = False           # Flag to pass to failed_GP in FuRBORestart
        
        # History
        self.history = history       # List where to store all relevant information
        
        # Utilities
        self.seed = seed            # Save seed
        self.sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        self.samples_evaluated: int = samples_evaluated
        
    # Update the status
    def update(self,
               X_next,
               Y_next,
               C_next,
               **tkwargs):
        
        # Debugging prints
        # print('Failure counter: ')
        # print(failure_counter)
        # print('Success counter: ')
        # print(success_counter)
        
        # Merge together current evaluated samples with previous samples
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
            # Find the best value among the candidates
        best_id = get_best_index_for_batch(n_tr=self.tr_number, Y=self.Y, C=self.C)
            # Update
            
        # Update success and failure counters
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
        
        self.best_X = self.X[best_id]
        self.best_Y = self.Y[best_id]
        self.best_C = self.C[best_id]
        
        # Debugging prints
        # print('X best: ')
        # print(self.best_X)
        # print('Y best: ')
        # print(self.best_Y)
        # print('C best: ')
        # print(self.best_C)
        
        # Update history
        event = {'iteration': self.it_counter,
                 'batch': {'X': self.batch_X,
                           'Y': self.batch_Y,
                           'C': self.batch_C},
                 'best': {'X': self.best_X,
                          'Y': self.best_Y,
                          'C': self.best_C},
                 'trust_region': {'lower_bound': self.tr_lb,
                                  'upper_bound': self.tr_ub,
                                  'radius': self.radius},
                 'performance': {'n_success': self.success_counter,
                                 'n_failures': self.failure_counter},
                 'seed': self.seed}
        
        self.history.append(event)
        
        # Update iteration counter
        self.it_counter += 1
        self.samples_evaluated += len(Y_next)
        