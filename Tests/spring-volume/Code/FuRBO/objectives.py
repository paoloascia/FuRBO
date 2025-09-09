# Class of objective functions for Teemo
# 
# March 2024
##########
# Imports
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

import numpy as np
import os
import torch

##########
# Objectives functions to test FuRBO with
###
# Ackley function, dimension 2 
class ack():
    
    def __init__(self, dim, negate, **tkwargs):
        
        self.fun = Ackley(dim = dim, negate = negate).to(**tkwargs)
        self.fun.bounds[0, :].fill_(-5)
        self.fun.bounds[1, :].fill_(10)
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        
    def eval_(self, x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return self.fun(unnormalize(x, self.fun.bounds))
    
class keane():
    
    def __init__(self, dimension, lower_bound, upper_bound, **tkwargs):
        
        self.dimension = dimension
        self.lower_bounds = lower_bound
        self.upper_bounds = upper_bound
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        sum_ = torch.sum(torch.cos(x) ** 4)
        prod_ = torch.prod(torch.cos(x) ** 2)
        
        num_ = sum_ - 2 * prod_
        den_ = torch.sqrt(torch.sum(((torch.arange(self.dimension, dtype = x.dtype) + 1) * x**2)))
        
        return torch.abs(num_/den_)
        
class mopta_obj():
    def __init__(self, dim, lower_bound, upper_bound):
        
        self.dim = dim
        self.lb = lower_bound * torch.ones(dim)
        self.ub = upper_bound * torch.ones(dim)
        return
        
    def eval_(self, x, **tkwargs):
        
        # transform to numpy
        x = x.cpu().numpy()
        np.save("mopta_in.npy", x)
        
        # Evaluate
        os.system("wsl python /home/paoloa/mopta_runner.py")
        
        # transform to torch
        y = torch.tensor(np.load("mopta_out.npy"), **tkwargs)
        
        return y[0]
    
class spring():
    def __init__(self):
        self.dimension = 3
        self.lower_bounds = torch.Tensor([ 2., 0.25, 0.05])
        self.upper_bounds = torch.Tensor([15., 1.3 , 2.  ])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        x1, x2, x3 = x
        
        return (x1 + 2) * x2 * x3**2
    
    def __repr__(self):
        return 'Tension/Compression Spring Design -> 3D minimization problem with 4 constraints'
 
class speed_reducer():
    def __init__(self):
        self.dimension = 7
        self.lower_bounds = torch.Tensor([2.6, 0.7, 17., 7.3, 7.3, 2.9, 2.9])
        self.upper_bounds = torch.Tensor([3.6, 0.8, 28., 8.3, 8.3, 3.9, 3.9])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        x1, x2, x3, x4, x5, x6, x7 = x
    
        term1 = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        term2 = 1.508 * x1 * (x6**2 + x7**2)
        term3 = 7.4777 * (x6**3 + x7**3)
        term4 = 0.7854 * (x4 * x6**2 + x5 * x7**2)
        
        return term1 - term2 + term3 + term4    

    def __repr__(self):
        return 'Speed Reducer design -> 7D minimization problem with 11 constraints'    
    
class welded_beam():
    def __init__(self):
        self.dimension = 4
        self.lower_bounds = torch.Tensor([ 0.125,  0.1,  0.1,  0.1])
        self.upper_bounds = torch.Tensor([10.   , 10. , 10. , 10. ])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        h, l, t, b = x
        
        return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l) 

    def __repr__(self):
        return 'Welded Beam design -> 4D minimization problem with 5 constraints'    
    
class pressure_vessel():
    def __init__(self):
        self.dimension = 4
        self.lower_bounds = torch.Tensor([0.0625, 0.0625,  10.,  10.])
        self.upper_bounds = torch.Tensor([5.    , 5.    , 200., 200.])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        T_s_raw, T_h_raw, R, L = x
        
        STEP_SIZE = 0.0625

        # Round T_s and T_h to the nearest multiple of STEP_SIZE
        T_s = torch.round(T_s_raw / STEP_SIZE) * STEP_SIZE
        T_h = torch.round(T_h_raw / STEP_SIZE) * STEP_SIZE

        # Ensure T_s and T_h stay within their bounds after rounding
        T_s = torch.clip(T_s, 0.0625, 5)
        T_h = torch.clip(T_h, 0.0625, 5)

        W = (0.6224 * T_s * R * L +
             1.7781 * T_h * R**2 +
             3.1661 * T_s**2 * L +
             19.84 * T_s**2 * R)
        return W

    def __repr__(self):
        return 'Pressure Vessel design -> 4D minimization problem with 4 constraints'
    
class two_d_toy_problem():
    def __init__(self):
        self.dimension = 2
        self.lower_bounds = torch.Tensor([0., 0.])
        self.upper_bounds = torch.Tensor([1., 1.])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        return torch.sum(x)
    
    def __repr__(self):
        return '2D Toy Problem -> 2D minimization problem with 2 constraints'
    
class rosenbrock():
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds #[-3., -3., -3., -3., -3.]
        self.upper_bounds = upper_bounds #[ 5.,  5.,  5.,  5.,  5.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        return torch.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    
    def __repr__(self):
        return 'X D Rosenbrock function -> X D minimization problem'
        
        
if __name__ == '__main__':
    
    fcn = mopta_obj()
    
    x = torch.tensor([1., 2., 3., 4., 5., 6.])
    
    y = fcn.eval_(x)