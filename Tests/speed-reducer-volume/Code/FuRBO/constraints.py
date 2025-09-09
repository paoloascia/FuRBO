# Class of constraints for Teemo
# 
# March 2024
##########
# Imports
from botorch.utils.transforms import unnormalize

import numpy as np
import os
import torch

##########
# All possible constraints
class sum_():
    # enforcing that sum(x) <= threshold
    def __init__(self, threshold, lb, ub):
        
        self.threshold = threshold
        self.lb = lb
        self.ub = ub
        return 
    
    def c(self, x):
        return x.sum() - self.threshold
    
    def eval_(self, x):
        return self.c(unnormalize(x, [self.lb, self.ub]))
###
class norm_():
    # enforcing that ||x||_2 <= threshold
    def __init__(self, threshold, lb, ub):
        
        self.threshold = threshold
        self.lb = lb
        self.ub = ub
        return 
    
    def c(self, x):
        return torch.norm(x, p=2) - self.threshold
    
    def eval_(self, x):
        return self.c(unnormalize(x, [self.lb, self.ub]))
    
class norm_rev():
    # enforcing that ||x||_2 >= threshold
    def __init__(self, threshold, lb, ub):
        
        self.threshold = threshold
        self.lb = lb
        self.ub = ub
        return 
    
    def c(self, x):
        return self.threshold - torch.norm(x, p=2)
    
    def eval_(self, x):
        return self.c(unnormalize(x, [self.lb, self.ub]))
    
class keane_cons():
    
    def __init__(self, dimension, lower_bound, upper_bound, **tkwargs):
        
        self.dimension = dimension
        self.lower_bounds = lower_bound
        self.upper_bounds = upper_bound
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
                
        g1 = 0.75 - torch.prod(x)
        g2 = torch.sum(x) - 7.5 * self.dimension
        
        return [g1, g2]
    
class mopta_cons():
    
    def __init__(self, i):
        self.i = i
        return
    
    def eval_(self, x, **tkwargs):
        
        # transform to numpy
        # x = x.cpu().numpy()
        # np.save("mopta_in.npy", x)
        
        # Evaluate
        # os.system("wsl python /home/paoloa/mopta_runner.py")
        
        # transform to torch
        y = torch.tensor(np.load("mopta_out.npy"), **tkwargs)
        
        return y[self.i]
    
class spring_cons():
    def __init__(self):
        self.dimension = 3
        self.lower_bounds = torch.Tensor([ 2., 0.25, 0.05])
        self.upper_bounds = torch.Tensor([15., 1.3 , 2.  ])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        x1, x2, x3 = x
        
        g1 = 1 - (x2**3 * x1) / (71785 * x3**4)
        g2 = (4 * x2**2 - x3 * x2) / (12566 * (x2 * x3**3 - x3**4)) + 1 / (5108 * x3**2) - 1
        g3 = 1 - (140.45 * x3) / (x2**2 * x1)
        g4 = (x2 + x3) / 1.5 - 1
        
        return [g1, g2, g3, g4]
    
    def __repr__(self):
        return 'Tension/Compression Spring Design -> 3D minimization problem with 4 constraints (neg=feasible)'

class speed_reducer_cons():
    def __init__(self):
        self.dimension = 7
        self.lower_bounds = torch.Tensor([2.6, 0.7, 17., 7.3, 7.3, 2.9, 4.9])
        self.upper_bounds = torch.Tensor([3.6, 0.8, 28., 8.3, 8.3, 3.9, 5.9])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        x1, x2, x3, x4, x5, x6, x7 = x
    
        g1 = 27.0 / (x1 * x2**2 * x3) - 1
        g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
        g3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
        g4 = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
        g5 = (1 / (0.1 * x6**3)) * np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) - 1100
        g6 = (1 / (0.1 * x7**3)) * np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) - 850
        g7 = x2 * x3 - 40
        g8 = 5 - x1 / x2
        g9 = x1 / x2 - 12
        g10 = (1.5 * x6 + 1.9) / x4 - 1
        g11 = (1.1 * x7 + 1.9) / x5 - 1
        
        return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]   

    def __repr__(self):
        return 'Speed Reducer design -> 7D minimization problem with 11 constraints (neg=feasible)'    
    
class welded_beam_cons():
    def __init__(self):
        self.dimension = 4
        self.lower_bounds = torch.Tensor([ 0.125,  0.1,  0.1,  0.1])
        self.upper_bounds = torch.Tensor([10.   , 10. , 10. , 10. ])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        h, l, t, b = x
        
        tau_p = 6000 / (np.sqrt(2) * h * l)
        alpha = np.sqrt(0.25 * (l**2 + (h + t)**2))
        tau_pp = (6000 * (14 + 0.5 * l) * alpha) / (2 * (0.707 * h * l * (l**2 / 12 + 0.25 * (h + t)**2)))
        tau = np.sqrt(tau_p**2 + tau_pp**2 + (l * tau_p * tau_pp) / alpha)
        
        sigma = 504000 / (t**2 * b)
        Pc = 64746.022 * (1 - 0.0282346 * t) * t * b**3
        delta = 2.1952 / (t**3 * b)
        
        g1 = tau - 13600
        g2 = sigma - 30000
        g3 = h - b
        g4 = 6000 - Pc
        g5 = delta - 0.25
        
        return [g1, g2, g3, g4, g5]

    def __repr__(self):
        return 'Welded Beam design -> 4D minimization problem with 5 constraints (neg=feasible)'    
    
class pressure_vessel_cons():
    def __init__(self):
        self.dimension = 4
        self.lower_bounds = torch.Tensor([0.0625, 0.0625, 10. , 10. ])
        self.upper_bounds = torch.Tensor([5.    , 5.    , 200., 200.])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        T_s_raw, T_h_raw, R, L = x
        
        STEP_SIZE = 0.0625

        # Round T_s and T_h to the nearest multiple of STEP_SIZE
        T_s = np.round(T_s_raw / STEP_SIZE) * STEP_SIZE
        T_h = np.round(T_h_raw / STEP_SIZE) * STEP_SIZE

        # Ensure T_s and T_h stay within their bounds after rounding
        T_s = np.clip(T_s, 0.0625, 5)
        T_h = np.clip(T_h, 0.0625, 5)
        
        g1 = 0.0193 * R - T_s
        g2 = 0.00954 * R - T_h
        g3 = 1296000 - np.pi * R**2 * L - (4/3) * np.pi * R**3
        g4 = L - 240
        
        return [g1, g2, g3, g4]

    def __repr__(self):
        return 'Pressure Vessel design -> 4D minimization problem with 4 constraints (neg=feasible)'
    
class two_d_toy_problem():
    def __init__(self):
        self.dimension = 2
        self.lower_bounds = torch.Tensor([0., 0.])
        self.upper_bounds = torch.Tensor([1., 1.])
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        x1, x2 = x
        
        g1 = 1.5 - x1 - 2* x2 - 0.5 * np.sin(2 * np.pi * (x1**2 - 2 * x2))
        g2 = x1**2 + x2**2 - 1.5
        
        return [g1, g2]
    
    def __repr__(self):
        return '2D Toy Problem -> 2D minimization problem with 2 constraints (neg=feasibility)'
    
class dixon_price():
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds #[-3., -3., -3., -3., -3.]
        self.upper_bounds = upper_bounds #[ 5.,  5.,  5.,  5.,  5.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        return (x[0] - 1)**2 + torch.sum((torch.arange(self.dimension - 2) + 2) * (2 * x[2:]**2 - x[:-1])**2)
    
    def __repr__(self):
        return 'Dixon price function'
    
class levy():
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds #[-3., -3., -3., -3., -3.]
        self.upper_bounds = upper_bounds #[ 5.,  5.,  5.,  5.,  5.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        w = 1 + (x - 1) / 4
        
        return ((torch.sin(torch.pi * w[0]))**2 + 
                torch.sum((w[:-1] - 1)**2 * (1 + 10 * (torch.sin(torch.pi * w[:-1]))**2)) +
                (w[-1] - 1)**2 * (1 + (torch.sin(2 * torch.pi * w[-1]))**2))
    
    def __repr__(self):
        return 'Levy function'
    
class rosenbrock_cons():
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds #[-3., -3., -3., -3., -3.]
        self.upper_bounds = upper_bounds #[ 5.,  5.,  5.,  5.,  5.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        dixon_ = dixon_price(self.dimension, self.lower_bounds, self.upper_bounds)
        levy_ = levy(self.dimension, self.lower_bounds, self.upper_bounds)        
        
        g1 = dixon_(x) - 10
        g2 = levy_(x) - 10
        
        return [g1, g2]
    
    def __repr__(self):
        return 'X D Rosenbrock function -> X D minimization problem (neg=feasible)'
    