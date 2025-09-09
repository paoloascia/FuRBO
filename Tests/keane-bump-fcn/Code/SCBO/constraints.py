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
        self.lower_bounds = [ 2., 0.25, 0.05]
        self.upper_bounds = [15., 1.3 , 2.  ]
        
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
        self.lower_bounds = [2.6, 0.7, 17., 7.3, 7.3, 2.9, 2.9]
        self.upper_bounds = [3.6, 0.8, 28., 8.3, 8.3, 3.9, 3.9]
        
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
        self.lower_bounds = [ 0.125,  0.1,  0.1,  0.1]
        self.upper_bounds = [10.   , 10. , 10. , 10. ]
        
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
        self.lower_bounds = [0.0625, 0.0625, 10. , 10. ]
        self.upper_bounds = [5.    , 5.    , 200., 200.]
        
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
        self.lower_bounds = [0., 0.]
        self.upper_bounds = [1., 1.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        x1, x2 = x
        
        g1 = x1 + 2*x2 + 0.5 * np.sin(2 * np.pi * (x1**2 - 2*x2)) - 1.5
        g2 = 1.5 - x1**2 - x2**2
        
        return [g1, g2]
    
    def __repr__(self):
        return '2D Toy Problem -> 2D minimization problem with 2 constraints (neg=feasibility)'
    
class dixon_price():
    def __init__(self):
        self.dimension = 5
        self.lower_bounds = [-3., -3., -3., -3., -3.]
        self.upper_bounds = [ 5.,  5.,  5.,  5.,  5.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        first_term = (x[0] - 1)**2
    
        sum_term = 0.0
        for i_python in range(1, self.dimension): # Loop from i_python = 1 to d-1 (corresponds to i=2 to d)
            term = (i_python + 1) * (2 * x[i_python]**2 - x[i_python - 1])**2
            sum_term += term
        
        return first_term + sum_term - 10
    
    def __repr__(self):
        return '5D dixon price constraint -> 5D constraint function (neg=feasible)'
    
class levy():
    def __init__(self):
        self.dimension = 5
        self.lower_bounds = [-3., -3., -3., -3., -3.]
        self.upper_bounds = [ 5.,  5.,  5.,  5.,  5.]
        
    def __call__(self, x):
        
        x = unnormalize(x, [self.lower_bounds, self.upper_bounds])
        
        w = 1 + (x - 1) / 4

        term1 = np.sin(np.pi * w[0])**2
        
        sum_terms = 0.0
        for i_python in range(self.dimension - 1): 
            current_w_i = w[i_python]
            sum_terms += (current_w_i - 1)**2 * (1 + 10 * np.sin(np.pi * current_w_i + 1)**2)
    
        last_w_d = w[self.dimension-1]
        term_last = (last_w_d - 1)**2 * (1 + np.sin(2 * np.pi * last_w_d)**2)

        return term1 + sum_terms + term_last - 10
    
    def __repr__(self):
        return '5D levy constraint -> 5D constraint function (neg=feasible)'
    