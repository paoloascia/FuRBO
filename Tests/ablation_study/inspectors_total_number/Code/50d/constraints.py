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
    
class keane_g1():
    
    def __init__(self, lb, ub):
        
        self.lb = lb
        self.ub = ub
        return
    
    def eval_(self, x):
        
        x = unnormalize(x, [self.lb, self.ub])
        
        return 0.75 - torch.prod(x)
    
class keane_g2():
    
    def __init__(self, dim, lb, ub):
        
        self.lb = lb
        self.ub = ub
        self.dim = dim
        return
    
    def eval_(self, x):
        
        x = unnormalize(x, [self.lb, self.ub])
        
        return torch.sum(x) - 7.5 * self.dim
    
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