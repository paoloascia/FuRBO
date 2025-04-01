# Full ode for FuRBO
#
# March 2024
##########
# Imports
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import torch

##########
# Load all information
states = []
states_torch = []

cwd_base = os.path.join(os.getcwd(), 'repetition')

for cwd_ in os.listdir(cwd_base):
    
    for torch_file in os.listdir(os.path.join(cwd_base, cwd_)):
        
        if 'torch' in torch_file:
            states_torch.append(torch.load(os.path.join(cwd_base, cwd_, torch_file), map_location=torch.device('cpu')))

##########
# Extract values of best
Y = []
C = []

for i, state in enumerate(states_torch):
    Y.append([event['best']['Y'].cpu().numpy() for event in state.history])
    C.append([np.max(event['best']['C'].cpu().numpy()) for event in state.history])
    
        
Y = np.array(Y)
Y = Y.reshape(Y.shape[0], Y.shape[1])
C = np.array(C)
C = C.reshape(C.shape[0], C.shape[1])

##########
# Convergence plot: obj - infeasible + feasible
fig_01 = plt.figure()
ax_01 = plt.gca()

ax_01.set_title('Best Obj value at each iteration')  

for yy, cc in zip(Y, C):
    xx = np.linspace(1, len(yy), len(yy))
    CLR = ['green' if np.all(c < 0) else 'red' for c in cc]
    ALPHA = [1.0 if np.all(c < 0) else 0.25 for c in cc]
    
    for j in range(len(yy)-1):
        x = [xx[j], xx[j+1]]
        y = [yy[j], yy[j+1]]
        clr = CLR[j]
        alpha = ALPHA[j]
        ax_01.plot(x,y, color=clr, alpha=alpha)
    ax_01.scatter(xx, yy, color=CLR, alpha=ALPHA)
    
ax_01.set_ylabel('Objective funtion')
ax_01.set_xlabel('Iteration')


##########
# Convergence plot: obj - all
fig_02 = plt.figure()
ax_02 = plt.gca()  

ax_02.set_title('Objective Function (maximization 2d Ackley fcn): median and 0.9-0.1 quantiles')  

lower_quantiles = 0.1
upper_quantiles = 0.9
mean = np.quantile(Y, 0.5, axis = 0)
lb = np.quantile(Y, lower_quantiles, axis = 0)
ub = np.quantile(Y, upper_quantiles, axis = 0)

x = np.linspace(1, len(mean), len(mean))

ax_02.plot(x, mean, color = 'darkorange', lw=2)
ax_02.fill_between(x, lb, ub, alpha = 0.2, color='darkorange', lw=2)

ax_02.set_ylabel('Objective funtion')
ax_02.set_xlabel('Iteration')

##########
# Convergence plot: cons - all
fig_03 = plt.figure()
ax_03 = plt.gca()

ax_03.set_title('Max violation (<= 0 -> feasible): median and 0.9-0.1 quantiles')  

lower_quantiles = 0.1
upper_quantiles = 0.9
mean = np.quantile(C, 0.5, axis = 0)
lb = np.quantile(C, lower_quantiles, axis = 0)
ub = np.quantile(C, upper_quantiles, axis = 0)

x = np.linspace(1, len(mean), len(mean))

ax_03.plot(x, mean, color = 'purple', lw=2)
ax_03.fill_between(x, lb, ub, alpha = 0.2, color='purple', lw=2)

ax_03.set_ylabel('Maximum constraint value')
ax_03.set_xlabel('Iteration')


Y_f = np.copy(Y)
C_f = np.copy(C)
Y_f[np.where(C_f > 0)[0], np.where(C_f > 0)[1]] = 0
Y_f[np.where(C_f > 0)[0], np.where(C_f > 0)[1]] = np.amin(Y_f) - .5

for j in range(Y_f.shape[0]):
    for i in range(1, Y_f.shape[1]):
        if Y_f[j, i] <= Y_f[j, i-1]:
            Y_f[j, i] = Y_f[j, i-1]

##########
# Convergence plot: obj - best feasible
fig_04 = plt.figure()
ax_04 = plt.gca()

ax_04.set_title('Best Obj value at each iteration - focus on improvement')  

for yy, cc in zip(Y_f, C_f):
    xx = np.linspace(1, len(yy), len(yy))
    CLR = ['green' if np.all(c < 0) else 'red' for c in cc]
    ALPHA = [1.0 if np.all(c < 0) else 0.25 for c in cc]
    
    for j in range(len(yy)-1):
        x = [xx[j], xx[j+1]]
        y = [yy[j], yy[j+1]]
        clr = CLR[j]
        alpha = ALPHA[j]
        ax_04.plot(x,y, color=clr, alpha=alpha)
    ax_04.scatter(xx, yy, color=CLR, alpha=ALPHA)
    
ax_04.set_ylabel('Objective funtion')
ax_04.set_xlabel('Iteration')

##########
# Convergence plot: obj - best feasible - median and interval
fig_05 = plt.figure()
ax_05 = plt.gca()  

ax_05.set_title('Objective Function (maximization 2d Ackley fcn): median and 0.9-0.1 quantiles')  

lower_quantiles = 0.1
upper_quantiles = 0.9
mean = np.quantile(Y_f, 0.5, axis = 0)
lb = np.quantile(Y_f, lower_quantiles, axis = 0)
ub = np.quantile(Y_f, upper_quantiles, axis = 0)

x = np.linspace(1, len(mean), len(mean))

ax_05.plot(x, mean, color = 'darkorange', lw=2)
ax_05.fill_between(x, lb, ub, alpha = 0.2, color='darkorange', lw=2)

ax_05.set_ylabel('Objective funtion')
ax_05.set_xlabel('Iteration')

##########
# All samples evaluated
Y_all = []
C_all = []
for state in states:
    y_all = []
    c_all = []
    for event in state.history:
        for y, c in zip(event['batch']['Y'], event['batch']['C']):
            y_all.append(y.cpu().numpy())
            c_all.append(np.amax(c.cpu().numpy()))
            
    Y_all.append(y_all)
    C_all.append(c_all)


fig_06 = plt.figure()
ax_06 = plt.gca()

ax_06.set_title('Obj function of all samples evaluated')

for y, c in zip(Y_all, C_all):
    
    clr = []
    alpha = []
    x = np.linspace(1, len(y), len(y))
    for i in range(len(y)):
        
        if c[i] > 0:
            clr.append('red')
            alpha.append(0.25)
        else:
            clr.append('green')
            alpha.append(1.0)
        
        if not i == len(y)-1:
            ax_06.plot([x[i], x[i+1]], [y[i], y[i+1]], color=clr[-1], alpha=alpha[-1])
    
    ax_06.scatter(x, y, color=clr, alpha=alpha)
    
ax_06.set_ylabel('Objective funtion')
ax_06.set_xlabel('Evaluations')
        
##########
# number of restarts
seeds = []

for state in states:
    seeds.append([event['seed'] for event in state.history])
    
seeds = np.array(seeds)

seeds_change = []
for seed in seeds:
    seed_change = [0]
    change_counter = 0
    for i in range(1, len(seed)):
        if not seed[i-1] == seed[i]:
            change_counter += 1
        seed_change.append(change_counter)
    seeds_change.append(seed_change)
    
seeds_change = np.array(seeds_change)
    
fig_07 = plt.figure()
ax_07 = plt.gca()

ax_07.set_title('Number of restarts')
for y in seeds_change:
    x = np.linspace(1, len(y), len(y))
    ax_07.plot(x, y)
    
    