# Function for plotting FuRBO status
# 
# March 2024
##########
# Imports
from botorch.utils.transforms import unnormalize

import matplotlib.pyplot as plt
import torch

def constraints_2d(state, no_save=False, cbar_trigger=True, **tkwargs):
    
    # Initiate figure
    title = 'Iteration '+str(state.it_counter - 1) + '\n Constraint model'
    fig = plt.figure('Number of Samples: '+str(state.X.shape[0])+title)
    ax = plt.gca()
    
    # Create mesh for contour plot
    resolution = 100
    x = torch.linspace(0, 1, resolution, **tkwargs)
    y = torch.linspace(0, 1, resolution, **tkwargs)
    X, Y = torch.meshgrid(x, y)
    
    # Reshape samples for function evaliation
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    xx = torch.transpose(torch.stack([X, Y]), 0, 1)
    
    # Evaluate the function at each combination of x and y
    state.C_model.eval()
    with torch.no_grad():
        posterior = state.C_model.posterior(xx)
        samples_cc = posterior.mean
    
    # Unnormalize xx
    X = unnormalize(X, [state.lb[0], state.ub[0]])
    Y = unnormalize(Y, [state.lb[1], state.ub[1]])
    Z = torch.max(samples_cc, dim=1).values
    
    # Reshape for plotting
    X, Y, Z = X.reshape((resolution, resolution)), Y.reshape((resolution, resolution)), Z.reshape((resolution, resolution))
    
    if tkwargs['device'].type == 'cuda':
        X, Y, Z = X.cpu(), Y.cpu(), Z.cpu()
        
    # Transform to numpy for plotting
    X, Y, Z = X.numpy(), Y.numpy(), Z.numpy()   
    
    # Create a contour plot
    # ax.contour(X, Y, Z, levels=20, cmap='viridis')  # Use plt.contourf for filled contours
    
    plt.title(title)
    
    # Create a contour plot
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        
    # Add trust regions
    color = ['blue', 'green', 'red', 'black']
    for ub, lb, c in zip(unnormalize(state.tr_ub, [state.lb, state.ub]), unnormalize(state.tr_lb, [state.lb, state.ub]), color):
        if tkwargs['device'].type == 'cuda':
            x_tr = torch.tensor([lb[0], lb[0], ub[0], ub[0], lb[0]]).cpu().numpy()
            y_tr = torch.tensor([lb[1], ub[1], ub[1], lb[1], lb[1]]).cpu().numpy()
        else:
            x_tr = torch.tensor([lb[0], lb[0], ub[0], ub[0], lb[0]]).numpy()
            y_tr = torch.tensor([lb[1], ub[1], ub[1], lb[1], lb[1]]).numpy()
        ax.plot(x_tr, y_tr, color=c)


    # Add colorbar
    if cbar_trigger:
        cbar = plt.colorbar(contour, ax=ax)
    
    # Add current best point
    if tkwargs['device'].type == 'cuda':
        x = unnormalize(state.batch_X[:,0], [state.lb[0], state.ub[0]]).cpu().numpy()
        y = unnormalize(state.batch_X[:,1], [state.lb[1], state.ub[1]]).cpu().numpy()
    else:
        x = unnormalize(state.batch_X[:,0], [state.lb[0], state.ub[0]]).numpy()
        y = unnormalize(state.batch_X[:,1], [state.lb[1], state.ub[1]]).numpy()
    ax.scatter(x, y, color='lime')  
    
    if no_save:
        return fig, ax
    
    # Save figure
    fig.savefig('It_'+str(state.it_counter - 1) + '_Cons' + '.png', format='png')
    
    return fig, ax
    
def constraints_2d_samples(X, state, no_save=False, **tkwargs):
    
    fig, ax = constraints_2d(state, no_save=True, cbar_trigger=False, **tkwargs)
    
    # Add samples on constraints
    if tkwargs['device'].type == 'cuda':
        x = unnormalize(X[:,0], [state.lb[0], state.ub[0]]).cpu().numpy()
        y = unnormalize(X[:,1], [state.lb[1], state.ub[1]]).cpu().numpy()
    else:
        x = unnormalize(X[:,0], [state.lb[0], state.ub[0]]).numpy()
        y = unnormalize(X[:,1], [state.lb[1], state.ub[1]]).numpy()
    ax.scatter(x, y, color='r', marker = 'x')  
    
    if no_save:
        return fig, ax
    
    fig.savefig('It_'+str(state.it_counter - 1) + '_Cons_Sampling' + '.png', format='png')
    return fig, ax

def objective_2d(state, no_save=False, cbar_trigger=True, **tkwargs):
    
    # Initiate figure
    title = 'Iteration '+str(state.it_counter - 1) + '\n Objective model'
    fig = plt.figure('Number of Samples: '+str(state.X.shape[0])+title)
    ax = plt.gca()
    
    # Plot contour plot of the function
    resolution = 100

    # Create a meshgrid from x and y
    x = torch.linspace(0, 1, resolution, **tkwargs)
    y = torch.linspace(0, 1, resolution, **tkwargs)
    X, Y = torch.meshgrid(x, y)
    
    # Reshape samples for function evaliation
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    xx = torch.transpose(torch.stack([X, Y]), 0, 1)
    
    # Evaluate the objective function at each combination of x and y
    Z = torch.tensor([state.obj.eval_(x_) for x_ in xx], **tkwargs).unsqueeze(-1)
    
    # Evaluate constraint functions at each combination of x and y
    C = []
    for c in state.cons:
        C.append(torch.tensor([c.eval_(x_, state.lb, state.ub) for x_ in xx], **tkwargs).unsqueeze(-1))
    
    # Unnormalize xx
    X = unnormalize(X, [state.lb[0], state.ub[0]])
    Y = unnormalize(Y, [state.lb[1], state.ub[1]])
    
    # Reshape for plotting
    X, Y = X.reshape((resolution, resolution)), Y.reshape((resolution, resolution))
    Z = Z.reshape((resolution, resolution))
    
    for i in range(len(C)):
        C[i] = C[i].reshape((resolution, resolution))
    
    if tkwargs['device'].type == 'cuda':
        X, Y = X.cpu(), Y.cpu()
        Z = Z.cpu()
        for i in range(len(C)):
            C[i] = C[i].cpu()
        
    # Transform to numpy for plotting
    X, Y = X.numpy(), Y.numpy() 
    Z = Z.numpy()
    for i in range(len(C)):
        C[i] = C[i].numpy()

    # Plot constraints
    for i in range(len(C)):
        ax.contourf(X, Y, C[i], levels=[0.0001, 1000], colors = 'blue', alpha = 0.2)
    
    # Create a contour plot
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')  # Use plt.contourf for filled contours
    
    # Add labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.title(title)
    
    # Add trust regions
    color = ['blue', 'green', 'red', 'black']
    for ub, lb, c in zip(unnormalize(state.tr_ub, [state.lb, state.ub]), unnormalize(state.tr_lb, [state.lb, state.ub]), color):
        if tkwargs['device'].type == 'cuda':
            x_tr = torch.tensor([lb[0], lb[0], ub[0], ub[0], lb[0]]).cpu().numpy()
            y_tr = torch.tensor([lb[1], ub[1], ub[1], lb[1], lb[1]]).cpu().numpy()
        else:
            x_tr = torch.tensor([lb[0], lb[0], ub[0], ub[0], lb[0]]).numpy()
            y_tr = torch.tensor([lb[1], ub[1], ub[1], lb[1], lb[1]]).numpy()
        ax.plot(x_tr, y_tr, color=c)


    # Add colorbar
    if cbar_trigger:
        cbar = plt.colorbar(contour, ax=ax)
    
    # Add current best point
    if tkwargs['device'].type == 'cuda':
        x = unnormalize(state.batch_X[:,0], [state.lb[0], state.ub[0]]).cpu().numpy()
        y = unnormalize(state.batch_X[:,1], [state.lb[1], state.ub[1]]).cpu().numpy()
    else:
        x = unnormalize(state.batch_X[:,0], [state.lb[0], state.ub[0]]).numpy()
        y = unnormalize(state.batch_X[:,1], [state.lb[1], state.ub[1]]).numpy()
    ax.scatter(x, y, color='lime')  
    
    if no_save:
        return fig, ax
    
    fig.savefig('It_'+str(state.it_counter - 1) + '_Obj' + '.png', format='png')
    return fig, ax
    