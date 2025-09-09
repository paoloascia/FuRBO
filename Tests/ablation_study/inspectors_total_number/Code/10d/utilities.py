# FuRBO utilities
# 
# March 2024
##########
# Imports
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from torch.quasirandom import SobolEngine

import torch
from torch import Tensor

import numpy as np

from scipy.stats import invgauss
from scipy.stats import ecdf

def get_initial_points(FuRBO, **tkwargs):
    X_init = FuRBO.sobol.draw(n=FuRBO.n_inits).to(**tkwargs)
    return X_init

def get_fitted_model(X, Y, dim, max_cholesky_size):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll, 
                         optimizer_kwargs={'method': 'L-BFGS-B'})

    return model

def multivariate_distribution(centre, 
                              n_samples,
                              lb = None,
                              ub = None,
                              **tkwargs):
    dim = centre.shape[0]
    
    # Generate a multivariate normal distribution centered at 0
    multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(centre, 0.025*torch.eye(dim, **tkwargs))

    # Draw samples torch.distributions.multivariate_normal import MultivariateNormal
    samples = multivariate_normal.sample(sample_shape=torch.Size([n_samples]))
    
    for dim in range(len(lb)):
        samples = samples[torch.where(samples[:,dim]>=lb[dim])]
        samples = samples[torch.where(samples[:,dim]<=ub[dim])]
    
    return samples

def get_best_index_for_batch(n_tr, Y: Tensor, C: Tensor):
    """Return the index for the best point. One for each trust region."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return torch.topk(score.reshape(-1), k=n_tr).indices
    return torch.topk(C.clamp(min=0).sum(dim=-1), k=n_tr, largest=False).indices # Return smallest violation
    
def gaussian_copula(y, **tkwargs):
    
    # Define percentiles
    shape = y.shape
    y = y.reshape(-1).cpu().numpy()
    res = ecdf(y)
    p = res.cdf.probabilities
    
    # Do not allow p=1 -> yields +inf
    p[p==1.0] = 0.99
    
    # Inverse gaussian
    inv = invgauss.ppf(p, 0.5)
    
    y = ((inv-np.amin(inv))*(np.amax(y)-np.amin(y))/(np.amax(inv)-np.amin(inv)))+np.amin(y)
    
    # Scale to range of y
    return torch.tensor(y, **tkwargs).reshape((shape[0], shape[1]))
    
def scaling_factor(y):
    
    sf = 10000
    
    return sf * y
    
def bilog(y):
    
    return torch.sign(y) * torch.log(1 + torch.abs(y))

def no_scaling(y):
    
    return y

def multivariate_circular(centre,
                          radius,
                          n_samples,
                          lb = None,
                          ub = None,
                          **tkwargs):
    
    dim = centre.shape[0]
    
    # Generate a multivariate normal distribution centered at 0
    multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim, **tkwargs), 0.025*torch.eye(dim, **tkwargs))
    
    #  Draw samples torch.distributions.multivariate_normal import MultivariateNormal
    samples = multivariate_normal.sample(sample_shape=torch.Size([n_samples]))
    
    # Normalize each sample to have unit norm, then scale by the radius
    norms = torch.norm(samples, dim=1, keepdim=True)  # Euclidean norms
    normalized_samples = samples / norms  # Normalize to unit hypersphere
    scaled_samples = normalized_samples * torch.rand(n_samples, 1, **tkwargs) * radius  # Scale by random factor within radius
    
    # Translate samples to be centered at centre
    samples = scaled_samples + centre
    
    
    # Trim samples outside domain
    for dim in range(len(lb)):
        samples = samples[torch.where(samples[:,dim]>=lb[dim])]
        samples = samples[torch.where(samples[:,dim]<=ub[dim])]
    
    return samples

def multivariate_circular_two(centre,
                          radius,
                          n_samples,
                          lb = None,
                          ub = None,
                          **tkwargs):
    
    dim = centre.shape[0]
    
    # Generate a multivariate normal distribution centered at 0
    multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(centre, radius**2*torch.eye(dim, **tkwargs))

    # Draw samples torch.distributions.multivariate_normal import MultivariateNormal
    samples = multivariate_normal.sample(sample_shape=torch.Size([n_samples]))
    
    for dim in range(len(lb)):
        samples = samples[torch.where(samples[:,dim]>=lb[dim])]
        samples = samples[torch.where(samples[:,dim]<=ub[dim])]
    
    return samples

