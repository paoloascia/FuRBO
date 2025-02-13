# FuRBO utilities
# 
# March 2024
##########
# Imports
import gpytorch
import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from scipy.stats import invgauss
from scipy.stats import ecdf

from torch import Tensor

##########
# GPR utilities
def get_fitted_model(X,
                     Y,
                     dim,
                     max_cholesky_size):
    '''Function to fit a GPR to a given set of data.
    For reference, see https://botorch.org/docs/tutorials/scalable_constrained_bo/'''
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

##########
# Utilities
def get_best_index_for_batch(n_tr, Y: Tensor, C: Tensor):
    """Return the index for the best point. One for each trust region.
    For reference, see https://botorch.org/docs/tutorials/scalable_constrained_bo/"""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return torch.topk(score.reshape(-1), k=n_tr).indices
    return torch.topk(C.clamp(min=0).sum(dim=-1), k=n_tr, largest=False).indices # Return smallest violation
    
##########
# Scaling functions
def gaussian_copula(y, **tkwargs):
    '''Function to scale given values with a Gaussian copula.'''
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
    
def scaling_factor(y, sf = 10000):
    '''Function to scale given values by a fixed value.'''
    return sf * y
    
def bilog(y):
    '''Function to scale given values with a bilog scale.'''
    return torch.sign(y) * torch.log(1 + torch.abs(y))

def no_scaling(y):
    '''Function to return no scaling.'''
    return y

##########
# Mulivariate distribution function
def multivariate_circular(centre,       # Centre of the multivariate distribution
                          radius,       # Radius of the multivariate distribution
                          n_samples,    # Number of samples to evaluate
                          lb = None,    # Domain lower bound
                          ub = None,    # Domain upper bound
                          **tkwargs):
    '''Function to generate multivariate distribution of given radius and centre within a given domain.'''
    # Dimension of the design domain
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

def multivariate_distribution(centre,       # Centre of the multivariate distribution
                              n_samples,    # Number of samples to evaluate
                              lb = None,    # Domain lower bound
                              ub = None,    # Domain upper bound
                              **tkwargs):
    '''Function to generate multivariate distribution of given centre within a given domain.'''
    # Dimension of the design domain
    dim = centre.shape[0]
    
    # Generate a multivariate normal distribution centered at 0
    multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(centre, 0.025*torch.eye(dim, **tkwargs))

    # Draw samples torch.distributions.multivariate_normal import MultivariateNormal
    samples = multivariate_normal.sample(sample_shape=torch.Size([n_samples]))
    
    for dim in range(len(lb)):
        samples = samples[torch.where(samples[:,dim]>=lb[dim])]
        samples = samples[torch.where(samples[:,dim]<=ub[dim])]
    
    return samples

