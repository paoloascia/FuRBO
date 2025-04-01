# FuRBO utilities
# 
# March 2024
##########
# Imports
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

import numpy as np

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from scipy.stats import invgauss
from scipy.stats import ecdf

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from typing import Optional

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


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()
