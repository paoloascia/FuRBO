# Full ode for FuRBO
#
# March 2024
##########
# Imports
import cocoex  # experimentation module
import math
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from abc import ABC, abstractmethod
from botorch.acquisition.objective import IdentityMCObjective
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.acquisition.objective import PosteriorTransform
from botorch.generation.utils import _flip_sub_unique

from torch.nn import Module
from torch import Tensor

from typing import Optional, Union

from botorch.utils.transforms import unnormalize

from torch import Tensor
from torch.quasirandom import SobolEngine

##########
# Custom imports
from utilities import get_fitted_model
from utilities import get_best_index_for_batch
from utilities import multivariate_circular

##########
### Modified for evaluating GPs in series and not in parallel

class SamplingStrategy(Module, ABC):
    """Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        self.objective = IdentityMCObjective() if objective is None else objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)


class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained max posterior sampling.

    Posterior sampling where we try to maximize an objective function while
    simulatenously satisfying a set of constraints c1(x) <= 0, c2(x) <= 0,
    ..., cm(x) <= 0 where c1, c2, ..., cm are black-box constraint functions.
    Each constraint function is modeled by a seperate GP model. We follow the
    procedure as described in https://doi.org/10.48550/arxiv.2002.08526.

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(
                model,
                constraint_model=ModelListGP(cmodel1, cmodel2),
            )
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform for the objective
                function (corresponding to `model`).
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel is a GP model for
                one constraint function, or a MultiTaskGP model where each task is one
                constraint function. All constraints are of the form c(x) <= 0. In the
                case when the constraint model predicts that all candidates
                violate constraints, we pick the candidates with minimum violation.
        """
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )

        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.constraint_model = constraint_model

    def _convert_samples_to_scores(self, Y_samples, C_samples) -> Tensor:
        r"""Convert the objective and constraint samples into a score.

        The logic is as follows:
            - If a realization has at least one feasible candidate we use the objective
                value as the score and set all infeasible candidates to -inf.
            - If a realization doesn't have a feasible candidate we set the score to
                the negative total violation of the constraints to incentivize choosing
                the candidate with the smallest constraint violation.

        Args:
            Y_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the objective function.
            C_samples: A `num_samples x batch_shape x num_cand x num_constraints`-dim
                Tensor of samples from the constraints.

        Returns:
            A `num_samples x batch_shape x num_cand x 1`-dim Tensor of scores.
        """
        is_feasible = (C_samples <= 0).all(
            dim=-1
        )  # num_samples x batch_shape x num_cand
        has_feasible_candidate = is_feasible.any(dim=-1)

        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            # Use negative total violation for samples where no candidate is feasible
            total_violation = (
                C_samples[~has_feasible_candidate]
                .clamp(min=0)
                .sum(dim=-1, keepdim=True)
            )
            scores[~has_feasible_candidate] = -total_violation
        return scores

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
                `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
            # Note: `posterior_transform` is only used for the objective
            posterior_transform=self.posterior_transform,
        )
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        C_tmp = []
        for c in self.constraint_model.models:
            c_posterior = c.posterior(
                X=X, observation_noise=observation_noise
                )
            C_tmp.append(c_posterior.rsample(sample_shape=torch.Size([num_samples])))
        
        C_samples = torch.cat(C_tmp, dim=2)
        
        # c_posterior = self.constraint_model.posterior(
        #     X=X, observation_noise=observation_noise
        # )
        # C_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))

        # Convert the objective and constraint samples into a scalar-valued "score"
        scores = self._convert_samples_to_scores(
            Y_samples=Y_samples, C_samples=C_samples
        )
        return self.maximize_samples(X=X, samples=scores, num_samples=num_samples)
    
    
    
###############################################################################
###############################################################################

##########
# Setting general MatPlotLib parameters 
cwd_save = os.path.join(os.getcwd())
matplotlib.use('Agg')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

#########
# Setting up PyTorch
device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

##########
# Opening optimization data

##########
# Selecting bbob function
# Define COCO input
suite_name = "bbob-constrained"
suite = cocoex.Suite(suite_name, "", "")
# Select p.id = bbob-constrained_f035_i01_d02
p = suite[510]

##########
# Plot of cons{tilde} + DoE + Best
# Initiate plot
fig = plt.figure(figsize = (6,6), 
                 dpi = 600)
ax = plt.gca()
    
# Plot contour plot of the function
resolution = 50
    
# Create a meshgrid from x and y
X, Y = torch.meshgrid(torch.linspace(0, 1, resolution), torch.linspace(0, 1, resolution), indexing="ij")
grid_x = torch.stack([X.flatten(), Y.flatten()], dim=-1)

# Train surrogate
seed = 24
sobolSampler = SobolEngine(dimension=p.dimension, scramble=True, seed=seed)
X_train = sobolSampler.draw(n=3 * p.dimension)
Y_train = Tensor([p(unnormalize(x_, [p.lower_bounds[0], p.upper_bounds[0]])) for x_ in X_train]).unsqueeze(-1)
C_train = Tensor([torch.amax(Tensor(p.constraint(unnormalize(x_, [p.lower_bounds[0], p.upper_bounds[0]])))) for x_ in X_train]).unsqueeze(-1)
Y_model = get_fitted_model(X_train, Y_train, p.dimension, max_cholesky_size = float("inf"))
C_model = get_fitted_model(X_train, C_train, p.dimension, max_cholesky_size = float("inf"))

# Sample surrogate
C_model.eval()
with torch.no_grad():
    C = C_model.posterior(grid_x)
    C = C.mean.view(resolution, resolution)

# Unnormalize xx
X = unnormalize(X, [p.lower_bounds[0], p.upper_bounds[0]])
Y = unnormalize(Y, [p.lower_bounds[1], p.upper_bounds[1]])
          
X, Y, C = X.cpu().numpy(), Y.cpu().numpy(), C.cpu().numpy()
                    
# Add multinormal distribution sampling
lb = torch.zeros(p.dimension, **tkwargs)
ub = torch.ones(p.dimension, **tkwargs)

X_best = X_train[get_best_index_for_batch(1, Y_train, C_train)]
torch.manual_seed(1)
samples = multivariate_circular(X_best[0], 0.5, 100 * p.dimension, lb=lb, ub=ub, **tkwargs).to(torch.float32)

# Identify top 10% of the samples
# Evaluate samples on the models of the objective -> yy Tensor
Y_model.eval()
with torch.no_grad():
    posterior = Y_model.posterior(samples)
    samples_yy = posterior.mean.squeeze()
        
# Evaluate samples on the models of the constraints -> yy Tensor
C_model.eval()
with torch.no_grad():
    posterior = C_model.posterior(samples)
    samples_cc = posterior.mean
        
# Combine the constraints values
# Normalize
samples_cc /= torch.abs(samples_cc).max(dim=0).values
samples_cc = torch.max(samples_cc, dim=1).values
        
# Take the best 10% of the drawn samples to define the trust region
n_samples = 100 * p.dimension
n_samples_tr = int(n_samples * 0.2)

# Order the samples for feasibility and for best objective
if torch.any(samples_cc < 0):
    
    feasible_samples_id = torch.where(samples_cc <= 0)[0]
    infeasible_samples_id = torch.where(samples_cc > 0)[0]
    
    feasible_cc = samples_yy[feasible_samples_id]
    infeasible_cc = samples_cc[infeasible_samples_id]
    
    feasible_sorted, feasible_sorted_id = torch.sort(feasible_cc)
    infeasible_sorted, infeasible_sorted_id = torch.sort(infeasible_cc)
    
    original_feasible_sorted_indices = feasible_samples_id[feasible_sorted_id]
    original_infeasible_sorted_indices = infeasible_samples_id[infeasible_sorted_id]
    
    top_indices = torch.cat((original_feasible_sorted_indices, original_infeasible_sorted_indices))[:n_samples_tr]
    
else:
    if n_samples_tr > len(samples_cc):
        n_samples_tr = len(samples_cc)
        
    if n_samples_tr < 4:
        n_samples_tr = 4
                
    top_values, top_indices = torch.topk(samples_cc, n_samples_tr, largest=False)
   
# Saving best samples
best_samples = samples[top_indices]

# Create a contour plot
contour = ax.contourf(X, Y, C, levels=10, cmap='viridis')  # Use plt.contourf for filled contours

# Add trust region
lower_bound = unnormalize(torch.min(samples[top_indices], dim=0).values.cpu().numpy(), [p.lower_bounds[0], p.upper_bounds[0]])
upper_bound = unnormalize(torch.max(samples[top_indices], dim=0).values.cpu().numpy(), [p.lower_bounds[0], p.upper_bounds[0]])
ax.plot([lower_bound[0], upper_bound[0]], [lower_bound[1], lower_bound[1]], color = 'r')
ax.plot([lower_bound[0], upper_bound[0]], [upper_bound[1], upper_bound[1]], color = 'r')
ax.plot([lower_bound[0], lower_bound[0]], [lower_bound[1], upper_bound[1]], color = 'r')
ax.plot([upper_bound[0], upper_bound[0]], [lower_bound[1], upper_bound[1]], color = 'r')

# Compute next sample
# Thompson Sampling w/ Constraints (like SCBO)
sobol = SobolEngine(dimension=p.dimension, scramble=True, seed = 24)
pert = sobol.draw(2000).to(**tkwargs)
pert = Tensor(lower_bound) + (Tensor(upper_bound) - Tensor(lower_bound)) * pert
pert = pert.to(torch.float32)

# Create a perturbation mask
prob_perturb = min(20.0 / p.dimension, 1.0)
mask = torch.rand(2000, p.dimension, **tkwargs) <= prob_perturb
ind = torch.where(mask.sum(dim=1) == 0)[0]
mask[ind, torch.randint(0, p.dimension - 1, size=(len(ind),), device=tkwargs['device'])] = 1

# Create candidate points from the perturbations and the mask
X_cand = X_best[0].expand(2000, p.dimension).clone()
X_cand[mask] = pert[mask]
        
# Sample on the candidate points using Constrained Max Posterior Sampling
constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
    model=Y_model, constraint_model=ModelListGP(C_model), replacement=False)
with torch.no_grad():
    X_next = constrained_thompson_sampling(X_cand, num_samples=1).cpu().numpy()[0]
    
ax.scatter(X_next[0], X_next[1], color = 'orange')
     
# Add labels and title
# ax.set_xlabel('X-axis')
ax.set_xticks([])
# ax.set_ylabel('Y-axis')
ax.set_yticks([])
# ax.set_title("bbob-constrained_f035_i01_d02\n"
#              "Constraints GPR and samples")
        
# Add colorbar
# cbar = plt.colorbar(contour, ax=ax)

# Save figure
fig.savefig(os.path.join(cwd_save, '2DconTnextTR' + '.png'))

# Close figure
plt.close(fig)

