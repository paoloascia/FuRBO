# FuRBO sampling strategies
# 
# March 2024
##########
# Imports
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

import torch

##########
# Custom imports

def get_initial_points_sobol(FuRBO,
                       **tkwargs):
    X_init = FuRBO.sobol.draw(n=FuRBO.n_init).to(**tkwargs)
    return X_init

def generate_batch_focus_on_feasibility(
    state,
    n_candidates,
    **tkwargs
):
    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))

    # Initialize tensor with samples to evaluate
    X_next = torch.ones((state.batch_size*state.tr_number, state.dim), **tkwargs)
    
    # Iterate over the several trust regions
    for i in range(state.tr_number):
        tr_lb = state.tr_lb[i]
        tr_ub = state.tr_ub[i]

        # Thompson Sampling w/ Constraints (like SCBO)
        pert = state.sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / state.dim, 1.0)
        mask = torch.rand(n_candidates, state.dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, state.dim - 1, size=(len(ind),), device=tkwargs['device'])] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = state.best_X[i].expand(n_candidates, state.dim).clone()
        X_cand[mask] = pert[mask]
        
        # If a feasible point has been identified:
        if torch.any(torch.max(state.C, dim=1).values <= 0):
            # Sample on the candidate points using Constrained Max Posterior Sampling
            constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
                model=state.Y_model, constraint_model=state.C_model, replacement=False
                )
            with torch.no_grad():
                X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constrained_thompson_sampling(X_cand, num_samples=state.batch_size)
        
        else:
            # Sample to minimize violation
            
            # First combine the constraints to train one surrogate
        
        # obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        # if self.replacement:
        #     # if we allow replacement then things are simple(r)
        #     idcs = torch.argmax(obj, dim=-1)
        # else:
        #     # if we need to deduplicate we have to do some tensor acrobatics
        #     # first we get the indices associated w/ the num_samples top samples
        #     _, idcs_full = torch.topk(obj, num_samples, dim=-1)
        #     # generate some indices to smartly index into the lower triangle of
        #     # idcs_full (broadcasting across batch dimensions)
        #     ridx, cindx = torch.tril_indices(num_samples, num_samples)
        #     # pick the unique indices in order - since we look at the lower triangle
        #     # of the index matrix and we don't sort, this achieves deduplication
        #     sub_idcs = idcs_full[ridx, ..., cindx]
        #     if sub_idcs.ndim == 1:
        #         idcs = _flip_sub_unique(sub_idcs, num_samples)
        #     elif sub_idcs.ndim == 2:
        #         # TODO: Find a better way to do this
        #         n_b = sub_idcs.size(-1)
        #         idcs = torch.stack(
        #             [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
        #             dim=-1,
        #         )
        #     else:
        #         # TODO: Find a general way to do this efficiently.
        #         raise NotImplementedError(
        #             "MaxPosteriorSampling without replacement for more than a single "
        #             "batch dimension is not yet implemented."
        #         )
        # # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # # to have shape batch_shape x num_samples
        # if idcs.ndim > 1:
        #     idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # # in order to use gather, we need to repeat the index tensor d times
        # idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # # now if the model is batched batch_shape will not necessarily be the
        # # batch_shape of X, so we expand X to the proper shape
        # Xe = X.expand(*obj.shape[1:], X.size(-1))
        # # finally we can gather along the N dimension
        # return torch.gather(Xe, -2, idcs)

            state.C_model.eval()
            with torch.no_grad():
                posterior = state.C_model.posterior(X_cand)
                C_cand = posterior.rsample(sample_shape=torch.Size([state.batch_size]))
                
            # Normalize
            for j in range(torch.abs(C_cand).max(dim=1).values.shape[0]):
                for k in range(torch.abs(C_cand).max(dim=1).values.shape[1]):
                    C_cand[j,:,k] /= torch.abs(C_cand).max(dim=1).values[j,k]
            # C /= torch.abs(C).max(dim=0).values
            
            # Combine into one tensor
            # C_cand = -1 * C_cand.max(dim=1).values
            C_cand[C_cand<0] = 0
            C_cand = -1 * C_cand.sum(dim=2)
            # C = -1 * C.sum(dim=1)
            # C = -1 * C.max(dim=1).values
            
            # Reshape to (-1, 1)
            # C_cand = C_cand.view(-1, 1)
            # C = C.view(-1, 1)
            
            # Train one model on the combination
            # constraint_model_united = get_fitted_model(X_cand, C_cand, dim=state.dim, max_cholesky_size = float("inf"))
            # constraint_model_united = get_fitted_model(state.X, C, dim=state.dim, max_cholesky_size = float("inf"))
            
            # Sample the candidate points (taken from MaxPosteriorSampling, adapted for multiple posteriors)
            _, idcs_full = torch.topk(C_cand, state.batch_size, dim=-1)
            ridx, cindx = torch.tril_indices(state.batch_size, state.batch_size)
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, state.batch_size)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                        [_flip_sub_unique(sub_idcs[:, i], state.batch_size) for i in range(n_b)],
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
            idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X_cand.size(-1))
            # now if the model is batched batch_shape will not necessarily be the
            # batch_shape of X, so we expand X to the proper shape
            Xe = X_cand.expand(*C_cand.shape[1:], X_cand.size(-1))
            # finally we can gather along the N dimension
            X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = torch.gather(Xe, -2, idcs)
            
            # Debugging
            # print(idcs)
            
            # constraint_sampling = MaxPosteriorSampling(
            #     model=constraint_model_united, replacement=False)
            # with torch.no_grad():
            #     X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constraint_sampling(X_cand, num_samples=state.batch_size)
                
            # X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = X_cand[idcs]

    return X_next

def generate_batch_thompson_sampling(
    state,
    n_candidates,
    **tkwargs
):
    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))

    # Initialize tensor with samples to evaluate
    X_next = torch.ones((state.batch_size*state.tr_number, state.dim), **tkwargs)
    
    # Iterate over the several trust regions
    for i in range(state.tr_number):
        tr_lb = state.tr_lb[i]
        tr_ub = state.tr_ub[i]

        # Thompson Sampling w/ Constraints (like SCBO)
        pert = state.sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / state.dim, 1.0)
        mask = torch.rand(n_candidates, state.dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, state.dim - 1, size=(len(ind),), device=tkwargs['device'])] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = state.best_X[i].expand(n_candidates, state.dim).clone()
        X_cand[mask] = pert[mask]
        
        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=state.Y_model, constraint_model=state.C_model, replacement=False
            )
        with torch.no_grad():
            X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constrained_thompson_sampling(X_cand, num_samples=state.batch_size)
        
        # print(X_next)
    return X_next

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
    
    
    
