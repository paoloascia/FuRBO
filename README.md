Feasible trust Region Bayesian Optimization (FuRBO) is a Bayesian optimization for high dimensional and highly constrained problems.

*This repository is intended to showcase the performance of FuRBO and to educate how to use/set up the method in Python*

## General information
FuRBO is a Bayesian optimization for high-dimensional black-box functions under black-box constraints. The method we propose uses trust regions to reduce the search space to only the area where all constraints are fulfilled (i.e., the feasible area). To do so, the algorithm relies on approximating the black-box objective function and constraints with Gaussian process regression to estimate the location of the feasible area with the best objective function. The trust region is placed in the estimated feasible region.

#### Workflow
1. Generate initial samples
2. Fit Gaussian processes to the objective function and constraints
3. Find the current optimum:
	- if a feasible sample is evaluated: feasible point with the best objective function
	- if no feasible sample is evaluated: point with the smallest violation
4. Sample with a multinormal distribution the Gaussian processes
5. Define trust region around best points according to estimated objective value and constraint violation
6. Draw a Thompson sample in the trust region and estimate the next optimum candidate according to objective value and constraint violation
7. Evaluate the next optimum candidate
8. Update multinormal distribution parameters:
	- if the last n_f optimum candidates are feasible and improve objective -> enlarge multinormal distribution
	- if the last n_f optimum candidates are infeasible or do not improve objective -> shrink multinormal distribution
9. Repeat steps 2 - 8 until the stopping criterion is met

*For further details, a paper will soon be published.*
