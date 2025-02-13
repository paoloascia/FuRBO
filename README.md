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

## Requirements
The following libraries are required to set up and run FuRBO. The algorithm has been tested only with the version of the libraries listed below.
- botorch (0.10.0)
- gpytorch (1.11)
- matplotlib (3.8.4)
- numpy (1.24.3)
- pytorch (2.3.0)

## Repository Structure
The repository is structured as follows:
```
└───FuRBO
    |   └───`FuRBO_restart.py`: Main optimization loop with restarts
    |   └───`FuRBO_single.py`: Main optimization loop without restarts
    |   └───fcn
    |       | └───`samplingStrategies.py`: script with all sampling strategies used during the optimization
    |       | └───`states.py`: script with the classes to hold and update the main information needed for the optimization
    |       | └───`stoppingNrestartCriterion.py`: script with stopping and restarting criteria
    |       | └───`trustRegionUpdate.py`: script to define the trust region
    |       | └───`utilities.py`: script with small utility functions
└───Tutorials
    |       └───`FuRBO_restart.ipynb`: Jupiter Notebook on how to set up FuRBO with restarts
    |       └───`FuRBO_single.ipynb`: Jupiter Notebook on how to set up FuRBO without restarts
└───Benchmarks
    |        └───SCBO: comparison between FuRBO and SCBO on the BBOB-Constrained benchmark
└───Figures: figures used throughout the git
```

## Examples and Benchmarking
- For examples on how to set up the FuRBO optimization loop, please refer to the Jupyter notebooks in the Tutorial folder. In this folder, the following examples are available:
	- Maximization of the 10D Ackley function under two easy constraint functions without restarts
	- Maximization of the 10D Ackley function under two easy constraint functions with restarts

- To understand how FuRBO performs, please refer to the benchmarks available. Currently, the FuRBO has been compared with
	- Scalable Constrained Bayesian Optimization (SCBO) [1]

[1]: David Eriksson and Matthias Poloczek. Scalable constrained Bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 730–738. PMLR, 2021. doi: [10.48550/arxiv.2002.08526](https://doi.org/10.48550/arxiv.2002.08526).

## Cite us
Coming soon
