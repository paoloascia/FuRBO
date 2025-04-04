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

![alt text](https://github.com/paoloascia/FuRBO/blob/main/Figures/workflow/graphical_abstract_furbo.png)

## Requirements
The following libraries are required to set up and run FuRBO. The algorithm has been tested only with the version of the libraries listed below.
- botorch (0.10.0)
- gpytorch (1.11)
- matplotlib (3.8.4)
- numpy (1.24.3)
- pytorch (2.3.0)

## How to run
To run the optimization, download the repository from [here](https://anonymous.4open.science/api/repo/FuRBO/zip) FuRBO and run `FuRBO_restart.py` to optimize with restarts or `FuRBO_single.py` to optimize without restarts.

1) Create a virtual environment
```bash
conda create -n FuRBO python=3.10
conda activate FuRBO
```
2) Extract the repository and install the required packages
```bash
cd FuRBO_repo
pip install -r requirements.txt
```
3) Run the optimization loop
```bash
cd FuRBO
python FuRBO_restart.py
```
or to run without restarts:
```bash
cd FuRBO
python FuRBO_single.py
```

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
└───Tests
    |   └───ablation_study: folder with the raw data to plot the ablation study performed
    |       |             └───batch_size: folder with raw data for the ablation study on the batch size. The following batches are evaluated: q=1, 1D, 2D, 3D, 4D, 5D
    |       |             └───doe_size: folder with raw data for the ablation study on the initial sample set size. The following sizes are evaluated: doe=1D, 3D, 5D, 10D
    |       |             └───inspector_percentage: folder with raw data for the ablation study on the percentage of inspectors used to define the trust region. The following percentages are evaluated: p=0.01, 0.05, 0.1, 0.2
    |   └───across-algorithm-performance: folder with the raw data to plot the comparison between FuRBO and other common constrained optimization algorithms
    |   └───bbob-constrained-suite: folder containing all the raw data to assess the performance of FuRBO against SCBO on the bbob-constrained benchmark suite.
└───Figures: figures used throughout the git
```

## Examples and Benchmarking
- For examples of how to set up the FuRBO optimization loop, please refer to the Jupyter notebooks in the Tutorial folder. In this folder, the following examples are available:
	- Maximization of the 10D Ackley function under two easy constraint functions without restarts
	- Maximization of the 10D Ackley function under two easy constraint functions with restarts

- To understand how FuRBO performs, please refer to the tests available. To load and plot the data related, please download the desired folder and unzip the folder named "Experiments". Then, run the python script within the folder. The plot will be saved directly in this folder. The following studies are available:
	- Performance of FuRBO on the bbob-constrained benchmark suite compared to SCBO with batch size 3D (folder: bbob-constrained-suite);
 	- Comparison between a random sampling, COBYLA, CMA-ES, Constrained-EI, SCBO and FuRBO on a sequential optimization on a selection of functions from the bbob-constrained benchmark suite (folder: across-algorithm-performance)
  	- An ablation study on the influence of the following hyper-parameters:
  		- initial sample set size (folder: ablation_study -> doe_size)
  	 	- batch size per iteration (folder: ablation_study -> batch_siye)
  	  	- the percentage of inspectors to define the trust region (folder: ablation_study -> inspectors_percentage)

[1]: David Eriksson and Matthias Poloczek. Scalable constrained Bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 730–738. PMLR, 2021. doi: [10.48550/arxiv.2002.08526](https://doi.org/10.48550/arxiv.2002.08526).

## Cite us
Coming soon
