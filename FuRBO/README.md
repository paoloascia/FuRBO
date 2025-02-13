## Scripts to run FuRBO
This folder contains all scripts for the optimization loop with and without restarts.

### Folder structure
```
└───`FuRBO_restart.py`: Main optimization loop with restarts
└───`FuRBO_single.py`: Main optimization loop without restarts
└───fcn
    | └───`samplingStrategies.py`: script with all sampling strategies used during the optimization
    |      |                   └───get_initial_points_sobol: function to generate the initial experimental design
    |      |                   └───generate_batch_thompson_sampling: function to find next candidate optimum
    | └───`states.py`: script with the classes to hold and update the main information needed for the optimization
    |      |       └───Furbo_state_single: class to track optimization status without restart
    |      |           |                └───update: function to update optimization status
    |      |       └───Furbo_state_restart: class to track optimization status with restart
    |      |           |                └───update: function to update optimization status
    |      |           |                └───reset_status: function to reset the status for the restart
    | └───`stoppingNrestartCriterion.py`: script with stopping and restarting criteria
    |      |                          └───max_iterations: function to evaluate if the maximum number of allowed iterations is reached
    |      |                          └───max_evaluations: function to evaluate if the maximum number of allowed evaluations is reached
    |      |                          └───failed_GP: function to evaluate if a Gaussian Process Regression failed during the optimization
    |      |                          └───min_radius: function to evaluate if Multinormal distribution radius is smaller than the minimum allowed radius
    | └───`trustRegionUpdate.py`: script to define the trust region
    |      |                  └───multinormal_radius: function to sample the Multinormal Distribution on Gaussian Process Regressions and define trust region
    | └───`utilities.py`: script with small utility functions
    |      |          └───get_fitted_model: function to fit a GPR to a given set of data. Taken from [SCBO](https://botorch.org/docs/tutorials/scalable_constrained_bo/)
    |      |          └───get_best_index_for_batch: return the index for the best point. One for each trust region. Taken from [SCBO](https://botorch.org/docs/tutorials/scalable_constrained_bo/)
    |      |          └───gaussian_copula: function to scale given values with a Gaussian copula
    |      |          └───scaling_factor: function to scale given values by a fixed value
    |      |          └───bilog: function to scale given values with a bilog scale
    |      |          └───no_scaling: function to return no scaling
    |      |          └───multivariate_circular: function to generate multivariate distribution of given radius and centre within a given domain
    |      |          └───multivariate_distribution: function to generate multivariate distribution of given centre within a given domain
```
