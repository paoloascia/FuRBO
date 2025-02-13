## Benchmarks
This folder contains all the benchmarks evaluated to assess the performance of FuRBO. FuRBO has been compared against the following algorithms on the the following benchmarking functions/suites:
- Scalable Constrained Bayesian Optimization (SCBO) [1] on BBOB-constrained [2] for 2d, 10d, 40d, and batch size 1, 1 x dimensionality, 3 x dimensionality

[1]: David Eriksson and Matthias Poloczek. Scalable constrained Bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 730–738. PMLR, 2021. doi: [10.48550/arxiv.2002.08526](https://doi.org/10.48550/arxiv.2002.08526).
[2]: Paul Dufossé, Nikolaus Hansen, Dimo Brockhoff, Phillipe R. Sampaio, Asma Atamna, and Anne Auger. Building scalable test problems for benchmarking constrained optimizers. 2022. To be submitted to the SIAM Journal of Optimization. [database](https://numbbo.github.io/coco/testsuites/bbob-constrained)
### Folder structure
```
└───FuRBO_Database: database with all the benchmarks evaluated on FuRBO
    |            └───bbob_constrained: results from optimizing the suite bbob-constrained (2d, 10d, 40d)
    |                |              └───batch_size_q1: optimization with batch size of 1
    |                |              └───batch_size_q1d: optimization with batch size of 1 x dimensionality
    |                |              └───batch_size_q3d: optimization with batch size of 3 x dimensionality
    |                |              └───code: scripts used to evaluate the database
    |                |              └───plots: plots of convergence for all functions evaluated
└───SCBO_Database: database with all the benchmarks evaluated on SCBO
    |            └───bbob_constrained: results from optimizing the suite bbob-constrained (2d, 10d, 40d)
    |                |              └───batch_size_q1: optimization with batch size of 1
    |                |              └───batch_size_q1d: optimization with batch size of 1 x dimensionality
    |                |              └───batch_size_q3d: optimization with batch size of 3 x dimensionality
    |                |              └───code: scripts used to evaluate the database
    |                |              └───plots: plots of comparison between SCBO and FuRBO for all functions evaluated
└───`Plot_FuRBO_bbob_constrained.py`
└───`Plot_compare_SCBO_bbob_constrained.py`: 
└───`bbob-constrained-targets.npy`: file storing optimum for all constrained functions of the suite bbob
```
