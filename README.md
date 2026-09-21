# Supplementary data for "Correlation-Free Transition Path Sampling through Shooting Point Generation Guided by Committor Learning"

This repository contains the simulation and analysis code and notebooks for

Negedly, M., Falkner, S., Coretti, A., & Dellago, C. (2026). Correlation-Free Transition Path Sampling through Shooting Point Generation Guided by Committor Learning. [arXiv:2609.20461](https://arxiv.org/abs/2609.20461). DOI: [10.48550/arXiv.2609.20461](https://doi.org/10.48550/arXiv.2609.20461)

In the paper, we introduce the GenAIMMD method and test it on two systems: the two-dimensional Wolfe-Quapp (WQ) potential and a polymer model with 11 degrees of freedom.

The code and notebooks for the two test systems are located in this repo's accordingly named branches --- `wolfe_quapp` and `polymer`. Since the corresponding simulation data are over 300 GB in size, they are available separately on Zenodo (DOI: [10.5281/zenodo.22875161](https://doi.org/10.5281/zenodo.22875161)). In each branch, there are multiple directories which only contain a README.md each. This is where the extracted data from Zenodo must be placed in for the scripts to find them. The respective README.md files provide more details. This README will update for each branch, describing the files contained therein.

## Installation

From a working Python installation (tested on 3.14), install the required packages by changing to the repo's root and running

```bash
$ pip install -r requirements.txt
```

## Wolfe-Quapp potential

This branch contains the code and notebooks for the two-dimensional Wolfe-Quapp potential. The following list explains the contents of each file in `src/`

- `aimmd.py`: Committor model definition and GenAIMMD training loop
- `genaimmd_wq.ipynb`: Running GenAIMMD on the two-dimensional model
- `generator.py`: Model definition of the conditioned Boltzmann Generator
- `mcmc.py`: MCMC propagator
- `md.py`: MD propagator
- `potential.py`: Potential energy and force functions, along with other functions related to the 2D system definition
- `si_benchmarking_plot.ipynb`: Plotting code for Fig. 4 in the paper's SI
- `tps.py`: Transition Path Sampling related functions
- `util.py`: General utility functions
- `wolfe_quapp.ipynb`: Definition of the system
- `wq_estimate_committor.py`: Numerical committor estimation utility
- `wq_sp_sample_generator.py`: Committor model training data generator
- `wq_tps_comparison_generated_paths.py`: Tool that propagates paths from given shooting points
- `wq_tps_comparison.ipynb`: Benchmarking plot against standard TPS for the 2D model (Fig. 4 in the main text)
- `wq_tps_sample_generator.py`: TPS simulation script
