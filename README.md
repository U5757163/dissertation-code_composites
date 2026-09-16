# dissertation-code_composites
Source code and computational pipeline for Bayesian permeability inference in microstructural damage models.
# Bayesian Permeability Inference in Microstructural Damage Models

This repository contains the full computational pipeline, finite element simulation tools, Gaussian Process (GP) surrogate modeling scripts, and MCMC inference routines supporting the Master's dissertation research.

## Repository Architecture

The project is structured into three sequential modules:

* **`01_data_generation/`**: Synthetic dataset generation using Gmsh and scikit-fem finite element permeability simulations. Includes pre-computed datasets (`microstructure_data_200.csv`).
* **`02_gp_surrogate/`**: JAX/tinygp Gaussian Process surrogate model training, cross-validation, and uncertainty quantification (UQ) diagnostics. Converts trained logic into `gp_model.py`.
* **`03_mcmc_inference/`**: NUTS MCMC sampling using NumPyro, prior GMM modeling via BIC, chain diagnostics, and out-of-distribution (OOD) testing.

---

## Quick Start & Dependencies

### Global Requirements
To install all necessary environment dependencies across all modules:

# General Numerical & Data Processing
numpy>=2.0.0
scipy
pandas
matplotlib
scikit-learn

# Finite Element & Mesh Generation (01_data_generation)
gmsh
scikit-fem

# JAX Ecosystem & GP Surrogate Modeling (02_gp_surrogate)
jax
jaxlib
jaxopt
tinygp

# Bayesian MCMC Inference (03_mcmc_inference)
numpyro
