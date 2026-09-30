# dissertation-code_composites
Source code and computational pipeline for Bayesian permeability inference in microstructural damage models.
# Bayesian Permeability Inference in Microstructural Damage Models

This repository contains the full computational pipeline, finite element simulation tools, Gaussian Process (GP) surrogate modeling scripts, and MCMC inference routines supporting the Master's dissertation research.

## Repository Architecture

The project is structured into three sequential modules:

* **`01_data_generation/`**: Synthetic dataset generation using Gmsh and scikit-fem finite element permeability simulations. Includes pre-computed datasets (`microstructure_data_200.csv`).
* **`02_gp_surrogate/`**: JAX/tinygp Gaussian Process surrogate model training, cross-validation, and uncertainty quantification (UQ) diagnostics. Converts trained logic into `gp_model.py`.
* **`03_mcmc_inference/`**: NUTS MCMC sampling using NumPyro, prior GMM modeling via BIC, chain diagnostics, and out-of-distribution (OOD) testing.
* **`docs/`**: Password-protected research outputs, including full Master's dissertation (`Dissertation_Protected.pdf`) and presentation poster (`Poster_Protected.pdf`). Contact `mostafa.gh1989@gmail.com` for access credentials.
---
