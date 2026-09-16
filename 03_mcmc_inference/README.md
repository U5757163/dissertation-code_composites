# 03. MCMC Inference Pipeline

This directory contains the Markov Chain Monte Carlo (MCMC) sampling scripts for Bayesian permeability inference, including Gaussian Mixture Model (GMM) prior fitting, chain convergence diagnostics, and out-of-distribution (OOD) validation tests.

## Execution Guide

* **Primary Notebook:** Run `MCMC_NUTS.ipynb` to execute the main sampling and inference pipeline.
* **Helper Module Imports:** The notebook directly imports and depends on the following helper modules:
  * `GP_Training.py` (for loading the trained `GPModel` surrogate)
  * `MeshGenerator.py` (for physical domain mesh operations)
* **Local Diagnostic & Validation Checks:**
  * **GMM Model Fitting:** Prior distribution fitting evaluated using Bayesian Information Criterion (BIC).
  * **Random Sample Inference:** Test runs performed on individual microstructural samples.
  * **Chain Mixing Diagnostics:** MCMC convergence plots (trace plots, autocorrelation, $\hat{R}$ metrics).
  * **Out-of-Distribution (OOD) Test:** Performance evaluation on out-of-distribution test cases.

Detailed markdown cells inside each notebook describe the exact function and output of each check.

## Environment & Dependency Requirements

Ensure your Python environment meets the required dependencies (notably NumPy 2.0+ compatibility):

```bash
# Core Dependencies
pip install "numpy>=2.0.0"
pip install numpyro jax jaxlib scikit-learn pandas matplotlib scipy

> **Note on Package Versions:** Please verify that your active Python environment meets or exceeds the required versions as specified in the root `requirements.txt`. Check for required package updates prior to running full inference runs.
