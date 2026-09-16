# 02. Gaussian Process Surrogate Modeling

This directory contains the training, hyperparameter optimization, and predictive uncertainty validation scripts for the JAX/tinygp surrogate model.

## Execution Guide

* **Primary Notebook:** Run `GP_Surrogate.ipynb` to execute the full surrogate modeling pipeline.
* **Module Conversion:** The trained GP surrogate model logic is converted into a module script (`gp_model.py`) for direct integration into downstream MCMC inference pipelines (`03_mcmc_inference`).
* **Local Diagnostic & Sensitivity Tests:** To inspect GP behavior and validate performance across specific features, run the individual test blocks inside the notebook:
  * **First Test:** Standard training run and hyperparameter optimization check.
  * **10-Fold Cross-Validation:** Robustness test across dataset splits.
  * **Sensitivity Feature UQ Plot:** Uncertainty Quantification (UQ) diagnostic along key spatial/microstructural features.
  * **Predictive Uncertainty vs. Distance:** Validation of GP variance growth relative to training data density.

Detailed markdown cells inside each notebook describe the exact function and output of each check.

## Environment & Dependency Requirements

Ensure your environment has the required JAX ecosystem packages installed:

```bash
# Core Dependencies
pip install --upgrade pip
pip install jax jaxlib jaxopt tinygp numpyro scikit-learn pandas matplotlib scipy
