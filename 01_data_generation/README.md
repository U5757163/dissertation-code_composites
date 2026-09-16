Data Generation
# 01. Data Generation Pipeline

This directory contains the mesh generation, finite element assembly, and synthetic data generation scripts for the microstructural permeability models.

## Execution Guide

* **Primary Notebook:** Run `DataGenerator.ipynb` to execute the full data generation pipeline end-to-end. 
* **Pre-Generated Data:** The resulting dataset `microstructure_data_200.csv` is already provided in this folder. You do not need to re-run the full simulation pipeline unless you wish to generate new samples. Example data simulations can be found in the folder.
* **Module Dependencies:** `DataGenerator.ipynb` automatically calls and imports functions from the helper scripts (`FEM_Solver.py` and `MeshGenerator.py`). You do not need to execute the helper `.py` files manually.
* **Local Testing & Troubleshooting:** If you wish to inspect or debug specific stages of the mesh generation or finite element setup individually, use the dedicated sub-notebooks (e.g., `test_mesh.ipynb`). Detailed markdown cells inside each notebook describe the exact function of each code block.

## Environment & Dependency Requirements

Ensure your Python environment has the necessary packages installed before running the notebooks:

```bash
# Core Dependencies
pip uninstall -y gmsh
pip install gmsh
pip install scikit-fem

22 ```
23 
24 > **Note on Package Versions:** Please verify that your active Python environment meets or exceeds the required versions for `gmsh` and `scikit-fem` as specified in the root `requirements.txt`. Check for required package updates prior to running full simulation runs.
