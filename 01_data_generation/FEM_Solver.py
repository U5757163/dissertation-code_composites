#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #run if unavailable in your environmnet
# !pip uninstall -y gmsh
# !pip install scikit-fem
# !pip install pygmsh


# In[ ]:


# #run if unavailable 
# !rm -rf ~/extra_libs && mkdir -p ~/extra_libs
# !cd ~/extra_libs && wget -q http://archive.ubuntu.com/ubuntu/pool/main/libg/libglu/libglu1-mesa_9.0.2-1.1build1_amd64.deb
# !cd ~/extra_libs && ar x libglu1-mesa_9.0.2-1.1build1_amd64.deb
# !cd ~/extra_libs && tar -xf data.tar.*
# !cd ~/extra_libs && cp usr/lib/x86_64-linux-gnu/libGLU.so.1* /opt/conda/lib/
# !ls -l /opt/conda/lib/libGLU*


# In[2]:


import numpy as np
import skfem as fem
from skfem.helpers import dot, grad
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def compute_P_eff(points, triangles, elements_D):

    '''
    skfem.MeshTri reads the pre-calculated nodes and triangles,
    converting them into a native scikit-fem object,
    so the library's internal algorithms can map and process the mesh.

    ONLY take columns 0 and 1 (X and Y), then transpose, then fix contiguity
    '''
    mesh = fem.MeshTri(np.ascontiguousarray(points[:, 0:2].T), 
                       np.ascontiguousarray(triangles[:, 0:3].T))

    # Apply Dirichlet Boundary Conditions (C=0 at top, C=1 at bottom)
    # Find nodes on boundaries using your coordinate points
    y_all = points[:, 1]
    x_all = points[:, 0]

    y_max, y_min = np.max(y_all), np.min(y_all)
    x_max, x_min = np.max(x_all), np.min(x_all)

    # floating point band, 1e-5:
    top_nodes = np.where(np.abs(y_all - y_max) < 1e-5)[0]
    bottom_nodes = np.where(np.abs(y_all - y_min) < 1e-5)[0]

    right_nodes = np.where(np.abs(x_all - x_max) < 1e-5)[0]
    left_nodes = np.where(np.abs(x_all - x_min) < 1e-5)[0]

    left_sorted = left_nodes[np.argsort(y_all[left_nodes])]
    right_sorted = right_nodes[np.argsort(y_all[right_nodes])]

    # Combine and enforce boundary values
    dirichlet_nodes = np.concatenate([top_nodes, bottom_nodes])
    boundary_values = np.concatenate([np.zeros(len(top_nodes)), np.ones(len(bottom_nodes))])

    ''' Sets out the mathematical framework 
    (Linear Triangular interploation)
    pass the pared values for periodicity to basis:
    '''
    min_len = min(len(left_sorted), len(right_sorted))
    # Map left nodes to right nodes directly inside the mesh object
    mesh.periodic_facets = np.column_stack((left_sorted[:min_len], right_sorted[:min_len]))

    # 3. Initialize Basis (automatically inherits mesh periodicity)
    basis = fem.Basis(mesh, fem.ElementTriP1())

    # Define the Weak Form bilinear
    @fem.BilinearForm
    def c_diffusion(u, v, w):
        # Read components directly from the mapped parameters
        flux_x = w.D00 * grad(u)[0] + w.D01 * grad(u)[1]
        flux_y = w.D10 * grad(u)[0] + w.D11 * grad(u)[1]
        return grad(v)[0] * flux_x + grad(v)[1] * flux_y

    # Excitation/Input force
    F = np.zeros(basis.N)
    # Global Stiffness Matrix K
    D_basis = fem.Basis(mesh, fem.ElementTriP0())
    D00 = D_basis.interpolate(elements_D[:, 0, 0])
    D01 = D_basis.interpolate(elements_D[:, 0, 1])
    D10 = D_basis.interpolate(elements_D[:, 1, 0])
    D11 = D_basis.interpolate(elements_D[:, 1, 1])

    K = c_diffusion.assemble(basis, D00=D00, D01=D01, D10=D10, D11=D11)

    #Regularisation term added for singularity flags
    K = K + sp.diags([1e-12 * np.ones(basis.N)], [0], shape=(basis.N, basis.N))

    # internal nodes:
    x_global = np.zeros(basis.N)
    x_global[dirichlet_nodes] = boundary_values
    K_cond, F_cond, u_cond, I = fem.condense(K, F, D=dirichlet_nodes, x=x_global)

    # Solve the internal part
    C_internal = spsolve(K_cond, F_cond)
    # initialise solution:
    #combine, distinguish array nodes, then assign correctly
    C_vector = u_cond.copy()
    C_vector[I] = C_internal

    # --------------------------  Effective Permeability from concentration gradient: --------------------------
    # ----------------------------------------------------------------------------------------------------------
    #cretae a filed that encondes both C & its topology
    C_field = basis.interpolate(C_vector)
    dg_basis = fem.Basis(mesh, fem.ElementTriP0())

    @fem.Functional
    def flux_y(w):
        grad_u = grad(w['C'])
        return -(w['D10'] * grad_u[0] + w['D11'] * grad_u[1])

    # Assemble the fluxes across all elements using quadrature
    J_y = flux_y.assemble(dg_basis, C=C_field, D10=elements_D[:, 1, 0], D11=elements_D[:, 1, 1])

    # Compute total area of the domain
    A_total = np.sum(basis.dx)

    # Compute average vertical flux: area-weighted (flux * area)
    J_avg = np.sum(J_y) / A_total
    # Calculate Peff: vertial component only as QoI
    H = y_max - y_min
    P_eff_yy = J_avg * H
    return P_eff_yy


# In[ ]:


def run_matrix_sanity_check(file_name):
    # Load saved example mesh data
    data = np.load(file_name, allow_pickle=True)
    points = data["points"]
    triangles = data["triangles"]

    # We force every element to have a basic D = [[1.0, 0.0], [0.0, 1.0]]
    num_elements = len(triangles)
    isotropic_D = np.zeros((num_elements, 2, 2))
    isotropic_D[:, 0, 0] = 1.0  # D_xx
    isotropic_D[:, 1, 1] = 1.0  # D_yy

    # Pass the geometry into the solver
    P_eff_yy = compute_P_eff(points, triangles, isotropic_D)

    print(f"\n--- RESULTS ---")
    print(f"Calculated P_eff_yy: {P_eff_yy:.6f}")
    print(f"Expected Baseline:   1.000000")

    if np.isclose(P_eff_yy, 1.0, atol=1e-4):
        print("SUCCESS")
    else:
        print("FAILURE")

if __name__ == "__main__":
    run_matrix_sanity_check("FEM_mesh_baseline.npz")


# In[ ]:


from MeshGenerator import Simulator
import random
def run_effective_RVE_size(Set):
    if Set is None:
        Set = np.arange(30,55,5)

    print(f"| L (length) | th (width) | Calculated P_eff | Rel. Error |")
    print(f"|----------|------------|------------------|------------|")

    P_eff_vals = []
    E_r = []
    for k, l in enumerate(Set):
        # random.seed(42)
        # np.random.seed(42)

        sim = Simulator(L = l, th= l/5, debond_fraction=0.3, crc_no=8, min_length=0.1, max_length=0.4)
        points, triangles, elements_D = sim.run_diffusion(mode="mixed")
        P_eff = compute_P_eff(points, triangles, elements_D)
        P_eff_vals.append(P_eff)
        # if k>0:
        #     E_r.append(abs(P_eff_vals[k] - P_eff_vals[k-1]) / P_eff_vals[k])
        #     print(f"| {Set[k]:8d} | {Set[k]/5:10.2f} | {P_eff:16.6f} | {E_r[-1]:10.4f} |")

        # Replace your loop error calculation with this:
        if k > 0:
            running_mean = np.mean(P_eff_vals[:k])
            error_calc = abs(P_eff - running_mean) / running_mean
            E_r.append(error_calc)
            print(f"| {Set[k]:8d} | {Set[k]/5:10.2f} | {P_eff:16.6f} | {E_r[-1]:10.4f} |")
    if E_r and E_r[-1] < 0.05:
        print("RVE Converged")

if __name__ == "__main__":
    RVE_set = np.arange(10,80,5)
    run_effective_RVE_size(RVE_set)


# In[ ]:


from MeshGenerator import Simulator
import random
import time
def run_effective_RVE_size(Set=None):
    if Set is None:
        Set = np.arange(30, 55, 5)
    print(f"| L (Length) | P_eff Calc | Rel. Error | Total Time (s) |")
    print(f"|------------|------------|------------|----------------|")

    P_eff_vals = []
    E_r = []

    for k, l in enumerate(Set):
        t0 = time.time()
        random.seed(42 + int(l))
        np.random.seed(42 + int(l))

        # Mesh Generation & FEM Solve
        sim = Simulator(L=l, th=l/5, debond_fraction=0.3, crc_no=8, min_length=0.1, max_length=0.4)
        points, triangles, elements_D = sim.run_diffusion(mode="mixed")
        P_eff = compute_P_eff(points, triangles, elements_D)

        t_total = time.time() - t0
        P_eff_vals.append(P_eff)

        if k == 0:
            print(f"| {l:10d} | {P_eff:10.6f} | {'N/A':>10s} | {t_total:14.3f} |")
        else:
            running_mean = np.mean(P_eff_vals[k-1])
            error_calc = abs(P_eff - running_mean) / running_mean
            E_r.append(error_calc)
            print(f"| {l:10d} | {P_eff:10.6f} | {error_calc:10.4f} | {t_total:14.3f} |")

if __name__ == "__main__":
    RVE_set = np.arange(10, 80, 5)
    run_effective_RVE_size(RVE_set)


# In[ ]:




