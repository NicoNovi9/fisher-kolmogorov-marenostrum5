# fisher-kolmogorov-marenostrum5
# Parallel Performance Analysis of the Fisher-Kolmogorov Equation Solver

## Overview
This project focuses on solving the Fisher-Kolmogorov equation, which models the propagation of misfolded proteins in the brain—an essential process in understanding neurodegenerative diseases like Alzheimer's and Parkinson's. The equation is solved using the **finite element method (FEM)** with an implicit time-stepping scheme, and parallelized using **MPI** for high-performance computing (HPC) environments.

## Mathematical Model
The Fisher-Kolmogorov equation is given by:


$\frac{\partial c}{\partial t} - \nabla \cdot (D \nabla c) - \alpha c (1 - c) = 0, \quad \text{in } \Omega$

where:
- $\( c(t, x) \)$ represents the concentration of misfolded proteins.
- $\( D \)$ is the diffusion coefficient.
- $\( \alpha \)$ controls the reaction rate (nonlinearity).
- Neumann boundary conditions $\( D \nabla c \cdot n = 0 \)$ ensure no flux across the domain.

The equation is discretized using **FEM in space** and a **backward Euler scheme in time**, solved iteratively with Newton’s method.

## Parallel Implementation
The solver is parallelized using **MPI**, distributing the computational domain among multiple processes:

- **Setup Phase:** Mesh partitioning, FEM space construction, and matrix assembly.
- **Solve Phase:** Nonlinear system solved using Newton’s method, with the Jacobian computed via the Fréchet derivative.
- **Linear Solver:** Iterative Krylov solver with parallel matrix-vector products and global norm computations via `MPI_Allreduce`.

## Performance Analysis
The solver was tested on **MareNostrum5**, evaluating scalability across up to **256 MPI processes**. Key insights:
- **Strong scaling** behavior observed until communication overhead becomes dominant.
- **Performance bottlenecks** identified in global reductions and boundary exchanges.
