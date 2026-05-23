# Operator Learning for Poisson/Diffusion (DeepONet + FNO)

This repository implements a complete operator-learning pipeline for the PDE:

- div(a(x) grad u(x)) = -f(x)  (implemented as -div(a grad u)=f)
on Ω = (0,1)^2 with homogeneous Dirichlet boundary conditions u=0 on ∂Ω.

**Input**: coefficient field a(x,y)  
**Output**: solution field u(x,y)

We provide:
- Dataset generation with a Gaussian random field prior for w and a pointwise transform a=T(w)
- A finite-elements solver for variable-coefficient diffusion in divergence form
- DeepONet training (sensor-based operator learning)
- FNO training (grid-based Fourier neural operator)
- Evaluation utilities


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

