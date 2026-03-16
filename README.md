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

## To run the pipeline:

From the repository root:

python -m src.data.generate_dataset --out_dir data/main --n_grid 33 --n_train 500 --n_val 100 --n_test 100 --n_sensors 128 --seed 0
python -m src.train.train_deeponet --data_dir data/main --epochs 200 --batch_size 32 --device cpu
python -m src.train.train_fno --data_dir data/main --epochs 200 --batch_size 16 --device cpu
python -m src.eval.evaluate --data_dir data/main --deeponet_ckpt runs/deeponet/best.pt --fno_ckpt runs/fno/best.pt --device cpu
python -m src.eval.benchmark --data_dir data/main --runs_dir runs --split test --device cpu
python -m src.eval.plot_benchmarks --results_csv runs/benchmarks/results.csv --out_dir runs/plots
python -m src.eval.qualitative --data_dir data/main --split test --idx 0 --deeponet_ckpt runs/deeponet/best.pt --fno_ckpt runs/fno/best.pt --device cpu --out_dir runs/plots/qualitative_fe
python -m src.eval.plot_iterations --deeponet_summary runs/deeponet/summary.json --fno_summary runs/fno/summary.json --out_path runs/plots/history_both.png
python -m src.eval.compare_times --data_dir data/main --split test --deeponet_ckpt runs/deeponet/best.pt --fno_ckpt runs/fno/best.pt --device cpu --out_dir runs/plots


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

