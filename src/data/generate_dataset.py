from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.prior.gaussian_rf import GaussianRF2D, GaussianRFParams
from src.pde.poisson_fe import VariableCoeffPoissonFE
from src.utils.seed import set_global_seed
from src.utils.io import ensure_dir, save_json


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_coords(n: int) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return coords.astype(np.float32)


def pick_sensor_indices(n: int, n_sensors: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n * n)
    choose = rng.choice(all_idx, size=n_sensors, replace=False)
    ii = choose // n
    jj = choose % n
    return np.stack([ii, jj], axis=1).astype(np.int64)


def extract_sensors(a: np.ndarray, sensor_ij: np.ndarray) -> np.ndarray:
    return a[sensor_ij[:, 0], sensor_ij[:, 1]]


def generate_split(
    n_samples: int,
    sampler: GaussianRF2D,
    solver: VariableCoeffPoissonFE,
    a_min: float,
    a_max: float,
    rhs_constant: float,
    sensor_ij: np.ndarray,
):
    n = solver.n
    coords = make_coords(n)
    nq = coords.shape[0]

    a_grid = np.zeros((n_samples, n, n), dtype=np.float32)
    u_grid = np.zeros((n_samples, n, n), dtype=np.float32)

    a_sensors = np.zeros((n_samples, sensor_ij.shape[0]), dtype=np.float32)
    u_query = np.zeros((n_samples, nq), dtype=np.float32)

    f = np.full((n, n), rhs_constant, dtype=np.float64)

    for i in tqdm(range(n_samples), desc=f"Generating {n_samples} samples"):
        w = sampler.sample_fft()

        a = a_min + (a_max - a_min) * sigmoid(w)
        a = a.astype(np.float64)

        res = solver.solve(a=a, f=f, method="cg", tol=1e-8, maxiter=3000)
        if not res.converged:
            raise RuntimeError(
                f"FE-CG did not converge at sample {i} (iters={res.iters}, info={res.info})."
            )

        u = res.u

        a_grid[i] = a.astype(np.float32)
        u_grid[i] = u.astype(np.float32)

        a_sensors[i] = extract_sensors(a, sensor_ij).astype(np.float32)
        u_query[i] = u.reshape(-1).astype(np.float32)

    return {
        "a_grid": a_grid,
        "u_grid": u_grid,
        "a_sensors": a_sensors,
        "u_query": u_query,
        "coords": coords.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_grid", type=int, default=33)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)

    ap.add_argument("--a_min", type=float, default=0.1)
    ap.add_argument("--a_max", type=float, default=1.0)

    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=1.0)

    ap.add_argument("--rhs_constant", type=float, default=1.0)
    ap.add_argument("--n_sensors", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.out_dir)

    solver = VariableCoeffPoissonFE(n=args.n_grid)
    sampler = GaussianRF2D(
        GaussianRFParams(n=args.n_grid, beta=args.beta, gamma=args.gamma, seed=args.seed)
    )

    sensor_ij = pick_sensor_indices(args.n_grid, args.n_sensors, seed=args.seed)

    meta = {
        "pde": "-div(a grad u)=f on (0,1)^2, u=0 on boundary",
        "discretization": "P1 finite elements on a structured triangulation; outputs resampled at mesh vertices (= regular grid)",
        "n_grid": args.n_grid,
        "a_min": args.a_min,
        "a_max": args.a_max,
        "beta": args.beta,
        "gamma": args.gamma,
        "rhs_constant": args.rhs_constant,
        "n_sensors": args.n_sensors,
        "sensor_ij": sensor_ij.tolist(),
        "seed": args.seed,
    }
    save_json(str(Path(args.out_dir) / "meta.json"), meta)

    train = generate_split(args.n_train, sampler, solver, args.a_min, args.a_max, args.rhs_constant, sensor_ij)
    val   = generate_split(args.n_val,   sampler, solver, args.a_min, args.a_max, args.rhs_constant, sensor_ij)
    test  = generate_split(args.n_test,  sampler, solver, args.a_min, args.a_max, args.rhs_constant, sensor_ij)

    np.savez_compressed(Path(args.out_dir) / "train.npz", **train)
    np.savez_compressed(Path(args.out_dir) / "val.npz", **val)
    np.savez_compressed(Path(args.out_dir) / "test.npz", **test)

    print(f"[OK] FE dataset written to: {args.out_dir}")


if __name__ == "__main__":
    main()