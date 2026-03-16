from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.deeponet import DeepONet
from src.models.fno import FNO2d
from src.pde.poisson_fe import VariableCoeffPoissonFE


def load_npz(data_dir: str, split: str):
    d = np.load(Path(data_dir) / f"{split}.npz")
    return {k: d[k] for k in d.files}


def load_meta(data_dir: str):
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@torch.inference_mode()
def time_deeponet(data, ckpt_path: str, device: str, batch_size: int = 32, warmup_batches: int = 5) -> float:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    a = torch.tensor(data["a_sensors"], dtype=torch.float32, device=device)
    coords = torch.tensor(data["coords"], dtype=torch.float32, device=device)

    model = DeepONet(
        n_sensors=a.shape[1],
        p=args["p"],
        branch_width=args["branch_width"],
        trunk_width=args["trunk_width"],
        depth=args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    N = a.shape[0]

    for _ in range(min(warmup_batches, max(1, N))):
        bx = a[: min(batch_size, N)]
        _ = model(bx, coords)

    t0 = time.perf_counter()
    n_seen = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx = a[start:end]
        _ = model(bx, coords)
        n_seen += (end - start)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / max(n_seen, 1)


@torch.inference_mode()
def time_fno(data, ckpt_path: str, device: str, batch_size: int = 16, warmup_batches: int = 5) -> float:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    a = torch.tensor(data["a_grid"], dtype=torch.float32, device=device)[:, None, :, :]

    model = FNO2d(width=args["width"], modes=args["modes"], layers=args["layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    N = a.shape[0]

    for _ in range(min(warmup_batches, max(1, N))):
        bx = a[: min(batch_size, N)]
        _ = model(bx)

    t0 = time.perf_counter()
    n_seen = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx = a[start:end]
        _ = model(bx)
        n_seen += (end - start)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / max(n_seen, 1)


def time_numerical_solver(data, n_grid: int, rhs_constant: float = 1.0,
                          method: str = "cg", tol: float = 1e-8, maxiter: int = 3000) -> float:
    solver = VariableCoeffPoissonFE(n=n_grid)
    a_all = data["a_grid"]
    N = a_all.shape[0]
    f = np.full((n_grid, n_grid), rhs_constant, dtype=np.float64)

    if N > 0:
        _ = solver.solve(a=a_all[0].astype(np.float64), f=f, method=method, tol=tol, maxiter=maxiter)

    t0 = time.perf_counter()
    for i in range(N):
        res = solver.solve(a=a_all[i].astype(np.float64), f=f, method=method, tol=tol, maxiter=maxiter)
        if not res.converged:
            raise RuntimeError(f"Numerical solver did not converge for sample {i} (iters={res.iters}).")
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / max(N, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--deeponet_ckpt", type=str, default=None)
    ap.add_argument("--fno_ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--rhs_constant", type=float, default=1.0)
    ap.add_argument("--solver_method", type=str, default="cg", choices=["cg", "spsolve"])
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data_dir, args.split)
    meta = load_meta(args.data_dir)
    n_grid = int(meta.get("n_grid", data["a_grid"].shape[1]))

    times = {}
    times["numerical_fe"] = time_numerical_solver(
        data=data,
        n_grid=n_grid,
        rhs_constant=args.rhs_constant,
        method=args.solver_method,
        tol=1e-8,
        maxiter=3000,
    )

    if args.deeponet_ckpt is not None:
        times["deeponet"] = time_deeponet(data, args.deeponet_ckpt, args.device)

    if args.fno_ckpt is not None:
        times["fno"] = time_fno(data, args.fno_ckpt, args.device)

    print("\nMean time per sample:")
    for k, v in times.items():
        print(f"  {k:>12s}: {v:.3f} ms/sample")

    out_json = Path(args.out_dir) / "time_compare.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(times, f, indent=2)

    plt.figure()
    plt.bar(list(times.keys()), list(times.values()))
    plt.ylabel("Time (ms / sample)")
    plt.title("Numerical solve vs trained-network inference")
    plt.tight_layout()
    out_png = Path(args.out_dir) / "time_compare_bar.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] Saved {out_json}")
    print(f"[OK] Saved {out_png}")


if __name__ == "__main__":
    main()