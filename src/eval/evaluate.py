from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from src.models.deeponet import DeepONet
from src.models.fno import FNO2d
from src.utils.metrics import relative_l2_error


def load_npz(data_dir: str, split: str):
    return np.load(Path(data_dir) / f"{split}.npz")


@torch.no_grad()
def eval_deeponet(data, ckpt_path: str, device: str) -> float:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    a_sensors = data["a_sensors"]  # (N, ns)
    u_true = data["u_query"]       # (N, nq)
    coords = torch.tensor(data["coords"], dtype=torch.float32, device=device)

    model = DeepONet(
        n_sensors=a_sensors.shape[1],
        p=args["p"],
        branch_width=args["branch_width"],
        trunk_width=args["trunk_width"],
        depth=args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    errs = []
    for i in range(a_sensors.shape[0]):
        a = torch.tensor(a_sensors[i:i+1], dtype=torch.float32, device=device)
        pred = model(a, coords).cpu().numpy()[0]
        err = relative_l2_error(pred, u_true[i])
        errs.append(err)
    return float(np.mean(errs))


@torch.no_grad()
def eval_fno(data, ckpt_path: str, device: str) -> float:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    a = data["a_grid"]  # (N,H,W)
    u = data["u_grid"]  # (N,H,W)

    model = FNO2d(width=args["width"], modes=args["modes"], layers=args["layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    errs = []
    for i in range(a.shape[0]):
        a_t = torch.tensor(a[i:i+1, None, :, :], dtype=torch.float32, device=device)
        pred = model(a_t).cpu().numpy()[0, 0]
        err = relative_l2_error(pred, u[i])
        errs.append(err)
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--deeponet_ckpt", type=str, required=True)
    ap.add_argument("--fno_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    test = load_npz(args.data_dir, "test")

    deeponet_err = eval_deeponet(test, args.deeponet_ckpt, args.device)
    fno_err = eval_fno(test, args.fno_ckpt, args.device)

    print(f"[TEST] DeepONet mean relative L2 error: {deeponet_err:.6e}")
    print(f"[TEST] FNO     mean relative L2 error: {fno_err:.6e}")


if __name__ == "__main__":
    main()