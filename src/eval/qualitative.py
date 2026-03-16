from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.deeponet import DeepONet
from src.models.fno import FNO2d


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_npz(data_dir: str, split: str = "test"):
    d = np.load(Path(data_dir) / f"{split}.npz")
    return {k: d[k] for k in d.files}


def load_meta(data_dir: str):
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_heatmap(arr2d: np.ndarray, out_path: str, title: str, vmin=None, vmax=None, scatter_ij=None):
    plt.figure()
    plt.imshow(arr2d, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar()
    if scatter_ij is not None and len(scatter_ij) > 0:
        ij = np.asarray(scatter_ij)
        plt.scatter(
            ij[:, 1], ij[:, 0],
            s=18, marker="o",
            facecolors="none", edgecolors="red", linewidths=0.8
        )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.inference_mode()
def predict_deeponet(sample, ckpt_path: str, device: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    a_sensors = sample["a_sensors"][None, :]
    coords = sample["coords"]

    model = DeepONet(
        n_sensors=a_sensors.shape[1],
        p=args["p"],
        branch_width=args["branch_width"],
        trunk_width=args["trunk_width"],
        depth=args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    a_t = torch.tensor(a_sensors, dtype=torch.float32, device=device)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    pred = model(a_t, coords_t).cpu().numpy()[0]
    return pred


@torch.inference_mode()
def predict_fno(sample, ckpt_path: str, device: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    a = sample["a_grid"][None, None, :, :]

    model = FNO2d(width=args["width"], modes=args["modes"], layers=args["layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    a_t = torch.tensor(a, dtype=torch.float32, device=device)
    pred = model(a_t).cpu().numpy()[0, 0]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--deeponet_ckpt", type=str, default=None)
    ap.add_argument("--fno_ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default="runs/plots/qualitative")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    data = load_npz(args.data_dir, args.split)
    meta = load_meta(args.data_dir)

    n = data["a_grid"].shape[1]
    idx = args.idx
    if idx < 0 or idx >= data["a_grid"].shape[0]:
        raise ValueError(f"idx out of range: {idx}")

    a_grid = data["a_grid"][idx]
    u_true_grid = data["u_grid"][idx]
    sensor_ij = meta.get("sensor_ij", None)

    save_heatmap(
        a_grid,
        str(Path(args.out_dir) / f"a_grid_idx{idx}.png"),
        f"Coefficient field a(x,y) (idx={idx})"
    )

    if sensor_ij is not None:
        save_heatmap(
            a_grid,
            str(Path(args.out_dir) / f"a_grid_with_sensors_idx{idx}.png"),
            f"Coefficient field a(x,y) + DeepONet sensors (idx={idx})",
            scatter_ij=sensor_ij
        )

    pred_don = None
    pred_fno = None
    err_don = None
    err_fno = None

    if args.deeponet_ckpt is not None:
        sample = {
            "a_sensors": data["a_sensors"][idx],
            "coords": data["coords"],
        }
        pred_flat = predict_deeponet(sample, args.deeponet_ckpt, args.device)
        pred_don = pred_flat.reshape(n, n)
        err_don = np.abs(pred_don - u_true_grid)

    if args.fno_ckpt is not None:
        sample = {"a_grid": data["a_grid"][idx]}
        pred_fno = predict_fno(sample, args.fno_ckpt, args.device)
        err_fno = np.abs(pred_fno - u_true_grid)

    u_vmin = float(u_true_grid.min())
    u_vmax = float(u_true_grid.max())
    if pred_don is not None:
        u_vmin = min(u_vmin, float(pred_don.min()))
        u_vmax = max(u_vmax, float(pred_don.max()))
    if pred_fno is not None:
        u_vmin = min(u_vmin, float(pred_fno.min()))
        u_vmax = max(u_vmax, float(pred_fno.max()))

    save_heatmap(
        u_true_grid,
        str(Path(args.out_dir) / f"true_u_idx{idx}.png"),
        f"True u (idx={idx})",
        vmin=u_vmin,
        vmax=u_vmax,
    )

    if pred_don is not None:
        save_heatmap(
            pred_don,
            str(Path(args.out_dir) / f"deeponet_pred_u_idx{idx}.png"),
            f"DeepONet pred u (idx={idx})",
            vmin=u_vmin,
            vmax=u_vmax,
        )

    if pred_fno is not None:
        save_heatmap(
            pred_fno,
            str(Path(args.out_dir) / f"fno_pred_u_idx{idx}.png"),
            f"FNO pred u (idx={idx})",
            vmin=u_vmin,
            vmax=u_vmax,
        )

    err_vmax = 0.0
    if err_don is not None:
        err_vmax = max(err_vmax, float(err_don.max()))
    if err_fno is not None:
        err_vmax = max(err_vmax, float(err_fno.max()))

    if err_don is not None:
        save_heatmap(
            err_don,
            str(Path(args.out_dir) / f"deeponet_abs_err_idx{idx}.png"),
            f"DeepONet |error| (idx={idx})",
            vmin=0.0,
            vmax=err_vmax,
        )

    if err_fno is not None:
        save_heatmap(
            err_fno,
            str(Path(args.out_dir) / f"fno_abs_err_idx{idx}.png"),
            f"FNO |error| (idx={idx})",
            vmin=0.0,
            vmax=err_vmax,
        )

    print(f"[OK] Qualitative plots written to {args.out_dir}")


if __name__ == "__main__":
    main()