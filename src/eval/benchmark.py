import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.deeponet import DeepONet
from src.models.fno import FNO2d
from src.utils.io import ensure_dir


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def load_meta(data_dir: str) -> Dict:
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_npz(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    path = Path(data_dir) / f"{split}.npz"
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _infer_model_type(ckpt_args: Dict) -> str:
    # Heuristic: DeepONet ckpt has 'p' and 'branch_width'; FNO has 'modes' and 'width'
    if "p" in ckpt_args and "branch_width" in ckpt_args:
        return "deeponet"
    if "modes" in ckpt_args and "width" in ckpt_args:
        return "fno"
    return "unknown"


@dataclass
class BenchResult:
    model: str
    ckpt: str
    data_dir: str
    split: str
    n_grid: int
    n_test: int
    n_train: int
    n_val: int
    n_sensors: Optional[int]
    mean_rel_l2: float
    median_rel_l2: float
    p90_rel_l2: float
    mean_mse: float
    infer_ms_per_sample: float
    extra: Dict


def _batch_rel_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    pred, true shape: (B, ...) -> returns (B,) relative l2 errors.
    """
    diff = (pred - true).reshape(pred.shape[0], -1)
    tru = true.reshape(true.shape[0], -1)
    num = torch.linalg.norm(diff, dim=1)
    den = torch.linalg.norm(tru, dim=1).clamp_min(eps)
    return num / den


def _batch_mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    diff = (pred - true).reshape(pred.shape[0], -1)
    return (diff * diff).mean(dim=1)


@torch.inference_mode()
def eval_deeponet(
    data: Dict[str, np.ndarray],
    ckpt_path: str,
    device: str,
    batch_size: int,
    timing_batches: int,
    warmup_batches: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      rel_l2 (N,)
      mse    (N,)
      infer_ms_per_sample
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    ns = data["a_sensors"].shape[1]
    coords = torch.tensor(data["coords"], dtype=torch.float32, device=device)  # (Nq,2)

    model = DeepONet(
        n_sensors=ns,
        p=args["p"],
        branch_width=args["branch_width"],
        trunk_width=args["trunk_width"],
        depth=args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    a = torch.tensor(data["a_sensors"], dtype=torch.float32, device=device)  # (N,ns)
    u_true = torch.tensor(data["u_query"], dtype=torch.float32, device=device)  # (N,Nq)

    N = a.shape[0]
    rels = []
    mses = []

    # --- inference timing (ms/sample) ---
    # Use fixed-size batches for timing
    def run_one_batch(bx):
        return model(bx, coords)

    # warmup
    if N > 0:
        for _ in range(warmup_batches):
            bx = a[: min(batch_size, N)]
            _ = run_one_batch(bx)

    # timed batches
    if N > 0:
        t0 = time.perf_counter()
        n_seen = 0
        for k in range(timing_batches):
            start = (k * batch_size) % N
            end = min(start + batch_size, N)
            bx = a[start:end]
            _ = run_one_batch(bx)
            n_seen += (end - start)
        t1 = time.perf_counter()
        infer_ms_per_sample = (t1 - t0) * 1000.0 / max(n_seen, 1)
    else:
        infer_ms_per_sample = float("nan")

    # --- full evaluation ---
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pred = model(a[start:end], coords)  # (B,Nq)
        rel = _batch_rel_l2(pred, u_true[start:end]).detach().cpu().numpy()
        mse = _batch_mse(pred, u_true[start:end]).detach().cpu().numpy()
        rels.append(rel)
        mses.append(mse)

    rels = np.concatenate(rels) if rels else np.array([], dtype=np.float64)
    mses = np.concatenate(mses) if mses else np.array([], dtype=np.float64)
    return rels, mses, float(infer_ms_per_sample)


@torch.inference_mode()
def eval_fno(
    data: Dict[str, np.ndarray],
    ckpt_path: str,
    device: str,
    batch_size: int,
    timing_batches: int,
    warmup_batches: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      rel_l2 (N,)
      mse    (N,)
      infer_ms_per_sample
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    model = FNO2d(width=args["width"], modes=args["modes"], layers=args["layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    a = torch.tensor(data["a_grid"], dtype=torch.float32, device=device)  # (N,H,W)
    u_true = torch.tensor(data["u_grid"], dtype=torch.float32, device=device)  # (N,H,W)

    # add channel dim
    a = a[:, None, :, :]      # (N,1,H,W)
    u_true = u_true[:, None]  # (N,1,H,W)

    N = a.shape[0]
    rels = []
    mses = []

    def run_one_batch(bx):
        return model(bx)

    # warmup
    if N > 0:
        for _ in range(warmup_batches):
            bx = a[: min(batch_size, N)]
            _ = run_one_batch(bx)

    # timed batches
    if N > 0:
        t0 = time.perf_counter()
        n_seen = 0
        for k in range(timing_batches):
            start = (k * batch_size) % N
            end = min(start + batch_size, N)
            bx = a[start:end]
            _ = run_one_batch(bx)
            n_seen += (end - start)
        t1 = time.perf_counter()
        infer_ms_per_sample = (t1 - t0) * 1000.0 / max(n_seen, 1)
    else:
        infer_ms_per_sample = float("nan")

    # full eval
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pred = model(a[start:end])  # (B,1,H,W)
        rel = _batch_rel_l2(pred, u_true[start:end]).detach().cpu().numpy()
        mse = _batch_mse(pred, u_true[start:end]).detach().cpu().numpy()
        rels.append(rel)
        mses.append(mse)

    rels = np.concatenate(rels) if rels else np.array([], dtype=np.float64)
    mses = np.concatenate(mses) if mses else np.array([], dtype=np.float64)
    return rels, mses, float(infer_ms_per_sample)


def discover_checkpoints(runs_dir: str) -> List[str]:
    runs = Path(runs_dir)
    if not runs.exists():
        return []
    pts = list(runs.rglob("best.pt"))
    return [str(p) for p in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--ckpt", type=str, action="append", default=None,
                    help="Checkpoint path. Can be passed multiple times. If omitted, auto-discovers runs/**/best.pt.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--timing_batches", type=int, default=30)
    ap.add_argument("--warmup_batches", type=int, default=5)
    ap.add_argument("--save_per_sample", action="store_true")
    ap.add_argument("--out_csv", type=str, default="runs/benchmarks/results.csv")
    args = ap.parse_args()

    ensure_dir(str(Path(args.out_csv).parent))

    data = load_npz(args.data_dir, args.split)
    meta = load_meta(args.data_dir)

    n_grid = int(meta.get("n_grid", data["a_grid"].shape[1] if "a_grid" in data else -1))
    n_test = int(data["a_grid"].shape[0]) if "a_grid" in data else int(data["a_sensors"].shape[0])
    n_train = int(load_npz(args.data_dir, "train")["a_grid"].shape[0])
    n_val = int(load_npz(args.data_dir, "val")["a_grid"].shape[0])
    n_sensors = int(meta.get("n_sensors", data["a_sensors"].shape[1] if "a_sensors" in data else -1))

    ckpts = args.ckpt if args.ckpt else discover_checkpoints(args.runs_dir)
    if not ckpts:
        raise RuntimeError("No checkpoints found. Use --ckpt or ensure runs/**/best.pt exists.")

    rows = []
    for ckpt_path in ckpts:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = ckpt_obj.get("args", {})
        model_type = _infer_model_type(ckpt_args)

        if model_type == "deeponet":
            rels, mses, t_ms = eval_deeponet(
                data=data, ckpt_path=ckpt_path, device=args.device,
                batch_size=args.batch_size, timing_batches=args.timing_batches, warmup_batches=args.warmup_batches
            )
            extra = {k: ckpt_args.get(k) for k in ["p", "branch_width", "trunk_width", "depth"]}
            extra["model_type"] = "deeponet"
        elif model_type == "fno":
            rels, mses, t_ms = eval_fno(
                data=data, ckpt_path=ckpt_path, device=args.device,
                batch_size=max(1, args.batch_size // 2),  # FNO can be heavier; smaller batch
                timing_batches=args.timing_batches, warmup_batches=args.warmup_batches
            )
            extra = {k: ckpt_args.get(k) for k in ["modes", "width", "layers"]}
            extra["model_type"] = "fno"
        else:
            print(f"[SKIP] Unknown checkpoint type: {ckpt_path}")
            continue

        mean_rel = float(np.mean(rels)) if rels.size else float("nan")
        med_rel = float(np.median(rels)) if rels.size else float("nan")
        p90_rel = float(np.quantile(rels, 0.90)) if rels.size else float("nan")
        mean_mse = float(np.mean(mses)) if mses.size else float("nan")

        r = BenchResult(
            model=model_type,
            ckpt=ckpt_path,
            data_dir=args.data_dir,
            split=args.split,
            n_grid=n_grid,
            n_test=n_test,
            n_train=n_train,
            n_val=n_val,
            n_sensors=n_sensors if n_sensors > 0 else None,
            mean_rel_l2=mean_rel,
            median_rel_l2=med_rel,
            p90_rel_l2=p90_rel,
            mean_mse=mean_mse,
            infer_ms_per_sample=t_ms,
            extra=extra,
        )
        row = {
            "model": r.model,
            "ckpt": r.ckpt,
            "ckpt_id": _sha1(r.ckpt),
            "data_dir": r.data_dir,
            "split": r.split,
            "n_grid": r.n_grid,
            "n_train": r.n_train,
            "n_val": r.n_val,
            "n_test": r.n_test,
            "n_sensors": r.n_sensors,
            "mean_rel_l2": r.mean_rel_l2,
            "median_rel_l2": r.median_rel_l2,
            "p90_rel_l2": r.p90_rel_l2,
            "mean_mse": r.mean_mse,
            "infer_ms_per_sample": r.infer_ms_per_sample,
        }
        for k, v in r.extra.items():
            row[k] = v
        rows.append(row)

        if args.save_per_sample:
            out_dir = Path("runs/benchmarks/per_sample")
            ensure_dir(str(out_dir))
            np.savez_compressed(
                out_dir / f"errors_{model_type}_{_sha1(ckpt_path)}.npz",
                rel_l2=rels.astype(np.float32),
                mse=mses.astype(np.float32),
            )

        print(f"[OK] {model_type}  mean_rel_l2={mean_rel:.4e}  time={t_ms:.3f} ms/sample  ckpt={ckpt_path}")

    df_new = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.to_csv(out_csv, index=False)
    else:
        df_new.to_csv(out_csv, index=False)

    print(f"[DONE] Results saved to {out_csv}")


if __name__ == "__main__":
    main()