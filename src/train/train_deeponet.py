from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.deeponet import DeepONet
from src.utils.seed import set_global_seed
from src.utils.io import ensure_dir, save_json


class DeepONetDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.a_sensors = d["a_sensors"]  # (N, n_sensors)
        self.u_query = d["u_query"]      # (N, Nq)
        self.coords = d["coords"]        # (Nq,2), shared across samples

    def __len__(self):
        return self.a_sensors.shape[0]

    def __getitem__(self, idx):
        return self.a_sensors[idx], self.u_query[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--branch_width", type=int, default=256)
    ap.add_argument("--trunk_width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path("runs/deeponet")
    ensure_dir(str(out_dir))

    train_ds = DeepONetDataset(str(Path(args.data_dir) / "train.npz"))
    val_ds   = DeepONetDataset(str(Path(args.data_dir) / "val.npz"))

    coords = torch.tensor(train_ds.coords, dtype=torch.float32, device=args.device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    n_sensors = train_ds.a_sensors.shape[1]
    model = DeepONet(
        n_sensors=n_sensors,
        p=args.p,
        branch_width=args.branch_width,
        trunk_width=args.trunk_width,
        depth=args.depth,
    ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    best_path = out_dir / "best.pt"

    train_history = []
    val_history = []

    ckpt_args = vars(args).copy()
    ckpt_args["n_sensors"] = int(n_sensors)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for a_s, u_q in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            a_s = a_s.to(args.device, dtype=torch.float32)
            u_q = u_q.to(args.device, dtype=torch.float32)

            pred = model(a_s, coords)  # (B, Nq)
            loss = loss_fn(pred, u_q)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for a_s, u_q in val_loader:
                a_s = a_s.to(args.device, dtype=torch.float32)
                u_q = u_q.to(args.device, dtype=torch.float32)
                pred = model(a_s, coords)
                val_losses.append(loss_fn(pred, u_q).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"[DeepONet] epoch={epoch:04d} train_mse={train_loss:.6e} val_mse={val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": ckpt_args}, best_path)

    save_json(
        str(out_dir / "summary.json"),
        {
            "model": "deeponet",
            "best_val_mse": best_val,
            "best_epoch": best_epoch,
            "best_ckpt": str(best_path),
            "train_mse_history": train_history,
            "val_mse_history": val_history,
            "n_sensors": int(n_sensors),
            "epochs": int(args.epochs),
            "data_dir": args.data_dir,
        },
    )

    print(f"[OK] Saved best checkpoint to {best_path}")
    print(f"[OK] Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()