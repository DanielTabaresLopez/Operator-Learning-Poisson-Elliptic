from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.fno import FNO2d
from src.utils.seed import set_global_seed
from src.utils.io import ensure_dir, save_json


class FNODataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.a = d["a_grid"]  # (N,H,W)
        self.u = d["u_grid"]  # (N,H,W)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        a = self.a[idx][None, :, :]  # (1,H,W)
        u = self.u[idx][None, :, :]  # (1,H,W)
        return a, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path("runs/fno")
    ensure_dir(str(out_dir))

    train_ds = FNODataset(str(Path(args.data_dir) / "train.npz"))
    val_ds   = FNODataset(str(Path(args.data_dir) / "val.npz"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = FNO2d(width=args.width, modes=args.modes, layers=args.layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    best_path = out_dir / "best.pt"

    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for a, u in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            a = a.to(args.device, dtype=torch.float32)
            u = u.to(args.device, dtype=torch.float32)

            pred = model(a)
            loss = loss_fn(pred, u)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for a, u in val_loader:
                a = a.to(args.device, dtype=torch.float32)
                u = u.to(args.device, dtype=torch.float32)
                pred = model(a)
                val_losses.append(loss_fn(pred, u).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"[FNO] epoch={epoch:04d} train_mse={train_loss:.6e} val_mse={val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    save_json(
        str(out_dir / "summary.json"),
        {
            "model": "fno",
            "best_val_mse": best_val,
            "best_epoch": best_epoch,
            "best_ckpt": str(best_path),
            "train_mse_history": train_history,
            "val_mse_history": val_history,
            "modes": int(args.modes),
            "epochs": int(args.epochs),
            "data_dir": args.data_dir,
        },
    )

    print(f"[OK] Saved best checkpoint to {best_path}")
    print(f"[OK] Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()