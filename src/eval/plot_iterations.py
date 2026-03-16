from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deeponet_summary", type=str, required=True)
    ap.add_argument("--fno_summary", type=str, required=True)
    ap.add_argument("--out_path", type=str, default="runs/plots/history_both.png")
    args = ap.parse_args()

    don = load_summary(args.deeponet_summary)
    fno = load_summary(args.fno_summary)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    if "train_mse_history" in don:
        plt.plot(don["train_mse_history"], label="DeepONet train")
    if "val_mse_history" in don:
        plt.plot(don["val_mse_history"], label="DeepONet val")

    if "train_mse_history" in fno:
        plt.plot(fno["train_mse_history"], label="FNO train")
    if "val_mse_history" in fno:
        plt.plot(fno["val_mse_history"], label="FNO val")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training/validation error vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved: {args.out_path}")


if __name__ == "__main__":
    main()