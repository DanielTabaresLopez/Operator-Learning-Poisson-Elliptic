import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, default="runs/benchmarks/results.csv")
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.results_csv)

    # Keep only rows with finite metrics
    df = df[np.isfinite(df["mean_rel_l2"].values)]
    if df.empty:
        raise RuntimeError("No valid rows found in results.csv")

    # -----------------------
    # 1) Summary bar: mean error by model (best per model)
    # -----------------------
    # Pick best checkpoint per model (lowest mean_rel_l2)
    best = df.sort_values("mean_rel_l2").groupby("model", as_index=False).first()

    plt.figure()
    plt.bar(best["model"], best["mean_rel_l2"])
    plt.ylabel("Mean relative L2 error (test)")
    plt.title("Best checkpoint per model")
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "mean_error_bar.png", dpi=200)
    plt.close()

    # -----------------------
    # 2) Summary bar: inference time per sample (best per model)
    # -----------------------
    plt.figure()
    plt.bar(best["model"], best["infer_ms_per_sample"])
    plt.ylabel("Inference time (ms / sample)")
    plt.title("Best checkpoint per model")
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "time_bar.png", dpi=200)
    plt.close()

    # -----------------------
    # 3) DeepONet sweep: error vs sensors (if present)
    # -----------------------
    don = df[df["model"] == "deeponet"].copy()
    if "n_sensors" in don.columns and don["n_sensors"].notna().any():
        # For each sensor count, keep best (lowest error)
        don_best = don.sort_values("mean_rel_l2").groupby("n_sensors", as_index=False).first()
        don_best = don_best.sort_values("n_sensors")

        plt.figure()
        plt.plot(don_best["n_sensors"], don_best["mean_rel_l2"], marker="o")
        plt.xlabel("Number of sensors (DeepONet)")
        plt.ylabel("Mean relative L2 error (test)")
        plt.title("DeepONet: error vs sensors (best per sensor count)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "error_vs_sensors.png", dpi=200)
        plt.close()

    # -----------------------
    # 4) FNO sweep: error vs modes (if present)
    # -----------------------
    fno = df[df["model"] == "fno"].copy()
    if "modes" in fno.columns and fno["modes"].notna().any():
        fno_best = fno.sort_values("mean_rel_l2").groupby("modes", as_index=False).first()
        fno_best = fno_best.sort_values("modes")

        plt.figure()
        plt.plot(fno_best["modes"], fno_best["mean_rel_l2"], marker="o")
        plt.xlabel("Number of Fourier modes (FNO)")
        plt.ylabel("Mean relative L2 error (test)")
        plt.title("FNO: error vs modes (best per modes)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "error_vs_modes.png", dpi=200)
        plt.close()

    # -----------------------
    # 5) Data efficiency: error vs n_train (if multiple datasets)
    # -----------------------
    if df["n_train"].nunique() > 1:
        # Choose best checkpoint per (model, n_train)
        eff = df.sort_values("mean_rel_l2").groupby(["model", "n_train"], as_index=False).first()
        plt.figure()
        for model_name in eff["model"].unique():
            sub = eff[eff["model"] == model_name].sort_values("n_train")
            plt.plot(sub["n_train"], sub["mean_rel_l2"], marker="o", label=model_name)
        plt.xlabel("Number of training samples")
        plt.ylabel("Mean relative L2 error (test)")
        plt.title("Data efficiency (best per n_train)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "error_vs_ntrain.png", dpi=200)
        plt.close()

    print(f"[OK] Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()