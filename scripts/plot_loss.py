#!/usr/bin/env python3
"""Plot Apollo training losses from the CSV produced by parse_train_log.py.

Usage
-----
    python scripts/plot_loss.py loss.csv -o loss.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ema(vals, alpha=0.002):
    out, s = [], None
    for v in vals:
        s = v if s is None else (1 - alpha) * s + alpha * v
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile")
    ap.add_argument("-o", "--out", default="loss.png")
    args = ap.parse_args()

    steps, d, g, val_steps, val = [], [], [], [], []
    with open(args.csvfile) as fh:
        for r in csv.DictReader(fh):
            if r["loss_d"] and r["loss_g"]:
                steps.append(int(r["step"]))
                d.append(float(r["loss_d"]))
                g.append(float(r["loss_g"]))
            if r["val_loss"]:
                val_steps.append(int(r["step"]))
                val.append(float(r["val_loss"]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(steps, d, lw=0.3, alpha=0.25, color="tab:blue")
    ax[0].plot(steps, ema(d), lw=1.6, color="tab:blue", label="D loss (EMA)")
    ax[0].plot(steps, g, lw=0.3, alpha=0.2, color="tab:red")
    ax[0].plot(steps, ema(g), lw=1.6, color="tab:red", label="G loss (EMA)")
    ax[0].set_title("losses")
    ax[0].set_xlabel("step")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    if val:
        ax[1].plot(val_steps, val, "o-", color="tab:green")
        ax[1].set_title("val_loss (negative SI-SDR, higher is better)")
        ax[1].set_xlabel("step")
        ax[1].grid(alpha=0.3)
    else:
        ax[1].text(0.5, 0.5, "no validation yet", ha="center", va="center")
        ax[1].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out} ({len(steps)} train points, {len(val)} val points)")


if __name__ == "__main__":
    main()
