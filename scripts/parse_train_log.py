#!/usr/bin/env python3
"""Parse a PyTorch-Lightning progress-bar log into a CSV of losses.

Lightning writes the progress bar with carriage returns, so a run's log is a
single "line" full of overwrites. This script recovers one record per logged
step and writes a tidy CSV that can be plotted anywhere.

Usage
-----
    python scripts/parse_train_log.py /data/.../logs/apollo-full_39108.out -o /tmp/loss.csv
"""
import argparse
import csv
import re
import sys

LINE = re.compile(
    r"Epoch (\d+):\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*"
    r"\[(\d+:\d+(?::\d+)?)<[^,\]]*,\s*([\d.]+)it/s"
    r"(?P<rest>[^\]]*)\]"
)
KV = re.compile(r"(\w+)=([^\s,\]]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("-o", "--out", default="loss.csv")
    args = ap.parse_args()

    rows = {}
    with open(args.logfile, errors="ignore") as fh:
        text = fh.read().replace("\r", "\n")

    for line in text.split("\n"):
        m = LINE.search(line)
        if not m:
            continue
        epoch, step, total, elapsed, it_s, rest = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            m.group(4), float(m.group(5)), m.group("rest"),
        )
        kv = dict(KV.findall(rest))
        row = {
            "epoch": epoch,
            "step": step,
            "pct": round(100.0 * step / total, 2),
            "elapsed": elapsed,
            "it_s": it_s,
            "loss_d": kv.get("train_loss_d_step"),
            "loss_g": kv.get("train_loss_g_step"),
            "val_loss": kv.get("val_loss"),
            "lr": kv.get("lr"),
        }
        rows[step] = row          # later render of same step wins

    if not rows:
        sys.exit(f"no progress-bar records found in {args.logfile}")

    order = ["epoch", "step", "pct", "elapsed", "it_s", "loss_d", "loss_g", "val_loss", "lr"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order)
        w.writeheader()
        for step in sorted(rows):
            w.writerow(rows[step])

    n = len(rows)
    last = rows[max(rows)]
    print(f"wrote {n} rows -> {args.out}")
    print(f"last step {last['step']} (epoch {last['epoch']}, {last['pct']}%) "
          f"d={last['loss_d']} g={last['loss_g']} val={last['val_loss']}")


if __name__ == "__main__":
    main()
