#!/usr/bin/env python3
"""Evaluate an Apollo checkpoint on held-out ``ori_wav/codec_wav`` pairs.

Reports, per clip and on average:
  * codec baseline SI-SDR   (the compressed input, i.e. "do nothing")
  * model SI-SDR            (the restored output)
  * val_loss as the training metric computes it (negative mean SI-SDR)

Usage
-----
    python scripts/eval_checkpoint.py --ckpt path/to/last.ckpt \
        --eval-dir /data/.../eval_holdout --out results.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from look2hear.losses import MultiSrcNegSDR  # noqa: E402
from look2hear.models.apollo import Apollo  # noqa: E402


def sisdr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum((est - target) ** 2) + eps))


def load_model(path, device):
    model = Apollo(sr=44100, win=20, feature_dim=256, layer=6)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {k[len("audio_model."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("audio_model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), ckpt.get("epoch"), ckpt.get("global_step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, epoch, step = load_model(args.ckpt, args.device)
    metric = MultiSrcNegSDR("sisdr")

    rows, base, model_sdr, val_losses = [], [], [], []
    for d in sorted(glob.glob(os.path.join(args.eval_dir, "*/"))):
        ori, _ = sf.read(os.path.join(d, "ori_wav.wav"))
        codec, _ = sf.read(os.path.join(d, "codec_wav.wav"))
        with torch.no_grad():
            est = model(torch.from_numpy(codec.T.astype(np.float32)).unsqueeze(0).to(args.device))
            est = est[0].cpu().numpy().T
        n = min(len(est), len(ori), len(codec))
        ori, codec, est = ori[:n], codec[:n], est[:n]

        b = float(np.mean([sisdr(codec[:, c], ori[:, c]) for c in range(ori.shape[1])]))
        m = float(np.mean([sisdr(est[:, c], ori[:, c]) for c in range(ori.shape[1])]))
        e_t = torch.from_numpy(est.T[None].astype(np.float32))
        r_t = torch.from_numpy(ori.T[None].astype(np.float32))
        vl = float(metric(e_t, r_t))
        base.append(b)
        model_sdr.append(m)
        val_losses.append(vl)
        rows.append({"clip": os.path.basename(d.rstrip("/")), "codec": b, "model": m, "delta": m - b})
        print(f"{rows[-1]['clip']:>8}  codec={b:6.2f}  model={m:6.2f}  delta={m - b:+5.2f}")

    summary = {
        "ckpt": args.ckpt,
        "epoch": epoch,
        "global_step": step,
        "n_clips": len(rows),
        "codec_sisdr_mean": float(np.mean(base)),
        "model_sisdr_mean": float(np.mean(model_sdr)),
        "delta_mean": float(np.mean(model_sdr) - np.mean(base)),
        "val_loss_mean": float(np.mean(val_losses)),
        "per_clip": rows,
    }
    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
