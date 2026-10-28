#!/usr/bin/env python3
"""PESQ (wideband) evaluation for an Apollo checkpoint.

SI-SDR only measures waveform fidelity; PESQ is a perceptual proxy for the
"does it sound better than the codec" question. Clips are resampled to 16 kHz
(PESQ-WB requirement) after downmixing to mono.

Usage
-----
    python scripts/eval_pesq.py --ckpt <path> --eval-dir <dir> --out <json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from pesq import pesq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from look2hear.models.apollo import Apollo  # noqa: E402


def to16k(x):
    return resample_poly(x, 16000, 44100)


def load_model(path, device):
    model = Apollo(sr=44100, win=20, feature_dim=256, layer=6)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {k[len("audio_model."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("audio_model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), ckpt.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    model, epoch = load_model(args.ckpt, args.device)
    dirs = sorted(glob.glob(os.path.join(args.eval_dir, "*/")))
    if args.limit:
        dirs = dirs[: args.limit]

    rows, base, mod = [], [], []
    for d in dirs:
        ori, _ = sf.read(os.path.join(d, "ori_wav.wav"))
        codec, _ = sf.read(os.path.join(d, "codec_wav.wav"))
        with torch.no_grad():
            est = model(torch.from_numpy(codec.T.astype(np.float32)).unsqueeze(0).to(args.device))
            est = est[0].cpu().numpy().T
        n = min(len(ori), len(codec), len(est))
        ref = to16k(ori[:n].mean(1))
        c16 = to16k(codec[:n].mean(1))
        e16 = to16k(est[:n].mean(1))
        m = min(len(ref), len(c16), len(e16))
        ref, c16, e16 = ref[:m], c16[:m], e16[:m]
        try:
            p_c = pesq(16000, ref, c16, "wb")
            p_m = pesq(16000, ref, e16, "wb")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {d}: {exc}")
            continue
        base.append(p_c)
        mod.append(p_m)
        rows.append({"clip": os.path.basename(d.rstrip("/")), "codec": p_c, "model": p_m})
        print(f"{rows[-1]['clip']:>8}  codec={p_c:5.2f}  model={p_m:5.2f}  delta={p_m - p_c:+.2f}", flush=True)

    summary = {
        "ckpt": args.ckpt,
        "epoch": epoch,
        "n_clips": len(rows),
        "pesq_codec_mean": float(np.mean(base)),
        "pesq_model_mean": float(np.mean(mod)),
        "delta_mean": float(np.mean(mod) - np.mean(base)),
        "n_model_better": int(sum(r["model"] > r["codec"] for r in rows)),
        "per_clip": rows,
    }
    print("\n=== PESQ summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
