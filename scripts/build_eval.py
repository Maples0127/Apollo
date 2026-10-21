#!/usr/bin/env python3
"""Build a held-out evaluation set for Apollo from MUSDB18-HQ test tracks.

Creates ``<out>/trackNN/{ori_wav.wav, codec_wav.wav}`` pairs, which is what
``look2hear.datas.MusdbMoisesdbEval`` reads. Only test tracks are used, so as
long as training HDF5 is built from ``train/`` only, the validation signal is
not contaminated.

Usage
-----
    python scripts/build_eval.py \
        --test-root /data/.../musdb18hq/test --out /data/.../eval_holdout --count 12
"""
import argparse
import os
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import apply_codec


def pick_segment(x, sr, seconds=6.0):
    """Pick the loudest non-silent window of ``seconds`` from ``x`` (T, C)."""
    seg = int(sr * seconds)
    if x.shape[0] <= seg:
        return x
    hop = seg // 2
    n = (x.shape[0] - seg) // hop
    best, best_e = 0, -1.0
    for i in range(n):
        e = float(np.mean(x[i * hop : i * hop + seg] ** 2))
        if e > best_e:
            best, best_e = i, e
    return x[best * hop : best * hop + seg]


def encode(x, sr, bitrate):
    wav = torch.from_numpy(x.T.astype(np.float32))  # (C, T)
    enc = apply_codec(wav, sr, format="mp3", compression=bitrate // 1000)
    if enc.shape[-1] >= wav.shape[-1]:
        enc = enc[..., : wav.shape[-1]]
    else:
        enc = torch.cat([enc, wav[..., enc.shape[-1] :]], -1)
    return enc.numpy().T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--bitrates", type=int, nargs="+", default=[32000, 48000, 64000, 96000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tracks = sorted(d for d in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, d)))
    if not tracks:
        raise SystemExit(f"no tracks under {args.test_root}")

    made = 0
    for i in range(args.count):
        track = tracks[i % len(tracks)]
        path = os.path.join(args.test_root, track, "mixture.wav")
        x, sr = sf.read(path)
        x = pick_segment(x, sr, args.seconds)
        br = args.bitrates[i % len(args.bitrates)]
        codec = encode(x, sr, br)
        d = os.path.join(args.out, f"track{i:02d}")
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, "ori_wav.wav"), x, sr)
        sf.write(os.path.join(d, "codec_wav.wav"), codec, sr)
        made += 1
        print(f"{d}: {track} | {br // 1000} kbps | {x.shape[0] / sr:.1f}s")

    print(f"wrote {made} eval pairs to {args.out}")


if __name__ == "__main__":
    main()
