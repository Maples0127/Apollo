#!/usr/bin/env python3
"""Perceptual + fidelity evaluation of Apollo checkpoints on held-out pairs.

SI-SDR measures waveform fidelity; restoration is judged perceptually, so this
reports PESQ (wideband, 16 kHz) alongside SI-SDR for the codec input and for the
model output. Results are appended per clip so a long run can be polled.

    python scripts/eval_pesq.py --ckpt a.ckpt b.ckpt \
        --eval-dir /data/.../eval_holdout50 --out /tmp/pesq.jsonl
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pesq import pesq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from look2hear.models.apollo import Apollo  # noqa: E402

TARGET_SR = 16000


def sisdr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    tgt = a * ref
    return 10 * np.log10((np.sum(tgt ** 2) + eps) / (np.sum((est - tgt) ** 2) + eps))


def to16k(x):
    t = torch.from_numpy(np.ascontiguousarray(x.T.astype(np.float32)))
    if 44100 != TARGET_SR:
        t = torchaudio.functional.resample(t, 44100, TARGET_SR)
    return t.mean(0).numpy()


def pesq_wb(ref, deg):
    """PESQ can reject clips it considers utterance-free; report NaN then."""
    n = min(len(ref), len(deg))
    try:
        return float(pesq(TARGET_SR, ref[:n], deg[:n], "wb"))
    except Exception:
        return float("nan")


def load_model(path):
    model = Apollo(sr=44100, win=20, feature_dim=256, layer=6)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = {k[len("audio_model."):]: v for k, v in ck["state_dict"].items()
             if k.startswith("audio_model.")}
    model.load_state_dict(state, strict=True)
    return model.eval(), ck.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N clips")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(args.eval_dir, "*/")))
    if args.limit:
        clips = clips[: args.limit]

    with open(args.out, "w") as fh:
        fh.write("")

    for ckpt in args.ckpt:
        model, epoch = load_model(ckpt)
        rows = []
        for d in clips:
            ori, _ = sf.read(os.path.join(d, "ori_wav.wav"))
            codec, _ = sf.read(os.path.join(d, "codec_wav.wav"))
            with torch.no_grad():
                est = model(torch.from_numpy(codec.T.astype(np.float32)).unsqueeze(0))[0].numpy().T
            n = min(len(est), len(ori), len(codec))
            ori, codec, est = ori[:n], codec[:n], est[:n]

            o16, c16, e16 = to16k(ori), to16k(codec), to16k(est)
            row = {
                "ckpt": os.path.basename(ckpt),
                "epoch": epoch,
                "clip": os.path.basename(d.rstrip("/")),
                "sisdr_codec": float(np.mean([sisdr(codec[:, c], ori[:, c]) for c in range(ori.shape[1])])),
                "sisdr_model": float(np.mean([sisdr(est[:, c], ori[:, c]) for c in range(ori.shape[1])])),
                "pesq_codec": pesq_wb(o16, c16),
                "pesq_model": pesq_wb(o16, e16),
            }
            rows.append(row)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")

        def mean(k):
            vals = [r[k] for r in rows if np.isfinite(r[k])]
            return float(np.mean(vals)) if vals else float("nan")

        n_pesq = sum(1 for r in rows if np.isfinite(r["pesq_model"]))
        print(f"epoch={epoch} n={len(rows)} (pesq ok on {n_pesq}) "
              f"SI-SDR codec={mean('sisdr_codec'):.2f} model={mean('sisdr_model'):.2f} "
              f"| PESQ codec={mean('pesq_codec'):.3f} model={mean('pesq_model'):.3f} "
              f"(delta PESQ {mean('pesq_model') - mean('pesq_codec'):+.3f})", flush=True)


if __name__ == "__main__":
    main()
