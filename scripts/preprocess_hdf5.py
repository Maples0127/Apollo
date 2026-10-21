#!/usr/bin/env python3
"""Build Apollo training HDF5 from MUSDB18-HQ / MoisesDB source trees.

Output layout matches what ``look2hear.datas.MusdbMoisesdbDataModule`` expects::

    <out>/<stem>/sampleN.hdf5      # each file: data (n_segments, 2, sr*session)

Every directory that looks like a track (contains ``mixture.wav`` or several
stem wavs) is segmented by the same energy VAD used by
``JusperLee/Apollo-data-preprocess`` and split into ``session``-second chunks.

Datasets without ``mixture.wav`` (MoisesDB) get the mixture synthesised as the
sample-wise sum of the track's stems, which is how the MoisesDB package defines
``track.audio``.

Usage
-----
    python scripts/preprocess_hdf5.py \
        --roots /data/.../musdb18hq/train --out /data/.../hdf5_train
"""
import argparse
import json
import os

import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm


def vad_segments(x, sr, session_length=6.0, win=441):
    """Split ``x`` (samples, channels) into non-silent ``session_length`` chunks."""
    if x.ndim == 1:
        x = x[:, None]
    n_ch = x.shape[1]
    win = int(win)
    x = x[: x.shape[0] // sr * sr]
    xp = x.reshape(-1, win, n_ch)
    pow_ = np.sum(np.power(xp, 2), (1, 2)).reshape(-1,)
    valid = pow_[pow_ > 1e-3]
    if valid.size == 0:
        return []

    thr = np.quantile(valid, 0.15)
    seg_len = int(sr * session_length)
    hop = seg_len // 2
    n = (x.shape[0] - seg_len) // hop
    out = []
    for s in range(n):
        seg = x[s * hop : s * hop + seg_len]
        sp = seg.reshape(-1, win, n_ch)
        sp = np.sum(np.power(sp, 2), (1, 2)).reshape(-1,)
        if np.sum(sp > thr) / sp.size > 0.5:
            out.append(seg)
    return out


def iter_tracks(roots):
    for root in roots:
        for dirpath, _dirnames, fnames in os.walk(root):
            wavs = {
                os.path.splitext(f)[0]: os.path.join(dirpath, f)
                for f in fnames
                if f.lower().endswith(".wav")
            }
            if "mixture" in wavs or len(wavs) > 1:
                yield dirpath, wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--session", type=float, default=6.0)
    ap.add_argument("--min-stems", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    counters = {}
    nseg = {}
    ntracks = 0
    skipped = []

    tracks = list(iter_tracks(args.roots))
    for dirpath, wavs in tqdm(tracks, desc="tracks", dynamic_ncols=True):
        stems = {k: v for k, v in wavs.items() if k != "mixture"}
        if len(stems) < args.min_stems:
            skipped.append(dirpath)
            continue

        arrays = {}
        for name, path in stems.items():
            x, sr = sf.read(path)
            if sr != args.sr:
                raise SystemExit(f"{path}: sample rate {sr} != {args.sr}")
            arrays[name] = x

        if "mixture" in wavs:
            mx, _ = sf.read(wavs["mixture"])
        else:
            length = min(a.shape[0] for a in arrays.values())
            mx = sum(a[:length] for a in arrays.values())
        arrays["mixture"] = mx

        ntracks += 1
        for name, x in arrays.items():
            segs = vad_segments(x, args.sr, args.session)
            if not segs:
                continue
            d = os.path.join(args.out, name)
            os.makedirs(d, exist_ok=True)
            c = counters.get(name, len(os.listdir(d)))
            counters[name] = c + 1
            data = np.stack([s.T for s in segs]).astype(np.float64)
            with h5py.File(os.path.join(d, f"sample{c + 1}.hdf5"), "w") as h:
                h.create_dataset("data", data=data)
            nseg[name] = nseg.get(name, 0) + len(segs)

    manifest = {
        "roots": args.roots,
        "sr": args.sr,
        "session_seconds": args.session,
        "tracks": ntracks,
        "segments_per_stem": nseg,
        "skipped_dirs": skipped,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: manifest[k] for k in ("tracks", "segments_per_stem")}, indent=2))


if __name__ == "__main__":
    main()
