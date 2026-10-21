#!/usr/bin/env python3
"""Build Apollo training HDF5 from MUSDB18-HQ / MoisesDB source trees.

Output layout matches what ``look2hear.datas.MusdbMoisesdbDataModule`` expects::

    <out>/<stem>/sampleN.hdf5      # each file: data (n_segments, 2, sr*session)

Two on-disk layouts are supported:

* MUSDB18-HQ: ``<track>/{mixture,vocals,drums,bass,other}.wav``
* MoisesDB:   ``<track>/<stem>/<source-uuid>.wav`` (several sources per stem)

For MoisesDB all sources of a stem are summed, and the mixture is synthesised as
the sum of all stems (this is how ``moisesdb`` package defines ``track.audio``,
MoisesDB ships no mixture.wav). Files are segmented with the same energy VAD as
``JusperLee/Apollo-data-preprocess``.

Usage
-----
    python scripts/preprocess_hdf5.py \
        --roots /data/.../musdb18hq/train /data/.../moisesdb/moisesdb_v0.1 \
        --out /data/.../hdf5_train
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


def _wavs(path):
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(path, f))
    )


STEM_NAMES = {
    "bass", "bowed_strings", "drums", "guitar", "other", "other_keys",
    "other_plucked", "percussion", "piano", "vocals", "wind",
}


def iter_tracks(roots):
    """Yield (dirpath, {stem: [wav, ...]}, mixture_wav_or_None).

    A directory counts as a track when it holds ``mixture.wav`` (MUSDB18-HQ),
    direct wavs named after known stems, or per-stem subdirectories
    (MoisesDB). Traversal is pruned inside recognised tracks.
    """
    for root in roots:
        for dirpath, dirnames, fnames in os.walk(root):
            dirnames.sort()
            direct = {os.path.splitext(f)[0]: os.path.join(dirpath, f)
                      for f in fnames if f.lower().endswith(".wav")}

            sub_stems = {}
            for d in dirnames:
                if d in STEM_NAMES:
                    sub = _wavs(os.path.join(dirpath, d))
                    if sub:
                        sub_stems[d] = sub

            if "mixture" in direct:
                stems = {k: [v] for k, v in direct.items() if k != "mixture"}
                stems.update(sub_stems)
            elif sub_stems and not direct:
                stems = sub_stems
            elif direct and set(direct) <= STEM_NAMES:
                stems = {k: [v] for k, v in direct.items()}
            else:
                continue

            dirnames[:] = []  # do not descend into a recognised track
            yield dirpath, stems, direct.get("mixture")


def load_group(paths, sr_expected):
    """Load and sum all wavs in a stem group, aligned to the shortest one."""
    arrays = []
    for p in paths:
        x, sr = sf.read(p)
        if sr != sr_expected:
            raise SystemExit(f"{p}: sample rate {sr} != {sr_expected}")
        arrays.append(x)
    if len(arrays) == 1:
        return arrays[0]
    length = min(a.shape[0] for a in arrays)
    return sum(a[:length] for a in arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--session", type=float, default=6.0)
    ap.add_argument("--min-stems", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0, help="shard index for parallel runs")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    counters, nseg = {}, {}
    ntracks = 0
    skipped = []

    tracks = list(iter_tracks(args.roots))
    if args.num_shards > 1:
        tracks = tracks[args.shard :: args.num_shards]
    for dirpath, stems, mixture in tqdm(tracks, desc="tracks", dynamic_ncols=True):
        if len(stems) < args.min_stems:
            skipped.append(dirpath)
            continue

        arrays = {name: load_group(paths, args.sr) for name, paths in stems.items()}
        if mixture is not None:
            mx, _ = sf.read(mixture)
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
