"""Stage 1: anchor frame extraction (no DLC).

Two modes:
  - uniform   every Nth frame
  - diverse   K most-different frames via k-means on RGB histograms

Plus a corrections-extract path for stage-4 re-labelling that prefills labels
from an existing tracks CSV.

All output is long-form CSV: ``frame,bodypart,x,y,vis``.

Module CLI:
    python -m napari_cotrack.pipeline.extract anchors \\
        --video VIDEO --output DIR --bodyparts thumb,index,wrist \\
        --mode diverse --n 30
    python -m napari_cotrack.pipeline.extract corrections \\
        --video VIDEO --output DIR --bodyparts thumb,index,wrist \\
        --tracks TRACKS_CSV --frames 12,45,67
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd

from napari_cotrack.pipeline._io import (
    LABELS_COLS, dense_to_arrays, read_labels, write_labels,
)


# --------------------------- Frame selection ------------------------------- #


def select_uniform(total_frames: int, stride: int) -> list[int]:
    """Every ``stride``-th frame: 0, stride, 2*stride, … up to total_frames."""
    if stride <= 0:
        return []
    return list(range(0, total_frames, stride))


def select_diverse(video_path: str, k: int,
                   stride: int = 5, hist_bins: int = 8,
                   batch_size: int = 256, seed: int = 0,
                   total_frames: int | None = None) -> list[int]:
    """Pick k visually distinct frames via k-means on RGB histograms.

    Pass ``total_frames`` if the caller already knows it — saves an
    open-and-count pass over the whole video."""
    if k <= 0:
        return []

    from sklearn.cluster import MiniBatchKMeans

    reader = imageio.get_reader(video_path)
    total = total_frames if total_frames is not None else reader.count_frames()
    sampled_indices: list[int] = list(range(0, total, max(1, stride)))

    if k >= len(sampled_indices):
        out = sorted(sampled_indices[:k])
        reader.close()
        return out

    feats = np.zeros((len(sampled_indices), hist_bins ** 3), dtype=np.float32)
    for i, idx in enumerate(sampled_indices):
        try:
            frame = reader.get_data(idx)
        except (IndexError, StopIteration):
            continue
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist(
            [small], [0, 1, 2], None,
            [hist_bins, hist_bins, hist_bins],
            [0, 256, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        feats[i] = hist.flatten()
    reader.close()

    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed,
        batch_size=min(batch_size, len(sampled_indices)),
        n_init="auto",
    )
    km.fit(feats)

    chosen: set[int] = set()
    for c in range(k):
        diffs = feats - km.cluster_centers_[c]
        d2 = (diffs * diffs).sum(axis=1)
        best = int(d2.argmin())
        chosen.add(sampled_indices[best])
        feats[best] = np.inf

    return sorted(chosen)


# --------------------------- Frame writing --------------------------------- #


def write_pngs(video_path: str, frame_indices: Sequence[int],
               output_dir: str | Path) -> list[int]:
    """Write the selected frames as imgNNNN.png. Returns the list of indices
    that were actually written (in case of unreadable frames)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path)
    written: list[int] = []
    for idx in frame_indices:
        try:
            frame = reader.get_data(idx)
        except (IndexError, StopIteration):
            print(f"  Warning: could not read frame {idx}, skipping")
            continue
        imageio.imwrite(output_dir / f"img{idx:04d}.png", frame)
        written.append(idx)
    reader.close()
    return written


# --------------------------- Anchor extraction ----------------------------- #


def extract_anchors(video_path: str, frame_indices: Sequence[int],
                    output_dir: str | Path, bodyparts: list[str]) -> Path:
    """Stage 1 (initial labelling): write PNGs + an empty labels CSV.

    The CSV contains no rows yet — the user will create them by labelling in
    napari. We write the file (with header only) so downstream code can
    blindly append/read.
    """
    output_dir = Path(output_dir)
    written = write_pngs(video_path, frame_indices, output_dir)
    if not written:
        raise RuntimeError("No frames extracted")
    labels_path = output_dir / "labels.csv"
    write_labels(pd.DataFrame(columns=LABELS_COLS), labels_path)
    print(f"  Saved {len(written)} frames + empty labels.csv to {output_dir}")
    return labels_path


def extract_corrections(video_path: str, frame_indices: Sequence[int],
                        output_dir: str | Path, tracks_csv: str | Path,
                        bodyparts: list[str]) -> Path:
    """Stage 4 (correction round): write PNGs + a labels CSV prefilled from
    the dense tracks CSV. Only the requested frames are prefilled."""
    output_dir = Path(output_dir)
    df = read_labels(tracks_csv)  # tracks.csv uses the same schema
    xs, ys, vs, df_bodyparts = dense_to_arrays(df)

    written = write_pngs(video_path, frame_indices, output_dir)
    if not written:
        raise RuntimeError("No frames extracted")

    rows = []
    for fi in written:
        for bp in bodyparts:
            if bp in df_bodyparts:
                bi = df_bodyparts.index(bp)
                if fi < xs.shape[0]:
                    rows.append((fi, bp, float(xs[fi, bi]), float(ys[fi, bi]),
                                 float(vs[fi, bi])))
    out_df = pd.DataFrame(rows, columns=LABELS_COLS)
    labels_path = output_dir / "labels.csv"
    write_labels(out_df, labels_path)
    print(f"  Saved {len(written)} frames + prefilled {len(out_df)} rows to {output_dir}")
    return labels_path


# --------------------------- CLI ------------------------------------------- #


def _parse_bodyparts(s: str) -> list[str]:
    return [bp.strip() for bp in s.split(",") if bp.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("anchors", help="Extract anchor frames for initial labelling")
    a.add_argument("--video", required=True)
    a.add_argument("--output", required=True, help="anchors/ folder")
    a.add_argument("--bodyparts", required=True, type=_parse_bodyparts)
    a.add_argument("--mode", choices=["uniform", "diverse"], default="diverse")
    a.add_argument("--n", type=int, default=30,
                   help="In 'uniform' mode this is the stride (every Nth frame). "
                        "In 'diverse' mode this is K, the number of frames to pick.")
    a.add_argument("--stride", type=int, default=5,
                   help="(diverse mode only) sample candidates every Nth frame")
    a.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("corrections",
                       help="Extract correction frames with tracks prefill")
    c.add_argument("--video", required=True)
    c.add_argument("--output", required=True, help="corrections/round_NNN/ folder")
    c.add_argument("--bodyparts", required=True, type=_parse_bodyparts)
    c.add_argument("--tracks", required=True, help="tracks.csv to prefill from")
    c.add_argument("--frames", required=True, type=_parse_int_list)

    args = p.parse_args(argv)

    if args.command == "anchors":
        reader = imageio.get_reader(args.video)
        total = reader.count_frames()
        reader.close()
        if args.mode == "uniform":
            indices = select_uniform(total, args.n)
        else:
            indices = select_diverse(args.video, args.n,
                                     stride=args.stride, seed=args.seed,
                                     total_frames=total)
        print(f"Selected {len(indices)} {args.mode} frames out of {total}")
        extract_anchors(args.video, indices, args.output, args.bodyparts)
        return 0

    if args.command == "corrections":
        extract_corrections(
            args.video, args.frames, args.output, args.tracks, args.bodyparts,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
