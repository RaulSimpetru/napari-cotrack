"""Stage 4: review-and-fix workflow (no DLC).

Three responsibilities, all CLI-exposed:

  - extract-all   dump every frame to ``review/img####.png``, prefill
                  ``review/labels.csv`` from the latest tracks CSV, snapshot
                  ``review/baseline.csv`` for diff-promote.
  - promote       diff ``review/labels.csv`` vs ``review/baseline.csv`` on
                  the (frame, bodypart) key, write only the changed rows to
                  ``corrections/round_NNN/labels.csv``.
  - jumps         detect multi-bodypart scene-confusion frames; either preview
                  or extract a representative frame per bad range as a fresh
                  corrections round for relabel.

No naive linear-interpolation _fixed.csv path — the interp wandered during

Module CLI:
    python -m napari_cotrack.pipeline.review extract-all --project P
    python -m napari_cotrack.pipeline.review promote --project P [--tolerance 1.0]
    python -m napari_cotrack.pipeline.review jumps --project P
        [--n-sigmas 3 --min-frac 0.5 --max-run 60] [--extract-corrections]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from napari_cotrack import project as P
from napari_cotrack.pipeline import _ffmpeg
from napari_cotrack.pipeline._io import (
    LABELS_COLS, dense_to_arrays, read_labels, write_labels,
)

VERBOSE = False


def _v(msg: str) -> None:
    if VERBOSE:
        print(msg)


# --------------------------- extract-all ----------------------------------- #


def do_extract_all(project: str | Path, *, force: bool = False) -> int:
    """Stage-4 extract-all as a kwarg-friendly callable. The CLI wraps this."""
    proj = P.load(Path(project))
    csv = proj.tracks_filtered_csv if proj.tracks_filtered_csv.exists() else proj.tracks_csv
    if not csv.exists():
        print(f"No tracks CSV ({proj.tracks_csv}); run stage 3 first", file=sys.stderr)
        return 1

    df = read_labels(csv)
    T = int(df["frame"].max()) + 1 if not df.empty else 0
    print(f"Tracks CSV {csv.name}: {T} frames")

    if proj.review_dir.exists():
        if not force:
            print(f"Review dir exists: {proj.review_dir}", file=sys.stderr)
            print("Pass --force to overwrite, or run 'promote' to keep your edits",
                  file=sys.stderr)
            return 1
        print(f"Overwriting {proj.review_dir}")
        shutil.rmtree(proj.review_dir)
    proj.review_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {T} frames to {proj.review_dir}/")
    progress_step = max(1, T // 10)
    written = 0
    aborted = False

    # Fast path: one ffmpeg pass writes all PNGs sequentially. Roughly 10-50x
    # faster than imageio per-frame seeks on H.264 sources.
    if _ffmpeg.have_ffmpeg():
        last_logged = -progress_step
        def _on_progress(frame: int) -> None:
            nonlocal last_logged
            if frame >= last_logged + progress_step:
                last_logged = frame
                print(f"  extracted {frame}/{T}  ({100 * frame // T}%)")
        try:
            written = _ffmpeg.extract_all_frames(
                proj.video, proj.review_dir, on_progress=_on_progress,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ffmpeg fast path failed ({e}); falling back to imageio",
                  file=sys.stderr)
            written = 0
    if written == 0:
        # Fallback: imageio per-frame seek. Slow but works without ffmpeg.
        reader = iio.imopen(proj.video, "r")
        try:
            for f in range(T):
                try:
                    frame = reader.read(index=f)
                except Exception:
                    print(f"  Could not read frame {f}, stopping early", file=sys.stderr)
                    aborted = True
                    break
                iio.imwrite(proj.review_dir / f"img{f:04d}.png", frame)
                written += 1
                if f == 0 or f % progress_step == 0:
                    print(f"  extracted {f}/{T}  ({100 * f // T}%)")
        finally:
            reader.close()

    if aborted:
        print(f"Extracted only {written}/{T} frames; not writing labels.csv. "
              f"Re-encode the video and retry.", file=sys.stderr)
        return 1

    write_labels(df, proj.review_labels)
    write_labels(df, proj.review_baseline)
    _v(f"Wrote labels: {proj.review_labels}")
    _v(f"Snapshot baseline: {proj.review_baseline}")
    print(f"Wrote review folder. Open via stage 4 → Open review in napari.")
    return 0


# --------------------------- promote --------------------------------------- #


def do_promote(project: str | Path, *, tolerance: float = 1.0) -> int:
    proj = P.load(Path(project))
    if not (proj.review_labels.exists() and proj.review_baseline.exists()):
        print(f"Missing review labels or baseline in {proj.review_dir}", file=sys.stderr)
        return 1

    cur = read_labels(proj.review_labels)
    base = read_labels(proj.review_baseline)

    # Key by (frame, bodypart). Any row whose x or y differs by more than
    # `tolerance` from the baseline counts as edited. A row absent from `cur`
    # but present in baseline is treated as a deletion → record with vis=0
    # so the next track sees "do not propagate from here." A row in `cur`
    # but absent from baseline is a brand-new manual label → keep.
    cur_idx = cur.set_index(["frame", "bodypart"])
    base_idx = base.set_index(["frame", "bodypart"])

    in_both = cur_idx.index.intersection(base_idx.index)
    diff = (cur_idx.loc[in_both, ["x", "y"]] - base_idx.loc[in_both, ["x", "y"]]).abs()
    edited_keys = diff[(diff > tolerance).any(axis=1)].index

    new_keys = cur_idx.index.difference(base_idx.index)
    deleted_keys = base_idx.index.difference(cur_idx.index)

    edited_rows = cur_idx.loc[edited_keys].reset_index() if len(edited_keys) else cur.iloc[0:0].copy()
    new_rows = cur_idx.loc[new_keys].reset_index() if len(new_keys) else cur.iloc[0:0].copy()
    deleted_rows = base_idx.loc[deleted_keys].reset_index() if len(deleted_keys) else base.iloc[0:0].copy()
    if not deleted_rows.empty:
        deleted_rows["vis"] = 0.0  # deletion sentinel

    out = pd.concat([edited_rows, new_rows, deleted_rows], ignore_index=True)
    n_total = len(out)
    print(f"Edited: {len(edited_rows)} rows  New: {len(new_rows)}  Deleted: {len(deleted_rows)}")
    if n_total == 0:
        print("Nothing to promote.")
        return 0

    target = proj.next_corrections_round()
    target.mkdir(parents=True, exist_ok=True)
    _v(f"Creating {target}/")

    for fi in sorted(out["frame"].unique()):
        png = proj.review_dir / f"img{int(fi):04d}.png"
        if png.exists():
            shutil.copy2(png, target / png.name)

    write_labels(out, target / "labels.csv")
    print(f"Wrote {n_total} edited rows + frame PNGs to {target}/")
    print("Re-run stage 3 to fold these corrections into the next pass.")
    return 0


# --------------------------- jump detection -------------------------------- #


def detect_jump_frames(xs: np.ndarray, ys: np.ndarray,
                       n_sigmas: float, min_frac: float):
    T, N = xs.shape
    dx = np.diff(xs, axis=0, prepend=xs[:1])
    dy = np.diff(ys, axis=0, prepend=ys[:1])
    speed = np.hypot(dx, dy)
    med = np.median(speed, axis=0)
    mad = np.median(np.abs(speed - med), axis=0)
    sigma = 1.4826 * np.maximum(mad, 1e-6)
    z = (speed - med) / sigma
    n_spike = (z > n_sigmas).sum(axis=1)
    threshold = max(2, int(np.ceil(min_frac * N)))
    return n_spike >= threshold, speed


def find_bad_ranges(is_jump: np.ndarray, max_run: int, max_gap: int = 3):
    ranges: list[tuple[int, int]] = []
    in_range = False
    start = 0
    last_jump = -1
    T = len(is_jump)
    for t in range(T):
        if is_jump[t]:
            if not in_range:
                in_range = True
                start = t
            last_jump = t
        elif in_range and t - last_jump > max_gap:
            end = last_jump + 1
            if end - start <= max_run:
                ranges.append((start, end))
            in_range = False
    if in_range:
        end = last_jump + 1
        if end - start <= max_run:
            ranges.append((start, end))
    return ranges


def do_jumps(project: str | Path, *,
             n_sigmas: float | None = None,
             min_frac: float | None = None,
             max_run: int | None = None,
             extract_corrections: bool = False,
             frames_per_range: int = 1) -> int:
    proj = P.load(Path(project))
    csv = proj.tracks_filtered_csv if proj.tracks_filtered_csv.exists() else proj.tracks_csv
    if not csv.exists():
        print(f"No tracks at {proj.tracks_csv}; run stage 3 first", file=sys.stderr)
        return 1

    df = read_labels(csv)
    xs, ys, _vs, bodyparts = dense_to_arrays(df)
    T, N = xs.shape
    print(f"Loaded {T} frames x {N} bodyparts")

    jp = proj.jumps
    n_sigmas = jp.n_sigmas if n_sigmas is None else n_sigmas
    min_frac = jp.min_frac if min_frac is None else min_frac
    max_run = jp.max_run if max_run is None else max_run

    is_jump, speed = detect_jump_frames(xs, ys, n_sigmas, min_frac)
    bad_ranges = find_bad_ranges(is_jump, max_run)
    n_bad = sum(end - start for start, end in bad_ranges)
    print(f"Detected {len(bad_ranges)} bad ranges ({n_bad} frames, "
          f"{100 * n_bad / T:.1f}% of video)")
    for start, end in bad_ranges:
        _v(f"  frames {start:>5d}-{end - 1:<5d}  ({end - start} frames)")

    plot_dir = proj.root / "plots"
    plot_dir.mkdir(exist_ok=True)
    score = (speed - np.median(speed, axis=0))
    score = np.clip(score, 0, None).sum(axis=1)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(score, color="black", linewidth=0.7)
    for start, end in bad_ranges:
        ax.axvspan(start, end, color="red", alpha=0.25)
    ax.set_xlabel("frame")
    ax.set_ylabel("global jump score")
    ax.set_title(f"Bad ranges (red shaded) — {len(bad_ranges)} ranges, {n_bad} frames")
    fig.tight_layout()
    fig.savefig(plot_dir / "jumps.png", dpi=130)
    plt.close(fig)
    print(f"Plot: {plot_dir}/jumps.png")

    if extract_corrections and bad_ranges:
        target = proj.next_corrections_round()
        target.mkdir(parents=True, exist_ok=True)
        frames_per = max(1, frames_per_range)
        picks: list[int] = []
        for start, end in bad_ranges:
            if frames_per == 1:
                picks.append((start + end - 1) // 2)
            else:
                step = max(1, (end - start) // frames_per)
                for i in range(frames_per):
                    picks.append(min(end - 1, start + i * step))
        picks = sorted(set(picks))

        # Reuse the tracks-prefill helper from extract.py
        from napari_cotrack.pipeline.extract import extract_corrections as _ext
        _ext(proj.video, picks, target, csv, proj.bodyparts)
        print(f"Extracted bad-range frames into {target}/. Re-label them in stage 2.")

    return 0


# Thin argparse wrappers — keep `cmd_*` API for the CLI, route through `do_*`.

def cmd_extract_all(args) -> int:
    return do_extract_all(args.project, force=args.force)


def cmd_promote(args) -> int:
    return do_promote(args.project, tolerance=args.tolerance)


def cmd_jumps(args) -> int:
    return do_jumps(
        args.project,
        n_sigmas=args.n_sigmas, min_frac=args.min_frac, max_run=args.max_run,
        extract_corrections=args.extract_corrections,
        frames_per_range=args.frames_per_range,
    )


# --------------------------- CLI ------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", "-v", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract-all")
    e.add_argument("--project", required=True)
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_extract_all)

    p = sub.add_parser("promote")
    p.add_argument("--project", required=True)
    p.add_argument("--tolerance", type=float, default=1.0)
    p.set_defaults(func=cmd_promote)

    j = sub.add_parser("jumps")
    j.add_argument("--project", required=True)
    j.add_argument("--n-sigmas", type=float, default=None)
    j.add_argument("--min-frac", type=float, default=None)
    j.add_argument("--max-run", type=int, default=None)
    j.add_argument("--extract-corrections", action="store_true")
    j.add_argument("--frames-per-range", type=int, default=1)
    j.set_defaults(func=cmd_jumps)

    args = ap.parse_args(argv)
    VERBOSE = args.verbose
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
