"""Schema-anchored I/O for napari-cotrack CSVs.

One long-form schema everywhere: ``frame,bodypart,x,y,vis``.
Anchor / corrections CSVs are sparse (only labelled rows). Tracks CSVs are
dense (every (frame, bodypart) pair). Same reader either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS_COLS = ["frame", "bodypart", "x", "y", "vis"]


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read a long-form labels/tracks CSV. Always returns the canonical
    columns; missing ``vis`` defaults to 1.0 for back-compat."""
    df = pd.read_csv(path)
    if "vis" not in df.columns:
        df["vis"] = 1.0
    # Validate frame column up front: it must be integer-valued. We allow
    # CSVs that store it as float (e.g. 12.0) but reject fractional or
    # non-numeric entries with row context — otherwise pandas truncates
    # silently and labels land on the wrong frames.
    frame_num = pd.to_numeric(df["frame"], errors="coerce")
    bad = frame_num.isna() | (frame_num != frame_num.round())
    if bad.any():
        bad_rows = df.index[bad].tolist()[:5]
        raise ValueError(
            f"{path}: frame column has non-integer entries at "
            f"row(s) {bad_rows}{'…' if bad.sum() > 5 else ''}",
        )
    df["frame"] = frame_num.astype(int)
    df["bodypart"] = df["bodypart"].astype(str)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["vis"] = df["vis"].astype(float)
    return df[LABELS_COLS]


def write_labels(df: pd.DataFrame, path: str | Path) -> None:
    """Write a long-form CSV, sorted by (frame, bodypart) for stable diffs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df[LABELS_COLS].sort_values(["frame", "bodypart"]).reset_index(drop=True)
    out.to_csv(path, index=False)


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABELS_COLS)


def labels_to_queries(df: pd.DataFrame, total_frames: int):
    """Convert sparse labels to a CoTracker query tensor (1, N, 3) and a
    parallel ``query_info`` list of (bodypart, frame_idx) for merge-back.
    Drops rows with frame >= total_frames, NaN coords, or vis <= 0
    (the deletion sentinel emitted by review.cmd_promote)."""
    df = df.dropna(subset=["x", "y"])
    df = df[df["frame"] < total_frames]
    df = df[df["vis"] > 0]
    if df.empty:
        raise ValueError("No valid labelled points within video frame range")
    rows: list[list[float]] = []
    info: list[tuple[str, int]] = []
    for _, r in df.iterrows():
        rows.append([float(r["frame"]), float(r["x"]), float(r["y"])])
        info.append((str(r["bodypart"]), int(r["frame"])))
    queries = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)
    return queries, info


def union_label_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate multiple labels CSVs, later rounds overriding earlier ones
    on the same (frame, bodypart) key."""
    dfs = [read_labels(p) for p in paths if Path(p).exists()]
    if not dfs:
        return empty_labels()
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["frame", "bodypart"], keep="last")
    return merged.reset_index(drop=True)


def tracks_to_dense(merged_tracks: torch.Tensor, merged_vis: torch.Tensor,
                    bodyparts: list[str]) -> pd.DataFrame:
    """``merged_tracks`` is (1, T, N_bp, 2); ``merged_vis`` is (1, T, N_bp).
    Returns a long-form DataFrame with one row per (frame, bodypart)."""
    T = merged_tracks.shape[1]
    xs = merged_tracks[0, :, :, 0].numpy()
    ys = merged_tracks[0, :, :, 1].numpy()
    vs = merged_vis[0].cpu().numpy() if merged_vis.dtype != torch.float32 else merged_vis[0].numpy()
    rows = []
    for t in range(T):
        for i, bp in enumerate(bodyparts):
            rows.append((int(t), bp, float(xs[t, i]), float(ys[t, i]), float(vs[t, i])))
    return pd.DataFrame(rows, columns=LABELS_COLS)


def dense_to_arrays(df: pd.DataFrame
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Inverse of ``tracks_to_dense``. Returns xs, ys, vis (each (T, N))
    and the bodypart order as encountered."""
    if df.empty:
        return (np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), [])
    bodyparts = list(dict.fromkeys(df["bodypart"].tolist()))
    T = int(df["frame"].max()) + 1
    N = len(bodyparts)
    xs = np.full((T, N), np.nan)
    ys = np.full((T, N), np.nan)
    vs = np.zeros((T, N))
    bp_idx = {bp: i for i, bp in enumerate(bodyparts)}
    for _, r in df.iterrows():
        t = int(r["frame"])
        i = bp_idx[r["bodypart"]]
        xs[t, i] = r["x"]
        ys[t, i] = r["y"]
        vs[t, i] = r["vis"]
    return xs, ys, vs, bodyparts
