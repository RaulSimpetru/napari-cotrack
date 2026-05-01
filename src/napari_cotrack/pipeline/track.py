"""Stage 3: CoTracker3 anchor-propagation driver (no DLC).

Reads a project (project.toml + anchors/labels.csv + corrections/round_*/labels.csv),
runs CoTracker3 with optional ROI crop, chunked stitching, and a refine pass,
and writes a dense long-form ``tracks.csv``.

Module CLI:
    python -m napari_cotrack.pipeline.track --project /path/to/foo.naparitracker
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch

from cotracker.predictor import CoTrackerPredictor

# Pinned model checkpoint URL. We use ``torch.hub.load_state_dict_from_url`` to
# fetch + cache it at ``~/.cache/torch/hub/checkpoints/scaled_offline.pth``;
# this avoids ``torch.hub.load("facebookresearch/co-tracker", ...)`` cloning
# upstream HEAD (which can drift past the version of ``cotracker`` we pin in
# ``pyproject.toml`` and silently break the predictor API).
COTRACKER3_OFFLINE_URL = (
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
)


def load_cotracker3_offline(device: str) -> CoTrackerPredictor:
    """Build the same CoTracker3 offline predictor that the upstream
    ``hubconf.cotracker3_offline()`` produces, but bound to the
    ``cotracker`` package version we pin."""
    state_dict = torch.hub.load_state_dict_from_url(
        COTRACKER3_OFFLINE_URL, map_location="cpu",
    )
    predictor = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
    predictor.model.load_state_dict(state_dict)
    return predictor.to(device)

from napari_cotrack import project as P
from napari_cotrack.pipeline._io import (
    LABELS_COLS, labels_to_queries, read_labels, tracks_to_dense, union_label_csvs,
    write_labels,
)

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

VERBOSE = False


def _v(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _progress_step(total_chunks: int) -> int:
    """Print every Nth chunk so the default log gets ~10 progress lines."""
    return max(1, total_chunks // 10)


# --------------------------- Video I/O ------------------------------------- #


def get_video_info(video_path: str) -> tuple[int, int, int]:
    """Frame count + resolution. Tries ffprobe first, falls back to imageio."""
    import subprocess

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        parts = result.stdout.strip().split(",")
        W, H, T = int(parts[0]), int(parts[1]), int(parts[2])
        return T, H, W
    except Exception:
        pass
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
    H, W = frame.shape[0], frame.shape[1]
    T = reader.count_frames()
    reader.close()
    return T, H, W


def read_video_chunk(video_path: str, start: int, end: int) -> np.ndarray:
    reader = imageio.get_reader(video_path)
    frames = []
    for i in range(start, end):
        try:
            frames.append(reader.get_data(i))
        except IndexError:
            break
    reader.close()
    return np.stack(frames)


def compute_roi_from_labels(labels_df: pd.DataFrame, W: int, H: int,
                            padding_pct: float = 0.30) -> tuple[int, int, int, int]:
    """Static bbox covering all labelled keypoints, padded and clamped."""
    if labels_df.empty:
        return 0, 0, W, H
    x_min, x_max = float(labels_df["x"].min()), float(labels_df["x"].max())
    y_min, y_max = float(labels_df["y"].min()), float(labels_df["y"].max())
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)
    pad_x = bw * padding_pct
    pad_y = bh * padding_pct
    x0 = max(0, int(x_min - pad_x))
    y0 = max(0, int(y_min - pad_y))
    x1 = min(W, int(x_max + pad_x) + 1)
    y1 = min(H, int(y_max + pad_y) + 1)
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    return x0, y0, x1, y1


# --------------------------- Tracking core --------------------------------- #


def downscale_for_gpu(video_tensor, max_short_side: int = 512):
    B, T, C, H, W = video_tensor.shape
    short_side = min(H, W)
    if short_side <= max_short_side:
        return video_tensor, 1.0, 1.0
    scale = max_short_side / short_side
    new_H, new_W = int(H * scale), int(W * scale)
    video_flat = video_tensor.reshape(B * T, C, H, W)
    video_flat = torch.nn.functional.interpolate(
        video_flat, size=(new_H, new_W), mode="bilinear", align_corners=False
    )
    return video_flat.reshape(B, T, C, new_H, new_W), W / new_W, H / new_H


def track_simple(video_tensor, queries, model, device, backward_tracking: bool):
    video_tensor, sx, sy = downscale_for_gpu(video_tensor)
    if sx != 1.0 or sy != 1.0:
        queries = queries.clone()
        queries[:, :, 1] /= sx
        queries[:, :, 2] /= sy
    video_tensor = video_tensor.to(device)
    queries = queries.to(device)
    pred_tracks, pred_visibility = model(
        video_tensor, queries=queries, backward_tracking=backward_tracking,
    )
    pred_tracks = pred_tracks.cpu()
    if sx != 1.0 or sy != 1.0:
        pred_tracks[:, :, :, 0] *= sx
        pred_tracks[:, :, :, 1] *= sy
    return pred_tracks, pred_visibility.cpu()


def track_chunked(video_path, queries, model, device, chunk_size, overlap,
                  backward_tracking, T, roi=None):
    """Triangle-weighted chunk stitching. ROI un-shift happens in the caller."""
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size; got "
            f"overlap={overlap}, chunk_size={chunk_size}"
        )
    N = queries.shape[1]
    stride = chunk_size - overlap

    all_tracks = torch.zeros(1, T, N, 2)
    all_weights = torch.zeros(1, T, N, 1)
    all_vis_sum = torch.zeros(1, T, N)
    all_vis_count = torch.zeros(1, T, N)

    total_chunks = max(1, (T + stride - 1) // stride)
    progress_step = _progress_step(total_chunks)

    chunk_idx = 0
    for chunk_start in range(0, T, stride):
        chunk_end = min(chunk_start + chunk_size, T)
        if chunk_end - chunk_start < 4:
            break
        chunk_idx += 1
        chunk_len = chunk_end - chunk_start
        _v(f"  Chunk {chunk_idx}: frames {chunk_start}-{chunk_end - 1} ({chunk_len} frames)")
        if chunk_idx == 1 or chunk_idx % progress_step == 0:
            print(f"  Chunk {chunk_idx}/{total_chunks}  ({100 * chunk_idx // total_chunks}%)")

        chunk_video = read_video_chunk(video_path, chunk_start, chunk_end)
        chunk_len = chunk_video.shape[0]
        if roi is not None:
            x0, y0, x1, y1 = roi
            chunk_video = chunk_video[:, y0:y1, x0:x1, :]
        chunk_video_tensor = (
            torch.from_numpy(chunk_video).permute(0, 3, 1, 2)[None].float()
        )
        del chunk_video

        chunk_queries = queries.clone()
        chunk_queries[:, :, 0] = (chunk_queries[:, :, 0] - chunk_start).clamp(
            0, chunk_len - 1
        )

        chunk_tracks, chunk_vis = track_simple(
            chunk_video_tensor, chunk_queries, model, device, backward_tracking
        )
        del chunk_video_tensor

        center = chunk_len / 2.0
        weight = 1.0 - (torch.arange(chunk_len, dtype=torch.float32) - center).abs() / center
        weight = weight.clamp(min=0.1)

        for local_t in range(chunk_len):
            global_t = chunk_start + local_t
            w = weight[local_t]
            all_tracks[0, global_t] += chunk_tracks[0, local_t] * w
            all_weights[0, global_t] += w
            all_vis_sum[0, global_t] += chunk_vis[0, local_t].float() * w
            all_vis_count[0, global_t] += w

    all_tracks /= all_weights.clamp(min=1e-8)
    all_visibility = (all_vis_sum / all_vis_count.clamp(min=1e-8)) > 0.5
    return all_tracks, all_visibility


def cluster_queries_spatially(pass1_tracks, query_info, T):
    median_pos = pass1_tracks[0].median(dim=0).values
    xs = median_pos[:, 0]
    sorted_idx = xs.argsort()
    sorted_xs = xs[sorted_idx]

    gaps = sorted_xs[1:] - sorted_xs[:-1]
    median_gap = gaps.median().item() if len(gaps) > 0 else 0.0
    threshold = max(median_gap * 5, 200)

    groups: list[list[int]] = []
    current_group = [sorted_idx[0].item()]
    for i in range(1, len(sorted_idx)):
        if gaps[i - 1].item() > threshold:
            groups.append(current_group)
            current_group = [sorted_idx[i].item()]
        else:
            current_group.append(sorted_idx[i].item())
    groups.append(current_group)

    if len(groups) > 1:
        names = []
        for g in groups:
            bps = set(query_info[qi][0] for qi in g)
            names.append("+".join(sorted(bps)))
        _v(f"  Split into {len(groups)} spatial groups: {', '.join(names)}")
    return groups


def refine_tracks(video_path, pass1_tracks, queries, model, device,
                  chunk_size, overlap, T, H, W, query_info):
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size; got "
            f"overlap={overlap}, chunk_size={chunk_size}"
        )
    N = queries.shape[1]
    stride = chunk_size - overlap
    pad_frac = 0.3

    groups = cluster_queries_spatially(pass1_tracks, query_info, T)

    all_tracks = torch.zeros(1, T, N, 2)
    all_weights = torch.zeros(1, T, N, 1)
    all_vis_sum = torch.zeros(1, T, N)
    all_vis_count = torch.zeros(1, T, N)

    total_chunks = max(1, (T + stride - 1) // stride)
    progress_step = _progress_step(total_chunks)

    for group_idx, query_indices in enumerate(groups):
        group_bps = sorted(set(query_info[qi][0] for qi in query_indices))
        _v(f"  Group {group_idx + 1}: {group_bps}")

        chunk_idx = 0
        for chunk_start in range(0, T, stride):
            chunk_end = min(chunk_start + chunk_size, T)
            if chunk_end - chunk_start < 4:
                break
            chunk_idx += 1

            ct = pass1_tracks[0, chunk_start:chunk_end][:, query_indices]
            x_min, x_max = ct[:, :, 0].min().item(), ct[:, :, 0].max().item()
            y_min, y_max = ct[:, :, 1].min().item(), ct[:, :, 1].max().item()
            x_pad = (x_max - x_min) * pad_frac + 50
            y_pad = (y_max - y_min) * pad_frac + 50
            crop_x1 = max(0, int(x_min - x_pad))
            crop_x2 = min(W, int(x_max + x_pad))
            crop_y1 = max(0, int(y_min - y_pad))
            crop_y2 = min(H, int(y_max + y_pad))
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            gain = min(W / crop_w, H / crop_h)

            if chunk_idx == 1:
                _v(f"    Chunk {chunk_idx}: crop {crop_w}x{crop_h} ({gain:.1f}x gain)")
            if chunk_idx == 1 or chunk_idx % progress_step == 0:
                pct = 100 * chunk_idx // total_chunks
                print(f"  refine group {group_idx + 1}/{len(groups)} chunk {chunk_idx}/{total_chunks}  ({pct}%)")

            chunk_video = read_video_chunk(video_path, chunk_start, chunk_end)
            actual_len = chunk_video.shape[0]
            chunk_video = chunk_video[:, crop_y1:crop_y2, crop_x1:crop_x2]
            chunk_video_tensor = (
                torch.from_numpy(chunk_video).permute(0, 3, 1, 2)[None].float()
            )
            del chunk_video

            group_queries_list = []
            for qi in query_indices:
                q = queries[0, qi].clone()
                q[0] = (q[0] - chunk_start).clamp(0, actual_len - 1)
                anchor_t = chunk_start + int(q[0].item())
                q[1:3] = pass1_tracks[0, anchor_t, qi]
                q[1] -= crop_x1
                q[2] -= crop_y1
                group_queries_list.append(q)
            group_queries = torch.stack(group_queries_list).unsqueeze(0)

            chunk_tracks, chunk_vis = track_simple(
                chunk_video_tensor, group_queries, model, device, True
            )
            del chunk_video_tensor

            chunk_tracks[:, :, :, 0] += crop_x1
            chunk_tracks[:, :, :, 1] += crop_y1

            center = actual_len / 2.0
            weight = 1.0 - (torch.arange(actual_len, dtype=torch.float32) - center).abs() / center
            weight = weight.clamp(min=0.1)

            for local_t in range(actual_len):
                global_t = chunk_start + local_t
                w = weight[local_t]
                for g_idx, qi in enumerate(query_indices):
                    all_tracks[0, global_t, qi] += chunk_tracks[0, local_t, g_idx] * w
                    all_weights[0, global_t, qi] += w
                    all_vis_sum[0, global_t, qi] += chunk_vis[0, local_t, g_idx].float() * w
                    all_vis_count[0, global_t, qi] += w

    all_tracks /= all_weights.clamp(min=1e-8)
    all_visibility = (all_vis_sum / all_vis_count.clamp(min=1e-8)) > 0.5
    return all_tracks, all_visibility


def merge_tracks(pred_tracks, pred_visibility, query_info, bodyparts, T):
    """Pick the temporally-nearest labelled query per (frame, bodypart)."""
    N_bp = len(bodyparts)
    merged_tracks = torch.full((1, T, N_bp, 2), float("nan"))
    merged_vis = torch.zeros(1, T, N_bp, dtype=torch.bool)
    bp_to_idx = {bp: i for i, bp in enumerate(bodyparts)}

    bp_queries: dict[str, list[tuple[int, int]]] = {}
    for q_idx, (bp, frame) in enumerate(query_info):
        bp_queries.setdefault(bp, []).append((q_idx, frame))

    for bp, entries in bp_queries.items():
        if bp not in bp_to_idx:
            continue
        bp_idx = bp_to_idx[bp]
        labeled_frames = torch.tensor([f for _, f in entries], dtype=torch.float32)
        query_indices = [q for q, _ in entries]

        for t in range(T):
            distances = (labeled_frames - t).abs()
            best = distances.argmin().item()
            q_idx = query_indices[best]
            merged_tracks[0, t, bp_idx] = pred_tracks[0, t, q_idx]
            merged_vis[0, t, bp_idx] = pred_visibility[0, t, q_idx]

    return merged_tracks, merged_vis


def postprocess_tracks(tracks, visibility, bodyparts,
                       median_window: int = 5, max_interp_gap: int = 60):
    """Spike-removal median filter + interpolation through invisible gaps,
    plus a continuous likelihood (decays away from visible frames). Returns
    (tracks_out, likelihood)."""
    from scipy.ndimage import median_filter
    from scipy.interpolate import interp1d

    T = tracks.shape[1]
    N_bp = tracks.shape[2]
    tracks_out = tracks.clone()
    likelihood = torch.zeros(1, T, N_bp)

    for bp_idx in range(N_bp):
        xs = tracks[0, :, bp_idx, 0].numpy().copy()
        ys = tracks[0, :, bp_idx, 1].numpy().copy()
        vis = visibility[0, :, bp_idx].numpy().copy()

        vis_mask = vis.astype(bool)
        spikes = np.zeros(T, dtype=bool)
        if vis_mask.sum() > median_window:
            xs_filt = median_filter(xs, size=median_window)
            ys_filt = median_filter(ys, size=median_window)
            diff = np.sqrt((xs - xs_filt) ** 2 + (ys - ys_filt) ** 2)
            dx = np.abs(np.diff(xs_filt[vis_mask]))
            dy = np.abs(np.diff(ys_filt[vis_mask]))
            typical_motion = np.median(np.sqrt(dx ** 2 + dy ** 2)) if len(dx) else 0.0
            spike_threshold = max(typical_motion * 3, 20)
            spikes = diff > spike_threshold
            xs[spikes & vis_mask] = xs_filt[spikes & vis_mask]
            ys[spikes & vis_mask] = ys_filt[spikes & vis_mask]

        visible_indices = np.where(vis_mask)[0]
        if len(visible_indices) >= 2:
            interp_x = interp1d(
                visible_indices, xs[visible_indices],
                kind="linear", bounds_error=False, fill_value="extrapolate",
            )
            interp_y = interp1d(
                visible_indices, ys[visible_indices],
                kind="linear", bounds_error=False, fill_value="extrapolate",
            )
            for idx in np.where(~vis_mask)[0]:
                nearest_dist = np.abs(visible_indices - idx).min()
                if nearest_dist <= max_interp_gap:
                    xs[idx] = interp_x(idx)
                    ys[idx] = interp_y(idx)

        lik = np.zeros(T)
        lik[vis_mask] = 1.0
        if len(visible_indices) >= 1:
            for idx in np.where(~vis_mask)[0]:
                nearest_dist = np.abs(visible_indices - idx).min()
                if nearest_dist <= max_interp_gap:
                    lik[idx] = max(0.1, 0.9 * np.exp(-nearest_dist / (max_interp_gap / 3)))

        tracks_out[0, :, bp_idx, 0] = torch.from_numpy(xs)
        tracks_out[0, :, bp_idx, 1] = torch.from_numpy(ys)
        likelihood[0, :, bp_idx] = torch.from_numpy(lik)

    return tracks_out, likelihood


# --------------------------- High-level orchestration ---------------------- #


def run_track(project_root: str | Path, *, device: str = DEFAULT_DEVICE) -> Path:
    """End-to-end stage-3 invocation. Returns the path to the saved tracks CSV."""
    proj = P.load(Path(project_root))
    _v(f"Project: {proj.root}")
    _v(f"Bodyparts: {proj.bodyparts}")
    _v(f"Video: {proj.video}")

    label_paths = proj.all_label_csvs()
    if not label_paths:
        raise FileNotFoundError(f"No labels found under {proj.anchors_dir} or {proj.corrections_dir}")
    labels_df = union_label_csvs(label_paths)
    for p in label_paths:
        n = len(read_labels(p))
        _v(f"Labels: {n} rows from {p}")
    print(f"Total: {len(labels_df)} merged label rows")

    T, H, W = get_video_info(proj.video)
    print(f"Video: {T} frames, {W}x{H}")

    queries, query_info = labels_to_queries(labels_df, T)
    print(f"Queries: {queries.shape[1]} points")

    tp = proj.track
    roi_box = None
    if tp.roi:
        x0, y0, x1, y1 = compute_roi_from_labels(labels_df, W, H)
        roi_box = (x0, y0, x1, y1)
        print(f"ROI: x={x0}-{x1}, y={y0}-{y1}  "
              f"[{(x1-x0)*(y1-y0)*100/(W*H):.1f}% of frame]")
        queries = queries.clone()
        queries[:, :, 1] -= x0
        queries[:, :, 2] -= y0

    print("Loading CoTracker3...")
    model = load_cotracker3_offline(device)

    if T <= tp.chunk_size:
        print("Tracking (single pass)...")
        video_np = read_video_chunk(proj.video, 0, T)
        if roi_box is not None:
            x0, y0, x1, y1 = roi_box
            video_np = video_np[:, y0:y1, x0:x1, :]
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
        del video_np
        pred_tracks, pred_visibility = track_simple(
            video_tensor, queries, model, device, backward_tracking=True
        )
        del video_tensor
    else:
        print(f"Tracking (chunked, {tp.chunk_size}/chunk, {tp.overlap} overlap)...")
        pred_tracks, pred_visibility = track_chunked(
            proj.video, queries, model, device,
            tp.chunk_size, tp.overlap, True, T, roi=roi_box,
        )

    if roi_box is not None:
        x0, y0, _, _ = roi_box
        pred_tracks[:, :, :, 0] += x0
        pred_tracks[:, :, :, 1] += y0
        queries = queries.clone()
        queries[:, :, 1] += x0
        queries[:, :, 2] += y0

    if tp.refine:
        print("Refining (pass 2, dynamic crop per group)...")
        pred_tracks, pred_visibility = refine_tracks(
            proj.video, pred_tracks, queries, model, device,
            tp.chunk_size, tp.overlap, T, H, W, query_info,
        )

    print("Merging tracks per bodypart...")
    merged_tracks, merged_vis = merge_tracks(
        pred_tracks, pred_visibility, query_info, proj.bodyparts, T,
    )

    track_sums = merged_tracks[0].abs().sum(dim=(1, 2))
    nz = (track_sums > 0).nonzero()
    last_valid = int(nz[-1].item()) + 1 if len(nz) else T
    if last_valid < T:
        print(f"Trimming {T - last_valid} trailing empty frames")
        merged_tracks = merged_tracks[:, :last_valid]
        merged_vis = merged_vis[:, :last_valid]

    if tp.postprocess:
        print("Postprocessing tracks (median + interpolation)...")
        merged_tracks, likelihood = postprocess_tracks(
            merged_tracks, merged_vis, proj.bodyparts,
        )
    else:
        likelihood = merged_vis.float()

    df = tracks_to_dense(merged_tracks, likelihood, proj.bodyparts)
    write_labels(df, proj.tracks_csv)
    print(f"Saved: {proj.tracks_csv}")
    return proj.tracks_csv


# --------------------------- CLI ------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    global VERBOSE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, help="Path to .naparitracker dir")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-chunk / per-group detail.")
    args = p.parse_args(argv)
    VERBOSE = args.verbose
    run_track(args.project, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
