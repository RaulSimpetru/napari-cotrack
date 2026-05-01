"""Stage 5b: render an overlay MP4 from a long-form tracks CSV.

Pure cv2. tab10 colours by base bodypart name (paired views — e.g. ``thumb_r``
and ``thumb_p`` — share a colour). ``--skip-jumps`` re-uses the jump detector
from ``napari_cotrack.pipeline.review`` to omit scene-confusion frames.

Module CLI:
    python -m napari_cotrack.pipeline.render --project /path/to/foo.naparitracker
        [--filtered] [--skip-jumps] [--trail N --radius R]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from napari_cotrack import project as P
from napari_cotrack.pipeline._io import dense_to_arrays, read_labels

VERBOSE = False


def _v(msg: str) -> None:
    if VERBOSE:
        print(msg)

TAB10_BGR = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
    (34, 189, 188), (207, 190, 23),
]


def _base_name(bp: str) -> str:
    s = bp.lower().replace("wirst", "wrist")
    if len(s) > 2 and s[-2] == "_":
        s = s[:-2]
    return s


def _bp_color_map(bodyparts):
    seen: list[str] = []
    for bp in bodyparts:
        b = _base_name(bp)
        if b not in seen:
            seen.append(b)
    color_for_base = {b: TAB10_BGR[i % len(TAB10_BGR)] for i, b in enumerate(seen)}
    return [color_for_base[_base_name(bp)] for bp in bodyparts], seen, color_for_base


def draw_legend(img, bases, color_for_base):
    x0, y0 = 20, 20
    pad = 12
    line_h = 34
    box_w = 210
    box_h = line_h * len(bases) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (240, 240, 240), 2,
                  lineType=cv2.LINE_AA)
    for i, base in enumerate(bases):
        c = color_for_base[base]
        y = y0 + pad + i * line_h + line_h // 2
        cv2.circle(img, (x0 + 22, y), 9, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x0 + 22, y), 9, (240, 240, 240), 1, lineType=cv2.LINE_AA)
        cv2.putText(img, base.title(), (x0 + 44, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2,
                    cv2.LINE_AA)


def run_render(project_root: str | Path, *,
               filtered: bool = True,
               trail: int | None = None, radius: int | None = None,
               no_legend: bool = False, skip_jumps: bool | None = None) -> Path:
    proj = P.load(Path(project_root))
    rp = proj.render
    trail = rp.trail if trail is None else trail
    radius = rp.radius if radius is None else radius
    skip_jumps = rp.skip_jumps if skip_jumps is None else skip_jumps

    csv = proj.tracks_filtered_csv if (filtered and proj.tracks_filtered_csv.exists()) else proj.tracks_csv
    if not csv.exists():
        raise FileNotFoundError(f"No tracks CSV; run stage 3 first")
    df = read_labels(csv)
    xs, ys, _vs, bodyparts = dense_to_arrays(df)
    n_frames = xs.shape[0]
    print(f"Loaded {n_frames} frames x {len(bodyparts)} bodyparts from {csv.name}")

    if not Path(proj.video).exists():
        raise FileNotFoundError(f"Video not found: {proj.video}")
    _v(f"Source video: {proj.video}")

    cap = cv2.VideoCapture(proj.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {proj.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} @ {fps:.1f}fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(proj.overlay_mp4), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter")

    bp_colors, base_names, color_for_base = _bp_color_map(bodyparts)

    skip_set: set[int] = set()
    if skip_jumps:
        from napari_cotrack.pipeline.review import detect_jump_frames, find_bad_ranges
        is_jump, _ = detect_jump_frames(xs, ys, proj.jumps.n_sigmas, proj.jumps.min_frac)
        for s, e in find_bad_ranges(is_jump, proj.jumps.max_run):
            skip_set.update(range(s, e))
        print(f"Skipping {len(skip_set)} frames")

    f = 0
    written = 0
    progress_step = max(1, n_frames // 10)
    try:
      while True:
        ok, frame = cap.read()
        if not ok or f >= n_frames:
            break
        if f in skip_set:
            f += 1
            continue
        if f == 0 or f % progress_step == 0:
            print(f"  rendered {f}/{n_frames}  ({100 * f // n_frames}%)")

        trail_layer = np.zeros_like(frame)
        for i in range(len(bodyparts)):
            c = bp_colors[i]
            for k in range(min(trail, f), 0, -1):
                px, py = xs[f - k, i], ys[f - k, i]
                if np.isnan(px) or np.isnan(py):
                    continue
                alpha = (1.0 - k / max(trail, 1)) ** 1.6
                r = max(2, int(round(radius * (0.4 + 0.6 * alpha))))
                p = (int(round(px)), int(round(py)))
                cv2.circle(trail_layer, p, r, c, -1, lineType=cv2.LINE_AA)
        if trail_layer.any():
            glow = cv2.GaussianBlur(trail_layer, (0, 0), sigmaX=4.0, sigmaY=4.0)
            cv2.addWeighted(glow, 0.35, frame, 1.0, 0, dst=frame)
            cv2.addWeighted(trail_layer, 0.55, frame, 1.0, 0, dst=frame)

        for i in range(len(bodyparts)):
            c = bp_colors[i]
            x, y = xs[f, i], ys[f, i]
            if np.isnan(x) or np.isnan(y):
                continue
            p = (int(round(x)), int(round(y)))
            cv2.circle(frame, p, radius + 3, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, p, radius + 3, c, 2, lineType=cv2.LINE_AA)
            cv2.drawMarker(frame, p, c, cv2.MARKER_CROSS,
                           markerSize=radius + 6, thickness=2,
                           line_type=cv2.LINE_AA)

        if not no_legend:
            draw_legend(frame, base_names, color_for_base)
        cv2.putText(frame, f"frame {f}/{n_frames}", (13, H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"frame {f}/{n_frames}", (12, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        written += 1
        f += 1
    finally:
        cap.release()
        writer.release()
    print(f"Saved {proj.overlay_mp4}  ({written} written, {f - written} skipped)")
    return proj.overlay_mp4


def main(argv: list[str] | None = None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--filtered", action="store_true", default=True,
                   help="(default) use tracks_filtered.csv if present")
    ap.add_argument("--no-filtered", dest="filtered", action="store_false")
    ap.add_argument("--trail", type=int, default=None)
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--skip-jumps", action="store_true", default=None)
    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = args.verbose
    run_render(
        args.project, filtered=args.filtered,
        trail=args.trail, radius=args.radius,
        no_legend=args.no_legend, skip_jumps=args.skip_jumps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
