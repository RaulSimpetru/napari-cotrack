"""Build a tiny synthetic napari-cotrack project for smoke tests.

Layout produced under <out>:
  smoke.naparitracker/
    project.toml
    clip.mp4                       # 60 frames, 320x240
    tracks.csv                     # synthetic tracks (long-form, 3 bodyparts)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from napari_cotrack import project as P
from napari_cotrack.pipeline._io import write_labels


def make_video(path: Path, n: int = 60, W: int = 320, H: int = 240, fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    rng = np.random.default_rng(0)
    for f in range(n):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = (np.linspace(0, 255, W).astype(np.uint8))[None, :]
        img[:, :, 1] = (np.linspace(0, 200, H).astype(np.uint8))[:, None]
        img[:, :, 2] = (f * 4) % 256
        cx = int(40 + 240 * (0.5 + 0.5 * np.sin(f / 8.0)))
        cy = int(40 + 160 * (0.5 + 0.5 * np.cos(f / 11.0)))
        cv2.circle(img, (cx, cy), 14, (255, 255, 255), -1)
        img = np.clip(img.astype(int) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
        writer.write(img)
    writer.release()


def make_tracks_csv(path: Path, n: int = 60, bodyparts=("thumb", "index", "wrist")) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for f in range(n):
        px = 100 + 50 * np.sin(f / 10.0)
        py = 120 + 30 * np.cos(f / 9.0)
        for k, bp in enumerate(bodyparts):
            jitter = rng.normal(0, 0.6)
            spike = 25.0 if (k == 0 and f == 22) else 0.0
            rows.append((
                f, bp, px + 25 * k + jitter + spike, py + 5 * k + jitter, 1.0,
            ))
    df = pd.DataFrame(rows, columns=["frame", "bodypart", "x", "y", "vis"])
    write_labels(df, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/napari-cotrack-fixture")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = out / "smoke.naparitracker"
    if root.exists():
        import shutil as _sh; _sh.rmtree(root)
    video = out / "clip.mp4"
    make_video(video, n=args.frames)
    p = P.create(root, video=str(video), bodyparts=["thumb", "index", "wrist"])
    make_tracks_csv(p.tracks_csv, n=args.frames)
    print(f"Wrote fixture to {root}/")
    print(f"  video: {video}")
    print(f"  tracks: {p.tracks_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
