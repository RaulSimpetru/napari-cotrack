"""Fast paths over ffmpeg for video → PNG dumps.

`imageio.read(index=f)` in a loop incurs per-frame seek+decode+pipe overhead.
A single ffmpeg subprocess with `img%04d.png` decodes sequentially and writes
PNGs in C, which is roughly 10-50x faster on H.264 sources.

The helpers here are best-effort: if ffmpeg isn't on PATH or the call fails,
the caller falls back to the imageio path.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_all_frames(video: str | Path, out_dir: str | Path,
                       on_progress: Callable[[int], None] | None = None,
                       ) -> int:
    """Dump every frame of ``video`` to ``out_dir/img####.png`` via ffmpeg.

    Returns the number of frames written. Raises ``FileNotFoundError`` if
    ffmpeg isn't on PATH and ``RuntimeError`` if ffmpeg exits non-zero.

    ``on_progress`` is called with the current frame number every time
    ffmpeg's ``-progress pipe:1`` emits a ``frame=N`` key — useful for
    streaming a percentage into the napari plugin log pane.
    """
    if not have_ffmpeg():
        raise FileNotFoundError("ffmpeg not on PATH")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-start_number", "0",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-stats_period", "0.5",
        str(out_dir / "img%04d.png"),
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    last_frame = 0
    assert proc.stdout is not None
    frame_re = re.compile(r"^frame=(\d+)$")
    for line in proc.stdout:
        m = frame_re.match(line.strip())
        if m:
            last_frame = int(m.group(1))
            if on_progress is not None:
                on_progress(last_frame)
    proc.wait()

    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "") or ""
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {err.strip()[:400]}")

    return last_frame
