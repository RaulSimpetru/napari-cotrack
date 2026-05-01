"""Smoke tests for napari_cotrack.pipeline.extract (no DLC)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from napari_cotrack.pipeline import extract
from napari_cotrack.pipeline._io import read_labels


def _ensure_fixture(tmp_dir: Path) -> Path:
    project_root = tmp_dir / "smoke.naparitracker"
    if not project_root.exists():
        script = Path(__file__).parent / "_make_fixture.py"
        subprocess.check_call([sys.executable, str(script), "--out", str(tmp_dir)])
    return project_root


def test_select_uniform_is_actual_stride() -> None:
    """N is a stride: 'every Nth frame' really means 0, N, 2N, ..."""
    assert extract.select_uniform(100, 10) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert extract.select_uniform(60, 8) == [0, 8, 16, 24, 32, 40, 48, 56]


def test_select_uniform_when_stride_exceeds_total() -> None:
    """Stride > total → just the first frame."""
    assert extract.select_uniform(5, 10) == [0]


def test_select_uniform_zero_or_negative_stride_returns_empty() -> None:
    assert extract.select_uniform(100, 0) == []
    assert extract.select_uniform(100, -1) == []


def test_select_diverse_returns_unique_sorted_indices(tmp_path: Path) -> None:
    project_root = _ensure_fixture(tmp_path)
    video = (project_root.parent / "clip.mp4")
    indices = extract.select_diverse(str(video), k=8, stride=2, seed=0)
    assert len(indices) == 8
    assert len(set(indices)) == 8
    assert indices == sorted(indices)
    assert all(0 <= i < 60 for i in indices)


def test_extract_anchors_writes_pngs_and_empty_labels(tmp_path: Path) -> None:
    project_root = _ensure_fixture(tmp_path)
    video = project_root.parent / "clip.mp4"
    out_dir = project_root / "anchors"
    indices = [0, 10, 20, 30]
    extract.extract_anchors(
        str(video), indices, str(out_dir),
        bodyparts=["thumb", "index"],
    )
    pngs = sorted(out_dir.glob("img*.png"))
    assert [p.name for p in pngs] == ["img0000.png", "img0010.png", "img0020.png", "img0030.png"]
    df = read_labels(out_dir / "labels.csv")
    assert df.empty


def test_extract_corrections_prefills_from_tracks(tmp_path: Path) -> None:
    project_root = _ensure_fixture(tmp_path)
    video = project_root.parent / "clip.mp4"
    tracks = project_root / "tracks.csv"
    out_dir = project_root / "corrections" / "round_001"
    extract.extract_corrections(
        str(video), [5, 15], str(out_dir),
        tracks_csv=str(tracks),
        bodyparts=["thumb", "index", "wrist"],
    )
    df = read_labels(out_dir / "labels.csv")
    # 2 frames × 3 bodyparts = 6 rows
    assert len(df) == 6
    assert sorted(df["frame"].unique().tolist()) == [5, 15]
    assert sorted(df["bodypart"].unique().tolist()) == ["index", "thumb", "wrist"]
