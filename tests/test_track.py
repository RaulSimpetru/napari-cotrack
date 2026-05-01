"""Tests for napari_cotrack.pipeline._io and project label union."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from napari_cotrack import project as P
from napari_cotrack.pipeline._io import (
    LABELS_COLS, read_labels, union_label_csvs, write_labels,
)


def test_union_label_csvs_later_round_wins(tmp_path: Path) -> None:
    a = tmp_path / "anchors.csv"
    b = tmp_path / "round_001.csv"
    write_labels(pd.DataFrame([
        (5, "thumb", 100.0, 50.0, 1.0),
        (5, "index", 200.0, 60.0, 1.0),
    ], columns=LABELS_COLS), a)
    write_labels(pd.DataFrame([
        (5, "thumb", 999.0, 999.0, 1.0),  # corrects the anchor
        (10, "thumb", 110.0, 55.0, 1.0),  # adds a new frame
    ], columns=LABELS_COLS), b)

    merged = union_label_csvs([a, b])
    # 5/thumb came from b (corrected), 5/index came from a (not corrected),
    # 10/thumb came from b (new).
    keys = sorted(zip(merged["frame"], merged["bodypart"]))
    assert keys == [(5, "index"), (5, "thumb"), (10, "thumb")]
    thumb5 = merged[(merged["frame"] == 5) & (merged["bodypart"] == "thumb")]
    assert float(thumb5["x"].iloc[0]) == 999.0


def test_project_all_label_csvs_orders_anchors_then_rounds(tmp_path: Path) -> None:
    p = P.create(tmp_path / "demo.naparitracker", video="/x.mp4", bodyparts=["a", "b"])
    p.anchors_dir.mkdir(exist_ok=True)
    write_labels(pd.DataFrame(columns=LABELS_COLS), p.anchors_labels)
    (p.corrections_dir / "round_001").mkdir(parents=True, exist_ok=True)
    write_labels(pd.DataFrame(columns=LABELS_COLS), p.corrections_dir / "round_001" / "labels.csv")
    (p.corrections_dir / "round_002").mkdir(exist_ok=True)
    write_labels(pd.DataFrame(columns=LABELS_COLS), p.corrections_dir / "round_002" / "labels.csv")

    paths = p.all_label_csvs()
    names = [str(x.relative_to(p.root)) for x in paths]
    assert names == [
        "anchors/labels.csv",
        "corrections/round_001/labels.csv",
        "corrections/round_002/labels.csv",
    ]


def test_next_corrections_round_increments(tmp_path: Path) -> None:
    p = P.create(tmp_path / "demo.naparitracker", video="/x.mp4", bodyparts=["a"])
    assert p.next_corrections_round().name == "round_001"
    (p.corrections_dir / "round_001").mkdir(parents=True)
    (p.corrections_dir / "round_002").mkdir()
    assert p.next_corrections_round().name == "round_003"
