"""Pure-function tests for `_layer.py`'s frame-mapping converters.

Spinning up a real ``napari.Viewer`` in CI is overkill (and pulls a Qt event
loop into pytest); these tests exercise the same conversion functions that
``attach_keypoint_layer`` uses at the boundary, plus a save round-trip via
a fake layer object that mirrors the napari Points layer interface.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from napari_cotrack._layer import _df_to_layer_arrays, _layer_to_df
from napari_cotrack.pipeline._io import LABELS_COLS, read_labels, write_labels


def test_df_to_layer_uses_stack_index_not_real_frame() -> None:
    """Sparse anchors at frames 0/10/20 must map to stack indices 0/1/2.
    A naive read would put them at z=0/10/20 and break alignment."""
    stack_to_frame = [0, 10, 20]
    frame_to_stack = {f: i for i, f in enumerate(stack_to_frame)}
    df = pd.DataFrame(
        [(0, "thumb", 100.0, 50.0, 1.0),
         (10, "thumb", 110.0, 55.0, 1.0),
         (20, "thumb", 120.0, 60.0, 1.0)],
        columns=LABELS_COLS,
    )
    data, bps = _df_to_layer_arrays(df, frame_to_stack)
    assert data.shape == (3, 3)
    assert list(data[:, 0]) == [0, 1, 2]      # stack indices, not 0/10/20
    assert list(bps) == ["thumb", "thumb", "thumb"]
    # napari order is (frame, y, x); confirm our swap.
    assert data[0, 1] == 50.0 and data[0, 2] == 100.0


def test_df_to_layer_drops_rows_outside_stack() -> None:
    """A label whose frame isn't in this stack must not silently land at
    stack-index 0 (that would corrupt other anchors)."""
    frame_to_stack = {0: 0, 10: 1}
    df = pd.DataFrame(
        [(0, "thumb", 1.0, 2.0, 1.0),
         (5, "thumb", 9.0, 9.0, 1.0),  # not in stack
         (10, "thumb", 3.0, 4.0, 1.0)],
        columns=LABELS_COLS,
    )
    data, bps = _df_to_layer_arrays(df, frame_to_stack)
    assert data.shape == (2, 3)
    assert list(data[:, 0]) == [0, 1]


def test_layer_to_df_translates_stack_to_real_frame() -> None:
    """The inverse: napari layer's stack index 0/1/2 must save as real
    frame numbers 0/10/20 in the CSV."""
    stack_to_frame = [0, 10, 20]
    layer = SimpleNamespace(
        # napari coords: (stack_idx, y, x)
        data=np.array([[0, 50.0, 100.0], [1, 55.0, 110.0], [2, 60.0, 120.0]]),
        properties={"bodypart": np.array(["thumb", "thumb", "thumb"])},
    )
    df = _layer_to_df(layer, ["thumb"], stack_to_frame)
    assert sorted(df["frame"].tolist()) == [0, 10, 20]
    # x/y must be re-stored from (y, x) napari order back to disk x/y order.
    row0 = df[df["frame"] == 0].iloc[0]
    assert (row0["x"], row0["y"]) == (100.0, 50.0)


def test_layer_to_df_drops_unknown_bodyparts() -> None:
    """A layer carrying a bodypart not in the project's list (e.g. legacy
    points after the user renamed bodyparts) must not pollute the CSV."""
    stack_to_frame = [0]
    layer = SimpleNamespace(
        data=np.array([[0, 1.0, 2.0], [0, 3.0, 4.0]]),
        properties={"bodypart": np.array(["thumb", "ghost"])},
    )
    df = _layer_to_df(layer, ["thumb"], stack_to_frame)
    assert len(df) == 1 and df["bodypart"].iloc[0] == "thumb"


def test_layer_to_df_skips_oob_stack_indices() -> None:
    """If a layer point ended up at stack idx 99 via some napari weirdness,
    we must drop it rather than map it to a phantom real-frame number."""
    stack_to_frame = [0, 10]
    layer = SimpleNamespace(
        data=np.array([[0, 1.0, 2.0], [99, 3.0, 4.0]]),
        properties={"bodypart": np.array(["thumb", "thumb"])},
    )
    df = _layer_to_df(layer, ["thumb"], stack_to_frame)
    assert len(df) == 1 and df["frame"].iloc[0] == 0


def test_save_round_trip_through_csv(tmp_path: Path) -> None:
    """End-to-end: synthesise a layer, drive _layer_to_df + write_labels,
    re-load via read_labels, confirm coords + frame numbers are intact."""
    stack_to_frame = [0, 100, 200]
    bodyparts = ["thumb", "index"]
    layer = SimpleNamespace(
        data=np.array([
            [0, 50.0, 100.0],
            [0, 60.0, 200.0],
            [1, 51.0, 101.0],
            [2, 52.0, 102.0],
        ]),
        properties={"bodypart": np.array(["thumb", "index", "thumb", "thumb"])},
    )
    df = _layer_to_df(layer, bodyparts, stack_to_frame)
    csv = tmp_path / "labels.csv"
    write_labels(df, csv)
    rt = read_labels(csv)

    # Every (frame, bodypart) pair survives with byte-exact x/y.
    rt_idx = rt.set_index(["frame", "bodypart"])[["x", "y"]]
    assert rt_idx.loc[(0, "thumb")].tolist() == [100.0, 50.0]
    assert rt_idx.loc[(0, "index")].tolist() == [200.0, 60.0]
    assert rt_idx.loc[(100, "thumb")].tolist() == [101.0, 51.0]
    assert rt_idx.loc[(200, "thumb")].tolist() == [102.0, 52.0]
