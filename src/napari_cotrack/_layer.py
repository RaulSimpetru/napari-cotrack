"""Reusable napari Points-layer setup for napari-cotrack.

Used both by the napari plugin (in-process) and by tests. Exposes one
function ``attach_keypoint_layer`` that:

- builds an image stack from a folder of img####.png frames,
- adds a single Points layer named ``labels`` whose colour is driven by a
  ``bodypart`` property,
- wires up move-or-add click semantics with auto-advance to the next bodypart,
- wires up a save callback that writes ``labels.csv`` (long-form, x/y in user
  coords; napari's ``(stack_idx, y, x)`` is translated at the boundary),
- returns a small ``LayerSession`` handle that the caller can use to drive
  the Prev/Next/Save buttons.

The dock widget owns the LayerSession and disposes of it on close.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Callable

import numpy as np
import pandas as pd

from napari_cotrack.pipeline._io import LABELS_COLS, read_labels, write_labels

# matplotlib tab10 as hex (vispy doesn't grok 'tab:blue').
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _frame_idx_from_name(p: Path) -> int:
    m = re.match(r"img(\d+)\.png$", p.name)
    return int(m.group(1)) if m else 0


def list_frame_pngs(folder: Path) -> tuple[list[Path], list[int]]:
    """Sorted list of img####.png files + their parsed real frame numbers."""
    pngs = sorted(folder.glob("img*.png"), key=_frame_idx_from_name)
    return pngs, [_frame_idx_from_name(p) for p in pngs]


def _df_to_layer_arrays(df: pd.DataFrame, frame_to_stack: dict[int, int]
                        ) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.zeros((0, 3), dtype=float), np.array([], dtype=object)
    df = df.sort_values(["frame", "bodypart"]).reset_index(drop=True)
    df = df[df["frame"].isin(frame_to_stack)]
    if df.empty:
        return np.zeros((0, 3), dtype=float), np.array([], dtype=object)
    stack_idx = df["frame"].map(frame_to_stack).astype(float).values
    data = np.column_stack([
        stack_idx, df["y"].astype(float).values, df["x"].astype(float).values,
    ])
    return data, df["bodypart"].astype(str).to_numpy()


def _layer_to_df(layer, bodyparts: list[str], stack_to_frame: list[int]) -> pd.DataFrame:
    if len(layer.data) == 0:
        return pd.DataFrame(columns=LABELS_COLS)
    bps = np.asarray(layer.properties.get("bodypart", np.empty(0, dtype=object)))
    rows: list[tuple[int, str, float, float, float]] = []
    for i, pt in enumerate(layer.data):
        bp = str(bps[i]) if i < len(bps) else (bodyparts[0] if bodyparts else "")
        if bp not in bodyparts:
            continue
        si = int(round(float(pt[0])))
        if not 0 <= si < len(stack_to_frame):
            continue
        rows.append((stack_to_frame[si], bp, float(pt[2]), float(pt[1]), 1.0))
    return pd.DataFrame(rows, columns=LABELS_COLS)


@dataclass
class LayerSession:
    """Handle returned by ``attach_keypoint_layer``. The dock widget uses
    these methods to wire the Prev/Next/Save buttons and to clean up."""
    viewer: object
    layer: object
    bodyparts: list[str]
    stack_to_frame: list[int]
    csv_path: Path
    set_active_idx: Callable[[int], None]
    next_kp: Callable[[], None]
    prev_kp: Callable[[], None]
    save: Callable[[], None]


def attach_keypoint_layer(viewer, frames_folder: Path, csv_path: Path,
                          bodyparts: list[str],
                          on_active_changed: Callable[[str], None] | None = None,
                          ) -> LayerSession:
    """Open ``frames_folder`` as an image stack and attach the labels layer.

    ``on_active_changed`` is called whenever the active bodypart changes,
    e.g. after a click auto-advance. The dock widget uses it to keep its
    dropdown in sync.
    """
    pngs, stack_to_frame = list_frame_pngs(frames_folder)
    if not pngs:
        raise FileNotFoundError(f"No img*.png files in {frames_folder}")
    frame_to_stack = {f: i for i, f in enumerate(stack_to_frame)}

    df = read_labels(csv_path) if csv_path.exists() else pd.DataFrame(columns=LABELS_COLS)
    df = df[df["bodypart"].isin(bodyparts)].reset_index(drop=True)

    viewer.open([str(p) for p in pngs], stack=True)

    color_cycle = [TAB10[i % len(TAB10)] for i in range(len(bodyparts))]
    data, bps_arr = _df_to_layer_arrays(df, frame_to_stack)

    layer = viewer.add_points(
        data,
        name="labels",
        properties={"bodypart": bps_arr},
        face_color="white",
        border_color="white",
        size=12,
        ndim=3,
        text={
            "string": "{bodypart}",
            "anchor": "upper_left",
            "translation": [0, 8, 8],
            "color": "white",
            "size": 10,
        },
    )
    layer.face_color_cycle = color_cycle
    layer.face_color = "bodypart"
    layer.feature_defaults = {"bodypart": bodyparts[0]}
    layer.mode = "add"

    state = {"idx": 0}

    def set_active_idx(i: int) -> None:
        i = i % len(bodyparts)
        state["idx"] = i
        bp = bodyparts[i]
        layer.feature_defaults = {"bodypart": bp}
        layer.current_properties = {"bodypart": np.array([bp])}
        viewer.status = f"Active keypoint: {bp}  ({i + 1}/{len(bodyparts)})"
        if on_active_changed is not None:
            on_active_changed(bp)

    def next_kp() -> None:
        set_active_idx(state["idx"] + 1)

    def prev_kp() -> None:
        set_active_idx(state["idx"] - 1)

    # Move-or-add + auto-advance click handler. See napari_session for the
    # rationale; we key on stack index here, translate to real frame on save.
    def _add(layer_self, coord) -> None:
        bp = bodyparts[state["idx"]]
        coord = np.asarray(coord, dtype=float)
        stack_idx = int(round(float(coord[0])))

        bps = np.asarray(layer_self.properties.get("bodypart", np.empty(0, dtype=object)))
        n_data = len(layer_self.data)
        # Defensive: keep `bps` aligned with `data` even if napari trimmed
        # properties unevenly during a Delete.
        if len(bps) < n_data:
            bps = np.concatenate([bps, np.array([bp] * (n_data - len(bps)), dtype=object)])
        elif len(bps) > n_data:
            bps = bps[:n_data]
        if n_data > 0:
            frames = layer_self.data[:, 0].astype(int)
            mask = (frames == stack_idx) & (bps == bp)
        else:
            mask = np.zeros(0, dtype=bool)

        if mask.any():
            idx = int(np.flatnonzero(mask)[0])
            new_data = layer_self.data.copy()
            new_data[idx] = coord
            layer_self.data = new_data
        else:
            new_data = np.append(layer_self.data, np.atleast_2d(coord), axis=0)
            new_bps = np.append(bps, bp)
            layer_self.data = new_data
            layer_self.properties = {"bodypart": new_bps}

        layer_self.selected_data = set()
        next_kp()

    layer.add = MethodType(_add, layer)

    def save() -> None:
        out = _layer_to_df(layer, bodyparts, stack_to_frame)
        write_labels(out, csv_path)
        viewer.status = f"Saved {len(out)} points -> {csv_path.name}"

    set_active_idx(0)

    # Layer-scoped key bindings — N / Shift-N to cycle, Ctrl/Cmd-S to save.
    layer.bind_key("N", lambda _v: next_kp(), overwrite=True)
    layer.bind_key("Shift-N", lambda _v: prev_kp(), overwrite=True)
    viewer.bind_key("Control-S", lambda _v: save(), overwrite=True)
    viewer.bind_key("Meta-S", lambda _v: save(), overwrite=True)

    return LayerSession(
        viewer=viewer, layer=layer, bodyparts=bodyparts,
        stack_to_frame=stack_to_frame, csv_path=csv_path,
        set_active_idx=set_active_idx, next_kp=next_kp, prev_kp=prev_kp,
        save=save,
    )
