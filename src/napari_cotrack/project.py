"""napari-cotrack project on disk.

A project is a directory ending in ``.naparitracker/`` that holds:

  project.toml                  # video path + bodyparts + per-stage knobs
  anchors/
    img0023.png ...
    labels.csv                  # frame,bodypart,x,y,vis
  corrections/
    round_001/labels.csv
    round_002/labels.csv
  tracks.csv                    # dense per-frame, all bodyparts
  tracks_filtered.csv
  overlay.mp4
  review/
    img0000.png ...
    labels.csv                  # prefilled from tracks.csv
    baseline.csv                # snapshot for diff-promote

This module owns the on-disk schema. Pipeline stages and the imgui app
read/write through here so the layout has exactly one source of truth.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import tomli_w


PROJECT_TOML = "project.toml"


@dataclass
class TrackParams:
    chunk_size: int = 50
    overlap: int = 30
    refine: bool = True
    roi: bool = True
    postprocess: bool = True
    scorer: str = "cotracker3"


@dataclass
class FilterParams:
    enabled: bool = True
    min_cutoff: float = 0.05
    beta: float = 0.01
    d_cutoff: float = 0.5
    no_hampel: bool = False


@dataclass
class RenderParams:
    trail: int = 12
    radius: int = 6
    skip_jumps: bool = False


@dataclass
class JumpParams:
    n_sigmas: float = 3.0
    min_frac: float = 0.5
    max_run: int = 60


@dataclass
class ExtractParams:
    mode: str = "diverse"           # "uniform" | "diverse"
    n: int = 30


@dataclass
class Project:
    """In-memory view of a project. ``root`` is the .naparitracker folder."""
    root: Path
    video: str = ""
    bodyparts: list[str] = field(default_factory=list)
    extract: ExtractParams = field(default_factory=ExtractParams)
    track: TrackParams = field(default_factory=TrackParams)
    filter: FilterParams = field(default_factory=FilterParams)
    render: RenderParams = field(default_factory=RenderParams)
    jumps: JumpParams = field(default_factory=JumpParams)

    # ------------------------- path helpers ---------------------------- #

    @property
    def toml_path(self) -> Path:
        return self.root / PROJECT_TOML

    @property
    def anchors_dir(self) -> Path:
        return self.root / "anchors"

    @property
    def anchors_labels(self) -> Path:
        return self.anchors_dir / "labels.csv"

    @property
    def corrections_dir(self) -> Path:
        return self.root / "corrections"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def review_labels(self) -> Path:
        return self.review_dir / "labels.csv"

    @property
    def review_baseline(self) -> Path:
        return self.review_dir / "baseline.csv"

    @property
    def tracks_csv(self) -> Path:
        return self.root / "tracks.csv"

    @property
    def tracks_filtered_csv(self) -> Path:
        return self.root / "tracks_filtered.csv"

    @property
    def overlay_mp4(self) -> Path:
        return self.root / "overlay.mp4"

    def latest_corrections_round(self) -> Path | None:
        if not self.corrections_dir.is_dir():
            return None
        rounds = sorted(self.corrections_dir.glob("round_*"))
        return rounds[-1] if rounds else None

    def next_corrections_round(self) -> Path:
        existing: list[int] = []
        if self.corrections_dir.is_dir():
            for d in self.corrections_dir.glob("round_*"):
                tail = d.name[len("round_"):]
                if tail.isdigit():
                    existing.append(int(tail))
        n = (max(existing) + 1) if existing else 1
        return self.corrections_dir / f"round_{n:03d}"

    def all_label_csvs(self) -> list[Path]:
        """Anchor labels + every corrections round, in order. Used by the
        track stage to union all human labels into the query set."""
        out: list[Path] = []
        if self.anchors_labels.exists():
            out.append(self.anchors_labels)
        if self.corrections_dir.is_dir():
            for d in sorted(self.corrections_dir.glob("round_*")):
                lbl = d / "labels.csv"
                if lbl.exists():
                    out.append(lbl)
        return out

    # ------------------------- I/O ------------------------------------- #

    def save(self) -> None:
        """Atomic write — a crash mid-write must not leave project.toml empty."""
        self.root.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "video": self.video,
            "bodyparts": list(self.bodyparts),
            "extract": _params_dict(self.extract),
            "track": _params_dict(self.track),
            "filter": _params_dict(self.filter),
            "render": _params_dict(self.render),
            "jumps": _params_dict(self.jumps),
        }
        tmp = self.toml_path.with_suffix(self.toml_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            tomli_w.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.toml_path)


def _params_dict(p) -> dict:
    return {k: v for k, v in p.__dict__.items()}


def load(root: Path | str) -> Project:
    """Load a project from a .naparitracker directory."""
    root = Path(root)
    toml_path = root / PROJECT_TOML
    if not toml_path.is_file():
        raise FileNotFoundError(f"No project.toml in {root}")
    with toml_path.open("rb") as f:
        raw = tomllib.load(f)
    p = Project(root=root)
    p.video = str(raw.get("video", ""))
    p.bodyparts = list(raw.get("bodyparts", []))
    p.extract = _merge(ExtractParams(), raw.get("extract", {}))
    p.track = _merge(TrackParams(), raw.get("track", {}))
    p.filter = _merge(FilterParams(), raw.get("filter", {}))
    p.render = _merge(RenderParams(), raw.get("render", {}))
    p.jumps = _merge(JumpParams(), raw.get("jumps", {}))
    return p


def create(root: Path | str, video: str, bodyparts: list[str]) -> Project:
    """Create a new project on disk and return it."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Project dir is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    p = Project(root=root, video=str(video), bodyparts=list(bodyparts))
    p.save()
    return p


def _merge(default, raw: dict):
    """Override the dataclass defaults with whatever was in the TOML, ignoring
    unknown keys (so loading an older project on a newer schema doesn't crash).
    tomllib gives us no defaults — that's our job here."""
    if not isinstance(raw, dict):
        return default
    fields = {k: v for k, v in raw.items() if hasattr(default, k)}
    return replace(default, **fields)
