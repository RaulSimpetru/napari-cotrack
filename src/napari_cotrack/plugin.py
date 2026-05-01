"""napari-cotrack — napari plugin entry point.

A single dock widget on the right with the five pipeline stages. The user
opens napari, opens or creates a project, and stays in this one window
through the whole anchor → track → review → render loop.

Long stages (track, render, extract-all) run via ``napari.qt.thread_worker``
so the UI stays responsive. Their stdout is redirected into our log pane.
"""

from __future__ import annotations

import contextlib
import io
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from napari_cotrack import _layer, project as P
from napari_cotrack.pipeline import extract as ex
from napari_cotrack.pipeline import filter as ft
from napari_cotrack.pipeline import render as rd
from napari_cotrack.pipeline import review as rv
from napari_cotrack.pipeline import track as tk

try:
    from napari.qt.threading import thread_worker
except ImportError:  # only happens when running unit tests outside napari
    thread_worker = None  # type: ignore


# --------------------------- log redirection ------------------------------ #


class _StdoutPipe(io.TextIOBase):
    """File-like that pushes complete lines onto a thread-safe queue.

    The dock widget owns the queue and drains it from the Qt event loop on a
    QTimer. This keeps in-process pipeline ``print`` calls visible in the
    log pane without changing any pipeline code.
    """

    def __init__(self, q: queue.Queue[str]):
        super().__init__()
        self._q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._q.put(self._buf)
            self._buf = ""


# --------------------------- new-project dialog --------------------------- #


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.parent_dir = QLineEdit()
        self.video = QLineEdit()
        self.bodyparts = QLineEdit("thumb, index, middle, ring, wrist")

        pick_dir = QPushButton("Browse…"); pick_dir.clicked.connect(self._pick_dir)
        pick_video = QPushButton("Browse…"); pick_video.clicked.connect(self._pick_video)
        dir_row = QHBoxLayout(); dir_row.addWidget(self.parent_dir, 1); dir_row.addWidget(pick_dir)
        vid_row = QHBoxLayout(); vid_row.addWidget(self.video, 1); vid_row.addWidget(pick_video)

        form = QFormLayout()
        form.addRow("Parent folder", dir_row)
        form.addRow("Video", vid_row)
        form.addRow("Bodyparts (comma-sep)", self.bodyparts)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form); layout.addWidget(buttons)

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Pick parent folder",
                                             self.parent_dir.text() or str(Path.home()))
        if d: self.parent_dir.setText(d)

    def _pick_video(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Pick video", str(Path.home()),
                                           "Video (*.mp4 *.mov *.mkv *.avi);;All (*)")
        if f: self.video.setText(f)

    def values(self) -> tuple[str, str, list[str]]:
        bps = [b.strip() for b in self.bodyparts.text().split(",") if b.strip()]
        return self.parent_dir.text().strip(), self.video.text().strip(), bps


# --------------------------- main dock widget ----------------------------- #


class NapariCotrackWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.proj: P.Project | None = None
        self._layer_session: _layer.LayerSession | None = None
        self._busy: bool = False
        self._log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()

        # Drain stdout-redirect queue into the log pane on the Qt event loop.
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(80)
        self._log_timer.timeout.connect(self._drain_log)
        self._log_timer.start()

    # ---- chrome --------------------------------------------------------- #

    def _build_ui(self) -> None:
        self.proj_label = QLabel("(no project)")
        self.proj_label.setWordWrap(True)
        self.bp_label = QLabel("")
        self.bp_label.setWordWrap(True)
        new_btn = QPushButton("New…"); new_btn.clicked.connect(self.new_project)
        open_btn = QPushButton("Open…"); open_btn.clicked.connect(self.open_project)
        edit_bps = QPushButton("Edit bodyparts…"); edit_bps.clicked.connect(self.edit_bodyparts)

        header = QGroupBox("Project")
        h = QVBoxLayout(header)
        row = QHBoxLayout(); row.addWidget(new_btn); row.addWidget(open_btn); row.addStretch()
        h.addLayout(row)
        h.addWidget(self.proj_label)
        h.addWidget(self.bp_label)
        h.addWidget(edit_bps, alignment=Qt.AlignmentFlag.AlignLeft)

        # Stage 1 — extract anchors
        self.mode_uniform = QRadioButton("Every Nth frame")
        self.mode_diverse = QRadioButton("Top K most diverse"); self.mode_diverse.setChecked(True)
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 100000); self.n_spin.setValue(30)
        s1 = QGroupBox("1. Extract anchor frames")
        s1l = QVBoxLayout(s1)
        s1l.addWidget(self.mode_uniform); s1l.addWidget(self.mode_diverse)
        _s1_guide = QLabel("Pick frames evenly through the video, or pick the most visually different ones. K diverse works better when motion is uneven.")
        _s1_guide.setStyleSheet("color: gray; font-style: italic;")
        _s1_guide.setWordWrap(True)
        s1l.addWidget(_s1_guide)
        nrow = QHBoxLayout(); nrow.addWidget(QLabel("N:")); nrow.addWidget(self.n_spin); nrow.addStretch()
        s1l.addLayout(nrow)
        b = QPushButton("Extract anchors"); b.clicked.connect(self.run_extract); s1l.addWidget(b)

        # Stage 2 — label
        s2 = QGroupBox("2. Label keypoints")
        s2l = QVBoxLayout(s2)
        _s2_guide = QLabel("Load anchors, then load corrections if any, then navigate frames and save edits.")
        _s2_guide.setStyleSheet("color: gray; font-style: italic;")
        _s2_guide.setWordWrap(True)
        s2l.addWidget(_s2_guide)
        self.kp_combo = QComboBox()
        self.kp_combo.currentIndexChanged.connect(self._on_kp_changed)
        kpn = QHBoxLayout()
        prev = QPushButton("3. ◀ Prev (Shift+N)"); prev.clicked.connect(self._prev_kp)
        nxt = QPushButton("4. Next (N) ▶"); nxt.clicked.connect(self._next_kp)
        save_btn = QPushButton("5. Save (Ctrl/Cmd+S)"); save_btn.clicked.connect(self._save_layer)
        kpn.addWidget(prev); kpn.addWidget(nxt); kpn.addWidget(save_btn)
        s2l.addWidget(QLabel("Active keypoint:"))
        s2l.addWidget(self.kp_combo)
        s2l.addLayout(kpn)
        load_row = QHBoxLayout()
        b = QPushButton("1. Load anchors"); b.clicked.connect(lambda: self.load_layer("anchors")); load_row.addWidget(b)
        b = QPushButton("2. Load latest corrections"); b.clicked.connect(lambda: self.load_layer("corrections")); load_row.addWidget(b)
        s2l.addLayout(load_row)

        # Stage 3 — track
        self.chunk = QSpinBox(); self.chunk.setRange(8, 10000); self.chunk.setValue(50)
        self.overlap = QSpinBox(); self.overlap.setRange(0, 9999); self.overlap.setValue(30)
        self.refine = QCheckBox("refine"); self.refine.setChecked(True)
        self.roi = QCheckBox("ROI auto-crop"); self.roi.setChecked(True)
        self.post = QCheckBox("postprocess"); self.post.setChecked(True)
        s3 = QGroupBox("3. Track with CoTracker")
        s3l = QFormLayout(s3)
        _s3_guide = QLabel("Propagates your anchor labels and corrections across all video frames.")
        _s3_guide.setStyleSheet("color: gray; font-style: italic;")
        _s3_guide.setWordWrap(True)
        s3l.addRow(_s3_guide)
        s3l.addRow("chunk size", self.chunk)
        s3l.addRow("overlap", self.overlap)
        opts = QHBoxLayout(); opts.addWidget(self.refine); opts.addWidget(self.roi); opts.addWidget(self.post); opts.addStretch()
        s3l.addRow("options", opts)
        b = QPushButton("Run tracking"); b.clicked.connect(self.run_track); s3l.addRow(b)

        # Stage 4 — review
        s4 = QGroupBox("4. Review and fix")
        s4l = QVBoxLayout(s4)
        _s4_guide = QLabel("All-frames path: 1a → 2a → fix in napari → 3a → re-run Track. Jumps-only path: 1b → 2b → load corrections in Stage 2 → fix → re-run Track.")
        _s4_guide.setStyleSheet("color: gray; font-style: italic;")
        _s4_guide.setWordWrap(True)
        s4l.addWidget(_s4_guide)
        row1 = QHBoxLayout()
        for label, slot in [
            ("1a. Extract all frames", self.run_review_extract),
            ("2a. Load review", lambda: self.load_layer("review")),
            ("3a. Promote review edits", self.run_review_promote),
        ]:
            b = QPushButton(label); b.clicked.connect(slot); row1.addWidget(b)
        s4l.addLayout(row1)
        row2 = QHBoxLayout()
        for label, slot in [
            ("1b. Detect jumps", self.run_jumps),
            ("2b. Extract jumps for relabel", self.run_jumps_extract),
        ]:
            b = QPushButton(label); b.clicked.connect(slot); row2.addWidget(b)
        s4l.addLayout(row2)

        # Stage 5 — render
        self.trail = QSpinBox(); self.trail.setRange(0, 200); self.trail.setValue(12)
        self.radius = QSpinBox(); self.radius.setRange(1, 50); self.radius.setValue(6)
        self.skip_jumps = QCheckBox("skip jump frames")
        s5 = QGroupBox("5. Render and inspect")
        s5l = QFormLayout(s5)
        _s5_guide = QLabel("Filter first (recommended), then render the overlay, then play to inspect.")
        _s5_guide.setStyleSheet("color: gray; font-style: italic;")
        _s5_guide.setWordWrap(True)
        s5l.addRow(_s5_guide)
        s5l.addRow("trail (frames)", self.trail)
        s5l.addRow("marker radius", self.radius)
        s5l.addRow("", self.skip_jumps)
        row = QHBoxLayout()
        for label, slot in [
            ("1. Filter (Hampel + 1Euro)", self.run_filter),
            ("2. Render overlay", self.run_render),
            ("3. Play with ffplay", self.play),
        ]:
            b = QPushButton(label); b.clicked.connect(slot); row.addWidget(b)
        s5l.addRow(row)

        # Stages live in a scroll area; log pane below in a vertical splitter.
        stages = QWidget()
        v = QVBoxLayout(stages); v.setContentsMargins(0, 0, 0, 0)
        for w in (header, s1, s2, s3, s4, s5):
            v.addWidget(w)
        v.addStretch()
        scroll = QScrollArea(); scroll.setWidget(stages); scroll.setWidgetResizable(True)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 10))
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(scroll); splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 1)

        outer = QVBoxLayout(self); outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)

    # ---- project lifecycle --------------------------------------------- #

    def new_project(self) -> None:
        d = NewProjectDialog(self)
        if d.exec() != QDialog.DialogCode.Accepted: return
        parent_dir, video, bps = d.values()
        if not parent_dir or not video or not bps:
            QMessageBox.warning(self, "Missing", "Need parent folder, video, and bodyparts.")
            return
        stem = Path(video).stem or "project"
        root = Path(parent_dir) / f"{stem}.naparitracker"
        try:
            self.proj = P.create(root, video=video, bodyparts=bps)
        except (FileExistsError, OSError) as e:
            QMessageBox.critical(self, "Create failed", str(e)); return
        self._refresh_header(); self._log(f"created {root}")

    def open_project(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open .naparitracker folder", str(Path.home()))
        if not d: return
        try:
            self.proj = P.load(Path(d))
        except (FileNotFoundError, OSError) as e:
            QMessageBox.critical(self, "Open failed", str(e)); return
        self._refresh_header(); self._log(f"opened {d}")

    def edit_bodyparts(self) -> None:
        if self.proj is None: return
        text, ok = QInputDialog.getMultiLineText(
            self, "Edit bodyparts", "One bodypart per line:",
            "\n".join(self.proj.bodyparts),
        )
        if not ok: return
        self.proj.bodyparts = [line.strip() for line in text.splitlines() if line.strip()]
        self.proj.save(); self._refresh_header()

    def _refresh_header(self) -> None:
        if self.proj is None:
            self.proj_label.setText("(no project)"); self.bp_label.setText("")
            self.kp_combo.clear(); return
        self.proj_label.setText(f"{self.proj.root}\n{self.proj.video}")
        self.bp_label.setText(", ".join(self.proj.bodyparts) or "(none)")
        # Sync stage controls from project.toml.
        self.mode_uniform.setChecked(self.proj.extract.mode == "uniform")
        self.mode_diverse.setChecked(self.proj.extract.mode != "uniform")
        self.n_spin.setValue(self.proj.extract.n)
        self.chunk.setValue(self.proj.track.chunk_size)
        self.overlap.setValue(self.proj.track.overlap)
        self.refine.setChecked(self.proj.track.refine)
        self.roi.setChecked(self.proj.track.roi)
        self.post.setChecked(self.proj.track.postprocess)
        self.trail.setValue(self.proj.render.trail)
        self.radius.setValue(self.proj.render.radius)
        self.skip_jumps.setChecked(self.proj.render.skip_jumps)
        self.kp_combo.blockSignals(True)
        self.kp_combo.clear()
        self.kp_combo.addItems(self.proj.bodyparts)
        self.kp_combo.blockSignals(False)

    def _persist(self) -> None:
        if self.proj is None: return
        self.proj.extract.mode = "uniform" if self.mode_uniform.isChecked() else "diverse"
        self.proj.extract.n = self.n_spin.value()
        self.proj.track.chunk_size = self.chunk.value()
        self.proj.track.overlap = self.overlap.value()
        self.proj.track.refine = self.refine.isChecked()
        self.proj.track.roi = self.roi.isChecked()
        self.proj.track.postprocess = self.post.isChecked()
        self.proj.render.trail = self.trail.value()
        self.proj.render.radius = self.radius.value()
        self.proj.render.skip_jumps = self.skip_jumps.isChecked()
        self.proj.save()

    # ---- in-process worker plumbing ------------------------------------ #

    def _run_in_thread(self, fn: Callable[[], object]) -> None:
        """Run a callable in a worker thread, redirecting its stdout into our
        log pane. One worker at a time."""
        if self._busy:
            self._log("[busy] another step is running"); return
        if thread_worker is None:
            self._log("[error] napari.qt.threading unavailable"); return
        if self.proj is None:
            self._log("[error] open or create a project first"); return
        self._persist()
        self._busy = True
        self._log(f"$ running {getattr(fn, '__name__', 'task')}")

        pipe = _StdoutPipe(self._log_queue)

        @thread_worker(connect={"returned": self._on_worker_done,
                                "errored": self._on_worker_error})
        def _w():
            with contextlib.redirect_stdout(pipe), contextlib.redirect_stderr(pipe):
                return fn()

        _w()

    def _on_worker_done(self, result) -> None:
        self._busy = False
        # Pipeline cmd_* funcs return an int exit code (0 = ok, 1 = recoverable
        # failure like "no tracks yet"). run_* funcs return paths (truthy) or
        # None on no-op. Treat anything non-zero as a failure.
        ok = result is None or result == 0 or not isinstance(result, int)
        self._log("[done]" if ok else f"[failed exit={result}]")

    def _on_worker_error(self, exc: Exception) -> None:
        self._busy = False
        self._log(f"[error] {type(exc).__name__}: {exc}")

    def _drain_log(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.log.appendPlainText(line)
        except queue.Empty:
            pass

    def _log(self, line: str) -> None:
        self.log.appendPlainText(line)

    # ---- click handlers ------------------------------------------------- #

    def run_extract(self) -> None:
        if self.proj is None: return
        proj = self.proj
        mode = "uniform" if self.mode_uniform.isChecked() else "diverse"
        n = self.n_spin.value()

        def _go():
            import imageio
            reader = imageio.get_reader(proj.video)
            total = reader.count_frames(); reader.close()
            if mode == "uniform":
                indices = ex.select_uniform(total, n)
            else:
                indices = ex.select_diverse(proj.video, n, total_frames=total)
            print(f"Selected {len(indices)} {mode} frames out of {total}")
            ex.extract_anchors(proj.video, indices, proj.anchors_dir, proj.bodyparts)
        self._run_in_thread(_go)

    def run_track(self) -> None:
        if self.proj is None: return
        proj = self.proj
        self._run_in_thread(lambda: tk.run_track(str(proj.root)))

    def run_filter(self) -> None:
        if self.proj is None: return
        proj = self.proj
        self._run_in_thread(lambda: ft.run_filter(str(proj.root)))

    def run_render(self) -> None:
        if self.proj is None: return
        proj = self.proj
        self._run_in_thread(lambda: rd.run_render(str(proj.root)))

    def run_review_extract(self) -> None:
        if self.proj is None: return
        proj = self.proj

        # Re-extracting nukes proj.review_dir and any user edits inside it.
        # If a baseline.csv is present the folder has been used at least once;
        # confirm before stomping it. New projects with no review folder yet
        # skip the prompt entirely.
        if proj.review_dir.exists() and proj.review_baseline.exists():
            ans = QMessageBox.warning(
                self, "Overwrite review folder?",
                f"{proj.review_dir} already contains review labels and a baseline.\n\n"
                f"Re-extracting will delete that folder and any unsaved edits inside.\n\n"
                f"Promote first if you want to keep the edits.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Discard,
                QMessageBox.StandardButton.Cancel,
            )
            if ans != QMessageBox.StandardButton.Discard:
                self._log("[skipped] re-extract cancelled by user")
                return

        self._run_in_thread(lambda: rv.do_extract_all(str(proj.root), force=True))

    def run_review_promote(self) -> None:
        if self.proj is None: return
        proj = self.proj
        self._run_in_thread(lambda: rv.do_promote(str(proj.root)))

    def run_jumps(self) -> None:
        if self.proj is None: return
        proj = self.proj
        self._run_in_thread(lambda: rv.do_jumps(str(proj.root)))

    def run_jumps_extract(self) -> None:
        if self.proj is None: return
        proj = self.proj
        self._run_in_thread(lambda: rv.do_jumps(str(proj.root), extract_corrections=True))

    def play(self) -> None:
        if self.proj is None: return
        if not self.proj.overlay_mp4.exists():
            self._log(f"no overlay yet at {self.proj.overlay_mp4}"); return
        ffplay = shutil.which("ffplay")
        if ffplay is None:
            self._log("ffplay not on PATH"); return
        try:
            subprocess.Popen([ffplay, "-autoexit", "-loglevel", "error",
                              str(self.proj.overlay_mp4)])
        except OSError as e:
            self._log(f"ffplay failed: {e}")

    # ---- in-place napari layer -------------------------------------- #

    def load_layer(self, target: str) -> None:
        """Replace the current viewer contents with a frames folder + labels
        layer for the given target ('anchors' | 'corrections' | 'review')."""
        if self.proj is None:
            self._log("[error] open or create a project first"); return
        if target == "anchors":
            folder, csv = self.proj.anchors_dir, self.proj.anchors_labels
        elif target == "review":
            folder, csv = self.proj.review_dir, self.proj.review_labels
        elif target == "corrections":
            latest = self.proj.latest_corrections_round()
            if latest is None:
                self._log(f"[error] no corrections rounds yet"); return
            folder, csv = latest, latest / "labels.csv"
        else:
            return
        if not folder.is_dir():
            self._log(f"[error] no frames folder at {folder}"); return

        # Tear down any prior layer session and clear the viewer.
        try:
            self.viewer.layers.clear()
        except Exception:
            pass
        try:
            self._layer_session = _layer.attach_keypoint_layer(
                self.viewer, folder, csv, self.proj.bodyparts,
                on_active_changed=self._sync_combo_to_active,
            )
        except (FileNotFoundError, ValueError) as e:
            self._log(f"[error] {e}"); return
        self._log(f"loaded {target}: {folder}")

    def _sync_combo_to_active(self, bp: str) -> None:
        idx = self.proj.bodyparts.index(bp) if bp in self.proj.bodyparts else 0
        self.kp_combo.blockSignals(True)
        self.kp_combo.setCurrentIndex(idx)
        self.kp_combo.blockSignals(False)

    def _on_kp_changed(self, idx: int) -> None:
        if self._layer_session is not None and idx >= 0:
            self._layer_session.set_active_idx(idx)

    def _next_kp(self) -> None:
        if self._layer_session is not None: self._layer_session.next_kp()

    def _prev_kp(self) -> None:
        if self._layer_session is not None: self._layer_session.prev_kp()

    def _save_layer(self) -> None:
        if self._layer_session is not None:
            self._layer_session.save()
            self._log(f"saved {self._layer_session.csv_path}")
        else:
            self._log("[error] no layer to save (load anchors/review/corrections first)")
