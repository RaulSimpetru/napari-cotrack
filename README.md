# napari-cotrack

A napari plugin for **anchor-based keypoint tracking through CoTracker3**.
Label a few frames in napari; CoTracker3 propagates those keypoints across
the rest of the video; iterate by fixing wrong predictions and re-running.

## Install

The recommended path on every platform is the one-shot bootstrap script. It
installs [`uv`](https://docs.astral.sh/uv/) if you don't already have it, then
`uv tool install napari-cotrack` — putting `napari-cotrack` on your PATH and
fetching the ~1.5 GB of Python deps (torch, napari, cv2, …) on first run.

**macOS / Linux:**
```bash
curl -LsSf https://raw.githubusercontent.com/RaulSimpetru/napari-cotrack/main/scripts/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/RaulSimpetru/napari-cotrack/main/scripts/install.ps1 | iex
```

Then run `napari-cotrack`. To upgrade later: `uv tool upgrade napari-cotrack`.

### Already have uv?
```bash
uv tool install napari-cotrack
napari-cotrack
```

### Working in a clone of the repo?
```bash
uv sync
uv run napari-cotrack
```

### Optional: install ffmpeg for the fast extract-all path

Stage 4's "Extract all frames" runs in a single ffmpeg pass (~10–50× faster on
H.264 sources) when ffmpeg is on `PATH`, otherwise falls back to a per-frame
imageio loop.

| OS | Install ffmpeg |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | `winget install ffmpeg` (or download from gyan.dev) |

## Tested platforms

- **macOS arm64 (Apple Silicon)** — primary development target, end-to-end
  tested on a real 1500-frame project.
- **Linux x86_64** and **Windows x86_64** — code is portable (PyQt6 +
  pathlib + ffmpeg subprocess + napari thread_worker), all deps publish
  wheels for these platforms, but I haven't run the GUI on them.
  Open an issue if anything breaks — should be a small fix.

## The five-stage loop

The dock widget mirrors the loop:

1. **Extract anchor frames** — every Nth frame, or top K most diverse via
   k-means on RGB histograms.
2. **Label keypoints** — load anchors into the viewer and click them in.
   One Points layer named `labels`, colour driven by a `bodypart` property.
   Click moves an existing point at the same `(frame, bodypart)` instead
   of duplicating, then auto-advances to the next bodypart.
3. **Track with CoTracker3** — propagates labels (anchors + every
   `corrections/round_*` folder) across the whole video. Runs in a napari
   `thread_worker`, log streams into the dock pane.
4. **Review and fix** — extract every video frame as PNG (ffmpeg fast path
   if available), load them into napari, fix wrong predictions, **Promote**
   to write only the changed rows into a new `corrections/round_NNN/`. Or
   detect scene-confusion jumps and extract a representative frame per bad
   range.
5. **Render and inspect** — Hampel + 1€ filter, then the cv2 overlay
   renderer with bodypart trails. `Play with ffplay` on the result.

Re-running stage 3 after a correction round folds those new labels in
automatically.

## On-disk format

A project is a directory ending in `.naparitracker/`:

```
<name>.naparitracker/
  project.toml          # video path, bodyparts, per-stage knobs
  anchors/
    img0023.png ...
    labels.csv          # long-form: frame,bodypart,x,y,vis
  corrections/
    round_001/labels.csv
    round_002/labels.csv
  tracks.csv            # dense: every (frame, bodypart)
  tracks_filtered.csv
  overlay.mp4
  review/
    img0000.png ...
    labels.csv          # prefilled from tracks.csv
    baseline.csv        # snapshot for diff-promote
```

`project.toml` is hand-editable TOML. Every stage's knobs version with the
project, so `git diff` on `project.toml` shows what changed.

## Headless CLIs

Every stage is also a module CLI for batch / scripted use:

```bash
uv run python -m napari_cotrack.pipeline.extract anchors \
    --video VIDEO --output PROJECT/anchors \
    --bodyparts thumb,index,middle,ring,wrist \
    --mode diverse --n 30

uv run python -m napari_cotrack.pipeline.track    --project PROJECT
uv run python -m napari_cotrack.pipeline.filter   --project PROJECT
uv run python -m napari_cotrack.pipeline.render   --project PROJECT
uv run python -m napari_cotrack.pipeline.review extract-all --project PROJECT --force
uv run python -m napari_cotrack.pipeline.review promote     --project PROJECT
uv run python -m napari_cotrack.pipeline.review jumps       --project PROJECT [--extract-corrections]
```

All accept `--verbose` for per-chunk / per-bodypart detail.

## Tests

```bash
uv run python -m pytest tests/
```

Covers extract uniform/diverse, label-CSV union semantics, sparse-frame
mapping, CSV round-trip. Stage 3 (track) needs a real GPU + checkpoint;
smoke against the synthetic fixture in `tests/_make_fixture.py`.

## Keyboard

While editing in napari:
- **N** / **Shift+N** — next / previous bodypart (avoids napari's
  default Tab/Up/Down)
- **Ctrl+S** / **Cmd+S** — save labels.csv
- **Right** / **Left** — next / prev frame (napari default)
- **Delete** — remove selected points (napari default)

Save also fires automatically when you close napari, as a safety net.
