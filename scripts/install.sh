#!/usr/bin/env bash
# napari-cotrack one-shot installer for macOS / Linux.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/RaulSimpetru/napari-cotrack/main/scripts/install.sh | bash
# or:
#   ./scripts/install.sh
#
# What this does:
#   1. Installs `uv` (Astral's Python package manager) if it isn't on PATH yet.
#      uv is a single ~30 MB binary; it has no system dependencies.
#   2. Runs `uv tool install napari-cotrack`, which puts the `napari-cotrack`
#      command on your PATH and installs ~1.5 GB of Python deps (torch, napari,
#      cv2, …) into an isolated venv on first run.
#   3. Prints how to launch the GUI.
#
# To upgrade later: `uv tool upgrade napari-cotrack`
# To uninstall:    `uv tool uninstall napari-cotrack`

set -euo pipefail

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!  \033[0m %s\n' "$*" >&2; }

if ! command -v uv >/dev/null 2>&1; then
    say "uv not found — installing the official Astral build…"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # The Astral installer adds ~/.local/bin to your shell's startup file but
    # the current shell session won't see it until it sources that file. Add
    # it to PATH for the rest of this script.
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv >/dev/null 2>&1; then
        warn "uv installed but isn't on PATH yet."
        warn "Open a new terminal and re-run this installer."
        exit 1
    fi
fi

say "Installing napari-cotrack via uv…"
uv tool install --upgrade napari-cotrack

say "Done."
echo
echo "  Run the GUI:    napari-cotrack"
echo "  Upgrade later:  uv tool upgrade napari-cotrack"
echo "  Uninstall:      uv tool uninstall napari-cotrack"
