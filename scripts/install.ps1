# napari-cotrack one-shot installer for Windows (PowerShell).
#
# Usage from a PowerShell prompt:
#   irm https://raw.githubusercontent.com/RaulSimpetru/napari-cotrack/main/scripts/install.ps1 | iex
# or:
#   .\scripts\install.ps1
#
# What this does:
#   1. Installs `uv` (Astral's Python package manager) if it isn't on PATH yet.
#      uv is a single ~30 MB binary; it has no system dependencies.
#   2. Runs `uv tool install napari-cotrack`, which puts `napari-cotrack` on
#      your PATH and installs ~1.5 GB of Python deps (torch, napari, cv2, …)
#      into an isolated venv on first run.
#   3. Prints how to launch the GUI.
#
# To upgrade later: `uv tool upgrade napari-cotrack`
# To uninstall:    `uv tool uninstall napari-cotrack`

$ErrorActionPreference = 'Stop'

function Say  { param($msg) Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn { param($msg) Write-Host "!   $msg" -ForegroundColor Yellow }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "uv not found - installing the official Astral build..."
    irm https://astral.sh/uv/install.ps1 | iex

    # The Astral installer puts uv in %USERPROFILE%\.local\bin and updates the
    # *user* PATH. The current PowerShell session won't see it until restart,
    # so prepend for the rest of this script.
    $localBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path $localBin) {
        $env:Path = "$localBin;$env:Path"
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Warn "uv installed but isn't on PATH yet."
        Warn "Open a new PowerShell window and re-run this installer."
        exit 1
    }
}

Say "Installing napari-cotrack via uv..."
uv tool install --upgrade napari-cotrack

Say "Done."
Write-Host ""
Write-Host "  Run the GUI:    napari-cotrack"
Write-Host "  Upgrade later:  uv tool upgrade napari-cotrack"
Write-Host "  Uninstall:      uv tool uninstall napari-cotrack"
