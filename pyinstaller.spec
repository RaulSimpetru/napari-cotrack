# pyinstaller.spec — napari-cotrack one-folder bundle
# Run with: pyinstaller pyinstaller.spec --noconfirm
#
# The CoTracker3 .pth checkpoint is NOT bundled; it is downloaded at first
# launch by torch.hub.load_state_dict_from_url.
# ffmpeg is NOT bundled; the pipeline falls back gracefully to imageio.

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# ── Collect whole packages ────────────────────────────────────────────────────
# collect_all() returns (datas, binaries, hiddenimports) for each package.

nc_d, nc_b, nc_h   = collect_all('napari_cotrack')
ct_d, ct_b, ct_h   = collect_all('cotracker')
np_d, np_b, np_h   = collect_all('napari')
npe2_d, npe2_b, npe2_h = collect_all('npe2')
app_d, app_b, app_h = collect_all('app_model')
mg_d, mg_b, mg_h   = collect_all('magicgui')
vp_d, vp_b, vp_h   = collect_all('vispy')
ic_d, ic_b, ic_h   = collect_all('imagecodecs')
sci_d, sci_b, sci_h = collect_all('scipy')
sk_d, sk_b, sk_h   = collect_all('sklearn')
psutil_d, psutil_b, psutil_h = collect_all('psutil')

all_datas    = (nc_d + ct_d + np_d + npe2_d + app_d + mg_d + vp_d + ic_d +
                sci_d + sk_d + psutil_d)
all_binaries = (nc_b + ct_b + np_b + npe2_b + app_b + mg_b + vp_b + ic_b +
                sci_b + sk_b + psutil_b)
all_hidden   = (nc_h + ct_h + np_h + npe2_h + app_h + mg_h + vp_h + ic_h +
                sci_h + sk_h + psutil_h)

# ── Extra data files ──────────────────────────────────────────────────────────
# napari.yaml must be resolvable by npe2 inside the bundle at the path
# napari_cotrack/napari.yaml (matching the wheel layout).
extra_datas = [
    ('src/napari_cotrack/napari.yaml', 'napari_cotrack'),
]

# Also grab vispy GLSL shaders which are pure-data and often missed.
extra_datas += collect_data_files('vispy', includes=['**/*.glsl', '**/*.vert',
                                                     '**/*.frag', '**/*.geo'])
# imageio plugin registry
extra_datas += collect_data_files('imageio')

all_datas = all_datas + extra_datas

# ── Hidden imports ────────────────────────────────────────────────────────────
# Torch JIT / cpp extensions loaded at runtime that PyInstaller cannot see.
torch_hidden = [
    'torch',
    'torch.jit',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torch._C',
    'torch._dynamo',
    'torch._inductor',
    'torch.backends',
    'torch.backends.cudnn',
    'torch.cuda',
    'torch.distributed',
    'torch.fx',
    'torch.jit._builtins',
    'torch.jit.supported_ops',
    'torch.onnx',
]
torchvision_hidden = [
    'torchvision',
    'torchvision.transforms',
    'torchvision.transforms.functional',
    'torchvision.io',
]
qt_hidden = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtOpenGL',
    'PyQt6.QtOpenGLWidgets',
    'PyQt6.sip',
]
napari_hidden = [
    'napari._qt',
    'napari._qt.qt_event_loop',
    'napari._qt.dialogs',
    'napari._app_model',
    'napari._app_model.context',
    'napari.components',
    'napari.layers',
    'napari.plugins',
    'napari.qt',
    'napari.settings',
    'napari.utils',
    'napari.utils.colormaps',
    'napari._vispy',
    'napari._vispy.layers',
]
misc_hidden = [
    'einops',
    'imageio',
    'imageio.plugins',
    'imageio.plugins.ffmpeg',
    'imageio.plugins.pyav',
    'imageio.v3',
    'cv2',
    'pandas',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qtagg',
    'tqdm',
    'tomli_w',
    'pint',
    'cachier',
    'typer',
    'rich',
    'superqt',
    'in_n_out',
]

all_hidden = (all_hidden + torch_hidden + torchvision_hidden + qt_hidden +
              napari_hidden + misc_hidden)

# Deduplicate
all_hidden = list(dict.fromkeys(all_hidden))

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ['src/napari_cotrack/__main__.py'],
    pathex=['src'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Keep the bundle smaller — things we definitely don't need at runtime.
        'IPython',
        'ipykernel',
        'jupyter',
        'notebook',
        'tkinter',
        '_tkinter',
        'doctest',
        'unittest',
        'pydoc',
        # Test infrastructure
        'pytest',
        'hypothesis',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='napari-cotrack',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX can corrupt Qt/torch shared libs — leave off
    console=False,       # no terminal window on macOS/Windows
    disable_windowed_traceback=False,
    target_arch=None,    # let PyInstaller auto-detect (arm64 on macos-14)
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='napari-cotrack',
)
