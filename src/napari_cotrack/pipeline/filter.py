"""Stage 5a: Hampel + 1€ post-filter for tracks (long-form CSV).

Hampel removes per-bodypart outliers (3σ MAD around a rolling median).
1€ filter is the standard speed-adaptive low-pass. Use ``--no-hampel`` on
manually-corrected data so spike rejection doesn't second-guess the user.

Module CLI:
    python -m napari_cotrack.pipeline.filter --project /path/to/foo.naparitracker
        [--no-hampel] [--min-cutoff F] [--beta B]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from napari_cotrack import project as P
from napari_cotrack.pipeline._io import dense_to_arrays, read_labels, write_labels

VERBOSE = False


def _v(msg: str) -> None:
    if VERBOSE:
        print(msg)


class OneEuroFilter:
    """1€ filter (Casiez, Roussel, Vogel 2012). Frame-indexed (dt=1)."""

    def __init__(self, min_cutoff: float = 0.05, beta: float = 0.01,
                 d_cutoff: float = 0.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float = 1.0) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = x - self.x_prev
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def filter_series(values: np.ndarray, min_cutoff: float, beta: float,
                  d_cutoff: float) -> np.ndarray:
    f = OneEuroFilter(min_cutoff, beta, d_cutoff)
    return np.array([f(v) for v in values])


def hampel(values: np.ndarray, window: int = 7, n_sigmas: float = 3.0
           ) -> tuple[np.ndarray, np.ndarray]:
    x = values.astype(float).copy()
    n = len(x)
    k = window // 2
    mask = np.zeros(n, dtype=bool)
    cleaned = x.copy()
    scale = 1.4826
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        w = x[a:b]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        sigma = scale * mad
        if sigma > 0 and abs(x[i] - med) > n_sigmas * sigma:
            cleaned[i] = med
            mask[i] = True
    return cleaned, mask


def comparison_plot(xs_raw, ys_raw, xs_flt, ys_flt, bodyparts, out_path):
    T = xs_raw.shape[0]
    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    cmap = plt.cm.tab10
    for i, bp in enumerate(bodyparts):
        c = cmap(i)
        axes[0].plot(t, xs_raw[:, i], color=c, alpha=0.35, linewidth=0.8)
        axes[0].plot(t, xs_flt[:, i], color=c, linewidth=1.4, label=bp)
        axes[1].plot(t, ys_raw[:, i], color=c, alpha=0.35, linewidth=0.8)
        axes[1].plot(t, ys_flt[:, i], color=c, linewidth=1.4, label=bp)
        spd_r = np.hypot(np.diff(xs_raw[:, i], prepend=xs_raw[0, i]),
                         np.diff(ys_raw[:, i], prepend=ys_raw[0, i]))
        spd_f = np.hypot(np.diff(xs_flt[:, i], prepend=xs_flt[0, i]),
                         np.diff(ys_flt[:, i], prepend=ys_flt[0, i]))
        axes[2].plot(t, spd_r, color=c, alpha=0.3, linewidth=0.8)
        axes[2].plot(t, spd_f, color=c, linewidth=1.2, label=bp)
    axes[0].set_ylabel("x (px)")
    axes[1].set_ylabel("y (px)")
    axes[2].set_ylabel("speed (px/frame)")
    axes[2].set_xlabel("frame")
    axes[0].set_title("1€ filter — faded = raw, solid = filtered")
    axes[0].legend(loc="upper right", ncol=len(bodyparts))
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_filter(project_root: str | Path, *,
               no_hampel: bool = False,
               min_cutoff: float | None = None,
               beta: float | None = None,
               d_cutoff: float | None = None,
               hampel_window: int = 11, hampel_sigmas: float = 3.0,
               hampel_iters: int = 3) -> Path:
    proj = P.load(Path(project_root))
    fp = proj.filter
    min_cutoff = fp.min_cutoff if min_cutoff is None else min_cutoff
    beta = fp.beta if beta is None else beta
    d_cutoff = fp.d_cutoff if d_cutoff is None else d_cutoff
    no_hampel = no_hampel or fp.no_hampel

    if not proj.tracks_csv.exists():
        raise FileNotFoundError(f"No tracks at {proj.tracks_csv}; run stage 3 first")

    df = read_labels(proj.tracks_csv)
    xs, ys, vs, bodyparts = dense_to_arrays(df)
    T = xs.shape[0]
    print(f"Loaded {T} frames x {len(bodyparts)} bodyparts")
    _v(f"min_cutoff={min_cutoff}  beta={beta}  d_cutoff={d_cutoff}")

    xs_raw, ys_raw = xs.copy(), ys.copy()
    xs_clean, ys_clean = xs.copy(), ys.copy()

    if not no_hampel:
        total = 0
        for i, bp in enumerate(bodyparts):
            x_arr = xs[:, i].copy()
            y_arr = ys[:, i].copy()
            for _ in range(hampel_iters):
                x_arr, mx = hampel(x_arr, hampel_window, hampel_sigmas)
                y_arr, my = hampel(y_arr, hampel_window, hampel_sigmas)
                if not (mx.any() or my.any()):
                    break
            xs_clean[:, i] = x_arr
            ys_clean[:, i] = y_arr
            replaced = int(((x_arr != xs[:, i]) | (y_arr != ys[:, i])).sum())
            total += replaced
            _v(f"  {bp:8s}: {replaced} outlier frame(s) replaced")
        print(f"Hampel total: {total} replacements")

    xs_flt = np.zeros_like(xs_clean)
    ys_flt = np.zeros_like(ys_clean)
    for i in range(len(bodyparts)):
        xs_flt[:, i] = filter_series(xs_clean[:, i], min_cutoff, beta, d_cutoff)
        ys_flt[:, i] = filter_series(ys_clean[:, i], min_cutoff, beta, d_cutoff)

    rows = []
    for t in range(T):
        for i, bp in enumerate(bodyparts):
            rows.append((int(t), bp, float(xs_flt[t, i]), float(ys_flt[t, i]),
                         float(vs[t, i])))
    out_df = pd.DataFrame(rows, columns=["frame", "bodypart", "x", "y", "vis"])
    write_labels(out_df, proj.tracks_filtered_csv)
    print(f"Saved filtered -> {proj.tracks_filtered_csv}")

    plot_dir = proj.root / "plots"
    plot_dir.mkdir(exist_ok=True)
    comparison_plot(xs_raw, ys_raw, xs_flt, ys_flt, bodyparts,
                    plot_dir / "filter_compare.png")
    print(f"Plots -> {plot_dir}/")
    return proj.tracks_filtered_csv


def main(argv: list[str] | None = None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--no-hampel", action="store_true")
    ap.add_argument("--min-cutoff", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--d-cutoff", type=float, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = args.verbose
    run_filter(
        args.project,
        no_hampel=args.no_hampel,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        d_cutoff=args.d_cutoff,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
