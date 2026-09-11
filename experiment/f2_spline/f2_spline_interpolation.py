"""
f2_spline_interpolation.py -- reconstruct F2's dissociation curve from a
coarse VQE scan using a cubic spline, and measure how good that
reconstruction actually is.

RUNS NO CHEMISTRY. This script loads a scan that
``experiment/f2/f2_ground_state_estimation.py`` already computed and saved,
and does pure numerical analysis on it. It takes seconds, and you can run it
as many times as you like without ever paying for another VQE point.

WHY THIS FOLDER EXISTS
------------------------
The scan script plots its curve by joining neighbouring points with straight
line segments -- the crudest possible reconstruction of a smooth function.
Fine for looking at, not fine for *using*: the lowest of the scanned samples
is not the minimum of the true curve, and anything you would want from the
curve -- the minimum, its curvature, the dissociation limit -- needs values
*between* the samples.

A cubic spline fits a separate cubic between each adjacent pair of points and
chooses their coefficients so the pieces agree in value, slope and curvature
wherever they meet [1]. The result interpolates every computed point exactly
and stays smooth, without the oscillation a single high-degree polynomial
suffers -- which is what the sibling ``f2_polynomial/`` folder demonstrates
going wrong.

HOW THE ERROR IS MEASURED, WITHOUT MORE VQE
---------------------------------------------
**Leave-one-out cross-validation.** Take one interior node out, build the
interpolant from the remaining nodes, and evaluate it at the node you
removed, where the true answer is known. Repeat for every interior node. The
resulting errors are genuine out-of-sample errors at real geometries, and
they cost nothing but arithmetic.

The same procedure is applied to plain linear interpolation, as a baseline.

Endpoints are excluded -- removing one turns interpolation into
extrapolation, a different and much harder problem.

OUTPUT
-------
  f2_spline_interp.json / .csv   per-node leave-one-out errors, plus summary
                                 metrics and the interpolated minimum in the
                                 metadata
  f2_spline_dense.csv            the smooth reconstructed curve, densely
                                 sampled, for your own plots
  f2_spline_curve.png            nodes, the spline through them, and the old
                                 straight-line version for comparison
  f2_spline_error.png            leave-one-out error per node, spline vs linear

REFERENCES
-----------
[1] scipy.interpolate.CubicSpline -- piecewise cubic, twice continuously
    differentiable, interpolating every data point:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
[2] scipy.optimize.minimize_scalar, used to locate the interpolant's minimum
    to better precision than the dense sampling alone:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

USAGE
------
Every case, explicitly. Runs no chemistry -- it reads a scan that
experiment/f2/ already saved.

  DEFAULT -- finds f2_scan.json (or .csv) in experiment/f2/
    python experiment/f2_spline/f2_spline_interpolation.py

  A PARTICULAR SCAN -- .json preferred, .csv accepted
    python experiment/f2_spline/f2_spline_interpolation.py --scan f2_scan.json
    python experiment/f2_spline/f2_spline_interpolation.py --scan f2_scan.csv

  WHICH ENERGY COLUMN -- 'vqe' by default. Interpolating 'exact' instead
  separates coarse-sampling error from VQE optimizer noise.
    python experiment/f2_spline/f2_spline_interpolation.py --column vqe
    python experiment/f2_spline/f2_spline_interpolation.py --column exact
    python experiment/f2_spline/f2_spline_interpolation.py --column hf

  OTHER KNOBS
    --dense 2000       number of points in the reconstructed curve
    --no-save          don't write results or figures
    --no-show          write the PNGs without opening a window
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

# F2's experimental equilibrium bond length, for comparison against whatever
# the interpolated minimum comes out as.
EQUILIBRIUM_BOND = 1.412   # Angstrom

DEFAULT_SCAN_NAMES = ["f2_scan.json", "f2_scan.csv"]


def find_scan(explicit: str | None, here: Path) -> Path:
    """Locate the saved F2 scan to interpolate. See the o2_spline twin."""
    f2_dir = here.parents[0] / "f2"
    if explicit is not None:
        candidates = [Path(explicit), here / explicit, f2_dir / explicit]
    else:
        candidates = [f2_dir / name for name in DEFAULT_SCAN_NAMES]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    looked = "\n  ".join(display_path(c) for c in candidates)
    raise SystemExit(
        f"no saved F2 scan found. Looked in:\n  {looked}\n\n"
        f"This script interpolates an existing scan; it does not run any chemistry.\n"
        f"Produce one first with:\n"
        f"  python experiment/f2/f2_ground_state_estimation.py"
    )


def _rows_from_csv(path: Path) -> tuple:
    """Read an F2 scan from its ``.csv``. Metadata is inferred; see the scan script."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("distance", "hf", "vqe", "exact")
    rows = [{k: float(v) for k, v in r.items() if k in numeric and v != ""} for r in raw]
    missing = {"distance", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} lacks column(s) {sorted(missing)} -- "
                         f"not an F2 scan CSV.")
    return {"molecule": "F2"}, rows


def _load(path: Path) -> tuple:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    if path.suffix.lower() == ".csv":
        return _rows_from_csv(path)
    return load_scan(path)


def leave_one_out(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Out-of-sample error at each interior node, for spline and for linear.

    For each interior index i: drop node i, rebuild each interpolant from the
    remaining nodes, evaluate at x[i], and compare with the y[i] we actually
    know. Endpoints are skipped -- dropping one of those leaves no data on
    that side, making it extrapolation rather than interpolation.

    Returns NaN at the endpoints so the arrays stay aligned with the nodes.
    """
    n = len(x)
    spline_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)

    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]

        # CubicSpline needs at least 4 points to define a cubic piece.
        if len(xk) >= 4:
            spline_err[i] = float(CubicSpline(xk, yk)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]

    return {"spline": spline_err, "linear": linear_err}


def metrics(errors: np.ndarray) -> Dict[str, float]:
    """Max / RMS / mean-absolute error in milli-Hartree, ignoring NaN slots."""
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {"max_mha": float("nan"), "rms_mha": float("nan"), "mae_mha": float("nan")}
    return {
        "max_mha": float(np.max(np.abs(finite)) * 1000),
        "rms_mha": float(np.sqrt(np.mean(finite ** 2)) * 1000),
        "mae_mha": float(np.mean(np.abs(finite)) * 1000),
    }


def interpolated_minimum(spline: CubicSpline, x: np.ndarray) -> Dict[str, float]:
    """Locate the spline's minimum properly, not just the lowest sample.

    Two stages: sample densely to find which bracket the minimum falls in,
    then refine inside that bracket with a bounded scalar minimizer [2]. The
    dense scan alone would only be as accurate as its own spacing, and the
    refinement is free.
    """
    dense_x = np.linspace(x[0], x[-1], 4001)
    dense_y = spline(dense_x)
    k = int(np.argmin(dense_y))
    lo = dense_x[max(k - 1, 0)]
    hi = dense_x[min(k + 1, len(dense_x) - 1)]

    result = minimize_scalar(lambda t: float(spline(t)), bounds=(lo, hi), method="bounded")
    return {"x": float(result.x), "energy": float(result.fun)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct F2's dissociation curve from a coarse VQE scan with a "
                    "cubic spline, and measure the reconstruction error by "
                    "leave-one-out cross-validation. Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE",
                        help="the saved F2 scan to interpolate (default: look for "
                             "f2_scan.json in experiment/f2/)")
    parser.add_argument("--column", choices=["vqe", "exact", "hf"], default="vqe",
                        help="which energy column to interpolate (default vqe)")
    parser.add_argument("--dense", type=int, default=800,
                        help="number of points in the reconstructed dense curve (default 800)")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save the results (default: f2_spline_interp.json/.csv "
                             "next to this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save results or figures")
    parser.add_argument("--no-show", action="store_true",
                        help="write the figures but don't open a window (useful over SSH)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.no_show:
        matplotlib.use("Agg")

    # -- load the nodes ------------------------------------------------
    scan_path = find_scan(args.scan, here)
    metadata, rows = _load(scan_path)
    rows = sorted(rows, key=lambda r: r["distance"])
    x = np.array([r["distance"] for r in rows], dtype=float)
    y = np.array([r[args.column] for r in rows], dtype=float)

    if len(x) < 4:
        raise SystemExit(
            f"{display_path(scan_path)} has only {len(x)} points; a cubic spline needs at "
            f"least 4. Re-run the F2 scan with a finer --step."
        )

    print(f"loaded {len(x)} nodes from {display_path(scan_path)}")
    print(f"  {metadata.get('molecule', 'F2')} "
          f"CAS({metadata.get('active_electrons')}, {metadata.get('active_orbitals')}), "
          f"column '{args.column}'")
    print(f"  {x[0]:.2f} to {x[-1]:.2f} A, spacing {np.mean(np.diff(x)):.3f} A")
    print("  no chemistry is run -- this is pure interpolation\n")

    # -- build the interpolant -----------------------------------------
    spline = CubicSpline(x, y)
    dense_x = np.linspace(x[0], x[-1], args.dense)
    dense_spline = spline(dense_x)
    dense_linear = np.interp(dense_x, x, y)

    # -- the minimum: interpolated vs best raw sample -------------------
    best_node = int(np.argmin(y))
    interp_min = interpolated_minimum(spline, x)

    print("MINIMUM OF THE CURVE")
    print(f"  lowest computed sample : {y[best_node]:.6f} Ha at {x[best_node]:.4f} A")
    print(f"  spline minimum         : {interp_min['energy']:.6f} Ha at "
          f"{interp_min['x']:.4f} A")
    print(f"  experimental reference : {EQUILIBRIUM_BOND} A")
    node_off = abs(x[best_node] - EQUILIBRIUM_BOND)
    interp_off = abs(interp_min["x"] - EQUILIBRIUM_BOND)
    print(f"  distance from experiment: sample {node_off:.4f} A, "
          f"spline {interp_off:.4f} A")
    if interp_off < node_off:
        print(f"  -> interpolating recovered the bond length "
              f"{node_off - interp_off:.4f} A closer than the best sample could")
    print()

    # -- leave-one-out error -------------------------------------------
    loo = leave_one_out(x, y)
    spline_metrics = metrics(loo["spline"])
    linear_metrics = metrics(loo["linear"])

    print("LEAVE-ONE-OUT ERROR  (drop a node, rebuild, predict it)")
    print(f"  {'':<10}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
    print(f"  {'spline':<10}{spline_metrics['max_mha']:>12.3f}"
          f"{spline_metrics['rms_mha']:>12.3f}{spline_metrics['mae_mha']:>12.3f}")
    print(f"  {'linear':<10}{linear_metrics['max_mha']:>12.3f}"
          f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    if spline_metrics["rms_mha"] < linear_metrics["rms_mha"]:
        factor = linear_metrics["rms_mha"] / max(spline_metrics["rms_mha"], 1e-12)
        print(f"  -> the spline is {factor:.1f}x more accurate than straight lines, "
              f"on the same nodes")
    else:
        print("  -> linear did as well or better here; with very few, widely spaced")
        print("     nodes the spline's extra freedom can work against it")
    print()

    # -- figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dense_x, dense_linear, "--", color="#999999", linewidth=1.4,
            label="linear (what the scan plots draw)")
    ax.plot(dense_x, dense_spline, "-", color="#4c5fd5", linewidth=2,
            label="cubic spline")
    ax.plot(x, y, "o", color="#1a9850", markersize=7, zorder=3,
            label=f"computed VQE nodes ({len(x)})")
    ax.plot([interp_min["x"]], [interp_min["energy"]], "r*", markersize=16, zorder=4,
            label=f"spline minimum @ {interp_min['x']:.3f} A")
    ax.axvline(EQUILIBRIUM_BOND, color="black", linestyle=":", linewidth=1,
               label=f"experiment {EQUILIBRIUM_BOND} A")
    ax.set_xlabel("F-F bond length (Angstrom)")
    ax.set_ylabel(f"Energy (Hartree)  [{args.column}]")
    ax.set_title("F$_2$ curve reconstructed by cubic spline from a coarse scan")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_err, ax_err = plt.subplots(figsize=(9, 4.5))
    interior = np.arange(len(x))[1:-1]
    width = 0.35
    ax_err.bar(interior - width / 2, np.abs(loo["linear"][1:-1]) * 1000, width,
               color="#999999", label="linear")
    ax_err.bar(interior + width / 2, np.abs(loo["spline"][1:-1]) * 1000, width,
               color="#4c5fd5", label="cubic spline")
    ax_err.set_xticks(interior)
    ax_err.set_xticklabels([f"{x[i]:.2f}" for i in interior])
    ax_err.set_xlabel("node left out (F-F bond length, Angstrom)")
    ax_err.set_ylabel("|predicted - true|  (mHa)")
    ax_err.set_title("Leave-one-out reconstruction error  (lower is better)")
    ax_err.set_yscale("log")
    ax_err.legend()
    ax_err.grid(alpha=0.3, axis="y")
    fig_err.tight_layout()

    # -- save -----------------------------------------------------------
    if not args.no_save:
        target = resolve_target(args.save, here, "f2_spline_interp")
        out_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            out_rows.append({
                "distance": float(x[i]),
                "energy": float(y[i]),
                "loo_spline": float(loo["spline"][i]),
                "loo_linear": float(loo["linear"][i]),
                "loo_spline_mha": float(loo["spline"][i] * 1000),
                "loo_linear_mha": float(loo["linear"][i] * 1000),
            })
        out_meta = {
            "molecule": metadata.get("molecule", "F2"),
            "source_scan": str(display_path(scan_path)),
            "method": "cubic_spline",
            "column": args.column,
            "n_nodes": int(len(x)),
            "node_spacing": float(np.mean(np.diff(x))),
            "spline_max_mha": spline_metrics["max_mha"],
            "spline_rms_mha": spline_metrics["rms_mha"],
            "linear_max_mha": linear_metrics["max_mha"],
            "linear_rms_mha": linear_metrics["rms_mha"],
            "interpolated_min_bond": interp_min["x"],
            "interpolated_min_energy": interp_min["energy"],
            "best_node_bond": float(x[best_node]),
            "best_node_energy": float(y[best_node]),
            "experimental_bond": EQUILIBRIUM_BOND,
        }
        json_path, csv_path = save_scan(target, out_meta, out_rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")

        dense_path = target.with_name(target.stem.replace("_interp", "") + "_dense.csv")
        with open(dense_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["distance", "spline", "linear"])
            writer.writerows(zip(dense_x, dense_spline, dense_linear))
        print(f"saved {display_path(dense_path)}")

        curve_png = here / "f2_spline_curve.png"
        error_png = here / "f2_spline_error.png"
        fig.savefig(curve_png, dpi=150, bbox_inches="tight")
        fig_err.savefig(error_png, dpi=150, bbox_inches="tight")
        print(f"saved {display_path(curve_png)}")
        print(f"saved {display_path(error_png)}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
