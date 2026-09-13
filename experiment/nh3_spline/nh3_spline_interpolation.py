"""
nh3_spline_interpolation.py -- reconstruct ammonia's potential-energy curves
and surface from a coarse VQE scan using splines, and measure how good the
reconstruction actually is.

RUNS NO CHEMISTRY. This loads a scan that
``experiment/nh3/nh3_ground_state_estimation.py`` already computed and saved.
Seconds, not minutes, and you can re-run it as often as you like without
paying for another VQE point.

WHAT IT DOES WITH EACH KIND OF SCAN
-------------------------------------
The script dispatches on the ``scan_type`` recorded in the save file.

**Curve scans** (stretch or bend) get a ``CubicSpline``: a separate cubic
between each adjacent pair of nodes, with coefficients chosen so the pieces
agree in value, slope and curvature where they meet [1]. It passes through
every computed point and stays smooth in between.

**Surface scans** get a ``RectBivariateSpline`` [2], the 2D analogue. This
one has a hard requirement: **a complete rectangular grid**. Every
(bond, angle) combination must be present. A scan that was interrupted
part-way leaves holes, and the fit simply cannot be built -- the script says
so rather than failing obscurely.

HOW THE ERROR IS MEASURED, WITHOUT SPENDING MORE VQE
-------------------------------------------------------
The honest question is how close the interpolant is to the truth at
geometries never computed. Answering it directly would mean computing more
geometries, which is the cost we are trying to avoid. So:

**Curves: leave-one-out.** Drop one interior node, rebuild from the rest,
predict the dropped node, where the true answer is known. Repeat. Endpoints
are skipped -- removing one leaves no data on that side, making it
extrapolation, a different and harder problem.

**Surfaces: half-grid holdout.** Leave-one-out is impossible on a grid.
Remove a single cell and the grid stops being rectangular, so
``RectBivariateSpline`` cannot be built at all. Instead keep every other
value on each axis, fit that subgrid, and score on everything held out.

That choice is not just a workaround -- it answers the question that
actually matters for a scan whose cost grows as N squared: **what would you
have lost by running half the grid?** A 7x7 NH3 surface at CAS(6,5) is
around half an hour. The 4x4 subgrid is about six minutes. If the holdout
error is small, the expensive scan was mostly redundant, and that is worth
knowing before the next molecule.

Both are compared against a linear (or bilinear) baseline, so the numbers
come with something to beat.

SCORE AGAINST THE MODEL, NOT AGAINST EXPERIMENT
-------------------------------------------------
There is an obvious-looking way to judge the interpolant: compare its
minimum against ammonia's experimental geometry (r = 1.0124 A,
theta = 106.67 deg [3]). **That comparison is confounded**, and this script
does not lead with it.

The scan is CAS(6,5)/STO-3G, and that model's own minimum is at
**r = 1.0500 A, theta = 102.048 deg**. The basis set is off by 0.028 A and
4.0 degrees before interpolation enters the picture. An interpolant cannot
recover accuracy its input data never contained, so scoring it against
experiment charges it for someone else's error.

Both references are printed. The **model** one is what measures
interpolation.

The same applies with more force to the inversion barrier, which the bend
scan exists to produce. STO-3G puts it near 4900 cm-1 against an
experimental ~1786 cm-1 [3] -- roughly 2.8x too high. The barrier this
script reports off the interpolated curve is a better estimate of *the
model's* barrier than the raw grid gives, and says nothing about whether
the model is right.

OUTPUT
-------
  nh3_<coordinate>_spline_interp.json / .csv   per-node or per-cell errors
                                               plus summary metrics
  nh3_<coordinate>_spline_dense.csv            the reconstructed curve,
                                               densely sampled (curves only)
  nh3_<coordinate>_spline_curve.png            nodes, spline, linear
  nh3_<coordinate>_spline_error.png            error per node (curves)
  nh3_surface_spline.png                       reconstructed surface and the
                                               holdout error map (surfaces)

REFERENCES
-----------
[1] scipy.interpolate.CubicSpline:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
[2] scipy.interpolate.RectBivariateSpline -- requires a complete rectangular grid:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html
[3] NIST CCCBDB, experimental NH3 geometry and inversion barrier:
    https://cccbdb.nist.gov/exp2x.asp?casno=7664417&charge=0

USAGE
------
  DEFAULT -- finds the first NH3 scan in experiment/nh3/
    python experiment/nh3_spline/nh3_spline_interpolation.py

  A PARTICULAR SCAN -- .json preferred, .csv accepted
    python experiment/nh3_spline/nh3_spline_interpolation.py --scan nh3_stretch_scan.json
    python experiment/nh3_spline/nh3_spline_interpolation.py --scan nh3_bend_scan.json
    python experiment/nh3_spline/nh3_spline_interpolation.py --scan nh3_surface_scan.json

  WHICH ENERGY COLUMN -- 'vqe' by default. Interpolating 'exact' separates
  coarse-sampling error from VQE optimizer noise.
    python experiment/nh3_spline/nh3_spline_interpolation.py --column exact

  OTHER KNOBS
    --dense 2000    points in the reconstructed curve
    --no-save       don't write results or figures
    --no-show       write the PNGs without opening a window
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

EXPERIMENTAL_BOND = 1.0124
EXPERIMENTAL_ANGLE = 106.67
EXPERIMENTAL_BARRIER_MHA = 8.14
# What the default CAS(6,5)/STO-3G model actually produces, from fine
# reference scans of the exact column. These -- not the experimental values
# -- are what an interpolant built on this data can be scored against.
#
#   stretch slice: theta held at the experimental 106.67 deg
MODEL_BOND_STRETCH = 1.0401     # Angstrom
MODEL_STRETCH_AT_ANGLE = 106.67
#   bend slice: r held at the experimental 1.0124 A
MODEL_ANGLE_BEND = 104.272      # degrees
MODEL_BEND_AT_BOND = 1.0124
#   the true 2D minimum, from direct Nelder-Mead on the exact energy.
#   Note it matches neither slice: coordinate descent from either one has
#   not converged after a single step.
MODEL_SURFACE_BOND = 1.0500     # Angstrom
MODEL_SURFACE_ANGLE = 102.048   # degrees
PLANAR_ANGLE = 120.0
MHA_PER_WAVENUMBER = 1000.0 / 219474.63

DEFAULT_SCAN_NAMES = [
    "nh3_stretch_scan.json", "nh3_bend_scan.json", "nh3_surface_scan.json",
    "nh3_stretch_scan.csv", "nh3_bend_scan.csv", "nh3_surface_scan.csv",
]


def check_active_space(metadata) -> None:
    """Warn when the scan's active space is not the one the MODEL_* constants
    describe.

    The MODEL_* reference geometries were measured on CAS(6,5)/STO-3G. Scoring a
    CAS(4,4) scan against them compares two different models and silently
    attributes the difference to interpolation, which is exactly the kind of
    confounded comparison this folder is careful about elsewhere.
    """
    ne = metadata.get("active_electrons")
    no = metadata.get("active_orbitals")
    if ne is None or no is None:
        return
    if (int(ne), int(no)) != (6, 5):
        print(f"  NOTE: this scan is CAS({ne},{no}), but the model reference geometry")
        print(f"  (r = {MODEL_SURFACE_BOND} A, theta = {MODEL_SURFACE_ANGLE} deg) came from CAS(6,5).")
        print(f"  Distances to it below mix an active-space difference into what is")
        print(f"  supposed to measure interpolation. The leave-one-out / holdout error")
        print(f"  numbers are unaffected -- they compare the interpolant against this")
        print(f"  scan's own data -- so read those instead.")


def curve_reference(coordinate: str, fixed_value: float | None = None):
    """The model minimum for a curve slice, and whether it actually applies.

    The stretch reference was measured with theta held at 106.67 deg and the
    bend reference with r held at 1.0124 A. Hold the other coordinate
    somewhere else and neither number is the right target any more -- the
    minimum of one coordinate moves when you move the other. Returns
    (reference, slice_value, applies).
    """
    if coordinate == "stretch":
        ref, at = MODEL_BOND_STRETCH, MODEL_STRETCH_AT_ANGLE
    else:
        ref, at = MODEL_ANGLE_BEND, MODEL_BEND_AT_BOND
    applies = fixed_value is None or abs(fixed_value - at) < 1e-3
    return ref, at, applies


def warn_if_slice_differs(coordinate: str, fixed_value: float | None) -> None:
    ref, at, applies = curve_reference(coordinate, fixed_value)
    if applies:
        return
    other = "theta" if coordinate == "stretch" else "r"
    unit = "deg" if coordinate == "stretch" else "A"
    print(f"  NOTE: the model reference {ref} for this coordinate was measured with")
    print(f"  {other} held at {at} {unit}, but this scan holds it at {fixed_value:g} {unit}.")
    print(f"  The minimum of one coordinate moves when the other moves, so the distance")
    print(f"  to that reference below is not a clean interpolation error. The")
    print(f"  leave-one-out numbers are unaffected.")


def find_scan(explicit: str | None, here: Path) -> Path:
    """Locate the saved NH3 scan to interpolate."""
    nh3_dir = here.parents[0] / "nh3"
    candidates = ([Path(explicit), here / explicit, nh3_dir / explicit] if explicit
                  else [nh3_dir / name for name in DEFAULT_SCAN_NAMES])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    looked = "\n  ".join(display_path(c) for c in candidates)
    raise SystemExit(
        f"no saved NH3 scan found. Looked in:\n  {looked}\n\n"
        f"This script interpolates an existing scan; it runs no chemistry.\n"
        f"Produce one first, e.g.:\n"
        f"  python experiment/nh3/nh3_ground_state_estimation.py --coordinate stretch"
    )


def _load(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    if path.suffix.lower() != ".csv":
        return load_scan(path)
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("bond", "angle", "hf", "vqe", "exact")
    rows = [{k: float(v) for k, v in r.items() if k in numeric and v != ""} for r in raw]
    missing = {"bond", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} lacks column(s) {sorted(missing)}.")
    bonds = {round(r["bond"], 6) for r in rows}
    angles = {round(r["angle"], 6) for r in rows}
    if len(bonds) > 1 and len(angles) > 1:
        meta = {"scan_type": "surface", "coordinate": "surface"}
    else:
        coord = "stretch" if len(bonds) > 1 else "bend"
        meta = {"scan_type": "curve", "coordinate": coord}
    meta["molecule"] = "NH3"
    return meta, rows


# ===========================================================================
#  Curves
# ===========================================================================

def leave_one_out(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Out-of-sample error at each interior node, spline and linear.

    NaN at the endpoints: dropping one of those leaves no data on that side,
    which is extrapolation rather than interpolation.
    """
    n = len(x)
    spline_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)
    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]
        if len(xk) >= 4:                        # a cubic piece needs 4 points
            spline_err[i] = float(CubicSpline(xk, yk)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]
    return {"spline": spline_err, "linear": linear_err}


def metrics(errors: np.ndarray) -> Dict[str, float]:
    """Max / RMS / mean-absolute error in milli-Hartree, ignoring NaN."""
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {"max_mha": float("nan"), "rms_mha": float("nan"), "mae_mha": float("nan")}
    return {
        "max_mha": float(np.max(np.abs(finite)) * 1000),
        "rms_mha": float(np.sqrt(np.mean(finite ** 2)) * 1000),
        "mae_mha": float(np.mean(np.abs(finite)) * 1000),
    }


def run_curve(metadata: Dict[str, Any], rows: List[Dict[str, Any]],
              args, here: Path, scan_path: Path) -> None:
    coordinate = metadata.get("coordinate", "stretch")
    key = "bond" if coordinate == "stretch" else "angle"
    unit = "A" if coordinate == "stretch" else "deg"
    fixed = (rows[0].get("angle") if coordinate == "stretch"
             else rows[0].get("bond"))
    model, _, _ = curve_reference(coordinate, fixed)
    experiment = EXPERIMENTAL_BOND if coordinate == "stretch" else EXPERIMENTAL_ANGLE

    rows = sorted(rows, key=lambda r: r[key])
    x = np.array([r[key] for r in rows], dtype=float)
    y = np.array([r[args.column] for r in rows], dtype=float)
    if len(x) < 4:
        raise SystemExit(f"{display_path(scan_path)} has only {len(x)} points; a cubic "
                         f"spline needs at least 4. Re-run with a finer step.")

    print(f"loaded {len(x)} nodes from {display_path(scan_path)}")
    print(f"  NH3 {coordinate} scan, CAS({metadata.get('active_electrons')}, "
          f"{metadata.get('active_orbitals')}), column '{args.column}'")
    print(f"  {x[0]:.4g} to {x[-1]:.4g} {unit}, spacing {np.mean(np.diff(x)):.4g} {unit}")
    check_active_space(metadata)
    print("  no chemistry is run -- this is pure interpolation\n")

    spline = CubicSpline(x, y)
    dense_x = np.linspace(x[0], x[-1], args.dense)
    dense_spline = spline(dense_x)
    dense_linear = np.interp(dense_x, x, y)

    # -- the minimum ---------------------------------------------------
    best_node = int(np.argmin(y))
    coarse = np.linspace(x[0], x[-1], 4001)
    k = int(np.argmin(spline(coarse)))
    lo, hi = coarse[max(k - 1, 0)], coarse[min(k + 1, len(coarse) - 1)]
    found = minimize_scalar(lambda t: float(spline(t)), bounds=(lo, hi), method="bounded")

    print("MINIMUM OF THE CURVE")
    print(f"  lowest computed sample : {y[best_node]:.6f} Ha at {x[best_node]:.4f} {unit}")
    print(f"  spline minimum         : {float(found.fun):.6f} Ha at {found.x:.4f} {unit}")
    node_off = abs(x[best_node] - model)
    interp_off = abs(found.x - model)
    warn_if_slice_differs(coordinate, fixed)
    print(f"\n  against the model's own minimum ({model} {unit}) -- the yardstick for")
    print(f"  interpolation, because the interpolant cannot beat its input data:")
    print(f"    best raw sample : off by {node_off:.4f} {unit}")
    print(f"    spline          : off by {interp_off:.4f} {unit}")
    if interp_off < node_off and interp_off > 0:
        print(f"    -> {node_off / interp_off:.0f}x more precise, from nodes "
              f"{np.mean(np.diff(x)):.4g} {unit} apart")
    print(f"  experiment is {experiment} {unit}; the model misses it by "
          f"{abs(model - experiment):.4f} {unit}")
    print(f"  on its own -- a basis-set error, not an interpolation one.\n")

    # -- inversion barrier, for the bend -------------------------------
    barrier_mha = float("nan")
    if coordinate == "bend" and x[-1] >= PLANAR_ANGLE - 1e-6:
        barrier_mha = (float(spline(PLANAR_ANGLE)) - float(found.fun)) * 1000
        raw_barrier = (y[-1] - y[best_node]) * 1000
        print("INVERSION BARRIER  E(planar 120 deg) - E(minimum)")
        print(f"  off the raw grid    : {raw_barrier:7.3f} mHa "
              f"({raw_barrier / MHA_PER_WAVENUMBER:6.0f} cm-1)")
        print(f"  off the spline      : {barrier_mha:7.3f} mHa "
              f"({barrier_mha / MHA_PER_WAVENUMBER:6.0f} cm-1)")
        print(f"  the raw-grid value is an UNDER-estimate: its floor is the lowest sample,")
        print(f"  which sits above the true minimum, so the gap to planar comes out short.")
        print(f"  experiment is {EXPERIMENTAL_BARRIER_MHA:.2f} mHa (~1786 cm-1); the model is")
        print(f"  ~{barrier_mha / EXPERIMENTAL_BARRIER_MHA:.1f}x too high, which is STO-3G's "
              f"fault, not the spline's.\n")

    # -- leave-one-out -------------------------------------------------
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
        print(f"  -> the spline is "
              f"{linear_metrics['rms_mha'] / max(spline_metrics['rms_mha'], 1e-12):.1f}x "
              f"more accurate than straight lines, on the same nodes")
    else:
        print("  -> linear did as well or better; with few, widely spaced nodes the")
        print("     spline's extra freedom can work against it")
    print(f"\n  compare with the global polynomial: python "
          f"experiment/nh3_polynomial/nh3_polynomial_interpolation.py "
          f"--scan {scan_path.name}\n")

    # -- figures --------------------------------------------------------
    label = "N-H bond length (Angstrom)" if coordinate == "stretch" else "H-N-H angle (degrees)"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dense_x, dense_linear, "--", color="#999999", linewidth=1.4,
            label="linear (what the scan plots draw)")
    ax.plot(dense_x, dense_spline, "-", color="#4c5fd5", linewidth=2, label="cubic spline")
    ax.plot(x, y, "o", color="#1a9850", markersize=7, zorder=3,
            label=f"computed VQE nodes ({len(x)})")
    ax.plot([found.x], [float(found.fun)], "r*", markersize=16, zorder=4,
            label=f"spline minimum @ {found.x:.3f} {unit}")
    ax.axvline(model, color="black", linestyle="-.", linewidth=1,
               label=f"model minimum {model} {unit}")
    ax.axvline(experiment, color="black", linestyle=":", linewidth=1,
               label=f"experiment {experiment} {unit}")
    ax.set_xlabel(label)
    ax.set_ylabel(f"Energy (Hartree)  [{args.column}]")
    ax.set_title(f"NH$_3$ {coordinate} curve reconstructed by cubic spline")
    ax.legend(loc="upper right", fontsize=8)
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
    ax_err.set_xticklabels([f"{x[i]:.3g}" for i in interior])
    ax_err.set_xlabel(f"node left out ({label})")
    ax_err.set_ylabel("|predicted - true|  (mHa)")
    ax_err.set_title("Leave-one-out reconstruction error  (lower is better)")
    ax_err.set_yscale("log")
    ax_err.legend()
    ax_err.grid(alpha=0.3, axis="y")
    fig_err.tight_layout()

    if not args.no_save:
        target = resolve_target(args.save, here, f"nh3_{coordinate}_spline_interp")
        out_rows = [{
            key: float(x[i]), "energy": float(y[i]),
            "loo_spline": float(loo["spline"][i]), "loo_linear": float(loo["linear"][i]),
            "loo_spline_mha": float(loo["spline"][i] * 1000),
            "loo_linear_mha": float(loo["linear"][i] * 1000),
        } for i in range(len(x))]
        out_meta = {
            "molecule": "NH3", "source_scan": str(display_path(scan_path)),
            "method": "cubic_spline", "scan_type": "curve", "coordinate": coordinate,
            "column": args.column, "n_nodes": int(len(x)),
            "node_spacing": float(np.mean(np.diff(x))),
            "spline_max_mha": spline_metrics["max_mha"],
            "spline_rms_mha": spline_metrics["rms_mha"],
            "linear_max_mha": linear_metrics["max_mha"],
            "linear_rms_mha": linear_metrics["rms_mha"],
            "interpolated_min": float(found.x),
            "interpolated_min_energy": float(found.fun),
            "best_node": float(x[best_node]), "best_node_energy": float(y[best_node]),
            "model_reference": model, "experimental_reference": experiment,
            "interp_error_vs_model": float(interp_off),
            "best_node_error_vs_model": float(node_off),
            "inversion_barrier_mha": barrier_mha,
        }
        json_path, csv_path = save_scan(target, out_meta, out_rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")

        dense_path = target.with_name(target.stem.replace("_interp", "") + "_dense.csv")
        with open(dense_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([key, "spline", "linear"])
            writer.writerows(zip(dense_x, dense_spline, dense_linear))
        print(f"saved {display_path(dense_path)}")

        for figure, name in ((fig, f"nh3_{coordinate}_spline_curve.png"),
                             (fig_err, f"nh3_{coordinate}_spline_error.png")):
            path = here / name
            figure.savefig(path, dpi=150, bbox_inches="tight")
            print(f"saved {display_path(path)}")

    if not args.no_show:
        plt.show()


# ===========================================================================
#  Surface
# ===========================================================================

def build_grid(rows: List[Dict[str, Any]], column: str
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows -> (bond_values, angle_values, energies[angle, bond]).

    RectBivariateSpline needs a complete rectangle [2], so any missing cell
    is fatal and is reported as such rather than left to fail cryptically
    inside scipy.
    """
    bonds = np.array(sorted({round(r["bond"], 6) for r in rows}), dtype=float)
    angles = np.array(sorted({round(r["angle"], 6) for r in rows}), dtype=float)
    grid = np.full((len(angles), len(bonds)), np.nan)
    for r in rows:
        i = int(np.argmin(np.abs(angles - r["angle"])))
        j = int(np.argmin(np.abs(bonds - r["bond"])))
        grid[i, j] = r[column]
    if not np.isfinite(grid).all():
        missing = int((~np.isfinite(grid)).sum())
        raise SystemExit(
            f"the surface grid has {missing} missing cell(s) out of {grid.size}. "
            f"RectBivariateSpline requires a complete rectangular grid, so this scan "
            f"cannot be interpolated in 2D. Re-run the surface scan to completion."
        )
    return bonds, angles, grid


def half_grid_holdout(bonds: np.ndarray, angles: np.ndarray, grid: np.ndarray
                      ) -> Dict[str, Any]:
    """Fit on every other node in each axis, score on everything held out.

    Leave-one-out cannot work here: removing one cell destroys the rectangle
    that RectBivariateSpline requires. This also answers the practical
    question -- what would half the grid have cost you?
    """
    bi = np.arange(0, len(bonds), 2)
    ai = np.arange(0, len(angles), 2)
    if len(bi) < 4 or len(ai) < 4:
        raise SystemExit(
            f"the half-grid subsample is {len(ai)}x{len(bi)}, but a bicubic spline needs "
            f"at least 4 nodes per axis. Your grid is {len(angles)}x{len(bonds)}; it needs "
            f"to be at least 7x7 for this cross-validation to be possible."
        )
    sub_bonds, sub_angles = bonds[bi], angles[ai]
    sub_grid = grid[np.ix_(ai, bi)]

    spline = RectBivariateSpline(sub_angles, sub_bonds, sub_grid, kx=3, ky=3)

    held = np.ones(grid.shape, dtype=bool)
    held[np.ix_(ai, bi)] = False
    errors = np.full(grid.shape, np.nan)
    for i in range(len(angles)):
        for j in range(len(bonds)):
            if held[i, j]:
                errors[i, j] = float(spline(angles[i], bonds[j])[0, 0]) - grid[i, j]

    # Bilinear baseline on the same subgrid.
    lin_err = np.full(grid.shape, np.nan)
    for i in range(len(angles)):
        for j in range(len(bonds)):
            if held[i, j]:
                lin_err[i, j] = _bilinear(sub_angles, sub_bonds, sub_grid,
                                          angles[i], bonds[j]) - grid[i, j]

    return {
        "spline_errors": errors, "linear_errors": lin_err,
        "held": held, "n_fit": int(sub_grid.size), "n_held": int(held.sum()),
        "sub_shape": (len(ai), len(bi)),
    }


def _bilinear(ys: np.ndarray, xs: np.ndarray, values: np.ndarray,
              y: float, x: float) -> float:
    """Bilinear interpolation on a rectangular grid, clamped at the edges."""
    j = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    i = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    tx = (x - xs[j]) / (xs[j + 1] - xs[j])
    ty = (y - ys[i]) / (ys[i + 1] - ys[i])
    tx, ty = float(np.clip(tx, 0, 1)), float(np.clip(ty, 0, 1))
    return float(
        values[i, j] * (1 - tx) * (1 - ty) + values[i, j + 1] * tx * (1 - ty)
        + values[i + 1, j] * (1 - tx) * ty + values[i + 1, j + 1] * tx * ty
    )


def run_surface(metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                args, here: Path, scan_path: Path) -> None:
    bonds, angles, grid = build_grid(rows, args.column)
    print(f"loaded a {len(angles)} x {len(bonds)} grid "
          f"({grid.size} points) from {display_path(scan_path)}")
    print(f"  NH3 surface, CAS({metadata.get('active_electrons')}, "
          f"{metadata.get('active_orbitals')}), column '{args.column}'")
    print(f"  r {bonds[0]:.3f}-{bonds[-1]:.3f} A, theta {angles[0]:.1f}-{angles[-1]:.1f} deg")
    check_active_space(metadata)
    print("  no chemistry is run -- this is pure interpolation\n")

    spline = RectBivariateSpline(angles, bonds, grid, kx=3, ky=3)

    # -- the minimum of the surface -------------------------------------
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    guess = np.array([angles[i], bonds[j]])
    found = minimize(lambda p: float(spline(p[0], p[1])[0, 0]), guess, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-12})
    min_angle, min_bond = float(found.x[0]), float(found.x[1])

    print("MINIMUM OF THE SURFACE")
    print(f"  lowest sampled point : {grid[i, j]:.6f} Ha at "
          f"r = {bonds[j]:.4f} A, theta = {angles[i]:.2f} deg")
    print(f"  spline minimum       : {float(found.fun):.6f} Ha at "
          f"r = {min_bond:.4f} A, theta = {min_angle:.2f} deg")
    print(f"\n  against the model's true 2D minimum (r = {MODEL_SURFACE_BOND} A, "
          f"theta = {MODEL_SURFACE_ANGLE} deg):")
    print(f"    best raw sample : {abs(bonds[j] - MODEL_SURFACE_BOND):.4f} A, "
          f"{abs(angles[i] - MODEL_SURFACE_ANGLE):.2f} deg")
    print(f"    spline          : {abs(min_bond - MODEL_SURFACE_BOND):.4f} A, "
          f"{abs(min_angle - MODEL_SURFACE_ANGLE):.2f} deg")
    print(f"  experiment is r = {EXPERIMENTAL_BOND} A, theta = {EXPERIMENTAL_ANGLE} deg;")
    print(f"  the model misses that on its own, which is the basis set, not the spline.\n")

    # -- half-grid holdout ----------------------------------------------
    hold = half_grid_holdout(bonds, angles, grid)
    spline_metrics = metrics(hold["spline_errors"])
    linear_metrics = metrics(hold["linear_errors"])
    print(f"HALF-GRID HOLDOUT  (fit on {hold['sub_shape'][0]}x{hold['sub_shape'][1]} "
          f"= {hold['n_fit']} nodes, score on the {hold['n_held']} held out)")
    print(f"  {'':<10}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
    print(f"  {'bicubic':<10}{spline_metrics['max_mha']:>12.3f}"
          f"{spline_metrics['rms_mha']:>12.3f}{spline_metrics['mae_mha']:>12.3f}")
    print(f"  {'bilinear':<10}{linear_metrics['max_mha']:>12.3f}"
          f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    print(f"\n  read this as a cost question: the {hold['n_fit']}-point subgrid is roughly")
    print(f"  {hold['n_fit'] / grid.size * 100:.0f}% of the scan's runtime. If the RMS above")
    print(f"  is small next to the features you care about, the other "
          f"{hold['n_held']} points were")
    print(f"  largely redundant and the next molecule's grid can be coarser.\n")

    # -- figures ---------------------------------------------------------
    dense_b = np.linspace(bonds[0], bonds[-1], 160)
    dense_a = np.linspace(angles[0], angles[-1], 160)
    dense = spline(dense_a, dense_b)

    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    levels = 24
    cf = ax1.contourf(dense_b, dense_a, dense, levels=levels, cmap="viridis")
    ax1.contour(dense_b, dense_a, dense, levels=levels, colors="white",
                linewidths=0.4, alpha=0.5)
    ax1.plot(bonds[np.tile(np.arange(len(bonds)), len(angles))],
             np.repeat(angles, len(bonds)), "k.", markersize=2, alpha=0.5,
             label="computed nodes")
    ax1.plot([min_bond], [min_angle], "r*", markersize=16,
             label=f"spline min\n{min_bond:.3f} A, {min_angle:.1f} deg")
    fig.colorbar(cf, ax=ax1, label="Energy (Hartree)")
    ax1.set_xlabel("r(N-H) (A)")
    ax1.set_ylabel("theta (degrees)")
    ax1.set_title("NH$_3$ surface, bicubic spline reconstruction")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    err_map = np.abs(hold["spline_errors"]) * 1000
    im = ax2.pcolormesh(bonds, angles, np.ma.masked_invalid(err_map),
                        cmap="magma", shading="nearest")
    fig.colorbar(im, ax=ax2, label="|holdout error| (mHa)")
    ax2.set_xlabel("r(N-H) (A)")
    ax2.set_ylabel("theta (degrees)")
    ax2.set_title(f"Half-grid holdout error\n(fit on {hold['n_fit']}, "
                  f"scored on {hold['n_held']})")
    fig.tight_layout()

    if not args.no_save:
        target = resolve_target(args.save, here, "nh3_surface_spline_interp")
        out_rows = []
        for a in range(len(angles)):
            for b in range(len(bonds)):
                out_rows.append({
                    "bond": float(bonds[b]), "angle": float(angles[a]),
                    "energy": float(grid[a, b]),
                    "held_out": bool(hold["held"][a, b]),
                    "holdout_spline_mha": float(hold["spline_errors"][a, b] * 1000),
                    "holdout_linear_mha": float(hold["linear_errors"][a, b] * 1000),
                })
        out_meta = {
            "molecule": "NH3", "source_scan": str(display_path(scan_path)),
            "method": "bicubic_spline", "scan_type": "surface", "coordinate": "surface",
            "column": args.column,
            "grid_shape": [int(len(angles)), int(len(bonds))],
            "subgrid_shape": [int(hold["sub_shape"][0]), int(hold["sub_shape"][1])],
            "n_fit": hold["n_fit"], "n_held": hold["n_held"],
            "spline_max_mha": spline_metrics["max_mha"],
            "spline_rms_mha": spline_metrics["rms_mha"],
            "linear_max_mha": linear_metrics["max_mha"],
            "linear_rms_mha": linear_metrics["rms_mha"],
            "interpolated_min_bond": min_bond, "interpolated_min_angle": min_angle,
            "interpolated_min_energy": float(found.fun),
            "best_node_bond": float(bonds[j]), "best_node_angle": float(angles[i]),
            "model_surface_bond": MODEL_SURFACE_BOND,
            "model_surface_angle": MODEL_SURFACE_ANGLE,
            "experimental_bond": EXPERIMENTAL_BOND, "experimental_angle": EXPERIMENTAL_ANGLE,
        }
        json_path, csv_path = save_scan(target, out_meta, out_rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        png = here / "nh3_surface_spline.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"saved {display_path(png)}")

    if not args.no_show:
        plt.show()


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct NH3's curves or surface from a coarse VQE scan using "
                    "splines, and measure the error by leave-one-out (curves) or half-grid "
                    "holdout (surface). Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE",
                        help="the saved NH3 scan to interpolate (default: the first one "
                             "found in experiment/nh3/)")
    parser.add_argument("--column", choices=["vqe", "exact", "hf"], default="vqe",
                        help="which energy column to interpolate (default vqe)")
    parser.add_argument("--dense", type=int, default=800,
                        help="points in the reconstructed dense curve (default 800)")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save the results")
    parser.add_argument("--no-save", action="store_true", help="don't save results or figures")
    parser.add_argument("--no-show", action="store_true",
                        help="write the figures but don't open a window")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.no_show:
        matplotlib.use("Agg")

    scan_path = find_scan(args.scan, here)
    metadata, rows = _load(scan_path)
    if metadata.get("scan_type", "curve") == "surface":
        run_surface(metadata, rows, args, here, scan_path)
    else:
        run_curve(metadata, rows, args, here, scan_path)


if __name__ == "__main__":
    main()