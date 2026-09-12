"""
h2o2_spline_interpolation.py -- reconstruct H2O2's energy curves and its
dihedral x angle energy surface from saved VQE scans using splines, and
measure how good the reconstruction is.

RUNS NO CHEMISTRY. This loads scans that
``experiment/h2o2/h2o2_ground_state_estimation.py`` already computed and
saved. It takes seconds and can be re-run freely.

By default it processes every H2O2 scan it can find -- stretch, bend, twist
and surface -- choosing a 1D or 2D method per file from the ``scan_type``
recorded in its metadata.

WHY THIS FOLDER EXISTS
------------------------
Every scan plot in this project joins neighbouring points with straight lines.
That is the crudest reconstruction of a smooth function: the minimum it
implies is always exactly one of your samples, and its second derivative is
zero everywhere except at the kinks, where it is undefined.

For the **twist** curve specifically, that matters more than it does for a
simple single well: the whole point of scanning the torsion is the cis and
trans barrier *heights*, and neither 0 deg nor 180 deg is guaranteed to land
exactly on a grid point. A spline evaluated at exactly 0 and 180 gives a far
better barrier estimate than reading off whichever raw sample happens to sit
closest -- see :func:`_report_barriers`.

For the **surface**, the stakes are the usual N^2-cost ones: the default grid
already costs dozens of VQE points, so knowing what a half-density grid would
have cost is a real, practical question, not an academic one.

WHAT A SPLINE IS, IN ONE AND TWO DIMENSIONS
---------------------------------------------
In 1D: instead of one polynomial through all the points, fit a separate cubic
between each adjacent pair and match value, slope and curvature where the
pieces meet [1]. The result interpolates every node exactly and stays smooth,
without the oscillation a single high-degree polynomial suffers -- see the
sibling ``../h2o2_polynomial/`` for that failure in action.

In 2D: the same idea on a rectangular grid, cubic in each direction [2]. This
needs a *complete* grid -- every (dihedral, angle) cell filled -- which is
why the script refuses a partially finished surface scan rather than quietly
filling holes.

HOW THE ERROR IS MEASURED, WITHOUT MORE VQE
---------------------------------------------
Two different schemes, because 1D and 2D admit different honest tests.

**1D -- leave-one-out.** Drop one interior node, rebuild the spline from the
rest, evaluate it at the node you removed, where the true value is known.
Repeat for every interior node. Endpoints are skipped: dropping one leaves no
data on that side, making it extrapolation rather than interpolation.

**2D -- half-grid holdout.** You cannot leave one point out of a rectangular
grid and still have a rectangular grid, so leave-one-out does not apply.
Instead keep every *other* dihedral value and every *other* angle value.
That subgrid is still rectangular, so a spline fits it, and every point not
in the subgrid is a genuine held-out test point with a known answer. It also
happens to answer the question that matters for an N^2-cost scan: what would
running half the grid have cost?

Linear interpolation gets the same treatment in both cases, so every number
comes with a baseline to beat.

OUTPUT (per scan processed)
-----------------------------
  h2o2_<coord>_spline_interp.json / .csv   per-node or per-point errors, with
                                           summary metrics in the metadata
  h2o2_<coord>_spline_dense.csv            the reconstructed curve/surface
  h2o2_<coord>_spline_curve.png            1D: nodes, spline, linear
  h2o2_<coord>_spline_error.png            1D: per-node error bars
  h2o2_surface_spline_surface.png          2D: reconstructed surface + contours
  h2o2_surface_spline_error.png            2D: held-out error heat map

REFERENCES
-----------
[1] scipy.interpolate.CubicSpline:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
[2] scipy.interpolate.RectBivariateSpline -- bivariate spline on a rectangular
    grid, which is why a complete grid is required:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html
[3] scipy.optimize.minimize_scalar:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

USAGE
------
Every case, explicitly. Runs no chemistry -- it reads scans that
experiment/h2o2/ already saved.

  ALL SCANS AT ONCE -- stretch, bend, twist and surface, whichever exist
    python experiment/h2o2_spline/h2o2_spline_interpolation.py

  ONE SCAN AT A TIME -- .json preferred, .csv accepted
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --scan h2o2_stretch_scan.json
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --scan h2o2_bend_scan.json
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --scan h2o2_twist_scan.json
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --scan h2o2_surface_scan.json

  WHICH ENERGY COLUMN -- 'vqe' by default. Interpolating 'exact' instead
  separates coarse-sampling error from VQE optimizer noise.
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --column vqe
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --column exact
    python experiment/h2o2_spline/h2o2_spline_interpolation.py --column hf

  OTHER KNOBS
    --dense 400        resolution of the reconstruction per axis
    --no-save          don't write results or figures
    --no-show          write the PNGs without opening a window
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
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

# H2O2's gas-phase equilibrium geometry -- see experiment/h2o2/'s own docstring
# for the literature citations.
EQUILIBRIUM_OO = 1.458        # Angstrom
EQUILIBRIUM_ANGLE = 101.9     # degrees
EQUILIBRIUM_DIHEDRAL = 111.5  # degrees, the chiral skew minimum

# Literature torsional barriers, for the cis/trans check on the twist curve.
LIT_CIS_BARRIER_KCAL = 7.32
LIT_TRANS_BARRIER_KCAL = 1.10
HARTREE_TO_KCAL = 627.5094740631

COORD_KEY = {"stretch": "oo", "bend": "angle", "twist": "dihedral"}
COORD_UNIT = {"stretch": "A", "bend": "deg", "twist": "deg"}
COORD_LABEL = {
    "stretch": "O-O bond length (Angstrom)",
    "bend": "H-O-O angle (degrees)",
    "twist": "H-O-O-H dihedral (degrees)",
}
COORD_REFERENCE = {"stretch": EQUILIBRIUM_OO, "bend": EQUILIBRIUM_ANGLE, "twist": EQUILIBRIUM_DIHEDRAL}

# Both suffixes: the .json is canonical (it carries the metadata), the .csv is
# accepted as a fallback with that metadata inferred from its columns.
CANDIDATE_SCANS = ["h2o2_stretch_scan.json", "h2o2_bend_scan.json", "h2o2_twist_scan.json",
                   "h2o2_surface_scan.json", "h2o2_stretch_scan.csv", "h2o2_bend_scan.csv",
                   "h2o2_twist_scan.csv", "h2o2_surface_scan.csv"]


# ===========================================================================
#  Locating the saved scans
# ===========================================================================

def find_scans(explicit: str | None, here: Path) -> List[Path]:
    """Which saved H2O2 scans to process.

    With no ``--scan``, every standard scan that exists gets processed, so
    one command covers the whole molecule.
    """
    h2o2_dir = here.parents[0] / "h2o2"
    if explicit is not None:
        for candidate in (Path(explicit), here / explicit, h2o2_dir / explicit):
            if candidate.exists():
                return [candidate]
        raise SystemExit(
            f"no scan found at {explicit}. Looked next to this script and in "
            f"{display_path(h2o2_dir)}."
        )

    found = []
    for name in CANDIDATE_SCANS:
        candidate = h2o2_dir / name
        if candidate.suffix == ".csv" and candidate.with_suffix(".json") in found:
            continue
        if candidate.exists():
            found.append(candidate)
    if not found:
        looked = "\n  ".join(display_path(h2o2_dir / n) for n in CANDIDATE_SCANS)
        raise SystemExit(
            f"no saved H2O2 scans found. Looked for:\n  {looked}\n\n"
            f"This script interpolates existing scans; it runs no chemistry.\n"
            f"Produce them first with:\n"
            f"  python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate stretch\n"
            f"  python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate bend\n"
            f"  python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate twist\n"
            f"  python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate surface"
        )
    return found


def _rows_from_csv(path: Path) -> tuple:
    """Read an H2O2 scan from its ``.csv``, inferring what the JSON records.

    Surface vs curve, and which coordinate moves, follow from whether angle
    and dihedral actually vary -- ``oh`` is never scanned by the ground-state
    script, so it plays no part in this inference.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("oo", "oh", "angle", "dihedral", "hf", "vqe", "exact")
    rows = [{k: float(v) for k, v in r.items() if k in numeric and v != ""} for r in raw]
    missing = {"oo", "oh", "angle", "dihedral", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} lacks column(s) {sorted(missing)} -- "
                         f"not an H2O2 scan CSV.")
    oos = sorted({r["oo"] for r in rows})
    angles = sorted({r["angle"] for r in rows})
    dihedrals = sorted({r["dihedral"] for r in rows})
    if len(angles) > 1 and len(dihedrals) > 1:
        meta = {"molecule": "H2O2", "scan_type": "surface", "coordinate": "surface",
                "dihedral_values": dihedrals, "angle_values": angles}
    elif len(oos) > 1:
        meta = {"molecule": "H2O2", "scan_type": "curve", "coordinate": "stretch"}
    elif len(angles) > 1:
        meta = {"molecule": "H2O2", "scan_type": "curve", "coordinate": "bend"}
    else:
        meta = {"molecule": "H2O2", "scan_type": "curve", "coordinate": "twist"}
    return meta, rows


def _load(path: Path) -> tuple:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    if path.suffix.lower() == ".csv":
        return _rows_from_csv(path)
    return load_scan(path)


def metrics(errors: np.ndarray) -> Dict[str, float]:
    """Max / RMS / mean-absolute error in milli-Hartree, ignoring NaN slots."""
    finite = np.asarray(errors)[np.isfinite(errors)]
    if finite.size == 0:
        return {"max_mha": float("nan"), "rms_mha": float("nan"), "mae_mha": float("nan")}
    return {
        "max_mha": float(np.max(np.abs(finite)) * 1000),
        "rms_mha": float(np.sqrt(np.mean(finite ** 2)) * 1000),
        "mae_mha": float(np.mean(np.abs(finite)) * 1000),
    }


def _print_metrics(label_a: str, a: Dict[str, float], label_b: str, b: Dict[str, float]) -> None:
    print(f"  {'':<10}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
    print(f"  {label_a:<10}{a['max_mha']:>12.3f}{a['rms_mha']:>12.3f}{a['mae_mha']:>12.3f}")
    print(f"  {label_b:<10}{b['max_mha']:>12.3f}{b['rms_mha']:>12.3f}{b['mae_mha']:>12.3f}")
    if a["rms_mha"] < b["rms_mha"]:
        factor = b["rms_mha"] / max(a["rms_mha"], 1e-12)
        print(f"  -> {label_a} is {factor:.1f}x more accurate than {label_b}, on the same data")
    else:
        print(f"  -> {label_b} did as well or better here; with very few, widely spaced")
        print(f"     points the spline's extra freedom can work against it")


def _report_barriers(evaluate, minimum_energy: float) -> None:
    """Cis/trans torsional barriers read off the smooth reconstruction at
    exactly 0 and 180 deg, rather than off whichever raw sample happens to be
    closest -- only meaningful for the twist curve. See the module docstring
    on the H2O2 ground-state script for why the trans barrier in particular
    is small enough that this precision matters."""
    cis_e = float(evaluate(0.0))
    trans_e = float(evaluate(180.0))
    cis_mha = (cis_e - minimum_energy) * 1000
    trans_mha = (trans_e - minimum_energy) * 1000
    print(f"  torsional barriers from the reconstruction (relative to its own minimum):")
    print(f"    cis   (0 deg):   {cis_mha:7.3f} mHa = {cis_mha / 1.5936:5.2f} kcal/mol   "
          f"(literature ~{LIT_CIS_BARRIER_KCAL} kcal/mol)")
    print(f"    trans (180 deg): {trans_mha:7.3f} mHa = {trans_mha / 1.5936:5.2f} kcal/mol   "
          f"(literature ~{LIT_TRANS_BARRIER_KCAL} kcal/mol)")


# ===========================================================================
#  1D: the stretch, bend and twist curves
# ===========================================================================

def leave_one_out_1d(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Out-of-sample error at each interior node, spline and linear."""
    n = len(x)
    spline_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)
    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]
        if len(xk) >= 4:        # a cubic piece needs 4 points
            spline_err[i] = float(CubicSpline(xk, yk)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]
    return {"spline": spline_err, "linear": linear_err}


def process_curve(scan_path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                  column: str, dense_n: int, here: Path, save: bool) -> None:
    """Spline-reconstruct a 1D stretch, bend or twist scan."""
    coordinate = metadata.get("coordinate", "stretch")
    key = COORD_KEY[coordinate]
    unit = COORD_UNIT[coordinate]
    axis_label = COORD_LABEL[coordinate]
    reference = COORD_REFERENCE[coordinate]

    rows = sorted(rows, key=lambda r: r[key])
    x = np.array([r[key] for r in rows], dtype=float)
    y = np.array([r[column] for r in rows], dtype=float)

    if len(x) < 4:
        print(f"  skipping: only {len(x)} points, a cubic spline needs at least 4\n")
        return

    print(f"  {len(x)} nodes, {x[0]:.2f} to {x[-1]:.2f} {unit}, "
          f"spacing {np.mean(np.diff(x)):.3f} {unit}")

    spline = CubicSpline(x, y)
    dense_x = np.linspace(x[0], x[-1], dense_n)
    dense_spline = spline(dense_x)
    dense_linear = np.interp(dense_x, x, y)

    # -- minimum: interpolated vs best raw sample ----------------------
    best_node = int(np.argmin(y))
    dense_k = int(np.argmin(dense_spline))
    bracket = (dense_x[max(dense_k - 1, 0)], dense_x[min(dense_k + 1, dense_n - 1)])
    refined = minimize_scalar(lambda t: float(spline(t)), bounds=bracket, method="bounded")

    print(f"  minimum: best sample {y[best_node]:.6f} Ha at {x[best_node]:.4f} {unit}; "
          f"spline {float(refined.fun):.6f} Ha at {float(refined.x):.4f} {unit}")
    print(f"           reference is {reference} {unit}")

    if coordinate == "twist" and x[0] <= 5.0 and x[-1] >= 175.0:
        _report_barriers(spline, float(refined.fun))

    # -- leave-one-out --------------------------------------------------
    loo = leave_one_out_1d(x, y)
    spline_metrics = metrics(loo["spline"])
    linear_metrics = metrics(loo["linear"])
    print("  leave-one-out error (drop a node, rebuild, predict it):")
    _print_metrics("spline", spline_metrics, "linear", linear_metrics)

    # -- figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dense_x, dense_linear, "--", color="#999999", linewidth=1.4,
            label="linear (what the scan plots draw)")
    ax.plot(dense_x, dense_spline, "-", color="#4c5fd5", linewidth=2, label="cubic spline")
    ax.plot(x, y, "o", color="#1a9850", markersize=7, zorder=3,
            label=f"computed VQE nodes ({len(x)})")
    ax.plot([float(refined.x)], [float(refined.fun)], "r*", markersize=16, zorder=4,
            label=f"spline minimum @ {float(refined.x):.3f} {unit}")
    ax.axvline(reference, color="black", linestyle=":", linewidth=1,
               label=f"reference {reference} {unit}")
    ax.set_xlabel(axis_label)
    ax.set_ylabel(f"Energy (Hartree)  [{column}]")
    ax.set_title(f"H$_2$O$_2$ {coordinate} curve, reconstructed by cubic spline")
    ax.legend(loc="best", fontsize=9)
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
    ax_err.set_xticklabels([f"{x[i]:.2f}" for i in interior], rotation=45, ha="right")
    ax_err.set_xlabel(f"node left out ({axis_label})")
    ax_err.set_ylabel("|predicted - true|  (mHa)")
    ax_err.set_title(f"H$_2$O$_2$ {coordinate}: leave-one-out error (lower is better)")
    ax_err.set_yscale("log")
    ax_err.legend()
    ax_err.grid(alpha=0.3, axis="y")
    fig_err.tight_layout()

    if not save:
        return

    target = resolve_target(None, here, f"h2o2_{coordinate}_spline_interp")
    out_rows = [{
        key: float(x[i]),
        "energy": float(y[i]),
        "loo_spline": float(loo["spline"][i]),
        "loo_linear": float(loo["linear"][i]),
        "loo_spline_mha": float(loo["spline"][i] * 1000),
        "loo_linear_mha": float(loo["linear"][i] * 1000),
    } for i in range(len(x))]
    out_meta = {
        "molecule": "H2O2", "source_scan": str(display_path(scan_path)),
        "method": "cubic_spline", "scan_type": "curve", "coordinate": coordinate,
        "column": column, "unit": unit,
        "n_nodes": int(len(x)), "node_spacing": float(np.mean(np.diff(x))),
        "spline_max_mha": spline_metrics["max_mha"],
        "spline_rms_mha": spline_metrics["rms_mha"],
        "linear_max_mha": linear_metrics["max_mha"],
        "linear_rms_mha": linear_metrics["rms_mha"],
        "interpolated_min_x": float(refined.x),
        "interpolated_min_energy": float(refined.fun),
        "best_node_x": float(x[best_node]), "best_node_energy": float(y[best_node]),
        "reference_x": float(reference),
    }
    json_path, csv_path = save_scan(target, out_meta, out_rows)
    print(f"  saved {display_path(json_path)}")
    print(f"  saved {display_path(csv_path)}")

    stem = target.stem.replace("_interp", "")
    dense_path = target.with_name(stem + "_dense.csv")
    with open(dense_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key, "spline", "linear"])
        writer.writerows(zip(dense_x, dense_spline, dense_linear))
    print(f"  saved {display_path(dense_path)}")

    for figure, name in ((fig, f"{stem}_curve.png"), (fig_err, f"{stem}_error.png")):
        path = here / name
        figure.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {display_path(path)}")


# ===========================================================================
#  2D: the dihedral x angle surface
# ===========================================================================

def rebuild_grid(metadata: Dict[str, Any],
                 rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassemble the surface scan into (dihedral_values, angle_values, Z).

    Z is indexed ``[dihedral, angle]``, which is the layout RectBivariateSpline
    wants. Rows are matched to cells by *nearest* value rather than equality,
    because the coordinates have been through a float round-trip via JSON, and
    the scan wrote its rows in serpentine order so the row list is not sorted.
    """
    dihedral_values = np.asarray(metadata["dihedral_values"], dtype=float)
    angle_values = np.asarray(metadata["angle_values"], dtype=float)
    Z = np.full((len(dihedral_values), len(angle_values)), np.nan)
    for row in rows:
        i = int(np.argmin(np.abs(dihedral_values - row["dihedral"])))
        j = int(np.argmin(np.abs(angle_values - row["angle"])))
        Z[i, j] = row["energy_value"]
    return dihedral_values, angle_values, Z


def half_grid_holdout(dihedral: np.ndarray, angle: np.ndarray,
                      Z: np.ndarray) -> Dict[str, Any]:
    """Fit on every other row and column; score on everything held out.

    Leave-one-out does not work on a grid interpolant -- remove a single cell
    and the grid is no longer rectangular, so RectBivariateSpline cannot be
    built at all. Taking every other value on each axis keeps the subgrid
    rectangular while holding out roughly three quarters of the points, and it
    answers the question that actually matters for an N^2-cost scan: what
    would we have lost by running half the grid?
    """
    di = np.arange(0, len(dihedral), 2)
    aj = np.arange(0, len(angle), 2)
    coarse_dihedral, coarse_angle = dihedral[di], angle[aj]
    coarse_Z = Z[np.ix_(di, aj)]

    kx = min(3, len(coarse_dihedral) - 1)
    ky = min(3, len(coarse_angle) - 1)
    if kx < 1 or ky < 1:
        return {"ok": False, "reason": "the half-grid is too small to fit a spline"}

    spline = RectBivariateSpline(coarse_dihedral, coarse_angle, coarse_Z, kx=kx, ky=ky)

    held_out = np.ones_like(Z, dtype=bool)
    held_out[np.ix_(di, aj)] = False

    spline_err = np.full_like(Z, np.nan)
    linear_err = np.full_like(Z, np.nan)
    for i in range(len(dihedral)):
        for j in range(len(angle)):
            if not held_out[i, j] or not np.isfinite(Z[i, j]):
                continue
            spline_err[i, j] = float(spline(dihedral[i], angle[j])[0, 0]) - Z[i, j]
            linear_err[i, j] = _bilinear(coarse_dihedral, coarse_angle, coarse_Z,
                                         dihedral[i], angle[j]) - Z[i, j]

    return {
        "ok": True, "kx": kx, "ky": ky,
        "n_fit": int(coarse_Z.size), "n_held_out": int(np.isfinite(spline_err).sum()),
        "spline_err": spline_err, "linear_err": linear_err,
        "coarse_dihedral": coarse_dihedral, "coarse_angle": coarse_angle,
    }


def _bilinear(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation on a rectangular grid -- the 2D straight-line
    baseline, i.e. what you get by joining grid neighbours with flat facets.
    """
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[j], ys[j + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    return float(
        Z[i, j] * (1 - tx) * (1 - ty) + Z[i + 1, j] * tx * (1 - ty)
        + Z[i, j + 1] * (1 - tx) * ty + Z[i + 1, j + 1] * tx * ty
    )


def process_surface(scan_path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                    column: str, dense_n: int, here: Path, save: bool) -> None:
    """Spline-reconstruct the dihedral x angle surface."""
    for row in rows:
        row["energy_value"] = row[column]
    dihedral, angle, Z = rebuild_grid(metadata, rows)

    missing = int(np.isnan(Z).sum())
    if missing:
        print(f"  skipping: the grid has {missing} of {Z.size} cells unfilled.")
        print("  A bivariate spline needs a complete rectangular grid [2]. Re-run the")
        print("  surface scan to completion, or interpolate a finished one.\n")
        return
    if len(dihedral) < 4 or len(angle) < 4:
        print(f"  skipping: grid is {len(dihedral)}x{len(angle)}; need at least 4x4 for a "
              f"bicubic spline\n")
        return

    print(f"  grid {len(dihedral)} dihedral x {len(angle)} angle = {Z.size} points")
    print(f"  dihedral {dihedral[0]:.1f} to {dihedral[-1]:.1f} deg, angle {angle[0]:.1f} to "
          f"{angle[-1]:.1f} deg")

    spline = RectBivariateSpline(dihedral, angle, Z)
    dense_dihedral = np.linspace(dihedral[0], dihedral[-1], dense_n)
    dense_angle = np.linspace(angle[0], angle[-1], dense_n)
    dense_Z = spline(dense_dihedral, dense_angle)

    # -- minimum: interpolated vs best raw grid point ------------------
    flat = int(np.argmin(Z))
    di, aj = np.unravel_index(flat, Z.shape)
    dflat = int(np.argmin(dense_Z))
    ddi, daj = np.unravel_index(dflat, dense_Z.shape)
    print(f"  minimum: best grid point {Z[di, aj]:.6f} Ha at "
          f"dihedral={dihedral[di]:.1f} deg, angle={angle[aj]:.1f} deg")
    print(f"           spline surface  {dense_Z[ddi, daj]:.6f} Ha at "
          f"dihedral={dense_dihedral[ddi]:.1f} deg, angle={dense_angle[daj]:.1f} deg")
    print(f"           reference is dihedral={EQUILIBRIUM_DIHEDRAL} deg, "
          f"angle={EQUILIBRIUM_ANGLE} deg")

    # -- half-grid holdout ----------------------------------------------
    holdout = half_grid_holdout(dihedral, angle, Z)
    if not holdout["ok"]:
        print(f"  holdout test skipped: {holdout['reason']}")
        spline_metrics = linear_metrics = metrics(np.array([np.nan]))
    else:
        spline_metrics = metrics(holdout["spline_err"])
        linear_metrics = metrics(holdout["linear_err"])
        print(f"  half-grid holdout: fit on {holdout['n_fit']} points "
              f"({len(holdout['coarse_dihedral'])}x{len(holdout['coarse_angle'])}), "
              f"scored on {holdout['n_held_out']} held out")
        _print_metrics("spline", spline_metrics, "bilinear", linear_metrics)
        print("  (i.e. this is what running half the grid would have cost you)")

    # -- figures --------------------------------------------------------
    grid_d, grid_a = np.meshgrid(dense_dihedral, dense_angle, indexing="ij")
    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(grid_d, grid_a, dense_Z, cmap="viridis", linewidth=0, antialiased=True)
    node_d, node_a = np.meshgrid(dihedral, angle, indexing="ij")
    ax3d.scatter(node_d.ravel(), node_a.ravel(), Z.ravel(), color="black", s=6,
                 depthshade=False, label="computed nodes")
    ax3d.set_xlabel("dihedral (deg)")
    ax3d.set_ylabel("H-O-O angle (deg)")
    ax3d.set_zlabel("Energy (Ha)")
    ax3d.set_title("Spline-reconstructed H$_2$O$_2$ surface")
    ax3d.view_init(elev=25, azim=-50)
    ax3d.legend(loc="upper left", fontsize=8)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(grid_d, grid_a, dense_Z, levels=30, cmap="viridis")
    ax2d.contour(grid_d, grid_a, dense_Z, levels=30, colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
    ax2d.plot(node_d.ravel(), node_a.ravel(), "k.", markersize=3, label="computed nodes")
    ax2d.plot([dense_dihedral[ddi]], [dense_angle[daj]], "r*", markersize=16,
              label=f"min {dense_Z[ddi, daj]:.4f} Ha")
    ax2d.set_xlabel("H-O-O-H dihedral (degrees)")
    ax2d.set_ylabel("H-O-O angle (degrees)")
    ax2d.set_title("Smooth contours from a coarse grid")
    ax2d.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    fig_err = None
    if holdout["ok"]:
        fig_err, ax_err = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, err, name, cmap in (
            (ax_err[0], holdout["spline_err"], "bicubic spline", "Blues"),
            (ax_err[1], holdout["linear_err"], "bilinear", "Greys"),
        ):
            mesh = ax.pcolormesh(dihedral, angle, np.abs(err.T) * 1000, cmap=cmap, shading="nearest")
            fig_err.colorbar(mesh, ax=ax, label="|error| (mHa)")
            ax.plot(*np.meshgrid(holdout["coarse_dihedral"], holdout["coarse_angle"]),
                    "r+", markersize=6, linestyle="none")
            ax.set_xlabel("H-O-O-H dihedral (degrees)")
            ax.set_ylabel("H-O-O angle (degrees)")
            ax.set_title(f"{name}: held-out error\n(red + = points it was fitted on)")
        fig_err.tight_layout()

    if not save:
        return

    target = resolve_target(None, here, "h2o2_surface_spline_interp")
    out_rows: List[Dict[str, Any]] = []
    for i in range(len(dihedral)):
        for j in range(len(angle)):
            s_err = holdout["spline_err"][i, j] if holdout["ok"] else float("nan")
            l_err = holdout["linear_err"][i, j] if holdout["ok"] else float("nan")
            out_rows.append({
                "dihedral": float(dihedral[i]), "angle": float(angle[j]),
                "energy": float(Z[i, j]),
                "held_out": bool(np.isfinite(s_err)),
                "holdout_spline_mha": float(s_err * 1000),
                "holdout_linear_mha": float(l_err * 1000),
            })
    out_meta = {
        "molecule": "H2O2", "source_scan": str(display_path(scan_path)),
        "method": "bicubic_spline", "scan_type": "surface", "column": column,
        "n_dihedral": int(len(dihedral)), "n_angle": int(len(angle)),
        "validation": "half_grid_holdout",
        "spline_max_mha": spline_metrics["max_mha"],
        "spline_rms_mha": spline_metrics["rms_mha"],
        "linear_max_mha": linear_metrics["max_mha"],
        "linear_rms_mha": linear_metrics["rms_mha"],
        "interpolated_min_dihedral": float(dense_dihedral[ddi]),
        "interpolated_min_angle": float(dense_angle[daj]),
        "interpolated_min_energy": float(dense_Z[ddi, daj]),
        "best_node_dihedral": float(dihedral[di]), "best_node_angle": float(angle[aj]),
        "best_node_energy": float(Z[di, aj]),
        "reference_dihedral": EQUILIBRIUM_DIHEDRAL, "reference_angle": EQUILIBRIUM_ANGLE,
    }
    json_path, csv_path = save_scan(target, out_meta, out_rows)
    print(f"  saved {display_path(json_path)}")
    print(f"  saved {display_path(csv_path)}")

    dense_path = here / "h2o2_surface_spline_dense.csv"
    with open(dense_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dihedral", "angle", "spline"])
        for i, d in enumerate(dense_dihedral):
            for j, a in enumerate(dense_angle):
                writer.writerow([d, a, dense_Z[i, j]])
    print(f"  saved {display_path(dense_path)}")

    surface_png = here / "h2o2_surface_spline_surface.png"
    fig.savefig(surface_png, dpi=150, bbox_inches="tight")
    print(f"  saved {display_path(surface_png)}")
    if fig_err is not None:
        error_png = here / "h2o2_surface_spline_error.png"
        fig_err.savefig(error_png, dpi=150, bbox_inches="tight")
        print(f"  saved {display_path(error_png)}")


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct H2O2's curves and surface from saved VQE scans using "
                    "splines, and measure the reconstruction error. Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE",
                        help="one saved H2O2 scan to interpolate (default: every standard "
                             "scan found in experiment/h2o2/)")
    parser.add_argument("--column", choices=["vqe", "exact", "hf"], default="vqe",
                        help="which energy column to interpolate (default vqe)")
    parser.add_argument("--dense", type=int, default=200,
                        help="resolution of the reconstructed curve/surface per axis "
                             "(default 200)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save results or figures")
    parser.add_argument("--no-show", action="store_true",
                        help="write the figures but don't open a window (useful over SSH)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.no_show:
        matplotlib.use("Agg")

    scans = find_scans(args.scan, here)
    print(f"interpolating {len(scans)} H2O2 scan(s) -- no chemistry is run\n")

    for scan_path in scans:
        metadata, rows = _load(scan_path)
        scan_type = metadata.get("scan_type", "curve")
        print(f"{display_path(scan_path)}  [{scan_type}]")
        if scan_type == "surface":
            process_surface(scan_path, metadata, rows, args.column, args.dense,
                            here, not args.no_save)
        else:
            process_curve(scan_path, metadata, rows, args.column, max(args.dense, 400),
                          here, not args.no_save)
        print()

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
