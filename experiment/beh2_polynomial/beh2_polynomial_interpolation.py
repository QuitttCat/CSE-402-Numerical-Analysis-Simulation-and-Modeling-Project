"""
beh2_polynomial_interpolation.py -- reconstruct BeH2's energy curves and its
energy surface from saved VQE scans using global polynomials, and show where
that approach breaks down.

RUNS NO CHEMISTRY. This loads scans that
``experiment/beh2/beh2_ground_state_estimation.py`` already computed and
saved. Seconds, not minutes.

By default it processes every BeH2 scan it can find -- stretch, bend and
surface -- choosing a 1D or 2D method per file from the ``scan_type`` recorded
in its metadata.

THE IDEA, AND THE PROBLEM WITH IT
-----------------------------------
Given n points there is exactly one polynomial of degree n-1 through all of
them, and in 2D there is a corresponding tensor-product polynomial through a
full grid. One smooth closed-form expression, exact at every node, infinitely
differentiable. Tempting.

It fails in two distinct ways, and this folder separates them.

**Oscillation (Runge's phenomenon).** A high-degree polynomial forced through
equally spaced points oscillates, worst near the ends of the interval, and it
gets worse as you add points and raise the degree [2]. The polynomial hits
every node exactly and misbehaves between them, so a zero fit residual tells
you nothing about accuracy. For BeH2's stretch the ends of the interval are
the repulsive wall and the dissociation plateau -- exactly where a spurious
wobble would invent physics that is not there.

**Conditioning.** The 2D case makes this acute. A full tensor-product
polynomial over the default 11x10 grid would need 110 coefficients determined
from 110 points: square, and hopelessly ill-conditioned. The fit would be
dominated by floating-point noise rather than by the data. So the 2D fits here
use **total degree** (all terms x^i y^j with i+j <= d) at a modest d, which is
a genuine least-squares approximation rather than a doomed exact
interpolation. That is a deliberate choice, not an oversight -- the honest
comparison against the spline is a well-conditioned polynomial, not a
numerically broken one.

Both axes are normalized to [-1, 1] before fitting, for the same reason
``numpy.polynomial.Polynomial.fit`` does it internally [1]: fitting in raw
Angstroms and degrees produces a Vandermonde matrix whose columns span many
orders of magnitude. Without the rescaling, some of the "oscillation" you
would see is arithmetic noise rather than the real Runge effect, and
attributing the failure correctly is the entire point.

Compare against ``../beh2_spline/``, which fits low-degree pieces locally and
stays well-behaved.

HOW THE ERROR IS MEASURED, WITHOUT MORE VQE
---------------------------------------------
**1D -- leave-one-out.** Drop one interior node, refit, predict the dropped
node. Repeat. This is the right test precisely because a full-degree
polynomial has zero in-sample residual by construction: the out-of-sample
error is the only thing that distinguishes a good reconstruction from a bad
one. Every usable degree is swept, which is what makes the degradation
visible.

**2D -- half-grid holdout.** Keep every other bond value and every other
angle, fit on that subgrid, and score on every point held out. Roughly three
quarters of the grid becomes test data. This also answers the question that
matters for a scan costing N^2 VQE points: what would running half the grid
have cost in accuracy?

Linear (and, in 2D, bilinear) interpolation gets the same treatment, so every
number has a baseline.

OUTPUT (per scan processed)
-----------------------------
  beh2_<coord>_polynomial_interp.json / .csv    per-point errors + metrics
  beh2_<coord>_polynomial_degrees.csv           error against degree
  beh2_<coord>_polynomial_dense.csv             the reconstruction
  beh2_<coord>_polynomial_curve.png             1D: nodes, polynomial, linear
  beh2_<coord>_polynomial_error.png             1D: per-node error
  beh2_<coord>_polynomial_degree_sweep.png      error vs degree
  beh2_surface_polynomial_surface.png           2D: surface + contours
  beh2_surface_polynomial_error.png             2D: held-out error heat map

REFERENCES
-----------
[1] numpy.polynomial.Polynomial.fit -- least squares with the domain mapped
    to [-1, 1] for conditioning:
    https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html
[2] Runge's phenomenon:
    https://en.wikipedia.org/wiki/Runge%27s_phenomenon
[3] numpy.linalg.lstsq -- least-squares solve used for the 2D total-degree fit:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

USAGE
------
Every case, explicitly. Runs no chemistry -- it reads scans that
experiment/beh2/ already saved.

  ALL SCANS AT ONCE -- stretch, bend and surface, whichever exist
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py

  ONE SCAN AT A TIME -- .json preferred, .csv accepted
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_stretch_scan.json
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_bend_scan.json
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_surface_scan.json
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_stretch_scan.csv
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_bend_scan.csv
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_surface_scan.csv

  WHICH ENERGY COLUMN -- 'vqe' by default. Interpolating 'exact' instead
  separates coarse-sampling error from VQE optimizer noise.
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --column vqe
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --column exact
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --column hf

  POLYNOMIAL DEGREE -- the default is the full interpolating degree (n_nodes-1),
  which is exactly the one degree leave-one-out cannot validate. Lower degrees
  can be cross-validated; the sweep covers all of them automatically.
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --degree 4
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --degree 7
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_surface_scan.json --degree 5

    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_stretch_scan.json
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_bend_scan.json
    python experiment/beh2_polynomial/beh2_polynomial_interpolation.py --scan beh2_surface_scan.json

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
from numpy.polynomial import Polynomial
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

EQUILIBRIUM_BOND = 1.3264   # Angstrom, linear BeH2
EQUILIBRIUM_ANGLE = 180.0   # degrees

# Both suffixes: the .json is canonical (it carries the metadata), the .csv is
# accepted as a fallback with that metadata inferred from its columns.
CANDIDATE_SCANS = ["beh2_stretch_scan.json", "beh2_bend_scan.json",
                   "beh2_surface_scan.json", "beh2_stretch_scan.csv",
                   "beh2_bend_scan.csv", "beh2_surface_scan.csv"]

# Default total degree for the 2D fits. Deliberately modest: see the module
# docstring on why an exact tensor-product interpolation of the full grid is
# numerically hopeless.
DEFAULT_SURFACE_DEGREE = 4


def find_scans(explicit: str | None, here: Path) -> List[Path]:
    """Which saved BeH2 scans to process. See the beh2_spline twin."""
    beh2_dir = here.parents[0] / "beh2"
    if explicit is not None:
        for candidate in (Path(explicit), here / explicit, beh2_dir / explicit):
            if candidate.exists():
                return [candidate]
        raise SystemExit(
            f"no scan found at {explicit}. Looked next to this script and in "
            f"{display_path(beh2_dir)}."
        )

    found = []
    for name in CANDIDATE_SCANS:
        candidate = beh2_dir / name
        # Skip a .csv whose .json twin is already in the list -- otherwise every
        # scan would be processed twice, once per format.
        if candidate.suffix == ".csv" and candidate.with_suffix(".json") in found:
            continue
        if candidate.exists():
            found.append(candidate)
    if not found:
        looked = "\n  ".join(display_path(beh2_dir / n) for n in CANDIDATE_SCANS)
        raise SystemExit(
            f"no saved BeH2 scans found. Looked for:\n  {looked}\n\n"
            f"This script interpolates existing scans; it runs no chemistry.\n"
            f"Produce them first with:\n"
            f"  python experiment/beh2/beh2_ground_state_estimation.py --coordinate stretch\n"
            f"  python experiment/beh2/beh2_ground_state_estimation.py --coordinate bend\n"
            f"  python experiment/beh2/beh2_ground_state_estimation.py --coordinate surface"
        )
    return found


def _rows_from_csv(path: Path) -> tuple:
    """Read a BeH2 scan from its ``.csv``, inferring what the JSON records.

    Surface vs curve, and which coordinate moves, follow from whether bond and
    angle actually vary -- exact for these three scan shapes. The active space
    is not a column, so it no longer appears in the headers.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("bond", "angle", "hf", "vqe", "exact")
    rows = [{k: float(v) for k, v in r.items() if k in numeric and v != ""} for r in raw]
    missing = {"bond", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} lacks column(s) {sorted(missing)} -- "
                         f"not a BeH2 scan CSV.")
    bonds = sorted({r["bond"] for r in rows})
    angles = sorted({r["angle"] for r in rows})
    if len(bonds) > 1 and len(angles) > 1:
        meta = {"molecule": "BeH2", "scan_type": "surface", "coordinate": "surface",
                "bond_values": bonds, "angle_values": angles}
    else:
        meta = {"molecule": "BeH2", "scan_type": "curve",
                "coordinate": "stretch" if len(bonds) > 1 else "bend"}
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
    print(f"  {'':<12}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
    print(f"  {label_a:<12}{a['max_mha']:>12.3f}{a['rms_mha']:>12.3f}{a['mae_mha']:>12.3f}")
    print(f"  {label_b:<12}{b['max_mha']:>12.3f}{b['rms_mha']:>12.3f}{b['mae_mha']:>12.3f}")


# ===========================================================================
#  1D: the stretch and bend curves
# ===========================================================================

def leave_one_out_1d(x: np.ndarray, y: np.ndarray, degree: int) -> Dict[str, np.ndarray]:
    """Out-of-sample error at each interior node, polynomial and linear."""
    n = len(x)
    poly_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)
    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]
        if len(xk) >= degree + 1:       # degree d needs d+1 points
            poly_err[i] = float(Polynomial.fit(xk, yk, degree)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]
    return {"polynomial": poly_err, "linear": linear_err}


def process_curve(scan_path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                  column: str, requested_degree: int | None, dense_n: int,
                  here: Path, save: bool) -> None:
    """Polynomial-reconstruct a 1D stretch or bend scan."""
    coordinate = metadata.get("coordinate", "stretch")
    key = "bond" if coordinate == "stretch" else "angle"
    unit = "A" if coordinate == "stretch" else "deg"
    axis_label = ("Be-H bond length (Angstrom)" if coordinate == "stretch"
                  else "H-Be-H angle (degrees)")

    rows = sorted(rows, key=lambda r: r[key])
    x = np.array([r[key] for r in rows], dtype=float)
    y = np.array([r[column] for r in rows], dtype=float)
    n = len(x)

    if n < 3:
        print(f"  skipping: only {n} points, not enough to say anything about degree\n")
        return

    full_degree = n - 1
    degree = full_degree if requested_degree is None else min(requested_degree, full_degree)
    print(f"  {n} nodes, {x[0]:.2f} to {x[-1]:.2f} {unit}, "
          f"spacing {np.mean(np.diff(x)):.3f} {unit}")
    print(f"  featured degree {degree}"
          + (f" (the full interpolating polynomial through all {n} nodes)"
             if degree == full_degree else ""))

    poly = Polynomial.fit(x, y, degree)
    dense_x = np.linspace(x[0], x[-1], dense_n)
    dense_poly = poly(dense_x)
    dense_linear = np.interp(dense_x, x, y)

    residual = float(np.max(np.abs(poly(x) - y)))
    print(f"  in-sample max residual: {residual * 1000:.6f} mHa"
          + ("  <- ~0 by construction, so it measures nothing"
             if degree == full_degree else ""))

    # -- minimum --------------------------------------------------------
    best_node = int(np.argmin(y))
    dense_k = int(np.argmin(dense_poly))
    bracket = (dense_x[max(dense_k - 1, 0)], dense_x[min(dense_k + 1, dense_n - 1)])
    refined = minimize_scalar(lambda t: float(poly(t)), bounds=bracket, method="bounded")
    reference = EQUILIBRIUM_BOND if coordinate == "stretch" else EQUILIBRIUM_ANGLE
    print(f"  minimum: best sample {y[best_node]:.6f} Ha at {x[best_node]:.4f} {unit}; "
          f"polynomial {float(refined.fun):.6f} Ha at {float(refined.x):.4f} {unit}")
    print(f"           reference is {reference} {unit}")

    # -- leave-one-out at the featured degree ---------------------------
    loo = leave_one_out_1d(x, y, degree)
    poly_metrics = metrics(loo["polynomial"])
    linear_metrics = metrics(loo["linear"])
    if degree > n - 2:
        # Dropping one node leaves n-1 points, and degree d needs d+1 to be
        # pinned down -- so the full interpolating polynomial is precisely the
        # one degree that cannot be cross-validated. A NaN table here would
        # read as a crash; it is really a property of the method.
        print(f"  leave-one-out at degree {degree}: undefined, and that is the point --")
        print(f"    dropping a node leaves {n - 1} points but degree {degree} needs "
              f"{degree + 1}. Zero")
        print(f"    in-sample residual, unmeasurable out-of-sample error. Use the sweep")
        print(f"    below, or --degree {n - 2} or less, to feature a validatable one.")
        print(f"  {'':<12}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
        print(f"  {'linear':<12}{linear_metrics['max_mha']:>12.3f}"
              f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    else:
        print(f"  leave-one-out error at degree {degree}:")
        _print_metrics("polynomial", poly_metrics, "linear", linear_metrics)

    # -- the degree sweep -----------------------------------------------
    sweep_degrees = list(range(1, n - 1))       # degree d on n-1 nodes needs d+1 <= n-1
    sweep = [{"degree": d, **metrics(leave_one_out_1d(x, y, d)["polynomial"])}
             for d in sweep_degrees]
    print(f"  error vs degree: " + "  ".join(
        f"d{int(e['degree'])}={e['rms_mha']:.2f}" for e in sweep) + "  (RMS mHa)")
    best_sweep = min(sweep, key=lambda e: e["rms_mha"])
    worst_sweep = max(sweep, key=lambda e: e["rms_mha"])
    print(f"  best degree {int(best_sweep['degree'])} "
          f"(RMS {best_sweep['rms_mha']:.3f} mHa), "
          f"worst {int(worst_sweep['degree'])} (RMS {worst_sweep['rms_mha']:.3f} mHa)")
    if worst_sweep["degree"] > best_sweep["degree"]:
        print("  -> accuracy degrades as the degree rises past the sweet spot: Runge [2]")

    # -- figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dense_x, dense_linear, "--", color="#999999", linewidth=1.4,
            label="linear (what the scan plots draw)")
    ax.plot(dense_x, dense_poly, "-", color="#d73027", linewidth=2,
            label=f"polynomial, degree {degree}")
    ax.plot(x, y, "o", color="#1a9850", markersize=7, zorder=3,
            label=f"computed VQE nodes ({n})")
    ax.plot([float(refined.x)], [float(refined.fun)], "r*", markersize=16, zorder=4,
            label=f"polynomial minimum @ {float(refined.x):.3f} {unit}")
    ax.axvline(reference, color="black", linestyle=":", linewidth=1,
               label=f"reference {reference} {unit}")
    pad = 0.15 * (y.max() - y.min())
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(f"Energy (Hartree)  [{column}]")
    ax.set_title(f"BeH$_2$ {coordinate} curve, degree-{degree} polynomial\n"
                 f"(y-axis clamped to the data range -- the fit may leave it)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_err, ax_err = plt.subplots(figsize=(9, 4.5))
    interior = np.arange(n)[1:-1]
    width = 0.35
    ax_err.bar(interior - width / 2, np.abs(loo["linear"][1:-1]) * 1000, width,
               color="#999999", label="linear")
    ax_err.bar(interior + width / 2, np.abs(loo["polynomial"][1:-1]) * 1000, width,
               color="#d73027", label=f"polynomial, degree {degree}")
    ax_err.set_xticks(interior)
    ax_err.set_xticklabels([f"{x[i]:.2f}" for i in interior], rotation=45, ha="right")
    ax_err.set_xlabel(f"node left out ({axis_label})")
    ax_err.set_ylabel("|predicted - true|  (mHa)")
    ax_err.set_title(f"BeH$_2$ {coordinate}: leave-one-out error (lower is better)")
    ax_err.set_yscale("log")
    ax_err.legend()
    ax_err.grid(alpha=0.3, axis="y")
    fig_err.tight_layout()

    fig_sweep, ax_sweep = plt.subplots(figsize=(8, 4.8))
    ds = [e["degree"] for e in sweep]
    ax_sweep.plot(ds, [e["rms_mha"] for e in sweep], "o-", color="#d73027", label="RMS")
    ax_sweep.plot(ds, [e["max_mha"] for e in sweep], "s--", color="#f4a582", label="max")
    ax_sweep.axhline(linear_metrics["rms_mha"], color="#999999", linestyle=":",
                     label="linear interpolation, RMS")
    ax_sweep.set_xlabel("polynomial degree")
    ax_sweep.set_ylabel("leave-one-out error (mHa)")
    ax_sweep.set_yscale("log")
    ax_sweep.set_xticks(ds)
    ax_sweep.set_title(f"BeH$_2$ {coordinate}: out-of-sample error against degree\n"
                       f"rising to the right is Runge's phenomenon")
    ax_sweep.legend()
    ax_sweep.grid(alpha=0.3)
    fig_sweep.tight_layout()

    if not save:
        return

    target = resolve_target(None, here, f"beh2_{coordinate}_polynomial_interp")
    out_rows = [{
        key: float(x[i]),
        "energy": float(y[i]),
        "loo_polynomial": float(loo["polynomial"][i]),
        "loo_linear": float(loo["linear"][i]),
        "loo_polynomial_mha": float(loo["polynomial"][i] * 1000),
        "loo_linear_mha": float(loo["linear"][i] * 1000),
    } for i in range(n)]
    out_meta = {
        "molecule": "BeH2", "source_scan": str(display_path(scan_path)),
        "method": "global_polynomial", "scan_type": "curve", "coordinate": coordinate,
        "column": column, "unit": unit,
        "n_nodes": int(n), "node_spacing": float(np.mean(np.diff(x))),
        "featured_degree": int(degree), "full_interpolating_degree": int(full_degree),
        "in_sample_max_residual_mha": residual * 1000,
        "polynomial_max_mha": poly_metrics["max_mha"],
        "polynomial_rms_mha": poly_metrics["rms_mha"],
        "linear_max_mha": linear_metrics["max_mha"],
        "linear_rms_mha": linear_metrics["rms_mha"],
        "best_degree": int(best_sweep["degree"]),
        "best_degree_rms_mha": best_sweep["rms_mha"],
        "worst_degree": int(worst_sweep["degree"]),
        "worst_degree_rms_mha": worst_sweep["rms_mha"],
        "interpolated_min_x": float(refined.x),
        "interpolated_min_energy": float(refined.fun),
        "best_node_x": float(x[best_node]), "best_node_energy": float(y[best_node]),
        "reference_x": float(reference),
    }
    json_path, csv_path = save_scan(target, out_meta, out_rows)
    print(f"  saved {display_path(json_path)}")
    print(f"  saved {display_path(csv_path)}")

    stem = target.stem.replace("_interp", "")
    degrees_path = target.with_name(stem + "_degrees.csv")
    with open(degrees_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["degree", "max_mha", "rms_mha", "mae_mha"])
        writer.writeheader()
        writer.writerows(sweep)
    print(f"  saved {display_path(degrees_path)}")

    dense_path = target.with_name(stem + "_dense.csv")
    with open(dense_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key, "polynomial", "linear"])
        writer.writerows(zip(dense_x, dense_poly, dense_linear))
    print(f"  saved {display_path(dense_path)}")

    for figure, name in ((fig, f"{stem}_curve.png"),
                         (fig_err, f"{stem}_error.png"),
                         (fig_sweep, f"{stem}_degree_sweep.png")):
        path = here / name
        figure.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {display_path(path)}")


# ===========================================================================
#  2D: the bond x angle surface
# ===========================================================================

def _normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map [lo, hi] onto [-1, 1]. See the docstring on conditioning."""
    if hi == lo:
        return np.zeros_like(values)
    return 2 * (values - lo) / (hi - lo) - 1


def _terms(degree: int) -> List[Tuple[int, int]]:
    """Exponent pairs for a total-degree-``degree`` bivariate polynomial.

    Total degree (i + j <= d) rather than tensor product (i <= d and j <= d):
    (d+1)(d+2)/2 coefficients instead of (d+1)^2, which is both better
    conditioned and a more honest approximation of a smooth surface.
    """
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def poly2d_fit(bond: np.ndarray, angle: np.ndarray, values: np.ndarray,
               degree: int, domain: Tuple[float, float, float, float]):
    """Least-squares total-degree bivariate polynomial [3].

    Takes flat arrays of scattered (bond, angle, value) so it can be fitted to
    a subgrid just as easily as a full grid. ``domain`` is the (bond_lo,
    bond_hi, angle_lo, angle_hi) box used for normalization -- passed in
    explicitly so that a fit made on a subgrid is evaluated on exactly the
    same scaling as the full grid it is tested against.

    Returns a callable ``f(bond, angle)`` accepting arrays.
    """
    b_lo, b_hi, a_lo, a_hi = domain
    u = _normalize(bond, b_lo, b_hi)
    v = _normalize(angle, a_lo, a_hi)
    terms = _terms(degree)
    design = np.column_stack([u ** i * v ** j for i, j in terms])
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)

    def evaluate(bond_q, angle_q):
        uq = _normalize(np.asarray(bond_q, dtype=float), b_lo, b_hi)
        vq = _normalize(np.asarray(angle_q, dtype=float), a_lo, a_hi)
        out = np.zeros(np.broadcast(uq, vq).shape, dtype=float)
        for c, (i, j) in zip(coeffs, terms):
            out = out + c * (uq ** i) * (vq ** j)
        return out

    return evaluate, len(terms)


def rebuild_grid(metadata: Dict[str, Any],
                 rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassemble the surface scan into (bond_values, angle_values, Z[bond, angle]).

    Cells are matched by nearest value, not equality -- the coordinates have
    been through a float round-trip via JSON, and the scan wrote its rows in
    serpentine order so the list is not sorted.
    """
    bond_values = np.asarray(metadata["bond_values"], dtype=float)
    angle_values = np.asarray(metadata["angle_values"], dtype=float)
    Z = np.full((len(bond_values), len(angle_values)), np.nan)
    for row in rows:
        i = int(np.argmin(np.abs(bond_values - row["bond"])))
        j = int(np.argmin(np.abs(angle_values - row["angle"])))
        Z[i, j] = row["energy_value"]
    return bond_values, angle_values, Z


def _bilinear(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation -- the 2D straight-line baseline. Edge-clamped."""
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


def half_grid_holdout(bond: np.ndarray, angle: np.ndarray, Z: np.ndarray,
                      degree: int) -> Dict[str, Any]:
    """Fit on every other row and column; score on everything held out."""
    bi = np.arange(0, len(bond), 2)
    aj = np.arange(0, len(angle), 2)
    coarse_bond, coarse_angle = bond[bi], angle[aj]
    coarse_Z = Z[np.ix_(bi, aj)]
    domain = (float(bond[0]), float(bond[-1]), float(angle[0]), float(angle[-1]))

    grid_b, grid_a = np.meshgrid(coarse_bond, coarse_angle, indexing="ij")
    n_terms = len(_terms(degree))
    if coarse_Z.size < n_terms:
        return {"ok": False,
                "reason": f"the half-grid has {coarse_Z.size} points but a degree-{degree} "
                          f"fit needs at least {n_terms}"}

    fit, _ = poly2d_fit(grid_b.ravel(), grid_a.ravel(), coarse_Z.ravel(), degree, domain)

    held_out = np.ones_like(Z, dtype=bool)
    held_out[np.ix_(bi, aj)] = False

    poly_err = np.full_like(Z, np.nan)
    linear_err = np.full_like(Z, np.nan)
    for i in range(len(bond)):
        for j in range(len(angle)):
            if not held_out[i, j] or not np.isfinite(Z[i, j]):
                continue
            poly_err[i, j] = float(fit(bond[i], angle[j])) - Z[i, j]
            linear_err[i, j] = _bilinear(coarse_bond, coarse_angle, coarse_Z,
                                         bond[i], angle[j]) - Z[i, j]

    return {
        "ok": True, "n_terms": n_terms,
        "n_fit": int(coarse_Z.size), "n_held_out": int(np.isfinite(poly_err).sum()),
        "polynomial_err": poly_err, "linear_err": linear_err,
        "coarse_bond": coarse_bond, "coarse_angle": coarse_angle,
    }


def process_surface(scan_path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                    column: str, requested_degree: int | None, dense_n: int,
                    here: Path, save: bool) -> None:
    """Polynomial-reconstruct the bond x angle surface."""
    for row in rows:
        row["energy_value"] = row[column]
    bond, angle, Z = rebuild_grid(metadata, rows)

    missing = int(np.isnan(Z).sum())
    if missing:
        print(f"  skipping: the grid has {missing} of {Z.size} cells unfilled.")
        print("  Re-run the surface scan to completion, or interpolate a finished one.\n")
        return

    degree = DEFAULT_SURFACE_DEGREE if requested_degree is None else requested_degree
    domain = (float(bond[0]), float(bond[-1]), float(angle[0]), float(angle[-1]))
    grid_b_nodes, grid_a_nodes = np.meshgrid(bond, angle, indexing="ij")
    n_terms = len(_terms(degree))

    print(f"  grid {len(bond)} bond x {len(angle)} angle = {Z.size} points")
    print(f"  total-degree {degree} fit -> {n_terms} coefficients "
          f"(least squares, not exact interpolation -- see the docstring)")
    if n_terms > Z.size:
        print(f"  skipping: degree {degree} needs {n_terms} coefficients but the grid has "
              f"only {Z.size} points\n")
        return

    fit, _ = poly2d_fit(grid_b_nodes.ravel(), grid_a_nodes.ravel(), Z.ravel(), degree, domain)
    residual = float(np.max(np.abs(fit(grid_b_nodes, grid_a_nodes) - Z)))
    print(f"  in-sample max residual: {residual * 1000:.3f} mHa")

    dense_bond = np.linspace(bond[0], bond[-1], dense_n)
    dense_angle = np.linspace(angle[0], angle[-1], dense_n)
    grid_b, grid_a = np.meshgrid(dense_bond, dense_angle, indexing="ij")
    dense_Z = fit(grid_b, grid_a)

    # -- minimum --------------------------------------------------------
    flat = int(np.argmin(Z))
    bi, aj = np.unravel_index(flat, Z.shape)
    dflat = int(np.argmin(dense_Z))
    dbi, daj = np.unravel_index(dflat, dense_Z.shape)
    print(f"  minimum: best grid point {Z[bi, aj]:.6f} Ha at "
          f"r={bond[bi]:.3f} A, angle={angle[aj]:.1f} deg")
    print(f"           polynomial      {dense_Z[dbi, daj]:.6f} Ha at "
          f"r={dense_bond[dbi]:.3f} A, angle={dense_angle[daj]:.1f} deg")
    print(f"           reference is r={EQUILIBRIUM_BOND} A, angle={EQUILIBRIUM_ANGLE} deg")

    # -- half-grid holdout ----------------------------------------------
    holdout = half_grid_holdout(bond, angle, Z, degree)
    if not holdout["ok"]:
        print(f"  holdout test skipped: {holdout['reason']}")
        poly_metrics = linear_metrics = metrics(np.array([np.nan]))
    else:
        poly_metrics = metrics(holdout["polynomial_err"])
        linear_metrics = metrics(holdout["linear_err"])
        print(f"  half-grid holdout: fit on {holdout['n_fit']} points, "
              f"scored on {holdout['n_held_out']} held out")
        _print_metrics("polynomial", poly_metrics, "bilinear", linear_metrics)
        print("  (i.e. this is what running half the grid would have cost you)")
        print("  compare: python experiment/beh2_spline/beh2_spline_interpolation.py")

    # -- figures --------------------------------------------------------
    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(grid_b, grid_a, dense_Z, cmap="magma", linewidth=0, antialiased=True)
    ax3d.scatter(grid_b_nodes.ravel(), grid_a_nodes.ravel(), Z.ravel(), color="black", s=6,
                 depthshade=False, label="computed nodes")
    ax3d.set_xlabel("r(Be-H) (A)")
    ax3d.set_ylabel("angle (deg)")
    ax3d.set_zlabel("Energy (Ha)")
    ax3d.set_title(f"Degree-{degree} polynomial fit to BeH$_2$")
    ax3d.view_init(elev=28, azim=-135)
    ax3d.legend(loc="upper left", fontsize=8)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(grid_b, grid_a, dense_Z, levels=30, cmap="magma")
    ax2d.contour(grid_b, grid_a, dense_Z, levels=30, colors="white",
                 linewidths=0.4, alpha=0.5)
    fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
    ax2d.plot(grid_b_nodes.ravel(), grid_a_nodes.ravel(), "k.", markersize=3,
              label="computed nodes")
    ax2d.plot([dense_bond[dbi]], [dense_angle[daj]], "c*", markersize=16,
              label=f"min {dense_Z[dbi, daj]:.4f} Ha")
    ax2d.set_xlabel("r(Be-H) (Angstrom)")
    ax2d.set_ylabel("H-Be-H angle (degrees)")
    ax2d.set_title("Contours -- a smooth fit, but it need not pass through the nodes")
    ax2d.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    fig_err = None
    if holdout["ok"]:
        fig_err, ax_err = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, err, name, cmap in (
            (ax_err[0], holdout["polynomial_err"], f"degree-{degree} polynomial", "Reds"),
            (ax_err[1], holdout["linear_err"], "bilinear", "Greys"),
        ):
            mesh = ax.pcolormesh(bond, angle, np.abs(err.T) * 1000, cmap=cmap,
                                 shading="nearest")
            fig_err.colorbar(mesh, ax=ax, label="|error| (mHa)")
            ax.plot(*np.meshgrid(holdout["coarse_bond"], holdout["coarse_angle"]),
                    "b+", markersize=6, linestyle="none")
            ax.set_xlabel("r(Be-H) (Angstrom)")
            ax.set_ylabel("H-Be-H angle (degrees)")
            ax.set_title(f"{name}: held-out error\n(blue + = points it was fitted on)")
        fig_err.tight_layout()

    if not save:
        return

    target = resolve_target(None, here, "beh2_surface_polynomial_interp")
    out_rows: List[Dict[str, Any]] = []
    for i in range(len(bond)):
        for j in range(len(angle)):
            p_err = holdout["polynomial_err"][i, j] if holdout["ok"] else float("nan")
            l_err = holdout["linear_err"][i, j] if holdout["ok"] else float("nan")
            out_rows.append({
                "bond": float(bond[i]), "angle": float(angle[j]),
                "energy": float(Z[i, j]),
                "held_out": bool(np.isfinite(p_err)),
                "holdout_polynomial_mha": float(p_err * 1000),
                "holdout_linear_mha": float(l_err * 1000),
            })
    out_meta = {
        "molecule": "BeH2", "source_scan": str(display_path(scan_path)),
        "method": "total_degree_polynomial", "scan_type": "surface", "column": column,
        "degree": int(degree), "n_coefficients": int(n_terms),
        "n_bond": int(len(bond)), "n_angle": int(len(angle)),
        "validation": "half_grid_holdout",
        "in_sample_max_residual_mha": residual * 1000,
        "polynomial_max_mha": poly_metrics["max_mha"],
        "polynomial_rms_mha": poly_metrics["rms_mha"],
        "linear_max_mha": linear_metrics["max_mha"],
        "linear_rms_mha": linear_metrics["rms_mha"],
        "interpolated_min_bond": float(dense_bond[dbi]),
        "interpolated_min_angle": float(dense_angle[daj]),
        "interpolated_min_energy": float(dense_Z[dbi, daj]),
        "best_node_bond": float(bond[bi]), "best_node_angle": float(angle[aj]),
        "best_node_energy": float(Z[bi, aj]),
        "reference_bond": EQUILIBRIUM_BOND, "reference_angle": EQUILIBRIUM_ANGLE,
    }
    json_path, csv_path = save_scan(target, out_meta, out_rows)
    print(f"  saved {display_path(json_path)}")
    print(f"  saved {display_path(csv_path)}")

    dense_path = here / "beh2_surface_polynomial_dense.csv"
    with open(dense_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bond", "angle", "polynomial"])
        for i, b in enumerate(dense_bond):
            for j, a in enumerate(dense_angle):
                writer.writerow([b, a, dense_Z[i, j]])
    print(f"  saved {display_path(dense_path)}")

    surface_png = here / "beh2_surface_polynomial_surface.png"
    fig.savefig(surface_png, dpi=150, bbox_inches="tight")
    print(f"  saved {display_path(surface_png)}")
    if fig_err is not None:
        error_png = here / "beh2_surface_polynomial_error.png"
        fig_err.savefig(error_png, dpi=150, bbox_inches="tight")
        print(f"  saved {display_path(error_png)}")


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct BeH2's curves and surface from saved VQE scans using "
                    "global polynomials, and measure the error. Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE",
                        help="one saved BeH2 scan to interpolate (default: every standard "
                             "scan found in experiment/beh2/)")
    parser.add_argument("--column", choices=["vqe", "exact", "hf"], default="vqe",
                        help="which energy column to interpolate (default vqe)")
    parser.add_argument("--degree", type=int, default=None,
                        help="polynomial degree. 1D default: n_nodes - 1, the full "
                             f"interpolating polynomial. 2D default: "
                             f"{DEFAULT_SURFACE_DEGREE} (total degree)")
    parser.add_argument("--dense", type=int, default=200,
                        help="resolution of the reconstruction per axis (default 200)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save results or figures")
    parser.add_argument("--no-show", action="store_true",
                        help="write the figures but don't open a window (useful over SSH)")
    args = parser.parse_args()

    if args.degree is not None and args.degree < 1:
        parser.error("--degree must be at least 1")

    here = Path(__file__).resolve().parent
    if args.no_show:
        matplotlib.use("Agg")

    scans = find_scans(args.scan, here)
    print(f"interpolating {len(scans)} BeH2 scan(s) -- no chemistry is run\n")

    for scan_path in scans:
        metadata, rows = _load(scan_path)
        scan_type = metadata.get("scan_type", "curve")
        print(f"{display_path(scan_path)}  [{scan_type}]")
        if scan_type == "surface":
            process_surface(scan_path, metadata, rows, args.column, args.degree,
                            args.dense, here, not args.no_save)
        else:
            process_curve(scan_path, metadata, rows, args.column, args.degree,
                          max(args.dense, 400), here, not args.no_save)
        print()

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
