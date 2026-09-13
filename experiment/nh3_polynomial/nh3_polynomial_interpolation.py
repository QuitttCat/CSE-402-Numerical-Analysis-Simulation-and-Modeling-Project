"""
nh3_polynomial_interpolation.py -- reconstruct ammonia's curves and surface
from a coarse VQE scan using global polynomials, and find out by measurement
whether that is a good idea.

RUNS NO CHEMISTRY. Loads a scan that
``experiment/nh3/nh3_ground_state_estimation.py`` already saved. Seconds.

THE IDEA, AND THE CLASSIC OBJECTION
-------------------------------------
Given n points there is exactly one polynomial of degree n-1 through all of
them. One closed-form expression, exact at every node, smooth everywhere --
no piecewise bookkeeping like a spline.

The classic objection is Runge's phenomenon [2]: a high-degree polynomial
forced through equally spaced points oscillates, worst near the ends of the
interval, and it gets *worse* as you add points and raise the degree. Since
such a polynomial hits every node exactly, its in-sample residual is zero
and tells you nothing. Only out-of-sample error does.

**Whether it actually bites is a property of the function, not a law.** On
LiH's dissociation curve in this project it does not bite at all -- the
global polynomial beats the cubic spline by a wide margin there. So this
folder measures rather than asserts, and prints what it finds. Run it on
both NH3 coordinates; there is no guarantee they behave the same way, and
the bend and the stretch are quite different functions.

TWO DIFFERENT FITS, FOR TWO DIFFERENT SCANS
---------------------------------------------
**Curves** use ``numpy.polynomial.Polynomial.fit``, sweeping every usable
degree. That class maps the data onto [-1, 1] internally before fitting [1],
which keeps the Vandermonde system far better conditioned. Fitting a
degree-10 polynomial in raw Angstroms is numerically nasty, and without the
rescaling some of the wobble you would see is floating-point noise rather
than the real Runge effect -- an important distinction when the whole point
is to attribute the failure correctly.

**Surfaces** use a **total-degree** fit: all terms x^i y^j with i + j <= d,
solved by least squares, with **both axes normalised to [-1, 1]**.

Total degree, not a full tensor product, and the reason is arithmetic. A
tensor-product fit of a 7x7 grid needs 49 coefficients from 49 points --
square, exactly determined, and hopelessly ill-conditioned. Any
"oscillation" it showed would be the linear solve falling apart, not a
statement about polynomials. Total degree keeps the coefficient count well
below the point count and the system honestly overdetermined.

HOW THE ERROR IS MEASURED, WITHOUT MORE VQE
---------------------------------------------
**Curves: leave-one-out**, swept across every degree the data can support.
Drop an interior node, refit, predict it. Endpoints are excluded because
dropping one turns interpolation into extrapolation.

Note the one degree that cannot be checked at all: the full interpolating
polynomial, degree n-1, needs n points, and leaving one out leaves n-1. Zero
in-sample residual, undefined out-of-sample error. That is a property of the
method, reported as such rather than as a table of NaN that looks like a
crash.

**Surfaces: half-grid holdout.** Keep every other node on each axis, fit
that subgrid, score on everything held out. Leave-one-out is not available
on a grid, and this additionally answers the question that matters for an
N-squared scan -- what would running half of it have cost?

SCORE AGAINST THE MODEL, NOT EXPERIMENT
-----------------------------------------
As in the spline folder. The scan is CAS(6,5)/STO-3G, whose own minimum is
at r = 1.0500 A, theta = 102.048 deg, against an experimental 1.0124 A and
106.67 deg [3]. The basis set's error is much larger than the interpolant's,
so scoring the interpolant against experiment measures the wrong thing.
Experiment is printed for context and flagged as the wrong yardstick.

OUTPUT
-------
  nh3_<coordinate>_polynomial_interp.json / .csv   per-node errors + metrics
  nh3_<coordinate>_polynomial_degrees.csv          error at every degree
  nh3_<coordinate>_polynomial_dense.csv            reconstructed curve
  nh3_<coordinate>_polynomial_curve.png            nodes, fit, linear
  nh3_<coordinate>_polynomial_error.png            per-node error
  nh3_<coordinate>_polynomial_degree_sweep.png     error vs degree
  nh3_surface_polynomial.png                       surface fit + error map

REFERENCES
-----------
[1] numpy.polynomial.Polynomial.fit -- least squares with the domain mapped
    to [-1, 1] for conditioning:
    https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html
[2] Runge's phenomenon:
    https://en.wikipedia.org/wiki/Runge%27s_phenomenon
[3] NIST CCCBDB, experimental NH3 geometry:
    https://cccbdb.nist.gov/exp2x.asp?casno=7664417&charge=0

USAGE
------
  DEFAULT -- finds the first NH3 scan in experiment/nh3/
    python experiment/nh3_polynomial/nh3_polynomial_interpolation.py

  A PARTICULAR SCAN
    python experiment/nh3_polynomial/nh3_polynomial_interpolation.py --scan nh3_stretch_scan.json
    python experiment/nh3_polynomial/nh3_polynomial_interpolation.py --scan nh3_bend_scan.json
    python experiment/nh3_polynomial/nh3_polynomial_interpolation.py --scan nh3_surface_scan.json

  DEGREE -- curves default to n_nodes-1 (the one degree that cannot be
  cross-validated); pass one lower for a usable error figure
    python experiment/nh3_polynomial/nh3_polynomial_interpolation.py --degree 9
    python experiment/nh3_polynomial/nh3_polynomial_interpolation.py --scan nh3_surface_scan.json --degree 4

  OTHER KNOBS
    --column vqe|exact|hf    which energy column (default vqe)
    --dense 2000             points in the reconstructed curve
    --no-save / --no-show
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
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

EXPERIMENTAL_BOND = 1.0124
EXPERIMENTAL_ANGLE = 106.67
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
        meta = {"scan_type": "curve",
                "coordinate": "stretch" if len(bonds) > 1 else "bend"}
    meta["molecule"] = "NH3"
    return meta, rows


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


# ===========================================================================
#  Curves
# ===========================================================================

def leave_one_out(x: np.ndarray, y: np.ndarray, degree: int) -> Dict[str, np.ndarray]:
    """Out-of-sample error at each interior node, polynomial and linear.

    NaN where the remaining data cannot support the requested degree, and at
    the endpoints (dropping one of those is extrapolation).
    """
    n = len(x)
    poly_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)
    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]
        if len(xk) >= degree + 1:
            poly_err[i] = float(Polynomial.fit(xk, yk, degree)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]
    return {"polynomial": poly_err, "linear": linear_err}


def count_interior_minima(poly: Polynomial, x: np.ndarray) -> int:
    """How many local minima the fit has inside the data range.

    The physical curve has one. Anything more is the fit inventing features
    -- a concrete, checkable symptom, more legible than an RMS going up.
    Counted on a dense sample rather than by root-finding, which has its own
    conditioning problems at high degree.
    """
    dense_x = np.linspace(x[0], x[-1], 4001)
    dense_y = poly(dense_x)
    inner = dense_y[1:-1]
    return int(np.sum((inner < dense_y[:-2]) & (inner < dense_y[2:])))


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
    n = len(x)
    if n < 3:
        raise SystemExit(f"{display_path(scan_path)} has only {n} points -- too few to say "
                         f"anything about degree.")

    full_degree = n - 1
    degree = full_degree if args.degree is None else args.degree
    if not 1 <= degree <= full_degree:
        raise SystemExit(f"--degree must be between 1 and n_nodes-1 = {full_degree}")

    print(f"loaded {n} nodes from {display_path(scan_path)}")
    print(f"  NH3 {coordinate} scan, CAS({metadata.get('active_electrons')}, "
          f"{metadata.get('active_orbitals')}), column '{args.column}'")
    print(f"  {x[0]:.4g} to {x[-1]:.4g} {unit}, spacing {np.mean(np.diff(x)):.4g} {unit}")
    print(f"  featured degree {degree}"
          + (f" (the full interpolating polynomial through all {n} nodes)"
             if degree == full_degree else ""))
    check_active_space(metadata)
    print("  no chemistry is run -- this is pure interpolation\n")

    poly = Polynomial.fit(x, y, degree)
    dense_x = np.linspace(x[0], x[-1], args.dense)
    dense_poly = poly(dense_x)
    dense_linear = np.interp(dense_x, x, y)

    residual = float(np.max(np.abs(poly(x) - y)))
    print("IN-SAMPLE RESIDUAL (at the nodes themselves)")
    print(f"  max |fit - data| = {residual * 1000:.6f} mHa")
    if degree == full_degree:
        print("  ...which is ~0 by construction: a degree-(n-1) polynomial passes through")
        print("  all n points exactly. Meaningless as an accuracy measure, which is exactly")
        print("  why leave-one-out below is the one to read.\n")
    else:
        print()

    n_minima = count_interior_minima(poly, x)
    print("SHAPE CHECK")
    print(f"  local minima the degree-{degree} fit has inside the data range: {n_minima}")
    if n_minima > 1:
        print(f"  this curve has ONE physical minimum. The other {n_minima - 1} are invented")
        print("  by the fit -- oscillation between nodes, i.e. Runge's phenomenon [2]")
        print("  showing up as something you can point at rather than a number going up.")
    elif degree < full_degree:
        print("  one minimum, as the physical curve has. Raise --degree "
              f"(up to {full_degree}) and watch.")
    else:
        print(f"  one minimum, as the physical curve has -- and this is already the full")
        print(f"  degree-{full_degree} interpolating polynomial, so it never rings on this")
        print("  data. A real result, not a missing one.")
    print()

    best_node = int(np.argmin(y))
    coarse = np.linspace(x[0], x[-1], 4001)
    k = int(np.argmin(poly(coarse)))
    lo, hi = coarse[max(k - 1, 0)], coarse[min(k + 1, len(coarse) - 1)]
    found = minimize_scalar(lambda t: float(poly(t)), bounds=(lo, hi), method="bounded")

    print("MINIMUM OF THE CURVE")
    print(f"  lowest computed sample : {y[best_node]:.6f} Ha at {x[best_node]:.4f} {unit}")
    print(f"  polynomial minimum     : {float(found.fun):.6f} Ha at {found.x:.4f} {unit}")
    warn_if_slice_differs(coordinate, fixed)
    print(f"  model reference        : {model} {unit}  (the yardstick for interpolation)")
    print(f"    best raw sample : off by {abs(x[best_node] - model):.4f} {unit}")
    print(f"    polynomial      : off by {abs(found.x - model):.4f} {unit}")
    print(f"  experiment is {experiment} {unit}; the model misses it by "
          f"{abs(model - experiment):.4f} {unit} on its")
    print(f"  own -- a basis-set error, not an interpolation one.\n")

    loo = leave_one_out(x, y, degree)
    poly_metrics = metrics(loo["polynomial"])
    linear_metrics = metrics(loo["linear"])

    if degree > n - 2:
        print(f"LEAVE-ONE-OUT ERROR at degree {degree}: undefined, and that is the point.")
        print(f"  Dropping one node leaves {n - 1} points, but degree {degree} needs "
              f"{degree + 1}.")
        print(f"  So the full interpolating polynomial is the one degree that cannot be")
        print(f"  validated at all: zero in-sample residual, unmeasurable out-of-sample")
        print(f"  error. Pass --degree {n - 2} or less to feature a validatable one.")
        print(f"  {'':<12}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
        print(f"  {'linear':<12}{linear_metrics['max_mha']:>12.3f}"
              f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    else:
        print(f"LEAVE-ONE-OUT ERROR at degree {degree}")
        print(f"  {'':<12}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
        print(f"  {'polynomial':<12}{poly_metrics['max_mha']:>12.3f}"
              f"{poly_metrics['rms_mha']:>12.3f}{poly_metrics['mae_mha']:>12.3f}")
        print(f"  {'linear':<12}{linear_metrics['max_mha']:>12.3f}"
              f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    print()

    sweep: List[Dict[str, float]] = []
    for d in range(1, n - 1):
        sweep.append({"degree": d, **metrics(leave_one_out(x, y, d)["polynomial"])})

    print("LEAVE-ONE-OUT ERROR vs DEGREE")
    print(f"  {'degree':>8}{'max (mHa)':>14}{'RMS (mHa)':>14}")
    for entry in sweep:
        print(f"  {int(entry['degree']):>8}{entry['max_mha']:>14.3f}{entry['rms_mha']:>14.3f}")
    best_sweep = min(sweep, key=lambda e: e["rms_mha"])
    worst_sweep = max(sweep, key=lambda e: e["rms_mha"])
    print(f"\n  best  out-of-sample degree: {int(best_sweep['degree'])} "
          f"(RMS {best_sweep['rms_mha']:.3f} mHa)")
    print(f"  worst out-of-sample degree: {int(worst_sweep['degree'])} "
          f"(RMS {worst_sweep['rms_mha']:.3f} mHa)")
    # Does the error RISE AFTER the best degree? That is the Runge signature.
    #
    # An earlier version tested whether the globally worst degree was higher
    # than the best one. That test is broken: the global worst is essentially
    # always degree 1, where the polynomial is simply too rigid to fit the
    # curve at all. Underfitting at the bottom of the sweep swamps
    # overfitting at the top, so the test almost never fired and reported "no
    # Runge failure" on data that plainly showed one -- NH3's stretch is 10x
    # worse at degree 9 than at its degree-6 optimum.
    tail = {e["degree"]: e["rms_mha"] for e in sweep if e["degree"] > best_sweep["degree"]}
    if tail:
        worst_tail = max(tail, key=tail.get)
        ratio = tail[worst_tail] / max(best_sweep["rms_mha"], 1e-12)
    else:
        worst_tail, ratio = None, 1.0

    if tail and ratio > 1.5:
        print(f"  -> accuracy gets WORSE past the sweet spot: degree {int(worst_tail)} is "
              f"{ratio:.1f}x worse")
        print(f"     than degree {int(best_sweep['degree'])}, even though the in-sample")
        print("     residual keeps shrinking toward zero. That is Runge's phenomenon [2],")
        print("     and it is why piecewise splines are usually preferred.")
    elif tail:
        print(f"  -> error is flat past degree {int(best_sweep['degree'])} "
              f"({ratio:.1f}x at worst): no")
        print("     meaningful Runge failure on this data. Whether it bites depends on the")
        print("     function, not on the degree alone.")
    else:
        print("  -> error falls all the way to the highest testable degree: no Runge")
        print("     failure at all here. Whether it bites depends on the function, not on")
        print("     the degree alone.")
    print(f"  compare with the spline: python experiment/nh3_spline/"
          f"nh3_spline_interpolation.py --scan {scan_path.name}\n")

    # -- figures --------------------------------------------------------
    label = "N-H bond length (Angstrom)" if coordinate == "stretch" else "H-N-H angle (degrees)"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dense_x, dense_linear, "--", color="#999999", linewidth=1.4, label="linear")
    ax.plot(dense_x, dense_poly, "-", color="#d73027", linewidth=2,
            label=f"polynomial, degree {degree}")
    ax.plot(x, y, "o", color="#1a9850", markersize=7, zorder=3,
            label=f"computed VQE nodes ({n})")
    ax.plot([found.x], [float(found.fun)], "r*", markersize=16, zorder=4,
            label=f"polynomial minimum @ {found.x:.3f} {unit}")
    ax.axvline(model, color="black", linestyle="-.", linewidth=1,
               label=f"model minimum {model} {unit}")
    pad = 0.15 * (y.max() - y.min())
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel(label)
    ax.set_ylabel(f"Energy (Hartree)  [{args.column}]")
    ax.set_title(f"NH$_3$ {coordinate} curve, degree-{degree} polynomial\n"
                 f"(y-axis clamped to the data range -- the fit may leave it)")
    ax.legend(loc="upper right", fontsize=8)
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
    ax_err.set_xticklabels([f"{x[i]:.3g}" for i in interior])
    ax_err.set_xlabel(f"node left out ({label})")
    ax_err.set_ylabel("|predicted - true|  (mHa)")
    ax_err.set_title("Leave-one-out reconstruction error  (lower is better)")
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
    ax_sweep.set_title(f"NH$_3$ {coordinate}: out-of-sample error against degree")
    ax_sweep.legend()
    ax_sweep.grid(alpha=0.3)
    fig_sweep.tight_layout()

    if not args.no_save:
        target = resolve_target(args.save, here, f"nh3_{coordinate}_polynomial_interp")
        out_rows = [{
            key: float(x[i]), "energy": float(y[i]),
            "loo_polynomial": float(loo["polynomial"][i]),
            "loo_linear": float(loo["linear"][i]),
            "loo_polynomial_mha": float(loo["polynomial"][i] * 1000),
            "loo_linear_mha": float(loo["linear"][i] * 1000),
        } for i in range(n)]
        out_meta = {
            "molecule": "NH3", "source_scan": str(display_path(scan_path)),
            "method": "global_polynomial", "scan_type": "curve", "coordinate": coordinate,
            "column": args.column, "n_nodes": int(n),
            "featured_degree": int(degree), "full_interpolating_degree": int(full_degree),
            "in_sample_max_residual_mha": residual * 1000,
            "interior_minima_at_featured_degree": int(n_minima),
            "polynomial_max_mha": poly_metrics["max_mha"],
            "polynomial_rms_mha": poly_metrics["rms_mha"],
            "linear_max_mha": linear_metrics["max_mha"],
            "linear_rms_mha": linear_metrics["rms_mha"],
            "best_degree": int(best_sweep["degree"]),
            "best_degree_rms_mha": best_sweep["rms_mha"],
            "worst_degree": int(worst_sweep["degree"]),
            "worst_degree_rms_mha": worst_sweep["rms_mha"],
            "interpolated_min": float(found.x),
            "interpolated_min_energy": float(found.fun),
            "best_node": float(x[best_node]),
            "model_reference": model, "experimental_reference": experiment,
        }
        json_path, csv_path = save_scan(target, out_meta, out_rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")

        stem = target.stem.replace("_interp", "")
        degrees_path = target.with_name(stem + "_degrees.csv")
        with open(degrees_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["degree", "max_mha", "rms_mha", "mae_mha"])
            writer.writeheader()
            writer.writerows(sweep)
        print(f"saved {display_path(degrees_path)}")

        dense_path = target.with_name(stem + "_dense.csv")
        with open(dense_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([key, "polynomial", "linear"])
            writer.writerows(zip(dense_x, dense_poly, dense_linear))
        print(f"saved {display_path(dense_path)}")

        for figure, name in (
            (fig, f"nh3_{coordinate}_polynomial_curve.png"),
            (fig_err, f"nh3_{coordinate}_polynomial_error.png"),
            (fig_sweep, f"nh3_{coordinate}_polynomial_degree_sweep.png"),
        ):
            path = here / name
            figure.savefig(path, dpi=150, bbox_inches="tight")
            print(f"saved {display_path(path)}")

    if not args.no_show:
        plt.show()


# ===========================================================================
#  Surface -- total-degree polynomial
# ===========================================================================

def normalise(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Map an axis onto [-1, 1]. Returns (scaled, centre, half-width)."""
    lo, hi = float(values[0]), float(values[-1])
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) or 1.0
    return (values - centre) / half, centre, half


def total_degree_terms(degree: int) -> List[Tuple[int, int]]:
    """Exponent pairs (i, j) with i + j <= degree."""
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def fit_total_degree(bs: np.ndarray, as_: np.ndarray, values: np.ndarray,
                     degree: int) -> np.ndarray:
    """Least-squares total-degree fit on normalised axes.

    ``bs`` and ``as_`` are already-normalised coordinate arrays, flattened
    alongside ``values``. Returns the coefficient vector matching
    ``total_degree_terms(degree)``.
    """
    terms = total_degree_terms(degree)
    design = np.column_stack([bs ** i * as_ ** j for i, j in terms])
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return coeffs


def eval_total_degree(coeffs: np.ndarray, degree: int,
                      b: np.ndarray, a: np.ndarray) -> np.ndarray:
    terms = total_degree_terms(degree)
    out = np.zeros(np.shape(b), dtype=float)
    for c, (i, j) in zip(coeffs, terms):
        out = out + c * (b ** i) * (a ** j)
    return out


def build_grid(rows: List[Dict[str, Any]], column: str
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bonds = np.array(sorted({round(r["bond"], 6) for r in rows}), dtype=float)
    angles = np.array(sorted({round(r["angle"], 6) for r in rows}), dtype=float)
    grid = np.full((len(angles), len(bonds)), np.nan)
    for r in rows:
        i = int(np.argmin(np.abs(angles - r["angle"])))
        j = int(np.argmin(np.abs(bonds - r["bond"])))
        grid[i, j] = r[column]
    if not np.isfinite(grid).all():
        raise SystemExit(
            f"the surface grid has {(~np.isfinite(grid)).sum()} missing cell(s). "
            f"Re-run the surface scan to completion."
        )
    return bonds, angles, grid


def run_surface(metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                args, here: Path, scan_path: Path) -> None:
    bonds, angles, grid = build_grid(rows, args.column)
    nb, na = len(bonds), len(angles)
    print(f"loaded a {na} x {nb} grid ({grid.size} points) from {display_path(scan_path)}")
    print(f"  NH3 surface, CAS({metadata.get('active_electrons')}, "
          f"{metadata.get('active_orbitals')}), column '{args.column}'")
    check_active_space(metadata)
    print("  no chemistry is run -- this is pure interpolation\n")

    nb_n, b_c, b_h = normalise(bonds)
    na_n, a_c, a_h = normalise(angles)
    B, A = np.meshgrid(nb_n, na_n)
    flat_b, flat_a, flat_v = B.ravel(), A.ravel(), grid.ravel()

    # Highest total degree that keeps the system honestly overdetermined.
    max_degree = 1
    while len(total_degree_terms(max_degree + 1)) < grid.size:
        max_degree += 1
    degree = args.degree if args.degree is not None else min(max_degree, 6)
    n_terms = len(total_degree_terms(degree))
    print(f"TOTAL-DEGREE FIT  (terms x^i y^j with i + j <= d)")
    print(f"  featured degree {degree}: {n_terms} coefficients from {grid.size} points")
    print(f"  a full tensor-product fit would need {nb * na} coefficients from "
          f"{grid.size} points --")
    print(f"  square and hopelessly ill-conditioned, which is why this uses total degree.\n")

    coeffs = fit_total_degree(flat_b, flat_a, flat_v, degree)
    fitted = eval_total_degree(coeffs, degree, B, A)
    residual = float(np.max(np.abs(fitted - grid)))
    print(f"  in-sample max |fit - data| = {residual * 1000:.4f} mHa\n")

    # -- half-grid holdout ----------------------------------------------
    bi = np.arange(0, nb, 2)
    ai = np.arange(0, na, 2)
    sub_b, sub_a = nb_n[bi], na_n[ai]
    sub_grid = grid[np.ix_(ai, bi)]
    SB, SA = np.meshgrid(sub_b, sub_a)

    held = np.ones(grid.shape, dtype=bool)
    held[np.ix_(ai, bi)] = False

    # Cap the sweep so the *subgrid* fit stays honestly overdetermined.
    #
    # This matters and it caught a bad result during development. The holdout
    # fits on the subgrid, not the full grid, so the relevant point count is
    # sub_grid.size. On a 7x7 grid the subgrid is 4x4 = 16 points, and total
    # degree 4 is 15 terms -- 15 unknowns from 16 equations. That fit is
    # effectively interpolating and numerically awful, and its holdout error
    # explodes by three orders of magnitude.
    #
    # Reporting that explosion as "Runge's phenomenon" would be wrong: it is
    # the linear solve falling apart on a near-square system, which is the
    # exact failure this script avoids on the full grid by using total degree
    # instead of a tensor product. Requiring terms <= 70% of the subgrid
    # points keeps the sweep measuring approximation quality rather than
    # conditioning.
    sweep: List[Dict[str, float]] = []
    max_sub_degree = 1
    while len(total_degree_terms(max_sub_degree + 1)) <= 0.7 * sub_grid.size:
        max_sub_degree += 1
    for d in range(1, max_sub_degree + 1):
        c = fit_total_degree(SB.ravel(), SA.ravel(), sub_grid.ravel(), d)
        pred = eval_total_degree(c, d, B, A)
        err = np.where(held, pred - grid, np.nan)
        sweep.append({"degree": d, **metrics(err)})

    featured = min(degree, max_sub_degree)
    c_hold = fit_total_degree(SB.ravel(), SA.ravel(), sub_grid.ravel(), featured)
    hold_err = np.where(held, eval_total_degree(c_hold, featured, B, A) - grid, np.nan)
    poly_metrics = metrics(hold_err)

    # Bilinear baseline on the same subgrid.
    lin_err = np.full(grid.shape, np.nan)
    for i in range(na):
        for j in range(nb):
            if held[i, j]:
                lin_err[i, j] = _bilinear(angles[ai], bonds[bi], sub_grid,
                                          angles[i], bonds[j]) - grid[i, j]
    linear_metrics = metrics(lin_err)

    print(f"HALF-GRID HOLDOUT  (fit on {len(ai)}x{len(bi)} = {sub_grid.size} nodes, "
          f"score on the {int(held.sum())} held out)")
    print(f"  {'':<14}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
    print(f"  {'poly deg %d' % featured:<14}{poly_metrics['max_mha']:>12.3f}"
          f"{poly_metrics['rms_mha']:>12.3f}{poly_metrics['mae_mha']:>12.3f}")
    print(f"  {'bilinear':<14}{linear_metrics['max_mha']:>12.3f}"
          f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    print(f"\n  error vs total degree:")
    print(f"  {'degree':>8}{'terms':>8}{'max (mHa)':>14}{'RMS (mHa)':>14}")
    for e in sweep:
        print(f"  {int(e['degree']):>8}{len(total_degree_terms(int(e['degree']))):>8}"
              f"{e['max_mha']:>14.3f}{e['rms_mha']:>14.3f}")
    best = min(sweep, key=lambda e: e["rms_mha"])
    worst = max(sweep, key=lambda e: e["rms_mha"])
    print(f"\n  best degree {int(best['degree'])} (RMS {best['rms_mha']:.3f} mHa), "
          f"worst degree {int(worst['degree'])} (RMS {worst['rms_mha']:.3f} mHa)")
    if worst["degree"] > best["degree"]:
        print("  -> error rises past the sweet spot: the 2D analogue of Runge's phenomenon.")
    else:
        print("  -> error falls with degree throughout: no oscillation failure on this grid.")
    print(f"  the sweep stops at degree {max_sub_degree} on purpose. The holdout fits the")
    print(f"  {sub_grid.size}-point subgrid, and degree {max_sub_degree + 1} would need "
          f"{len(total_degree_terms(max_sub_degree + 1))} terms -- close enough to")
    print(f"  square that the least-squares solve becomes ill-conditioned. The error blows")
    print(f"  up there, but that is the linear algebra failing, not the polynomial")
    print(f"  oscillating, so it is excluded rather than reported as a Runge result.")
    print()

    # -- minimum ----------------------------------------------------------
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    guess = np.array([na_n[i], nb_n[j]])
    found = minimize(lambda p: float(eval_total_degree(coeffs, degree,
                                                       np.array(p[1]), np.array(p[0]))),
                     guess, method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-12})
    min_angle = float(found.x[0]) * a_h + a_c
    min_bond = float(found.x[1]) * b_h + b_c
    print("MINIMUM OF THE SURFACE")
    print(f"  lowest sampled point : {grid[i, j]:.6f} Ha at "
          f"r = {bonds[j]:.4f} A, theta = {angles[i]:.2f} deg")
    print(f"  polynomial minimum   : {float(found.fun):.6f} Ha at "
          f"r = {min_bond:.4f} A, theta = {min_angle:.2f} deg")
    print(f"  model reference      : r = {MODEL_SURFACE_BOND} A, theta = {MODEL_SURFACE_ANGLE} deg")
    print(f"  experiment           : r = {EXPERIMENTAL_BOND} A, "
          f"theta = {EXPERIMENTAL_ANGLE} deg (basis-set error, not the fit's)\n")

    # -- figures -----------------------------------------------------------
    dense_b = np.linspace(bonds[0], bonds[-1], 160)
    dense_a = np.linspace(angles[0], angles[-1], 160)
    DB, DA = np.meshgrid((dense_b - b_c) / b_h, (dense_a - a_c) / a_h)
    dense = eval_total_degree(coeffs, degree, DB, DA)

    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    cf = ax1.contourf(dense_b, dense_a, dense, levels=24, cmap="viridis")
    ax1.contour(dense_b, dense_a, dense, levels=24, colors="white",
                linewidths=0.4, alpha=0.5)
    ax1.plot([min_bond], [min_angle], "r*", markersize=16,
             label=f"poly min\n{min_bond:.3f} A, {min_angle:.1f} deg")
    fig.colorbar(cf, ax=ax1, label="Energy (Hartree)")
    ax1.set_xlabel("r(N-H) (A)")
    ax1.set_ylabel("theta (degrees)")
    ax1.set_title(f"NH$_3$ surface, total-degree-{degree} polynomial")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ds = [e["degree"] for e in sweep]
    ax2.plot(ds, [e["rms_mha"] for e in sweep], "o-", color="#d73027", label="RMS")
    ax2.plot(ds, [e["max_mha"] for e in sweep], "s--", color="#f4a582", label="max")
    ax2.axhline(linear_metrics["rms_mha"], color="#999999", linestyle=":",
                label="bilinear, RMS")
    ax2.set_xlabel("total degree")
    ax2.set_ylabel("half-grid holdout error (mHa)")
    ax2.set_yscale("log")
    ax2.set_xticks(ds)
    ax2.set_title("Out-of-sample error against total degree")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    if not args.no_save:
        target = resolve_target(args.save, here, "nh3_surface_polynomial_interp")
        out_rows = []
        for a in range(na):
            for b in range(nb):
                out_rows.append({
                    "bond": float(bonds[b]), "angle": float(angles[a]),
                    "energy": float(grid[a, b]), "held_out": bool(held[a, b]),
                    "holdout_polynomial_mha": float(hold_err[a, b] * 1000),
                    "holdout_linear_mha": float(lin_err[a, b] * 1000),
                })
        out_meta = {
            "molecule": "NH3", "source_scan": str(display_path(scan_path)),
            "method": "total_degree_polynomial", "scan_type": "surface",
            "coordinate": "surface", "column": args.column,
            "grid_shape": [int(na), int(nb)],
            "featured_degree": int(degree), "n_terms": int(n_terms),
            "holdout_degree": int(featured),
            "in_sample_max_residual_mha": residual * 1000,
            "polynomial_max_mha": poly_metrics["max_mha"],
            "polynomial_rms_mha": poly_metrics["rms_mha"],
            "linear_max_mha": linear_metrics["max_mha"],
            "linear_rms_mha": linear_metrics["rms_mha"],
            "best_degree": int(best["degree"]), "best_degree_rms_mha": best["rms_mha"],
            "worst_degree": int(worst["degree"]), "worst_degree_rms_mha": worst["rms_mha"],
            "interpolated_min_bond": min_bond, "interpolated_min_angle": min_angle,
            "interpolated_min_energy": float(found.fun),
            "model_surface_bond": MODEL_SURFACE_BOND,
            "model_surface_angle": MODEL_SURFACE_ANGLE,
            "experimental_bond": EXPERIMENTAL_BOND, "experimental_angle": EXPERIMENTAL_ANGLE,
        }
        json_path, csv_path = save_scan(target, out_meta, out_rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")

        degrees_path = target.with_name(target.stem.replace("_interp", "") + "_degrees.csv")
        with open(degrees_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["degree", "max_mha", "rms_mha", "mae_mha"])
            writer.writeheader()
            writer.writerows(sweep)
        print(f"saved {display_path(degrees_path)}")

        png = here / "nh3_surface_polynomial.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"saved {display_path(png)}")

    if not args.no_show:
        plt.show()


def _bilinear(ys: np.ndarray, xs: np.ndarray, values: np.ndarray,
              y: float, x: float) -> float:
    """Bilinear interpolation on a rectangular grid, clamped at the edges."""
    j = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    i = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    tx = float(np.clip((x - xs[j]) / (xs[j + 1] - xs[j]), 0, 1))
    ty = float(np.clip((y - ys[i]) / (ys[i + 1] - ys[i]), 0, 1))
    return float(
        values[i, j] * (1 - tx) * (1 - ty) + values[i, j + 1] * tx * (1 - ty)
        + values[i + 1, j] * (1 - tx) * ty + values[i + 1, j + 1] * tx * ty
    )


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct NH3's curves or surface from a coarse VQE scan using "
                    "global polynomials, and measure the error by leave-one-out (curves) "
                    "or half-grid holdout (surface). Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE")
    parser.add_argument("--column", choices=["vqe", "exact", "hf"], default="vqe")
    parser.add_argument("--degree", type=int, default=None,
                        help="polynomial degree to feature (curves default to n_nodes-1; "
                             "surfaces default to total degree 6 or the largest that keeps "
                             "the fit overdetermined, whichever is smaller)")
    parser.add_argument("--dense", type=int, default=800)
    parser.add_argument("--save", type=str, default=None, metavar="FILE")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
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