"""
lih_polynomial_interpolation.py -- reconstruct LiH's dissociation curve from a
coarse VQE scan using a single global polynomial, and show why that is a worse
idea than it sounds.

RUNS NO CHEMISTRY. This script loads a scan that
``experiment/lih/lih_ground_state_estimation.py`` already computed and saved,
and does pure numerical analysis on it. Seconds, not minutes.

THE IDEA, AND THE PROBLEM WITH IT
-----------------------------------
Given n computed points, there is exactly one polynomial of degree n-1 that
passes through all of them. That sounds like the natural answer: a single
smooth closed-form curve, no piecewise bookkeeping, exact at every node,
infinitely differentiable everywhere.

It is usually a trap. A high-degree polynomial forced through equally spaced
points **oscillates**, worst near the ends of the interval, and it gets
*worse*, not better, as you add points and raise the degree [2]. The
polynomial hits every node exactly and does something wild in between -- so a
fit residual of zero tells you nothing at all about accuracy between the
nodes.

WHAT ACTUALLY HAPPENS FOR LiH -- AND IT IS NOT WHAT THIS FOLDER EXPECTED
--------------------------------------------------------------------------
This folder was written expecting to reproduce the failure above. Measured on
the default 13-node scan, **it does not happen**, and saying so is more useful
than quietly shipping the expected story.

The measurement, on the exact column of the default grid (0.9 to 3.9 A, step
0.25 A, so the full interpolating polynomial is degree 12):

    degree      leave-one-out RMS (mHa)
      2              31.640
      4               7.395
      6               1.713
      8               0.210
     10               0.230
     11               0.053     <-- best, and it is the highest testable degree

The error falls essentially monotonically all the way to the top. The
degree-12 fit has **exactly one** interior minimum -- it invents no spurious
features at all. And the headline comparison comes out backwards:

    cubic spline, leave-one-out RMS      1.814 mHa
    best polynomial (degree 11)          0.053 mHa   -- about 34x better

So on this data the global polynomial beats the piecewise spline decisively.
That is the opposite of the moral the sibling ``../lih_spline/`` folder and
the O2 pair are usually read as teaching, and it is the real result here.

WHY, AND WHERE THE TRAP ACTUALLY STARTS
-----------------------------------------
Runge's phenomenon is a property of the **function**, not an automatic
consequence of high degree. It bites when the function being interpolated has
a singularity in the complex plane close to the real interval -- Runge's own
1/(1+25x^2) has poles at +/- i/5, right next to it. LiH's CAS(2,5) curve over
this interval is smooth, gently varying and has nothing like that nearby, so a
high-degree polynomial approximates it extremely well.

The trap is real, it just starts later. Repeating the sweep on a 25-node scan
of the same interval (step 0.125 A):

    best degree 13     RMS 0.0004 mHa
    degree 18          RMS 0.0005 mHa
    degree 23          RMS 0.0065 mHa   <-- 16x worse than the sweet spot

There it is: error bottoms out and then climbs as the degree keeps rising,
which is the Runge signature. Note the magnitude, though -- the whole effect
plays out below a hundredth of a milli-Hartree, so it is a clean textbook
trend rather than the catastrophe the phenomenon is usually illustrated with.

To see it yourself:

    python experiment/lih/lih_ground_state_estimation.py --step 0.125 --save lih_fine
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --scan lih_fine.json

That costs about twice the default scan. The degree-sweep plot is the one to
look at.

The honest summary for a write-up: *a global polynomial is not automatically
wrong, and "use a spline" is not automatically right.* Which wins depends on
the smoothness of the function and the number of nodes, and leave-one-out is
how you find out rather than guess. The script prints the comparison either
way and does not claim a Runge failure it has not measured.

HOW THE ERROR IS MEASURED, WITHOUT MORE VQE
---------------------------------------------
**Leave-one-out cross-validation.** Drop one interior node, refit the
polynomial from the remaining nodes, and evaluate it at the node you removed,
where the true answer is known. Repeat for every interior node.

This is the right tool here specifically *because* a full-degree polynomial
interpolates its nodes exactly. In-sample residuals are identically zero and
carry no information; the leave-one-out error is what actually distinguishes a
good reconstruction from a bad one. The script sweeps every usable degree and
reports the error for each, which is what makes the degradation visible:
raising the degree drives in-sample residual to zero while driving
out-of-sample error up.

Endpoints are excluded -- removing one leaves no data on that side, which is
extrapolation, a different and much harder problem.

A NOTE ON CONDITIONING
------------------------
This uses ``numpy.polynomial.Polynomial.fit`` rather than the older
``numpy.polyfit``. The modern class maps the data onto [-1, 1] internally
before fitting [1], which keeps the Vandermonde system far better
conditioned. Fitting a degree-12 polynomial in raw Angstroms is numerically
nasty, and without that rescaling some of the oscillation you would see would
be floating-point noise rather than the genuine Runge effect this script is
trying to show. Getting that distinction right matters when the whole point is
to attribute the failure correctly.

ON COMPARING THE MINIMUM TO EXPERIMENT
----------------------------------------
Same confound as the spline folder, and the same handling. The scan is
CAS(2,5)/STO-3G, whose own minimum is at **1.5474 A** against an experimental
1.5949 A [3]. The interpolant cannot recover accuracy its input data never
had, so the minimum is scored against the model's reference, not against
experiment. Experiment is printed for context and explicitly flagged as the
wrong yardstick for this particular question.

OUTPUT
-------
  lih_polynomial_interp.json / .csv    per-node leave-one-out errors at the
                                       chosen degree, plus summary metrics
  lih_polynomial_degrees.csv           leave-one-out error for every degree
  lih_polynomial_dense.csv             the reconstructed curve, densely sampled
  lih_polynomial_curve.png             nodes, the polynomial, and linear for
                                       comparison
  lih_polynomial_error.png             per-node error, polynomial vs linear
  lih_polynomial_degree_sweep.png      error against degree -- the Runge story

REFERENCES
-----------
[1] numpy.polynomial.Polynomial.fit -- least-squares fit, with the domain
    mapped to [-1, 1] for conditioning:
    https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html
[2] Runge's phenomenon -- oscillation of high-degree polynomial interpolants
    on equally spaced nodes:
    https://en.wikipedia.org/wiki/Runge%27s_phenomenon
[3] NIST Chemistry WebBook, constants of diatomic molecules -- LiH r_e = 1.5949 A:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7580678&Units=SI&Mask=1000

USAGE
------
Every case, explicitly. Runs no chemistry -- it reads a scan that
experiment/lih/ already saved.

  DEFAULT -- finds lih_scan.json (or .csv) in experiment/lih/
    python experiment/lih_polynomial/lih_polynomial_interpolation.py

  A PARTICULAR SCAN -- .json preferred, .csv accepted
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --scan lih_scan.json
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --scan lih_scan.csv

  WHICH ENERGY COLUMN -- 'vqe' by default
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --column vqe
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --column exact
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --column hf

  POLYNOMIAL DEGREE -- the default is n_nodes-1 (12 for the default scan),
  which is exactly the one degree leave-one-out cannot validate. The sweep
  covers every degree that can be.
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --degree 8
    python experiment/lih_polynomial/lih_polynomial_interpolation.py --degree 4

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
from numpy.polynomial import Polynomial
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

EQUILIBRIUM_BOND = 1.5949   # Angstrom, experimental [3]
MODEL_BOND = 1.5474         # Angstrom, the CAS(2,5)/STO-3G model's own minimum

DEFAULT_SCAN_NAMES = ["lih_scan.json", "lih_scan.csv"]


def find_scan(explicit: str | None, here: Path) -> Path:
    """Locate the saved LiH scan to interpolate. See the lih_spline twin."""
    lih_dir = here.parents[0] / "lih"
    if explicit is not None:
        candidates = [Path(explicit), here / explicit, lih_dir / explicit]
    else:
        candidates = [lih_dir / name for name in DEFAULT_SCAN_NAMES]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    looked = "\n  ".join(display_path(c) for c in candidates)
    raise SystemExit(
        f"no saved LiH scan found. Looked in:\n  {looked}\n\n"
        f"This script interpolates an existing scan; it does not run any chemistry.\n"
        f"Produce one first with:\n"
        f"  python experiment/lih/lih_ground_state_estimation.py"
    )


def _rows_from_csv(path: Path) -> tuple:
    """Read a LiH scan from its ``.csv``. Metadata is inferred; see the scan script."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("distance", "hf", "vqe", "exact")
    rows = [{k: float(v) for k, v in r.items() if k in numeric and v != ""} for r in raw]
    missing = {"distance", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} lacks column(s) {sorted(missing)} -- "
                         f"not a LiH scan CSV.")
    return {"molecule": "LiH"}, rows


def _load(path: Path) -> tuple:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    if path.suffix.lower() == ".csv":
        return _rows_from_csv(path)
    return load_scan(path)


def leave_one_out(x: np.ndarray, y: np.ndarray, degree: int) -> Dict[str, np.ndarray]:
    """Out-of-sample error at each interior node, for a polynomial of ``degree``
    and for plain linear interpolation as a baseline.

    Returns NaN at the endpoints (dropping one of those is extrapolation) and
    at any node where the remaining data cannot support the requested degree.
    """
    n = len(x)
    poly_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)

    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]
        # Fitting degree d needs at least d+1 points.
        if len(xk) >= degree + 1:
            poly_err[i] = float(Polynomial.fit(xk, yk, degree)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]

    return {"polynomial": poly_err, "linear": linear_err}


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


def interpolated_minimum(poly: Polynomial, x: np.ndarray) -> Dict[str, float]:
    """Locate the polynomial's minimum inside the scanned interval.

    Dense sample to bracket it, then refine with a bounded scalar minimizer.
    Restricted to the data range on purpose: outside it a high-degree
    polynomial diverges to +/- infinity and any minimum found there is an
    artefact of the fit, not a property of the molecule.
    """
    dense_x = np.linspace(x[0], x[-1], 4001)
    dense_y = poly(dense_x)
    k = int(np.argmin(dense_y))
    lo = dense_x[max(k - 1, 0)]
    hi = dense_x[min(k + 1, len(dense_x) - 1)]
    result = minimize_scalar(lambda t: float(poly(t)), bounds=(lo, hi), method="bounded")
    return {"x": float(result.x), "energy": float(result.fun)}


def count_interior_minima(poly: Polynomial, x: np.ndarray) -> int:
    """How many local minima the fit has inside the data range.

    The physical curve has exactly one. Anything more is the fit inventing a
    feature -- the concrete, checkable symptom of the oscillation this folder
    is about, and more legible than "the RMS went up".

    Counted on a dense sample rather than from the derivative's roots: root
    finding on a degree-12 polynomial has its own conditioning problems, and
    here we only need to know how many dips a reader would see.
    """
    dense_x = np.linspace(x[0], x[-1], 4001)
    dense_y = poly(dense_x)
    interior = dense_y[1:-1]
    return int(np.sum((interior < dense_y[:-2]) & (interior < dense_y[2:])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct LiH's dissociation curve from a coarse VQE scan with a "
                    "global polynomial, and measure the error by leave-one-out "
                    "cross-validation across every usable degree. Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE",
                        help="the saved LiH scan to interpolate (default: look for "
                             "lih_scan.json in experiment/lih/)")
    parser.add_argument("--column", choices=["vqe", "exact", "hf"], default="vqe",
                        help="which energy column to interpolate (default vqe)")
    parser.add_argument("--degree", type=int, default=None,
                        help="degree of the polynomial to feature in the plots "
                             "(default: n_nodes - 1, the full interpolating polynomial, "
                             "which is the worst case and the one worth seeing)")
    parser.add_argument("--dense", type=int, default=800,
                        help="number of points in the reconstructed dense curve (default 800)")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save the results (default: "
                             "lih_polynomial_interp.json/.csv next to this script)")
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
    n = len(x)

    if n < 3:
        raise SystemExit(
            f"{display_path(scan_path)} has only {n} points -- not enough to say anything "
            f"about polynomial degree. Re-run the LiH scan with a finer --step."
        )

    full_degree = n - 1
    degree = full_degree if args.degree is None else args.degree
    if degree < 1:
        parser.error("--degree must be at least 1")
    if degree > full_degree:
        parser.error(f"--degree cannot exceed n_nodes - 1 = {full_degree} "
                     f"({n} nodes cannot pin down a higher-degree polynomial)")

    model_bond = float(metadata.get("model_bond", MODEL_BOND))

    print(f"loaded {n} nodes from {display_path(scan_path)}")
    print(f"  {metadata.get('molecule', 'LiH')} "
          f"CAS({metadata.get('active_electrons')}, {metadata.get('active_orbitals')}), "
          f"column '{args.column}'")
    print(f"  {x[0]:.2f} to {x[-1]:.2f} A, spacing {np.mean(np.diff(x)):.3f} A")
    print(f"  featured degree {degree}"
          + (f" (the full interpolating polynomial through all {n} nodes)"
             if degree == full_degree else ""))
    print("  no chemistry is run -- this is pure interpolation\n")

    # -- build the interpolant -----------------------------------------
    poly = Polynomial.fit(x, y, degree)
    dense_x = np.linspace(x[0], x[-1], args.dense)
    dense_poly = poly(dense_x)
    dense_linear = np.interp(dense_x, x, y)

    residual = float(np.max(np.abs(poly(x) - y)))
    print("IN-SAMPLE RESIDUAL (at the nodes themselves)")
    print(f"  max |fit - data| = {residual * 1000:.6f} mHa")
    if degree == full_degree:
        print("  ...which is ~0 by construction: a degree-(n-1) polynomial passes through")
        print("  all n points exactly. This number is meaningless as an accuracy measure,")
        print("  and that is exactly why leave-one-out below is the one to read.\n")
    else:
        print()

    # -- how many minima did the fit invent? ----------------------------
    n_minima = count_interior_minima(poly, x)
    print("SHAPE CHECK")
    print(f"  local minima the degree-{degree} fit has inside the data range: {n_minima}")
    if n_minima > 1:
        print(f"  LiH's curve has exactly ONE minimum. The other {n_minima - 1} are invented")
        print("  by the fit -- oscillation between nodes, not physics. This is the Runge")
        print("  phenomenon [2] showing up as something you can point at rather than as a")
        print("  number going up.")
    elif degree < full_degree:
        print("  one minimum, as the physical curve has -- this degree has not started")
        print(f"  inventing features. Raise --degree (up to {full_degree}) and watch.")
    else:
        print("  one minimum, as the physical curve has -- and this is already the full")
        print(f"  degree-{full_degree} interpolating polynomial, so it never rings on this")
        print("  data at all. That is a real result, not a missing one: see the module")
        print("  docstring for the measurement and for the finer scan where it does start.")
    print()

    # -- the minimum ----------------------------------------------------
    best_node = int(np.argmin(y))
    interp_min = interpolated_minimum(poly, x)

    print("MINIMUM OF THE CURVE")
    print(f"  lowest computed sample : {y[best_node]:.6f} Ha at {x[best_node]:.4f} A")
    print(f"  polynomial minimum     : {interp_min['energy']:.6f} Ha at "
          f"{interp_min['x']:.4f} A")
    print(f"  model reference        : {model_bond:.4f} A  "
          f"(CAS/STO-3G's own minimum -- the yardstick for interpolation)")
    print(f"    best raw sample : off by {abs(x[best_node] - model_bond):.4f} A")
    print(f"    polynomial      : off by {abs(interp_min['x'] - model_bond):.4f} A")
    print(f"  experimental bond length is {EQUILIBRIUM_BOND} A, which the model misses by")
    print(f"  {abs(model_bond - EQUILIBRIUM_BOND):.4f} A on its own -- a basis-set error, not "
          f"an interpolation one.")
    print()

    # -- leave-one-out, at the featured degree --------------------------
    loo = leave_one_out(x, y, degree)
    poly_metrics = metrics(loo["polynomial"])
    linear_metrics = metrics(loo["linear"])

    if degree > n - 2:
        # Leaving one node out leaves n-1 points, and degree d needs d+1 points
        # to be pinned down. So the full interpolating polynomial is precisely
        # the one degree that cannot be cross-validated: its in-sample residual
        # is zero and its out-of-sample error is undefined. Printing a table of
        # NaN here would look like a crash; it is actually a property of the
        # method, and a rather pointed one.
        print(f"LEAVE-ONE-OUT ERROR at degree {degree}: undefined, and that is the point.")
        print(f"  Dropping one node leaves {n - 1} points, but a degree-{degree} polynomial "
              f"needs {degree + 1}")
        print(f"  to be determined. So the full interpolating polynomial is the one degree")
        print(f"  that cannot be validated at all: zero in-sample residual, unmeasurable")
        print(f"  out-of-sample error. Every lower degree can be checked -- see the sweep")
        print(f"  below, or pass --degree {n - 2} or less to feature a validatable one.")
        print(f"  {'':<12}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
        print(f"  {'linear':<12}{linear_metrics['max_mha']:>12.3f}"
              f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    else:
        print(f"LEAVE-ONE-OUT ERROR at degree {degree}  (drop a node, refit, predict it)")
        print(f"  {'':<12}{'max':>12}{'RMS':>12}{'mean abs':>12}   (mHa)")
        print(f"  {'polynomial':<12}{poly_metrics['max_mha']:>12.3f}"
              f"{poly_metrics['rms_mha']:>12.3f}{poly_metrics['mae_mha']:>12.3f}")
        print(f"  {'linear':<12}{linear_metrics['max_mha']:>12.3f}"
              f"{linear_metrics['rms_mha']:>12.3f}{linear_metrics['mae_mha']:>12.3f}")
    print()

    # -- the degree sweep: the actual result ----------------------------
    # A polynomial of degree d fitted on n-1 nodes needs d+1 <= n-1.
    sweep_degrees = list(range(1, n - 1))
    sweep: List[Dict[str, float]] = []
    for d in sweep_degrees:
        d_metrics = metrics(leave_one_out(x, y, d)["polynomial"])
        sweep.append({"degree": d, **d_metrics})

    print("LEAVE-ONE-OUT ERROR vs DEGREE  (this is the Runge story)")
    print(f"  {'degree':>8}{'max (mHa)':>14}{'RMS (mHa)':>14}")
    for entry in sweep:
        print(f"  {int(entry['degree']):>8}{entry['max_mha']:>14.3f}{entry['rms_mha']:>14.3f}")

    best_sweep = min(sweep, key=lambda e: e["rms_mha"])
    worst_sweep = max(sweep, key=lambda e: e["rms_mha"])
    print(f"\n  best  out-of-sample degree: {int(best_sweep['degree'])} "
          f"(RMS {best_sweep['rms_mha']:.3f} mHa)")
    print(f"  worst out-of-sample degree: {int(worst_sweep['degree'])} "
          f"(RMS {worst_sweep['rms_mha']:.3f} mHa)")
    # Does the error RISE AFTER the best degree? An earlier version compared
    # the globally worst degree against the best, but the global worst is
    # essentially always degree 1 -- underfitting at the bottom of the sweep
    # swamps overfitting at the top, so the test almost never fired.
    tail = {e["degree"]: e["rms_mha"] for e in sweep if e["degree"] > best_sweep["degree"]}
    worst_tail = max(tail, key=tail.get) if tail else None
    ratio = tail[worst_tail] / max(best_sweep["rms_mha"], 1e-12) if tail else 1.0

    if tail and ratio > 1.5:
        print(f"  -> accuracy gets WORSE past the sweet spot: degree {int(worst_tail)} is "
              f"{ratio:.1f}x worse")
        print(f"     than degree {int(best_sweep['degree'])}, even though the in-sample")
        print("     residual keeps shrinking to zero. That is Runge's phenomenon [2],")
        print("     and it is why piecewise splines are preferred.")
    print("  compare with the cubic spline: "
          "python experiment/lih_spline/lih_spline_interpolation.py")
    print()

    # -- figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dense_x, dense_linear, "--", color="#999999", linewidth=1.4,
            label="linear (what the scan plots draw)")
    ax.plot(dense_x, dense_poly, "-", color="#d73027", linewidth=2,
            label=f"polynomial, degree {degree}")
    ax.plot(x, y, "o", color="#1a9850", markersize=7, zorder=3,
            label=f"computed VQE nodes ({n})")
    ax.plot([interp_min["x"]], [interp_min["energy"]], "r*", markersize=16, zorder=4,
            label=f"polynomial minimum @ {interp_min['x']:.3f} A")
    ax.axvline(model_bond, color="black", linestyle="-.", linewidth=1,
               label=f"model minimum {model_bond:.3f} A")
    ax.axvline(EQUILIBRIUM_BOND, color="black", linestyle=":", linewidth=1,
               label=f"experiment {EQUILIBRIUM_BOND} A")
    # The polynomial can shoot far outside the data range; clamp the view to
    # the data with a little headroom, or the nodes become invisible specks.
    pad = 0.15 * (y.max() - y.min())
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel("Li-H bond length (Angstrom)")
    ax.set_ylabel(f"Energy (Hartree)  [{args.column}]")
    ax.set_title(f"LiH curve reconstructed by a degree-{degree} polynomial\n"
                 f"(y-axis clamped to the data range -- the fit may leave it)")
    ax.legend(loc="upper right", fontsize=9)
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
    ax_err.set_xticklabels([f"{x[i]:.2f}" for i in interior])
    ax_err.set_xlabel("node left out (Li-H bond length, Angstrom)")
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
    ax_sweep.set_title("Out-of-sample error against degree\n"
                       "rising to the right is Runge's phenomenon")
    ax_sweep.legend()
    ax_sweep.grid(alpha=0.3)
    fig_sweep.tight_layout()

    # -- save -----------------------------------------------------------
    if not args.no_save:
        target = resolve_target(args.save, here, "lih_polynomial_interp")
        out_rows: List[Dict[str, Any]] = []
        for i in range(n):
            out_rows.append({
                "distance": float(x[i]),
                "energy": float(y[i]),
                "loo_polynomial": float(loo["polynomial"][i]),
                "loo_linear": float(loo["linear"][i]),
                "loo_polynomial_mha": float(loo["polynomial"][i] * 1000),
                "loo_linear_mha": float(loo["linear"][i] * 1000),
            })
        out_meta = {
            "molecule": metadata.get("molecule", "LiH"),
            "source_scan": str(display_path(scan_path)),
            "method": "global_polynomial",
            "column": args.column,
            "n_nodes": int(n),
            "node_spacing": float(np.mean(np.diff(x))),
            "featured_degree": int(degree),
            "full_interpolating_degree": int(full_degree),
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
            "interpolated_min_bond": interp_min["x"],
            "interpolated_min_energy": interp_min["energy"],
            "best_node_bond": float(x[best_node]),
            "best_node_energy": float(y[best_node]),
            "model_bond": model_bond,
            "experimental_bond": EQUILIBRIUM_BOND,
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
            writer.writerow(["distance", "polynomial", "linear"])
            writer.writerows(zip(dense_x, dense_poly, dense_linear))
        print(f"saved {display_path(dense_path)}")

        for figure, name in ((fig, "lih_polynomial_curve.png"),
                             (fig_err, "lih_polynomial_error.png"),
                             (fig_sweep, "lih_polynomial_degree_sweep.png")):
            path = here / name
            figure.savefig(path, dpi=150, bbox_inches="tight")
            print(f"saved {display_path(path)}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()