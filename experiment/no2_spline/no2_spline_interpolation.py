"""
no2_spline_interpolation.py -- reconstruct NO2's energy curves and its
bond/angle energy surface from saved VQE scans using splines, and measure
how good the reconstruction is.

RUNS NO CHEMISTRY. This loads scans that
``experiment/no2/no2_ground_state_estimation.py`` already computed and
saved. Seconds, not minutes.

By default it processes every NO2 scan it can find -- stretch, bend and
surface -- choosing a 1D or 2D method per file from the ``scan_type``
recorded in its metadata. This is functionally identical to
``../beh2_spline/``, since NO2's bond/angle field layout is the same as
BeH2's (both molecules are symmetric, so both reduce to one bond number and
one angle number) -- the only real difference is what is *in* the saved
numbers, not how they are interpolated. See the NO2 ground-state script's
docstring for why those numbers come from an open-shell ROHF/UCCSD
calculation rather than a closed-shell one; interpolation does not care
either way, it only sees energies.

WHY THIS FOLDER EXISTS
------------------------
Every scan plot in this project joins neighbouring points with straight
lines -- the crudest possible reconstruction. For NO2 the surface is the
expensive one: N_bond x N_angle VQE points, each one an open-shell UCCSD
optimization that the ground-state script's docstring is explicit about not
having benchmarked. So the practical question is not academic: if we had
only run half the grid, how well could we have reconstructed the rest?

WHAT A SPLINE IS, IN ONE AND TWO DIMENSIONS
---------------------------------------------
1D: a separate cubic between each adjacent pair of points, matching value,
slope and curvature where pieces meet [1]. Exact at every node, no
oscillation -- contrast ``../no2_polynomial/``, which fits one global
polynomial and shows that oscillation happening.

2D: the same idea on a rectangular grid, cubic in each direction [2],
which needs a *complete* grid -- an unfinished surface scan is refused
rather than silently patched.

HOW THE ERROR IS MEASURED, WITHOUT MORE VQE
---------------------------------------------
**1D -- leave-one-out.** Drop one interior node, rebuild from the rest,
evaluate at the dropped node. Repeat for every interior node. Endpoints are
skipped, since dropping one is extrapolation.

**2D -- half-grid holdout.** Leave-one-out does not survive on a
rectangular grid, so instead: keep every other bond value and every other
angle value, fit on that subgrid, and score on everything held out. Still
rectangular, so a bicubic spline applies, and it directly answers "what
would running half the grid have cost us."

Linear (bilinear in 2D) interpolation gets the same treatment in both cases,
so every number comes with a baseline to beat.

OUTPUT (per scan processed)
-----------------------------
  no2_<coord>_spline_interp.json / .csv   per-node/per-point errors + metrics
  no2_<coord>_spline_dense.csv            the reconstructed curve/surface
  no2_<coord>_spline_curve.png            1D: nodes, spline, linear
  no2_<coord>_spline_error.png            1D: per-node error bars
  no2_surface_spline_surface.png          2D: reconstructed surface + contours
  no2_surface_spline_error.png            2D: held-out error heat map

REFERENCES
-----------
[1] scipy.interpolate.CubicSpline:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
[2] scipy.interpolate.RectBivariateSpline:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html
[3] scipy.optimize.minimize_scalar:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

USAGE
------
Every case, explicitly. Runs no chemistry -- it reads scans that
experiment/no2/ already saved.

  ALL SCANS AT ONCE -- stretch, bend, surface, whichever exist
    python experiment/no2_spline/no2_spline_interpolation.py

  ONE SCAN AT A TIME -- .json preferred, .csv accepted
    python experiment/no2_spline/no2_spline_interpolation.py --scan no2_stretch_scan.json
    python experiment/no2_spline/no2_spline_interpolation.py --scan no2_bend_scan.json
    python experiment/no2_spline/no2_spline_interpolation.py --scan no2_surface_scan.json

  WHICH ENERGY COLUMN -- 'vqe' by default. Interpolating 'exact' instead
  separates coarse-sampling error from VQE optimizer noise.
    python experiment/no2_spline/no2_spline_interpolation.py --column exact

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

EQUILIBRIUM_BOND = 1.194     # Angstrom, r(N-O)
EQUILIBRIUM_ANGLE = 134.3    # degrees, O-N-O

CANDIDATE_SCANS = [
    "no2_stretch_scan.json", "no2_bend_scan.json", "no2_surface_scan.json",
    "no2_stretch_scan.csv", "no2_bend_scan.csv", "no2_surface_scan.csv",
]

_AXIS = {
    "stretch": ("bond", "A", "N-O bond length (Angstrom)", EQUILIBRIUM_BOND),
    "bend": ("angle", "deg", "O-N-O angle (degrees)", EQUILIBRIUM_ANGLE),
}


# ===========================================================================
#  Locating the saved scans
# ===========================================================================

def find_scans(explicit: str | None, here: Path) -> List[Path]:
    """Which saved NO2 scans to process. With no ``--scan``, every standard
    scan that exists gets processed, so one command covers the molecule."""
    no2_dir = here.parents[0] / "no2"
    if explicit is not None:
        for candidate in (Path(explicit), here / explicit, no2_dir / explicit):
            if candidate.exists():
                return [candidate]
        raise SystemExit(
            f"no scan found at {explicit}. Looked next to this script and in "
            f"{display_path(no2_dir)}."
        )

    found = []
    for name in CANDIDATE_SCANS:
        candidate = no2_dir / name
        if candidate.suffix == ".csv" and candidate.with_suffix(".json") in found:
            continue
        if candidate.exists():
            found.append(candidate)
    if not found:
        looked = "\n  ".join(display_path(no2_dir / n) for n in CANDIDATE_SCANS)
        raise SystemExit(
            f"no saved NO2 scans found. Looked for:\n  {looked}\n\n"
            f"This script interpolates existing scans; it runs no chemistry.\n"
            f"Produce them first with:\n"
            f"  python experiment/no2/no2_ground_state_estimation.py --coordinate stretch\n"
            f"  python experiment/no2/no2_ground_state_estimation.py --coordinate bend\n"
            f"  python experiment/no2/no2_ground_state_estimation.py --coordinate surface"
        )
    return found


def _rows_from_csv(path: Path) -> tuple:
    """Read an NO2 scan from its ``.csv``, inferring what the JSON records."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("bond", "angle", "hf", "vqe", "exact")
    rows = [{k: float(v) for k, v in r.items() if k in numeric and v != ""} for r in raw]
    missing = {"bond", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} lacks column(s) {sorted(missing)} -- "
                         f"not an NO2 scan CSV.")
    bonds = sorted({r["bond"] for r in rows})
    angles = sorted({r["angle"] for r in rows})
    if len(bonds) > 1 and len(angles) > 1:
        meta = {"molecule": "NO2", "scan_type": "surface", "coordinate": "surface",
                "bond_values": bonds, "angle_values": angles}
    elif len(bonds) > 1:
        meta = {"molecule": "NO2", "scan_type": "curve", "coordinate": "stretch"}
    else:
        meta = {"molecule": "NO2", "scan_type": "curve", "coordinate": "bend"}
    return meta, rows


def _load(path: Path) -> tuple:
    if path.suffix.lower() == ".csv":
        return _rows_from_csv(path)
    return load_scan(path)


def metrics(errors: np.ndarray) -> Dict[str, float]:
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


# ===========================================================================
#  1D: stretch, bend
# ===========================================================================

def leave_one_out_1d(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(x)
    spline_err = np.full(n, np.nan)
    linear_err = np.full(n, np.nan)
    for i in range(1, n - 1):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xk, yk = x[keep], y[keep]
        if len(xk) >= 4:
            spline_err[i] = float(CubicSpline(xk, yk)(x[i])) - y[i]
        linear_err[i] = float(np.interp(x[i], xk, yk)) - y[i]
    return {"spline": spline_err, "linear": linear_err}


def process_curve(scan_path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                  column: str, dense_n: int, here: Path, save: bool) -> None:
    coordinate = metadata.get("coordinate", "stretch")
    key, unit, axis_label, reference = _AXIS[coordinate]

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

    best_node = int(np.argmin(y))
    dense_k = int(np.argmin(dense_spline))
    bracket = (dense_x[max(dense_k - 1, 0)], dense_x[min(dense_k + 1, dense_n - 1)])
    refined = minimize_scalar(lambda t: float(spline(t)), bounds=bracket, method="bounded")

    print(f"  minimum: best sample {y[best_node]:.6f} Ha at {x[best_node]:.4f} {unit}; "
          f"spline {float(refined.fun):.6f} Ha at {float(refined.x):.4f} {unit}")
    print(f"           reference is {reference} {unit}")

    loo = leave_one_out_1d(x, y)
    spline_metrics = metrics(loo["spline"])
    linear_metrics = metrics(loo["linear"])
    print("  leave-one-out error (drop a node, rebuild, predict it):")
    _print_metrics("spline", spline_metrics, "linear", linear_metrics)

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
    ax.set_title(f"NO$_2$ {coordinate} curve, reconstructed by cubic spline")
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
    ax_err.set_title(f"NO$_2$ {coordinate}: leave-one-out error (lower is better)")
    ax_err.set_yscale("log")
    ax_err.legend()
    ax_err.grid(alpha=0.3, axis="y")
    fig_err.tight_layout()

    if not save:
        return

    target = resolve_target(None, here, f"no2_{coordinate}_spline_interp")
    out_rows = [{
        key: float(x[i]),
        "energy": float(y[i]),
        "loo_spline": float(loo["spline"][i]),
        "loo_linear": float(loo["linear"][i]),
        "loo_spline_mha": float(loo["spline"][i] * 1000),
        "loo_linear_mha": float(loo["linear"][i] * 1000),
    } for i in range(len(x))]
    out_meta = {
        "molecule": "NO2", "source_scan": str(display_path(scan_path)),
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
#  2D: the bond x angle surface
# ===========================================================================

def rebuild_grid(metadata: Dict[str, Any],
                 rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassemble the surface scan into (bond_values, angle_values, Z[bond, angle])."""
    bond_values = np.asarray(metadata["bond_values"], dtype=float)
    angle_values = np.asarray(metadata["angle_values"], dtype=float)
    Z = np.full((len(bond_values), len(angle_values)), np.nan)
    for row in rows:
        i = int(np.argmin(np.abs(bond_values - row["bond"])))
        j = int(np.argmin(np.abs(angle_values - row["angle"])))
        Z[i, j] = row["energy_value"]
    return bond_values, angle_values, Z


def _bilinear(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, x: float, y: float) -> float:
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


def half_grid_holdout(bond: np.ndarray, angle: np.ndarray, Z: np.ndarray) -> Dict[str, Any]:
    bi = np.arange(0, len(bond), 2)
    aj = np.arange(0, len(angle), 2)
    coarse_bond, coarse_angle = bond[bi], angle[aj]
    coarse_Z = Z[np.ix_(bi, aj)]

    kx = min(3, len(coarse_bond) - 1)
    ky = min(3, len(coarse_angle) - 1)
    if kx < 1 or ky < 1:
        return {"ok": False, "reason": "the half-grid is too small to fit a spline"}

    spline = RectBivariateSpline(coarse_bond, coarse_angle, coarse_Z, kx=kx, ky=ky)

    held_out = np.ones_like(Z, dtype=bool)
    held_out[np.ix_(bi, aj)] = False

    spline_err = np.full_like(Z, np.nan)
    linear_err = np.full_like(Z, np.nan)
    for i in range(len(bond)):
        for j in range(len(angle)):
            if not held_out[i, j] or not np.isfinite(Z[i, j]):
                continue
            spline_err[i, j] = float(spline(bond[i], angle[j])[0, 0]) - Z[i, j]
            linear_err[i, j] = _bilinear(coarse_bond, coarse_angle, coarse_Z,
                                         bond[i], angle[j]) - Z[i, j]

    return {
        "ok": True, "kx": kx, "ky": ky,
        "n_fit": int(coarse_Z.size), "n_held_out": int(np.isfinite(spline_err).sum()),
        "spline_err": spline_err, "linear_err": linear_err,
        "coarse_bond": coarse_bond, "coarse_angle": coarse_angle,
    }


def process_surface(scan_path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]],
                    column: str, dense_n: int, here: Path, save: bool) -> None:
    for row in rows:
        row["energy_value"] = row[column]
    bond, angle, Z = rebuild_grid(metadata, rows)

    missing = int(np.isnan(Z).sum())
    if missing:
        print(f"  skipping: the grid has {missing} of {Z.size} cells unfilled.")
        print("  A bivariate spline needs a complete rectangular grid [2]. Re-run the")
        print("  surface scan to completion, or interpolate a finished one.\n")
        return
    if len(bond) < 4 or len(angle) < 4:
        print(f"  skipping: grid is {len(bond)}x{len(angle)}; need at least 4x4 for a "
              f"bicubic spline\n")
        return

    print(f"  grid {len(bond)} bond x {len(angle)} angle = {Z.size} points")
    print(f"  bond {bond[0]:.2f} to {bond[-1]:.2f} A, angle {angle[0]:.1f} to "
          f"{angle[-1]:.1f} deg")

    spline = RectBivariateSpline(bond, angle, Z)
    dense_bond = np.linspace(bond[0], bond[-1], dense_n)
    dense_angle = np.linspace(angle[0], angle[-1], dense_n)
    dense_Z = spline(dense_bond, dense_angle)

    flat = int(np.argmin(Z))
    bi, aj = np.unravel_index(flat, Z.shape)
    dflat = int(np.argmin(dense_Z))
    dbi, daj = np.unravel_index(dflat, dense_Z.shape)
    print(f"  minimum: best grid point {Z[bi, aj]:.6f} Ha at "
          f"bond={bond[bi]:.3f} A, angle={angle[aj]:.1f} deg")
    print(f"           spline surface  {dense_Z[dbi, daj]:.6f} Ha at "
          f"bond={dense_bond[dbi]:.3f} A, angle={dense_angle[daj]:.1f} deg")
    print(f"           reference is bond={EQUILIBRIUM_BOND} A, angle={EQUILIBRIUM_ANGLE} deg")

    holdout = half_grid_holdout(bond, angle, Z)
    if not holdout["ok"]:
        print(f"  holdout test skipped: {holdout['reason']}")
        spline_metrics = linear_metrics = metrics(np.array([np.nan]))
    else:
        spline_metrics = metrics(holdout["spline_err"])
        linear_metrics = metrics(holdout["linear_err"])
        print(f"  half-grid holdout: fit on {holdout['n_fit']} points "
              f"({len(holdout['coarse_bond'])}x{len(holdout['coarse_angle'])}), "
              f"scored on {holdout['n_held_out']} held out")
        _print_metrics("spline", spline_metrics, "bilinear", linear_metrics)
        print("  (i.e. this is what running half the grid would have cost you)")

    grid_bond, grid_angle = np.meshgrid(dense_bond, dense_angle, indexing="ij")
    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(grid_bond, grid_angle, dense_Z, cmap="viridis", linewidth=0,
                      antialiased=True)
    node_bond, node_angle = np.meshgrid(bond, angle, indexing="ij")
    ax3d.scatter(node_bond.ravel(), node_angle.ravel(), Z.ravel(), color="black", s=6,
                 depthshade=False, label="computed nodes")
    ax3d.set_xlabel("bond (A)")
    ax3d.set_ylabel("angle (deg)")
    ax3d.set_zlabel("Energy (Ha)")
    ax3d.set_title("Spline-reconstructed NO$_2$ surface")
    ax3d.view_init(elev=28, azim=-135)
    ax3d.legend(loc="upper left", fontsize=8)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(grid_bond, grid_angle, dense_Z, levels=30, cmap="viridis")
    ax2d.contour(grid_bond, grid_angle, dense_Z, levels=30, colors="white",
                 linewidths=0.4, alpha=0.5)
    fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
    ax2d.plot(node_bond.ravel(), node_angle.ravel(), "k.", markersize=3, label="computed nodes")
    ax2d.plot([dense_bond[dbi]], [dense_angle[daj]], "r*", markersize=16,
              label=f"min {dense_Z[dbi, daj]:.4f} Ha")
    ax2d.set_xlabel("N-O bond length (Angstrom)")
    ax2d.set_ylabel("O-N-O angle (degrees)")
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
            mesh = ax.pcolormesh(bond, angle, np.abs(err.T) * 1000, cmap=cmap,
                                 shading="nearest")
            fig_err.colorbar(mesh, ax=ax, label="|error| (mHa)")
            ax.plot(*np.meshgrid(holdout["coarse_bond"], holdout["coarse_angle"]),
                    "r+", markersize=6, linestyle="none")
            ax.set_xlabel("N-O bond length (Angstrom)")
            ax.set_ylabel("O-N-O angle (degrees)")
            ax.set_title(f"{name}: held-out error\n(red + = points it was fitted on)")
        fig_err.tight_layout()

    if not save:
        return

    target = resolve_target(None, here, "no2_surface_spline_interp")
    out_rows: List[Dict[str, Any]] = []
    for i in range(len(bond)):
        for j in range(len(angle)):
            s_err = holdout["spline_err"][i, j] if holdout["ok"] else float("nan")
            l_err = holdout["linear_err"][i, j] if holdout["ok"] else float("nan")
            out_rows.append({
                "bond": float(bond[i]), "angle": float(angle[j]),
                "energy": float(Z[i, j]),
                "held_out": bool(np.isfinite(s_err)),
                "holdout_spline_mha": float(s_err * 1000),
                "holdout_linear_mha": float(l_err * 1000),
            })
    out_meta = {
        "molecule": "NO2", "source_scan": str(display_path(scan_path)),
        "method": "bicubic_spline", "scan_type": "surface", "column": column,
        "n_bond": int(len(bond)), "n_angle": int(len(angle)),
        "validation": "half_grid_holdout",
        "spline_max_mha": spline_metrics["max_mha"],
        "spline_rms_mha": spline_metrics["rms_mha"],
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

    dense_path = here / "no2_surface_spline_dense.csv"
    with open(dense_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bond", "angle", "spline"])
        for i, b in enumerate(dense_bond):
            for j, a in enumerate(dense_angle):
                writer.writerow([b, a, dense_Z[i, j]])
    print(f"  saved {display_path(dense_path)}")

    surface_png = here / "no2_surface_spline_surface.png"
    fig.savefig(surface_png, dpi=150, bbox_inches="tight")
    print(f"  saved {display_path(surface_png)}")
    if fig_err is not None:
        error_png = here / "no2_surface_spline_error.png"
        fig_err.savefig(error_png, dpi=150, bbox_inches="tight")
        print(f"  saved {display_path(error_png)}")


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct NO2's curves and bond/angle surface from saved VQE "
                    "scans using splines, and measure the reconstruction error. "
                    "Runs no chemistry."
    )
    parser.add_argument("--scan", type=str, default=None, metavar="FILE",
                        help="one saved NO2 scan to interpolate (default: every standard "
                             "scan found in experiment/no2/)")
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
    print(f"interpolating {len(scans)} NO2 scan(s) -- no chemistry is run\n")

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