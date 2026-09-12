"""
h2o2_twist_gif.py -- render a saved H2O2 twist (torsion) scan as an animated
GIF: the energy curve draws itself point by point as the molecule twists
from cis (0 deg) through the skew minimum to trans (180 deg), then the
finished curve is swept back to the start so the loop is seamless.

RUNS NO CHEMISTRY. This reads the ``h2o2_twist_scan.json`` that
``h2o2_ground_state_estimation.py --coordinate twist`` saved and never calls
VQE. It takes well under a minute and can be re-run freely.

WHY A SEPARATE SCRIPT, AND WHY IT LOOKS DIFFERENT FROM THE SURFACE GIF
---------------------------------------------------------------------------
``h2o2_surface_gif.py`` grows a 3D sheet and ends with a camera spin --
that structure exists because a *surface* has a shape worth turning around
to look at. A curve has no such shape; what is worth watching here is the
molecule itself rotating through the torsion while the curve traces out
underneath it. So the "closing move" here is not a spin but a **sweep**:
once the curve is fully drawn, the marker and the molecule travel back
across it from trans to cis, pausing on the true minimum along the way,
landing back at frame one so the loop repeats cleanly.

It borrows the 3D molecule-drawing helpers straight from
``h2o2_ground_state_estimation`` (import only, runs nothing) -- the same
``_build_molecule_panel_3d`` / ``_draw_molecule_3d`` / ``_mol_label_text``
the live viewer and the surface GIF both use.

WHAT THE ANIMATION SHOWS
-------------------------
  grow    One frame per scanned point, cis -> trans, in scan order. HF, VQE
          and exact all draw simultaneously (same colours as the live
          viewer), with a black diamond marking the point just reached. The
          molecule panel shows the two O-H bonds swinging out of the screen
          plane and back -- genuine 3D motion, not a projection trick.

  sweep   The curve is now complete and stops changing. The diamond marker
          and the molecule instead walk *backward* across the finished
          curve, trans -> cis, so the whole thing loops without a jump cut.
          It pauses noticeably longer on the true minimum on the way past,
          labelled, since that is the physically important point on this
          curve.

The GIF is quantized to one shared 256-colour palette taken from the
finished curve, so colours do not flicker between frames, and the blank
canvas margin is cropped away.

REQUIREMENTS
-------------
Pillow, already installed as a dependency of matplotlib.

USAGE
------
  DEFAULTS -- h2o2_twist_scan.json next to this script -> assets/h2o2_twist_scan.gif
    python experiment/h2o2/h2o2_twist_gif.py

  A DIFFERENT SCAN, OR A DIFFERENT OUTPUT
    python experiment/h2o2/h2o2_twist_gif.py --scan fine_twist.json
    python experiment/h2o2/h2o2_twist_gif.py --out ~/slides/h2o2_twist.gif

  TUNING
    --dpi 72              render resolution (72 -> ~860 px wide)
    --frame-ms 90          time per scan point during the grow phase
    --sweep-ms 45          time per point during the return sweep
    --elev 18 --azim -60   molecule panel camera, same as the live viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")                       # offscreen: no window, ever
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
from PIL import Image, ImageChops           # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import h2o2_ground_state_estimation as h2o2  # noqa: E402  -- import only, runs nothing
from scan_io import display_path, load_scan  # noqa: E402

DEFAULT_SCAN = "h2o2_twist_scan.json"
DEFAULT_OUT = HERE.parents[1] / "assets" / "h2o2_twist_scan.gif"


# ===========================================================================
#  Frame capture
# ===========================================================================

def _grab(fig) -> Image.Image:
    """Rasterize the current figure state into a PIL image."""
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


def _crop_common(frames: List[Image.Image], samples: List[int], margin: int = 12) -> List[Image.Image]:
    """Trim the blank border shared by every frame -- the union of ink over a
    few representative frames, so nothing gets clipped."""
    bbox = None
    for k in samples:
        b = ImageChops.invert(frames[k]).getbbox()
        bbox = b if bbox is None else (min(bbox[0], b[0]), min(bbox[1], b[1]),
                                       max(bbox[2], b[2]), max(bbox[3], b[3]))
    w, h = frames[0].size
    bbox = (max(bbox[0] - margin, 0), max(bbox[1] - margin, 0),
            min(bbox[2] + margin, w), min(bbox[3] + margin, h))
    return [f.crop(bbox) for f in frames]


# ===========================================================================
#  The render
# ===========================================================================

def render(scan_path: Path, out: Path, dpi: int, frame_ms: int, sweep_ms: int,
           elev: float, azim: float) -> None:
    metadata, rows = load_scan(scan_path, default_dir=HERE)
    if metadata.get("scan_type") != "curve" or metadata.get("coordinate") != "twist":
        raise SystemExit(f"{display_path(scan_path)} is not a twist curve scan "
                         f"(got scan_type={metadata.get('scan_type')!r}, "
                         f"coordinate={metadata.get('coordinate')!r}).")
    rows = sorted(rows, key=lambda r: r["dihedral"])
    print(f"loaded {len(rows)} points from {display_path(scan_path)} -- no VQE re-run")

    oo_fixed = float(rows[0]["oo"])
    oh_fixed = float(rows[0]["oh"])
    dihedrals = np.array([r["dihedral"] for r in rows])
    best = min(rows, key=lambda r: r["vqe"])
    best_idx = rows.index(best)

    # -- figure: energy curve left, 3D molecule right --------------------
    fig = plt.figure(figsize=(12, 5.5), dpi=dpi)
    ax = fig.add_axes([0.08, 0.12, 0.38, 0.76])
    # Narrower than a plain 1x2 split, and shifted left, so the molecule
    # label's longest text (with the "(minimum)" note appended) has enough
    # right-hand margin not to run off the canvas -- see h2o2_surface_gif.py,
    # which hit the identical problem.
    ax_mol = fig.add_axes([0.52, 0.05, 0.36, 0.88], projection="3d")

    ax.set_xlim(dihedrals[0], dihedrals[-1])
    pad = 0.06 * (max(r["hf"] for r in rows) - min(r["exact"] for r in rows))
    ax.set_ylim(min(r["exact"] for r in rows) - pad, max(r["hf"] for r in rows) + pad)
    (line_hf,) = ax.plot([], [], "-", color="#d73027", label="Hartree-Fock")
    (line_vqe,) = ax.plot([], [], "o-", color="#4c5fd5", label="VQE",
                          markerfacecolor="none", markeredgewidth=1.5)
    (line_exact,) = ax.plot([], [], "x", color="#1a9850", label="Exact", markersize=5, zorder=3)
    (marker,) = ax.plot([], [], "D", color="black", markersize=11, fillstyle="none",
                        markeredgewidth=2)
    ax.set_xlabel("H-O-O-H dihedral (degrees)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("H$_2$O$_2$ twist scan   (r(O-O), r(O-H), angle fixed at equilibrium)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    atom_lines, bond_lines, label = h2o2._build_molecule_panel_3d(ax_mol, oo_fixed, oh_fixed)
    ax_mol.view_init(elev=elev, azim=azim)

    def set_geometry(row: Dict[str, Any], note: str = "") -> None:
        h2o2._draw_molecule_3d(atom_lines, bond_lines, row)
        label.set_text(h2o2._mol_label_text(row) + note)

    frames: List[Image.Image] = []
    durations: List[int] = []

    # -- phase 1: grow, cis -> trans --------------------------------------
    xs: List[float] = []
    hf: List[float] = []
    vqe: List[float] = []
    exact: List[float] = []
    for n, row in enumerate(rows):
        xs.append(row["dihedral"]); hf.append(row["hf"]); vqe.append(row["vqe"]); exact.append(row["exact"])
        line_hf.set_data(xs, hf)
        line_vqe.set_data(xs, vqe)
        line_exact.set_data(xs, exact)
        marker.set_data([row["dihedral"]], [row["vqe"]])
        set_geometry(row)
        frames.append(_grab(fig))
        durations.append(frame_ms)
        if (n + 1) % 5 == 0 or n == len(rows) - 1:
            print(f"  grow   {n + 1:3d}/{len(rows)} points")
    durations[-1] = 900   # hold on the finished curve at trans

    # -- phase 2: sweep back, trans -> cis ---------------------------------
    minimum_frame_idx = None
    for n, i in enumerate(range(len(rows) - 2, -1, -1)):
        row = rows[i]
        marker.set_data([row["dihedral"]], [row["vqe"]])
        note = "   (minimum)" if i == best_idx else ""
        if note:
            minimum_frame_idx = len(frames)   # captured explicitly below, not guessed at
        set_geometry(row, note)
        frames.append(_grab(fig))
        durations.append(sweep_ms if i != best_idx else max(sweep_ms * 6, 400))
        if (n + 1) % 5 == 0 or i == 0:
            print(f"  sweep  {n + 1:3d}/{len(rows) - 1} points")
    durations[-1] = 1200   # hold back at cis before the loop restarts
    plt.close(fig)

    # -- encode -------------------------------------------------------------
    n_grow = len(rows)
    # minimum_frame_idx is included explicitly -- it carries the longest label
    # text (the appended "(minimum)" note), and a bounding box computed from
    # any *other* frame can end up too narrow and clip it.
    samples = [0, n_grow // 2, n_grow - 1, len(frames) - 1]
    if minimum_frame_idx is not None:
        samples.append(minimum_frame_idx)
    frames = _crop_common(frames, samples)
    palette = frames[n_grow - 1].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    w, h = frames[0].size
    print(f"saved {display_path(out)}  --  {len(frames)} frames, {w}x{h} px, "
          f"{out.stat().st_size / 1e6:.2f} MB")


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a saved H2O2 twist scan as a growing curve with the molecule "
                    "twisting alongside, then a return sweep. Runs no chemistry.",
    )
    parser.add_argument("--scan", default=DEFAULT_SCAN, metavar="FILE",
                        help=f"twist scan .json to read (default: {DEFAULT_SCAN}). A bare "
                             f"name is looked for here and next to this script.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), metavar="FILE",
                        help="GIF to write (default: assets/h2o2_twist_scan.gif at the repo root)")
    parser.add_argument("--dpi", type=int, default=72,
                        help="render resolution; 72 gives ~860 px wide (default: 72)")
    parser.add_argument("--frame-ms", type=int, default=90,
                        help="milliseconds per scan point during the grow phase (default: 90)")
    parser.add_argument("--sweep-ms", type=int, default=45,
                        help="milliseconds per point during the return sweep (default: 45, "
                             "faster than grow since nothing new is being revealed)")
    parser.add_argument("--elev", type=float, default=18.0,
                        help="molecule panel camera elevation (default: 18, same as the live viewer)")
    parser.add_argument("--azim", type=float, default=-60.0,
                        help="molecule panel camera azimuth (default: -60, same as the live viewer)")
    args = parser.parse_args()

    if args.dpi <= 0 or args.frame_ms <= 0 or args.sweep_ms <= 0:
        parser.error("--dpi, --frame-ms and --sweep-ms must be positive")

    render(Path(args.scan), Path(args.out).expanduser(), args.dpi, args.frame_ms,
           args.sweep_ms, args.elev, args.azim)


if __name__ == "__main__":
    main()
