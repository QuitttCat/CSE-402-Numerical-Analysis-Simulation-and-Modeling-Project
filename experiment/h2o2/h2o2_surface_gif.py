"""
h2o2_surface_gif.py -- render a saved H2O2 surface scan as an animated GIF:
the 3D energy surface rises out of the grid one VQE point at a time, in the
order the scan actually visited them, with the molecule twisting alongside;
then the finished surface turns once on the spot.

RUNS NO CHEMISTRY. This reads the ``h2o2_surface_scan.json`` that
``h2o2_ground_state_estimation.py --coordinate surface`` saved and never
calls VQE. It takes about a minute, nearly all of it matplotlib drawing,
and can be re-run freely.

The output is meant for the README, the same role
``assets/beh2_surface_scan_3d.gif`` plays for BeH2. Re-run this after
re-scanning the surface and the picture updates itself.

WHY A SEPARATE SCRIPT
----------------------
``--load h2o2_surface_scan.json`` on the main script replays the scan live as
a 2D heat map filling in, then pops the finished 3D surface as a static
figure -- right for sitting at the keyboard, since re-rendering
``plot_surface`` on every point is slow and a half-empty 3D surface is hard
to read interactively. For a picture meant to be watched rather than driven,
the trade-off flips: this script rebuilds the 3D view offscreen, frame by
frame, and never touches the interactive viewer.

It borrows the 3D molecule-drawing helpers straight from
``h2o2_ground_state_estimation`` (import only, runs nothing) -- the same
``_build_molecule_panel_3d`` / ``_draw_molecule_3d`` / ``_mol_label_text``
the live viewer uses, so the Geometry panel here is pixel-for-pixel what the
live window draws, just captured frame by frame instead of drawn once.

WHAT THE ANIMATION SHOWS -- AND WHY IT LOOKS DIFFERENT FROM BeH2's
-----------------------------------------------------------------------
  grow   One frame per scanned geometry, in serpentine order (the warm-start
         order the scan ran in). A black dot lands on every solved point and
         a white diamond marks the newest one. ``plot_surface`` needs all
         four corners of a cell, so the sheet lags the dots by one row. The z
         range and colour scale are fixed to the *final* data from frame one,
         so nothing rescales while it fills.

         Unlike BeH2, the Geometry panel here is genuinely 3D: as the
         dihedral axis moves, the two O-H bonds visibly rotate out of the
         plane of the screen rather than just swinging within it. That is
         the entire reason H2O2 was chosen over a simpler bond+angle
         molecule -- see the module docstring on the ground-state script.

  spin   One full turn about the vertical axis of the *energy surface*,
         ending back at the starting view so the loop is seamless. The
         Geometry panel parks on the minimum-energy geometry found by the
         scan for the whole spin, camera fixed -- the spin is about showing
         the shape of the surface, not the molecule.

The GIF is quantized to one shared 256-colour palette taken from the finished
surface, so colours do not flicker between frames, and the blank strip the
canvas leaves round the axes is cropped away.

REQUIREMENTS
-------------
Pillow, for writing the GIF. Already installed as a dependency of
matplotlib.

USAGE
------
  DEFAULTS -- h2o2_surface_scan.json next to this script -> assets/h2o2_surface_scan_3d.gif
    python experiment/h2o2/h2o2_surface_gif.py

  A DIFFERENT SCAN, OR A DIFFERENT OUTPUT
    python experiment/h2o2/h2o2_surface_gif.py --scan fine_surface.json
    python experiment/h2o2/h2o2_surface_gif.py --out ~/slides/h2o2_surface.gif

  TUNING
    --dpi 72              render resolution (72 -> ~890 px wide)
    --frame-ms 70          time per scan point during the grow phase
    --spin-frames 90       frames in the final rotation; 0 skips the spin
    --elev 25 --azim -50   starting camera, same as the main script's 3D view
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")                       # offscreen: no window, ever
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
from matplotlib import cm, colors           # noqa: E402
from PIL import Image, ImageChops           # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import h2o2_ground_state_estimation as h2o2  # noqa: E402  -- import only, runs nothing
from scan_io import display_path, load_scan  # noqa: E402

DEFAULT_SCAN = "h2o2_surface_scan.json"
DEFAULT_OUT = HERE.parents[1] / "assets" / "h2o2_surface_scan_3d.gif"


# ===========================================================================
#  Frame capture
# ===========================================================================

def _grab(fig) -> Image.Image:
    """Rasterize the current figure state into a PIL image."""
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


def _crop_common(frames: List[Image.Image], samples: List[int], margin: int = 12) -> List[Image.Image]:
    """Trim the blank border shared by every frame.

    The bounding box is the union over a few representative frames -- the
    empty grid, mid-fill, the finished surface, and a mid-spin view whose
    rotated axes stick out furthest -- so nothing gets clipped as the camera
    moves.
    """
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

def render(scan_path: Path, out: Path, dpi: int, frame_ms: int, spin_frames: int,
           elev: float, azim: float) -> None:
    metadata, rows = load_scan(scan_path, default_dir=HERE)
    if metadata.get("scan_type") != "surface":
        raise SystemExit(f"{display_path(scan_path)} is a {metadata.get('coordinate')} scan, "
                         f"not a surface -- there is no 3D sheet to draw.")
    print(f"loaded {len(rows)} points from {display_path(scan_path)} -- no VQE re-run")

    dihedrals = np.asarray(metadata["dihedral_values"], dtype=float)
    angles = np.asarray(metadata["angle_values"], dtype=float)
    oo_fixed = float(metadata["fixed_oo"])
    oh_fixed = float(metadata["fixed_oh"])
    grid_dihedral, grid_angle = np.meshgrid(dihedrals, angles)   # [angle, dihedral], same as the live viewer
    Z = np.full(grid_angle.shape, np.nan)

    energies = np.array([r["vqe"] for r in rows])
    zmin, zmax = float(energies.min()), float(energies.max())
    zpad = 0.04 * (zmax - zmin)
    norm = colors.Normalize(zmin, zmax)
    cmap = plt.get_cmap("viridis")

    # -- figure: 3D surface left, 3D molecule right ---------------------
    fig = plt.figure(figsize=(12.5, 5.8), dpi=dpi)
    ax = fig.add_axes([-0.02, 0.0, 0.66, 1.0], projection="3d", computed_zorder=False)
    ax.set_xlim(dihedrals[0], dihedrals[-1])
    ax.set_ylim(angles[0], angles[-1])
    ax.set_zlim(zmin - zpad, zmax + zpad)
    ax.set_xlabel("H-O-O-H dihedral (deg)", labelpad=8)
    ax.set_ylabel("H-O-O angle (deg)", labelpad=8)
    ax.set_zlabel("Energy (Ha)", labelpad=10)
    ax.set_title("H$_2$O$_2$ ground-state surface (VQE)", y=0.96)
    # Wider along the dihedral axis than the angle axis -- it spans 3x the
    # range and is the axis this molecule was chosen to show off.
    ax.set_box_aspect((1.7, 1.0, 0.75), zoom=1.02)
    ax.view_init(elev=elev, azim=azim)

    cax = fig.add_axes([0.625, 0.26, 0.012, 0.48])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Energy (Hartree)")

    ax_mol = fig.add_axes([0.66, 0.08, 0.28, 0.86], projection="3d")
    atom_lines, bond_lines, label = h2o2._build_molecule_panel_3d(ax_mol, oo_fixed, oh_fixed)

    # computed_zorder=False so the dots and the diamond always draw on top of
    # the sheet instead of being depth-sorted behind it.
    (marker,) = ax.plot([], [], [], "D", color="white", markeredgecolor="black",
                        markersize=8, markeredgewidth=1.5, zorder=10)
    dots = ax.scatter([], [], [], s=10, c="black", depthshade=False, zorder=9)
    surface = None

    def redraw_surface() -> None:
        nonlocal surface
        if surface is not None:
            surface.remove()
            surface = None
        if np.isfinite(Z).sum() >= 4:            # fewer than one cell's corners: nothing to knit
            surface = ax.plot_surface(grid_dihedral, grid_angle, np.ma.masked_invalid(Z),
                                      cmap=cmap, norm=norm, linewidth=0.3,
                                      edgecolor=(1, 1, 1, 0.25), antialiased=True,
                                      alpha=0.97, zorder=1)

    def set_geometry(row, note: str = "") -> None:
        h2o2._draw_molecule_3d(atom_lines, bond_lines, row)
        label.set_text(h2o2._mol_label_text(row) + note)

    frames: List[Image.Image] = []
    durations: List[int] = []

    # -- phase 1: grow --------------------------------------------------
    frames.append(_grab(fig))
    durations.append(500)                        # a beat on the empty grid
    done_d, done_a, done_z = [], [], []
    for n, row in enumerate(rows, start=1):
        j = int(np.argmin(np.abs(dihedrals - row["dihedral"])))   # nearest, not ==: float round-trip via JSON
        i = int(np.argmin(np.abs(angles - row["angle"])))
        Z[i, j] = row["vqe"]
        done_d.append(row["dihedral"]); done_a.append(row["angle"]); done_z.append(row["vqe"])
        dots._offsets3d = (done_d, done_a, done_z)
        marker.set_data([row["dihedral"]], [row["angle"]])
        marker.set_3d_properties([row["vqe"]])
        redraw_surface()
        set_geometry(row)
        frames.append(_grab(fig))
        durations.append(frame_ms)
        if n % 20 == 0 or n == len(rows):
            print(f"  grow  {n:3d}/{len(rows)} points")
    durations[-1] = 900                          # hold on the finished sheet

    # -- phase 2: spin --------------------------------------------------
    best = min(rows, key=lambda r: r["vqe"])
    marker.set_data([best["dihedral"]], [best["angle"]])
    marker.set_3d_properties([best["vqe"]])
    set_geometry(best, note="   (minimum)")
    for k in range(1, spin_frames + 1):
        ax.view_init(elev=elev, azim=azim + 360.0 * k / spin_frames)
        frames.append(_grab(fig))
        durations.append(60)
        if k % 30 == 0 or k == spin_frames:
            print(f"  spin  {k:3d}/{spin_frames} frames")
    durations[-1] = 1500                         # hold before the loop restarts
    plt.close(fig)

    # -- encode ---------------------------------------------------------
    n_grow = len(rows)
    samples = [0, n_grow // 2, n_grow, len(frames) - 1]
    if spin_frames:
        samples.append(n_grow + spin_frames // 2)
    frames = _crop_common(frames, samples)
    palette = frames[n_grow].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
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
        description="Animate a saved H2O2 surface scan as a growing, then rotating, 3D GIF. "
                    "Runs no chemistry.",
    )
    parser.add_argument("--scan", default=DEFAULT_SCAN, metavar="FILE",
                        help=f"surface scan .json to read (default: {DEFAULT_SCAN}). A bare "
                             f"name is looked for here and next to this script.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), metavar="FILE",
                        help="GIF to write (default: assets/h2o2_surface_scan_3d.gif at the repo root)")
    parser.add_argument("--dpi", type=int, default=72,
                        help="render resolution; 72 gives ~890 px wide (default: 72)")
    parser.add_argument("--frame-ms", type=int, default=70,
                        help="milliseconds per scan point during the grow phase (default: 70)")
    parser.add_argument("--spin-frames", type=int, default=90,
                        help="frames in the closing rotation; 0 to skip it (default: 90)")
    parser.add_argument("--elev", type=float, default=25.0, help="camera elevation (default: 25)")
    parser.add_argument("--azim", type=float, default=-50.0,
                        help="starting camera azimuth; the spin returns here (default: -50)")
    args = parser.parse_args()

    if args.dpi <= 0 or args.frame_ms <= 0 or args.spin_frames < 0:
        parser.error("--dpi and --frame-ms must be positive, --spin-frames non-negative")

    render(Path(args.scan), Path(args.out).expanduser(), args.dpi, args.frame_ms,
           args.spin_frames, args.elev, args.azim)


if __name__ == "__main__":
    main()
