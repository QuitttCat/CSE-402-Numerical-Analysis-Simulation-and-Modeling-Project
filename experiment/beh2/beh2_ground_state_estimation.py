"""
beh2_ground_state_estimation.py -- BeH2's ground-state energy by VQE, scanned
along its symmetric internal coordinates, plotted live next to a picture of
the molecule deforming.

Simulator only. No IBM account, no queue, no quota.

SYMMETRIC MOTION ONLY -- AND WHY THAT IS THE WHOLE POINT
---------------------------------------------------------
BeH2 is triatomic, so it has 3N-6 = 3 internal degrees of freedom: the two
Be-H bond lengths and the H-Be-H angle. This script deliberately constrains
the two bonds to be equal, which collapses those three coordinates down to
**two**:

    bond    the shared Be-H distance  (both bonds always move together)
    angle   the H-Be-H angle          (both hydrogens always swing together)

That constraint is structural, not a check applied afterwards.
:func:`beh2_positions` takes exactly one bond number and one angle number and
places H1 and H2 as mirror images of each other through the bisector at equal
radius. There is no way to express an asymmetric geometry through this
interface, so every geometry in every mode below keeps the molecule's mirror
symmetry -- C2v while bent, and genuinely linear (D-infinity-h) at 180 deg.

Two knobs cross into exactly **one** surface, which is why there is a single
``--coordinate surface`` mode rather than several.

THREE MODES, AND WHAT EACH CURVE SHOULD LOOK LIKE
--------------------------------------------------
  --coordinate stretch   Both Be-H bonds stretched together, angle held at
                         180 deg. A classic dissociation curve: steep
                         repulsive wall on the left, a minimum near 1.33 A,
                         then a climb that flattens toward Be + 2H.

                         The interesting part is the gap between the red
                         Hartree-Fock curve and the blue/green VQE and exact
                         points. Near the minimum HF is decent; out in the
                         stretched region it fails badly and peels visibly
                         upward, because a single Slater determinant cannot
                         describe two bonds breaking at once. That gap *is*
                         the correlation energy VQE recovers and HF misses.

  --coordinate bend      The H-Be-H angle varied with both bonds held at
                         1.33 A. **The minimum is at 180 deg, not in the
                         middle of the range** -- BeH2 is linear at
                         equilibrium, because the central atom has no lone
                         pairs to push the hydrogens down. This is the
                         opposite of water, which bends to 104.5 deg. So the
                         curve is not a symmetric well: it falls from left to
                         right and bottoms out at the right-hand *edge* of
                         the scan, looking like half a parabola. It is also
                         flat exactly at 180 deg, since bending either
                         direction costs the same -- the slope there is zero
                         by symmetry.

                         Two other things to expect. The energy range is far
                         smaller than the stretch -- tens of milli-Hartree
                         across the whole bend against hundreds across the
                         stretch, because bending is a soft motion and
                         stretching is stiff. And the molecule never comes
                         apart, so the electronic structure stays comfortably
                         single-reference and HF tracks the exact curve
                         closely instead of blowing up.

  --coordinate surface   Both coordinates varied on a grid. A long curved
                         valley: soft along the angle direction, stiff along
                         the bond direction, so there is a trough running
                         along the angle axis at bond ~1.33 A with steep
                         walls rising either side of it. The contours come
                         out stretched along the angle axis and squeezed
                         along the bond axis, and that anisotropy is the
                         headline result -- it is the thing a surface shows
                         you that two separate 1D slices cannot.

                         The floor of the trough also drifts: the preferred
                         bond length contracts as the molecule straightens.
                         Measured on the default grid via the spline in
                         ../beh2_spline/, it runs 1.3045 A at 90 deg down to
                         1.2895 A at 180 deg -- monotonic, and a total drift
                         of only 0.015 A.

                         Note what that means. The drift is one tenth of the
                         0.15 A grid step, so on the raw grid the optimal bond
                         comes out as exactly 1.35 A at *every* angle and the
                         coupling is invisible. You can only see it through
                         the interpolant. That is worth a line in a write-up:
                         a real physical effect an order of magnitude below
                         the sampling resolution, recovered by interpolation
                         rather than by computing more points.

Run the stretch and the bend and compare: one coordinate where HF fails
badly, one where it is fine, in the same molecule. That contrast is the
experiment.

THE SANITY CHECK THAT MATTERS
-------------------------------
The stretch curve must rise **monotonically** after its minimum. If it dips
back down as you pull the atoms further apart, the active space is too small
-- that is a failure of the model, not of VQE, and it is the exact trap
documented in the H2O script's docstring, where CAS(4,4) produced a visibly
kinked water curve while reproducing the (wrong) active-space energy to under
0.1 mHa. An "exact" number is only exact *within the model you chose*.

ACTIVE SPACE
-------------
BeH2 in STO-3G is 6 electrons in 7 spatial orbitals -> 14 qubits if you take
everything, far too slow to scan and hopeless for a surface. The default here
is CAS(4,4) -- 4 electrons in 4 spatial orbitals, which freezes the Be 1s
core -- giving 6 qubits, 26 UCCSD parameters, and a few seconds per point.
Six qubits is also the size Kandala et al. used for their BeH2 hardware
demonstration [3].

``--active-electrons`` / ``--active-orbitals`` open this up for a convergence
spot-check on a coarse grid. Bear in mind a surface costs N^2 points, so the
default grid is already 110 VQE optimizations.

COST
-----
What dominates runtime is not circuit simulation (one energy evaluation is
well under a second) but the *number* of evaluations: SLSQP estimates its
gradient by finite differences, so each optimizer step costs roughly one
evaluation per parameter.

Measured on a laptop CPU at the default CAS(4,4) -- 6 qubits, 26 UCCSD
parameters -- a point takes about 2-6 s, averaging roughly 4:

    stretch   16 points   ~1 min
    bend      19 points   ~1.5 min
    surface  110 points   ~8 min

So the whole molecule is about ten minutes of compute, not the forty-odd an
earlier estimate here claimed.

Each geometry warm-starts its optimizer from the previous geometry's
solution, since neighbouring points have similar wavefunctions. The surface
walks its grid in serpentine order (left to right, then right to left on the
next row) specifically so that the warm start always comes from a
geometrically *adjacent* point instead of jumping back across the grid.
``--cold-start`` disables warm starting.

REFERENCES
-----------
[1] Be-H equilibrium bond length 1.3264 A for linear BeH2, and the CAS
    treatments of it, as surveyed in:
    https://www.nature.com/articles/s42005-023-01312-y
[2] ActiveSpaceTransformer, and hand-picking orbitals with active_orbitals:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html
[3] Kandala et al., "Hardware-efficient variational quantum eigensolver for
    small molecules and quantum magnets", Nature 549, 242 (2017) -- BeH2 on
    six qubits: https://www.nature.com/articles/nature23879

USAGE
------
Every case, explicitly. All three modes move symmetrically -- see the note at
the top of this docstring.

  RUN A SCAN -- symmetric stretch (both Be-H bonds together, angle fixed)
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate stretch
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate stretch --start 0.9 --stop 2.4 --step 0.1
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate stretch --angle 170
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate stretch --start 1.1 --stop 1.6 --step 0.25 --no-save

  RUN A SCAN -- symmetric bend (H-Be-H angle, both bonds fixed)
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate bend
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate bend --start 90 --stop 180 --step 5
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate bend --bond 1.4
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate bend --start 150 --stop 180 --step 15 --no-save

  RUN A SCAN -- the surface (bond x angle, both symmetric)
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate surface
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate surface --step 0.15 --angle-step 10
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate surface --angle-start 90 --angle-stop 180 --angle-step 10
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate surface --step 0.5 --angle-step 30 --no-save   # quick 4x4

  RELOAD A FINISHED SCAN -- no VQE re-run. Either format works.
    python experiment/beh2/beh2_ground_state_estimation.py --load beh2_stretch_scan.json
    python experiment/beh2/beh2_ground_state_estimation.py --load beh2_bend_scan.json
    python experiment/beh2/beh2_ground_state_estimation.py --load beh2_surface_scan.json
    python experiment/beh2/beh2_ground_state_estimation.py --load beh2_stretch_scan.csv
    python experiment/beh2/beh2_ground_state_estimation.py --load beh2_bend_scan.csv
    python experiment/beh2/beh2_ground_state_estimation.py --load beh2_surface_scan.csv

  A LARGER ACTIVE SPACE -- for a convergence spot-check on a coarse grid
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate stretch --active-electrons 6 --active-orbitals 6
    python experiment/beh2/beh2_ground_state_estimation.py --coordinate surface --active-electrons 6 --active-orbitals 6 --step 0.5 --angle-step 30

  OTHER KNOBS
    --cold-start          no warm start from the previous geometry
    --save myname         write to myname.json/.csv beside this script
    --no-save             don't write anything
    --no-show             exit when the scan finishes instead of waiting on the window
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_io import display_path, load_scan, resolve_target, save_scan  # noqa: E402

from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

BASIS = "sto3g"

# BeH2's equilibrium geometry. The bond length is the literature value for
# linear BeH2 [1]; the angle is 180 deg because BeH2 *is* linear -- the central
# atom carries no lone pairs, so there is nothing to bend the hydrogens away
# from the straight-through arrangement. Contrast water at 104.5 deg.
EQUILIBRIUM_BOND = 1.3264   # Angstrom
EQUILIBRIUM_ANGLE = 180.0   # degrees


# ===========================================================================
#  Geometry -- symmetric by construction
# ===========================================================================

def beh2_positions(bond: float, angle_deg: float) -> Dict[str, tuple]:
    """Atom positions for BeH2, in the x-z plane.

    Beryllium sits at the origin. The two hydrogens are placed symmetrically
    about the +z axis, **both** at `bond` Angstrom from Be, with `angle_deg`
    between the two Be-H bonds.

    This signature is the symmetry constraint. There is one bond number and
    one angle number, and H1/H2 come out as mirror images through the
    bisector at equal radius -- so stretching necessarily moves both bonds
    together, and bending necessarily swings both hydrogens together. An
    asymmetric geometry is not expressible here, which is exactly what we
    want: every point of every scan, including both axes of the surface,
    stays symmetric.

    Keeping the bisector along +z means the molecule stays centred and
    upright however it is deformed, which makes the animation easy to read.
    At angle_deg = 180 the hydrogens land at (+/-bond, 0) with Be between
    them -- genuinely linear, drawn horizontally.
    """
    half = math.radians(angle_deg) / 2
    dx, dz = bond * math.sin(half), bond * math.cos(half)
    return {"Be": (0.0, 0.0), "H1": (dx, dz), "H2": (-dx, dz)}


def geometry_string(bond: float, angle_deg: float) -> str:
    pos = beh2_positions(bond, angle_deg)
    return (f"Be 0 0 0; "
            f"H {pos['H1'][0]} 0 {pos['H1'][1]}; "
            f"H {pos['H2'][0]} 0 {pos['H2'][1]}")


# ===========================================================================
#  The quantum chemistry -- one geometry at a time
# ===========================================================================

def build_problem(bond: float, angle_deg: float,
                  active_electrons: int, active_orbitals: int) -> ElectronicStructureProblem:
    """BeH2 at this geometry, restricted to a CAS(active_electrons, active_orbitals)
    active space. See [2] for the transformer's orbital-selection rules --
    without an explicit ``active_orbitals`` list it takes the orbitals around
    the Fermi level, which for BeH2 means freezing the Be 1s core."""
    problem = PySCFDriver(atom=geometry_string(bond, angle_deg), basis=BASIS).run()
    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). Miss these and the
    energy comes out wrong by tens of Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_geometry(bond: float, angle_deg: float, active_electrons: int,
                     active_orbitals: int, max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one BeH2 geometry.

    ``theta0`` warm-starts the optimizer from a previous geometry's solution.
    Neighbouring points on a scan have very similar wavefunctions, so
    starting from the previous answer instead of from scratch cuts the number
    of energy evaluations -- which matters here, because evaluation count
    (not simulation speed) dominates the runtime.

    Note what "exact" means: the lowest eigenvalue of the *mapped,
    active-space-reduced* qubit Hamiltonian. It is exact within the chosen
    model, not a full-basis FCI energy. "hf" likewise is the Hartree-Fock
    circuit's expectation value on that same Hamiltonian, not PySCF's SCF
    energy. Both are therefore directly comparable with the VQE number, which
    is the point -- but neither is a benchmark-quality absolute energy.
    """
    problem = build_problem(bond, angle_deg, active_electrons, active_orbitals)
    mapper = ParityMapper(num_particles=problem.num_particles)
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                   initial_state=hf_circuit)
    # UCCSD's excitations arrive as PauliEvolutionGate black boxes, which the
    # estimator would otherwise matrix-exponentiate on every call. Transpile
    # once here, outside the optimization loop.
    ansatz_t = transpile(ansatz, basis_gates=["cx", "rz", "sx", "x", "h"], optimization_level=1)

    estimator = StatevectorEstimator()
    exact_electronic = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian).eigenvalue.real
    hf_electronic = float(estimator.run([(hf_circuit, hamiltonian, [])]).result()[0].data.evs)

    def cost(theta: np.ndarray) -> float:
        return float(estimator.run([(ansatz_t, hamiltonian, theta)]).result()[0].data.evs)

    # theta=0 is exactly the HF state -- the safe cold start.
    start = np.zeros(ansatz.num_parameters) if theta0 is None else np.asarray(theta0)
    if start.shape != (ansatz.num_parameters,):
        start = np.zeros(ansatz.num_parameters)
    result = minimize(cost, start, method="SLSQP", options={"maxiter": max_iterations})

    # Same row shape as the H2O script, so both write files the other could
    # read. The classical and quantum errors are not stored as columns --
    # they are differences of these three numbers, so they are derived where
    # they are reported instead of duplicated into every row.
    return {
        "bond": float(bond),
        "angle": float(angle_deg),
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
        "n_qubits": hamiltonian.num_qubits,
        "n_params": ansatz.num_parameters,
        "vqe_params": np.asarray(result.x),
    }


# ===========================================================================
#  Shared drawing constants
# ===========================================================================

ATOM_RADII = {"Be": 0.40, "H": 0.22}      # drawn radii, Angstrom
ATOM_COLORS = {"Be": "#c2ff00", "H": "#f2f2f2"}   # CPK-ish: beryllium is green


def _atom_element(name: str) -> str:
    """'H1' -> 'H', 'Be' -> 'Be'. Strips the trailing index off a label."""
    return name.rstrip("0123456789")


def _draw_molecule(ax, patches, bond_lines, row) -> None:
    """Move the atom circles and bond lines to match one scanned geometry."""
    pos = beh2_positions(row["bond"], row["angle"])
    for name, (x, z) in pos.items():
        patches[name].center = (x, z)
    for bond_line, h in zip(bond_lines, ("H1", "H2")):
        bond_line.set_data([pos["Be"][0], pos[h][0]], [pos["Be"][1], pos[h][1]])


def _build_molecule_panel(ax, span: float):
    """Set up the right-hand geometry panel. Returns (patches, bond_lines, label)."""
    ax.set_xlim(-span, span)
    ax.set_ylim(-span * 0.75, span * 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Geometry")
    bond_lines = [ax.plot([], [], "-", color="#888888", linewidth=3, zorder=1)[0]
                  for _ in range(2)]
    patches = {
        name: Circle((0, 0), ATOM_RADII[_atom_element(name)],
                     facecolor=ATOM_COLORS[_atom_element(name)],
                     edgecolor="black", linewidth=1.5, zorder=2)
        for name in ("Be", "H1", "H2")
    }
    for patch in patches.values():
        ax.add_patch(patch)
    label = ax.text(0, -span * 0.62, "", ha="center", fontsize=11)
    return patches, bond_lines, label


# ===========================================================================
#  Viewer 1 -- the 1D modes: live curve left, live molecule right
# ===========================================================================

class BeH2ScanViewer:
    """Live scan plot plus, once finished, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called per geometry as its energies come in
    -- the curve grows and the molecule redraws at the geometry just
    computed. Phase 2: :meth:`enable_review` turns the window into a
    scrubber over the finished scan.
    """

    def __init__(self, coordinate: str, fixed_label: str, max_extent: float) -> None:
        self.coordinate = coordinate
        plt.ion()
        self.fig, (self.ax_energy, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        # -- left panel: the energy curve ------------------------------
        (self.line_hf,) = self.ax_energy.plot([], [], "-", color="#d73027", label="Hartree-Fock")
        # Hollow VQE markers so the green exact crosses stay visible
        # underneath -- the two agree to well under a milli-Hartree, so a
        # filled marker would hide the exact curve entirely and make it look
        # as though it were never plotted.
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2, label="current")
        self.ax_energy.set_xlabel(
            "Be-H bond length (Angstrom)" if coordinate == "stretch" else "H-Be-H angle (degrees)"
        )
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title(f"BeH$_2$ {coordinate} scan   ({fixed_label})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        # -- right panel: the molecule ---------------------------------
        span = max_extent + 0.5
        self.atom_patches, self.bonds, self.mol_label = _build_molecule_panel(self.ax_mol, span)

        self._redraw()

    # -- phase 1: live during the scan ---------------------------------

    def _x_of(self, row: Dict[str, Any]) -> float:
        return row["bond"] if self.coordinate == "stretch" else row["angle"]

    def add_point(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        xs = [self._x_of(r) for r in self.rows]
        self.line_hf.set_data(xs, [r["hf"] for r in self.rows])
        self.line_vqe.set_data(xs, [r["vqe"] for r in self.rows])
        self.line_exact.set_data(xs, [r["exact"] for r in self.rows])
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        self.index = len(self.rows) - 1
        self._show(self.index)

    def _show(self, index: int) -> None:
        """Point the marker and the molecule at ``rows[index]``."""
        row = self.rows[index]
        self.marker.set_data([self._x_of(row)], [row["vqe"]])
        _draw_molecule(self.ax_mol, self.atom_patches, self.bonds, row)
        self.mol_label.set_text(
            f"r(Be-H) = {row['bond']:.2f} A     angle = {row['angle']:.1f}$\\degree$\n"
            f"E = {row['vqe']:.4f} Ha"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -- phase 2: review the finished scan ------------------------------

    def enable_review(self) -> None:
        # A slider needs a range. With a single-point scan (a smoke test, say)
        # its low and high bounds would be identical, which matplotlib warns
        # about and cannot render sensibly -- so skip it, since there is
        # nothing to scrub through anyway.
        if len(self.rows) > 1:
            slider_ax = self.fig.add_axes([0.12, 0.06, 0.5, 0.04])
            self.slider = Slider(slider_ax, "point", 0, len(self.rows) - 1,
                                 valinit=self.index, valstep=1)
            self.slider.on_changed(lambda v: self._show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax_energy.set_title(
            f"{self.ax_energy.get_title()}  --  LEFT/RIGHT arrows or slider"
        )
        print("\nScan complete. The window is now interactive:")
        print("  LEFT / RIGHT arrow keys  -- step backward / forward along the curve")
        print("  slider                   -- jump anywhere on the curve")
        print("  close the window         -- quit")
        plt.ioff()
        self._show(self.index)
        plt.show()

    def _on_key(self, event) -> None:
        if event.key == "right":
            new_index = min(self.index + 1, len(self.rows) - 1)
        elif event.key == "left":
            new_index = max(self.index - 1, 0)
        else:
            return
        self.index = new_index
        # Driving the slider re-enters _show() through on_changed, keeping
        # the slider handle and the plotted marker in sync.
        if self.slider is not None:
            self.slider.set_val(new_index)
        else:
            self._show(new_index)


# ===========================================================================
#  Viewer 2 -- the surface mode
# ===========================================================================

class BeH2SurfaceViewer:
    """Live filling-in energy map on the left, live molecule on the right.

    The left panel is a 2D heat map rather than a 3D surface *while the scan
    runs*, for a practical reason: re-rendering a ``plot_surface`` on every
    point is slow and, with most of the grid still empty, unreadable. Cells
    that have not been computed yet are left as NaN and drawn transparent, so
    you literally watch the map fill in. The real 3D surface is rendered once
    at the end by :meth:`show_surface`, which is also what gets saved for the
    report.
    """

    def __init__(self, bond_values: np.ndarray, angle_values: np.ndarray) -> None:
        self.bond_values = np.asarray(bond_values, dtype=float)
        self.angle_values = np.asarray(angle_values, dtype=float)
        # Z is indexed [angle, bond] so that imshow's rows run up the y axis.
        self.Z = np.full((len(self.angle_values), len(self.bond_values)), np.nan)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        plt.ion()
        self.fig, (self.ax_map, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        # -- left panel: the energy map --------------------------------
        # Half-step padding on each edge so that each grid point sits in the
        # centre of its own cell rather than on a corner.
        db = self._half_step(self.bond_values)
        da = self._half_step(self.angle_values)
        self.extent = (self.bond_values[0] - db, self.bond_values[-1] + db,
                       self.angle_values[0] - da, self.angle_values[-1] + da)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)          # uncomputed cells stay transparent
        self.im = self.ax_map.imshow(self.Z, origin="lower", extent=self.extent,
                                     aspect="auto", cmap=cmap, interpolation="nearest")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_map, label="Energy (Hartree)")
        (self.map_marker,) = self.ax_map.plot([], [], "D", color="white", markersize=9,
                                              fillstyle="none", markeredgewidth=2)
        self.ax_map.set_xlabel("Be-H bond length (Angstrom)   [both bonds, symmetric]")
        self.ax_map.set_ylabel("H-Be-H angle (degrees)   [symmetric bend]")
        self.ax_map.set_title("BeH$_2$ energy surface")

        # -- right panel: the molecule ---------------------------------
        span = float(self.bond_values[-1]) + 0.5
        self.atom_patches, self.bonds, self.mol_label = _build_molecule_panel(self.ax_mol, span)

        self._redraw()

    @staticmethod
    def _half_step(values: np.ndarray) -> float:
        """Half the grid spacing -- or a sane stand-in for a single-value axis."""
        if len(values) < 2:
            return 0.5
        return float(values[1] - values[0]) / 2

    def _cell_of(self, row: Dict[str, Any]) -> Tuple[int, int]:
        """Which grid cell a row belongs in, matched by nearest value.

        Matching by nearest rather than by exact equality because the bond and
        angle values have been through a float round-trip via JSON, so `==`
        against the regenerated axis is not reliable.
        """
        j = int(np.argmin(np.abs(self.bond_values - row["bond"])))
        i = int(np.argmin(np.abs(self.angle_values - row["angle"])))
        return i, j

    # -- phase 1: live during the scan ---------------------------------

    def add_point(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        i, j = self._cell_of(row)
        self.Z[i, j] = row["vqe"]
        self.im.set_data(np.ma.masked_invalid(self.Z))
        finite = self.Z[np.isfinite(self.Z)]
        if finite.size:
            self.im.set_clim(float(finite.min()), float(finite.max()))
        self.index = len(self.rows) - 1
        self._show(self.index)

    def _show(self, index: int) -> None:
        row = self.rows[index]
        self.map_marker.set_data([row["bond"]], [row["angle"]])
        _draw_molecule(self.ax_mol, self.atom_patches, self.bonds, row)
        self.mol_label.set_text(
            f"r(Be-H) = {row['bond']:.2f} A     angle = {row['angle']:.1f}$\\degree$\n"
            f"E = {row['vqe']:.4f} Ha"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -- phase 2: the 3D surface, and review ----------------------------

    def show_surface(self, png_path: Path | None = None) -> None:
        """Render the finished grid as a real 3D surface beside a contour map.

        Two panels because they answer different questions: the 3D view shows
        the shape of the valley, the contour view shows the anisotropy (widely
        spaced lines along the soft angle axis, tightly packed along the stiff
        bond axis) and is far easier to read a number off.
        """
        grid_bond, grid_angle = np.meshgrid(self.bond_values, self.angle_values)
        masked = np.ma.masked_invalid(self.Z)

        fig = plt.figure(figsize=(13, 5.5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot_surface(grid_bond, grid_angle, masked, cmap="viridis",
                          linewidth=0, antialiased=True, alpha=0.95)
        ax3d.set_xlabel("r(Be-H) (A)")
        ax3d.set_ylabel("angle (deg)")
        ax3d.set_zlabel("Energy (Ha)")
        ax3d.set_title("BeH$_2$ ground-state surface (VQE)")
        # Looking along the trough rather than across it, so the valley reads
        # as a valley instead of a wall.
        ax3d.view_init(elev=28, azim=-135)

        ax2d = fig.add_subplot(1, 2, 2)
        levels = 25
        contour = ax2d.contourf(grid_bond, grid_angle, masked, levels=levels, cmap="viridis")
        ax2d.contour(grid_bond, grid_angle, masked, levels=levels,
                     colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
        if self.rows:
            best = min(self.rows, key=lambda r: r["vqe"])
            ax2d.plot([best["bond"]], [best["angle"]], "r*", markersize=16,
                      label=f"min {best['vqe']:.4f} Ha")
            ax2d.legend(loc="lower left")
        ax2d.set_xlabel("r(Be-H) (Angstrom)")
        ax2d.set_ylabel("H-Be-H angle (degrees)")
        ax2d.set_title("Contours -- note the wide spacing along the angle axis")

        fig.tight_layout()
        if png_path is not None:
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"saved {display_path(png_path)}")

    def enable_review(self) -> None:
        # A slider needs a range. With a single-point scan (a smoke test, say)
        # its low and high bounds would be identical, which matplotlib warns
        # about and cannot render sensibly -- so skip it, since there is
        # nothing to scrub through anyway.
        if len(self.rows) > 1:
            slider_ax = self.fig.add_axes([0.12, 0.06, 0.5, 0.04])
            self.slider = Slider(slider_ax, "point", 0, len(self.rows) - 1,
                                 valinit=self.index, valstep=1)
            self.slider.on_changed(lambda v: self._show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax_map.set_title(f"{self.ax_map.get_title()}  --  LEFT/RIGHT arrows or slider")
        print("\nScan complete. The window is now interactive:")
        print("  LEFT / RIGHT arrow keys  -- step through the grid points in scan order")
        print("  slider                   -- jump anywhere in the grid")
        print("  close the windows        -- quit")
        plt.ioff()
        self._show(self.index)
        plt.show()

    def _on_key(self, event) -> None:
        if event.key == "right":
            new_index = min(self.index + 1, len(self.rows) - 1)
        elif event.key == "left":
            new_index = max(self.index - 1, 0)
        else:
            return
        self.index = new_index
        if self.slider is not None:
            self.slider.set_val(new_index)
        else:
            self._show(new_index)


# ===========================================================================
#  Scan drivers
# ===========================================================================

def inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    """``start`` to ``stop`` inclusive.

    The ``+ step/2`` is what makes ``stop`` itself land in the array despite
    floating-point drift -- the same trick the H2 and H2O scripts use.
    """
    return np.arange(start, stop + step / 2, step)


def serpentine(bond_values: np.ndarray, angle_values: np.ndarray) -> List[Tuple[float, float]]:
    """Grid geometries ordered so consecutive points are always adjacent.

    Scanning every row left-to-right would make the optimizer jump from the
    right-hand edge of one row back to the left-hand edge of the next, where
    the warm start is worthless. Reversing alternate rows -- boustrophedon,
    the way an ox ploughs a field -- keeps every step a small one.
    """
    geometries: List[Tuple[float, float]] = []
    for i, angle in enumerate(angle_values):
        row_bonds = bond_values if i % 2 == 0 else bond_values[::-1]
        geometries.extend((float(b), float(angle)) for b in row_bonds)
    return geometries


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BeH2 ground-state energy by VQE (local simulator), scanned along its "
                    "symmetric stretch, its symmetric bend, or both at once as a surface."
    )
    parser.add_argument("--coordinate", choices=["stretch", "bend", "surface"], default="stretch",
                        help="'stretch': vary both Be-H bonds together at fixed angle (default). "
                             "'bend': vary the H-Be-H angle at fixed bond length. "
                             "'surface': vary both on a grid. Every mode is symmetric.")
    parser.add_argument("--start", type=float, default=None,
                        help="scan start -- Angstrom for stretch/surface (default 0.9), "
                             "degrees for bend (default 90)")
    parser.add_argument("--stop", type=float, default=None,
                        help="scan stop, inclusive -- default 2.4 A for stretch/surface, "
                             "180 deg for bend")
    parser.add_argument("--step", type=float, default=None,
                        help="scan step -- default 0.1 A for stretch, 5 deg for bend, "
                             "0.15 A for the surface's bond axis")
    parser.add_argument("--angle-start", type=float, default=90.0,
                        help="[surface mode] first H-Be-H angle in degrees (default 90)")
    parser.add_argument("--angle-stop", type=float, default=180.0,
                        help="[surface mode] last H-Be-H angle in degrees (default 180)")
    parser.add_argument("--angle-step", type=float, default=10.0,
                        help="[surface mode] angle step in degrees (default 10)")
    parser.add_argument("--angle", type=float, default=EQUILIBRIUM_ANGLE,
                        help=f"[stretch mode] fixed H-Be-H angle in degrees "
                             f"(default {EQUILIBRIUM_ANGLE}, i.e. linear)")
    parser.add_argument("--bond", type=float, default=EQUILIBRIUM_BOND,
                        help=f"[bend mode] fixed Be-H bond length in Angstrom "
                             f"(default {EQUILIBRIUM_BOND})")
    parser.add_argument("--active-electrons", type=int, default=4,
                        help="electrons in the active space (default 4)")
    parser.add_argument("--active-orbitals", type=int, default=4,
                        help="spatial orbitals in the active space (default 4 -> 6 qubits). "
                             "Larger is more accurate but much slower, and a surface pays "
                             "for it N^2 times -- see the module docstring.")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each geometry's optimization from the Hartree-Fock state "
                             "instead of warm-starting from the previous geometry's solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json written by an earlier run) straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: beh2_<coordinate>_scan.json/.csv "
                             "next to this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save the results of this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="exit when the scan finishes instead of staying open in "
                             "review mode. The live plot still draws as the scan runs; it "
                             "just doesn't block at the end waiting for you to close the "
                             "window. Use this to run several scans back to back "
                             "unattended -- everything is still saved, and --load reopens "
                             "it interactively later.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # -- replay a finished scan, no VQE at all -------------------------
    if args.load:
        _replay(Path(args.load), here, args.no_show)
        return

    if args.coordinate == "surface":
        _run_surface(args, parser, here)
    else:
        _run_curve(args, parser, here)


def _finish(viewer, no_show: bool) -> None:
    """Either hand the window over for review, or return so the process exits.

    ``enable_review`` calls ``plt.show()``, which blocks until the window is
    closed -- fine when you are sitting there, useless when you have queued
    four scans and gone to bed.
    """
    if no_show:
        print("\nScan complete. --no-show was passed, so exiting instead of waiting.")
        return
    viewer.enable_review()


def _load_csv(path: Path) -> tuple:
    """Read a scan back from its ``.csv`` instead of its ``.json``.

    The JSON is the canonical save -- it carries the metadata needed to rebuild
    the plot. The CSV is a flat table with no metadata at all, so everything
    the viewer needs has to be *inferred* from the columns:

      scan type   both bond and angle varying  -> surface, else a curve
      coordinate  whichever of the two actually moves
      grid axes   the sorted unique values of each column

    That inference is exact for these three scan shapes, so a CSV round-trips
    faithfully. What is genuinely lost is the active space (CAS(n,m) no longer
    appears in the title) and ``vqe_params``. Neither matters for replay --
    warm starts are only used while scanning.

    Worth having because the CSV is the file that ends up in a report, a
    spreadsheet or a git diff, and being able to reopen the plot straight from
    it beats having to keep the JSON alongside.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")

    numeric = ("bond", "angle", "hf", "vqe", "exact")
    rows = []
    for r in raw:
        row = {k: float(v) for k, v in r.items() if k in numeric and v != ""}
        for k in ("n_qubits", "n_params"):
            if r.get(k):
                row[k] = int(float(r[k]))
        rows.append(row)

    missing = {"bond", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(
            f"{display_path(path)} is missing the column(s) {sorted(missing)}, so it is "
            f"not a BeH2 scan CSV. Expected columns: bond, angle, hf, vqe, exact."
        )

    bonds = sorted({r["bond"] for r in rows})
    angles = sorted({r["angle"] for r in rows})

    if len(bonds) > 1 and len(angles) > 1:
        metadata = {
            "scan_type": "surface", "coordinate": "surface",
            "bond_values": bonds, "angle_values": angles,
        }
    elif len(bonds) > 1:
        metadata = {
            "scan_type": "curve", "coordinate": "stretch",
            "fixed_label": f"angle fixed at {angles[0]}$\\degree$",
            "max_extent": bonds[-1],
        }
    else:
        metadata = {
            "scan_type": "curve", "coordinate": "bend",
            "fixed_label": f"r(Be-H) fixed at {bonds[0]} A",
            "max_extent": bonds[0],
        }
    return metadata, rows


def _load_any(load_path: Path, here: Path) -> tuple:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    candidates = [load_path, here / load_path.name]
    if load_path.suffix.lower() == ".csv":
        for candidate in candidates:
            if candidate.exists():
                print(f"reading the CSV (metadata inferred from the columns; "
                      f"the .json carries it explicitly)")
                return _load_csv(candidate)
        looked = "\n  ".join(display_path(c) for c in candidates)
        raise SystemExit(f"no scan CSV found. Looked in:\n  {looked}")
    return load_scan(load_path, default_dir=here)


def _replay(load_path: Path, here: Path, no_show: bool = False) -> None:
    """Reopen a saved scan in review mode without running any chemistry.

    ``scan_type`` tells us which viewer to build. It defaults to "curve" for
    files written before the surface mode existed, so older saves still load.
    """
    metadata, rows = _load_any(load_path, here)
    scan_type = metadata.get("scan_type", "curve")
    # The active space is metadata, so a CSV-sourced scan has none. Report the
    # qubit count from the rows instead of printing "CAS(None, None)".
    if metadata.get("active_electrons") is not None:
        space = f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
    elif rows and rows[0].get("n_qubits"):
        space = f"{rows[0]['n_qubits']} qubits, active space not recorded in the csv"
    else:
        space = "active space not recorded"
    print(f"loaded {len(rows)} points from {load_path} "
          f"({metadata.get('coordinate')} scan, {space}) -- no VQE re-run")

    if scan_type == "surface":
        viewer = BeH2SurfaceViewer(np.asarray(metadata["bond_values"], dtype=float),
                                   np.asarray(metadata["angle_values"], dtype=float))
        for row in rows:
            viewer.add_point(row)
        viewer.show_surface()
        _finish(viewer, no_show)
    else:
        viewer = BeH2ScanViewer(metadata["coordinate"], metadata["fixed_label"],
                                float(metadata["max_extent"]))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, no_show)


def _run_curve(args, parser, here: Path) -> None:
    """The 1D modes: stretch (bonds move together) or bend (angle moves)."""
    if args.coordinate == "stretch":
        start = 0.9 if args.start is None else args.start
        stop = 2.4 if args.stop is None else args.stop
        step = 0.1 if args.step is None else args.step
        unit = "A"
    else:
        start = 90.0 if args.start is None else args.start
        stop = 180.0 if args.stop is None else args.stop
        step = 5.0 if args.step is None else args.step
        unit = "deg"

    if step <= 0:
        parser.error("--step must be positive")
    if stop < start:
        parser.error("--stop must be >= --start")

    values = inclusive_range(start, stop, step)

    if args.coordinate == "stretch":
        geometries = [(float(v), args.angle) for v in values]
        fixed_label = f"angle fixed at {args.angle}$\\degree$"
        max_extent = float(values[-1])
    else:
        geometries = [(args.bond, float(v)) for v in values]
        fixed_label = f"r(Be-H) fixed at {args.bond} A"
        max_extent = float(args.bond)

    print(f"BeH2 {args.coordinate} scan: {len(values)} points from {values[0]:.2f} to "
          f"{values[-1]:.2f} {unit} (step {step} {unit})")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    print(f"estimated runtime: ~{len(values) * 5 / 60:.0f} min "
          f"(~5s per point; the plot updates as each point lands)")

    viewer = BeH2ScanViewer(args.coordinate, fixed_label, max_extent)
    rows = _scan(geometries, args, viewer, unit, label_index=0 if args.coordinate == "stretch" else 1)

    best = min(rows, key=lambda r: r["vqe"])
    best_x = best["bond"] if args.coordinate == "stretch" else best["angle"]
    reference = (f"literature r(Be-H) for linear BeH2 is ~{EQUILIBRIUM_BOND} A"
                 if args.coordinate == "stretch"
                 else "BeH2 is linear, so the minimum belongs at 180 deg")
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at {best_x:.2f} {unit}  ({reference})")
    _report_errors(rows)

    if args.coordinate == "stretch":
        _report_monotonicity(rows)

    if not args.no_save:
        target = resolve_target(args.save, here, f"beh2_{args.coordinate}_scan")
        metadata = {
            "molecule": "BeH2", "basis": BASIS,
            "scan_type": "curve",
            "coordinate": args.coordinate,
            "symmetric": True,
            "start": float(values[0]), "stop": float(values[-1]), "step": float(step),
            "unit": unit,
            "fixed_angle": float(args.angle), "fixed_bond": float(args.bond),
            "fixed_label": fixed_label,
            "max_extent": float(max_extent),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
        }
        _save(target, metadata, rows)

    _finish(viewer, args.no_show)


def _run_surface(args, parser, here: Path) -> None:
    """The 2D mode: bond and angle both varied, both symmetrically."""
    bond_start = 0.9 if args.start is None else args.start
    bond_stop = 2.4 if args.stop is None else args.stop
    bond_step = 0.15 if args.step is None else args.step

    if bond_step <= 0 or args.angle_step <= 0:
        parser.error("--step and --angle-step must be positive")
    if bond_stop < bond_start:
        parser.error("--stop must be >= --start")
    if args.angle_stop < args.angle_start:
        parser.error("--angle-stop must be >= --angle-start")

    bond_values = inclusive_range(bond_start, bond_stop, bond_step)
    angle_values = inclusive_range(args.angle_start, args.angle_stop, args.angle_step)
    geometries = serpentine(bond_values, angle_values)

    print(f"BeH2 surface scan: {len(bond_values)} bond x {len(angle_values)} angle "
          f"= {len(geometries)} points")
    print(f"  bond  {bond_values[0]:.2f} to {bond_values[-1]:.2f} A (step {bond_step} A) "
          f"-- both Be-H bonds, symmetric")
    print(f"  angle {angle_values[0]:.1f} to {angle_values[-1]:.1f} deg "
          f"(step {args.angle_step} deg) -- symmetric bend")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    print(f"estimated runtime: ~{len(geometries) * 5 / 60:.0f} min "
          f"(~5s per point; the map fills in as each point lands)")

    viewer = BeH2SurfaceViewer(bond_values, angle_values)
    rows = _scan(geometries, args, viewer, "A", label_index=None)

    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at "
          f"r(Be-H) = {best['bond']:.2f} A, angle = {best['angle']:.1f} deg")
    print(f"  (literature: r ~{EQUILIBRIUM_BOND} A and linear, i.e. 180 deg)")
    _report_errors(rows)

    png_path = None
    if not args.no_save:
        target = resolve_target(args.save, here, "beh2_surface_scan")
        metadata = {
            "molecule": "BeH2", "basis": BASIS,
            "scan_type": "surface",
            "coordinate": "surface",
            "symmetric": True,
            "bond_values": [float(v) for v in bond_values],
            "angle_values": [float(v) for v in angle_values],
            "bond_step": float(bond_step), "angle_step": float(args.angle_step),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
        }
        _save(target, metadata, rows)
        png_path = target.with_name(target.stem + "_3d.png")

    viewer.show_surface(png_path)
    _finish(viewer, args.no_show)


def _scan(geometries, args, viewer, unit: str, label_index) -> List[Dict[str, Any]]:
    """Run VQE over a list of (bond, angle) geometries, feeding the viewer.

    ``label_index`` picks which coordinate to print per line: 0 for bond,
    1 for angle, or None to print both (the surface case).
    """
    rows: List[Dict[str, Any]] = []
    previous_params = None
    total = len(geometries)
    for i, (bond, angle) in enumerate(geometries):
        t0 = time.perf_counter()
        row = run_one_geometry(bond, angle, args.active_electrons, args.active_orbitals,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        if label_index is None:
            where = f"r={bond:5.2f} A a={angle:6.1f} deg"
        elif label_index == 0:
            where = f"{bond:6.2f} {unit}"
        else:
            where = f"{angle:6.1f} {unit}"
        print(f"  [{i + 1:4d}/{total}] {where}   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({time.perf_counter() - t0:.1f}s)")
    return rows


def _report_errors(rows: List[Dict[str, Any]]) -> None:
    """The classical-vs-quantum error summary, in milli-Hartree.

    These two numbers measure different things and it is worth keeping them
    apart. |HF - exact| is *physics*: the correlation energy a single Slater
    determinant cannot represent, which is what makes a correlated method
    necessary at all. |VQE - exact| is *numerics*: how close the optimizer
    got to the exact answer for this Hamiltonian. A large HF error is the
    expected, interesting result; a large VQE error means the optimization
    struggled.

    Both are derived here rather than stored per row, which keeps the saved
    row shape identical to the H2O script's -- they are just differences of
    the hf/vqe/exact columns already in the file.
    """
    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    print(f"largest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")
    if max_vqe * 1000 > 1.0:
        print("  note: |VQE - exact| above 1 mHa suggests the optimizer did not fully "
              "converge somewhere -- try --cold-start or a finer --step")


def _report_monotonicity(rows: List[Dict[str, Any]]) -> None:
    """Flag the active-space failure mode described in the module docstring.

    Past its minimum a dissociation curve must rise as the atoms separate. A
    curve that turns back downward is the signature of an active space too
    small to describe the bonds breaking -- the H2O script documents the same
    trap. Worth catching automatically, because it is easy to miss by eye on
    a coarse scan and it invalidates everything downstream.
    """
    ordered = sorted(rows, key=lambda r: r["bond"])
    best_at = min(range(len(ordered)), key=lambda i: ordered[i]["exact"])
    dips = [
        (ordered[i]["bond"], ordered[i + 1]["bond"])
        for i in range(best_at, len(ordered) - 1)
        if ordered[i + 1]["exact"] < ordered[i]["exact"]
    ]
    if dips:
        print("\n  WARNING: the exact curve dips downward past its minimum, between "
              + ", ".join(f"{a:.2f}-{b:.2f} A" for a, b in dips))
        print("  A ground-state curve must rise monotonically toward dissociation. This is")
        print("  an active space too small to describe two breaking bonds, not a VQE failure.")
        print("  Try a larger --active-orbitals / --active-electrons.")
    else:
        print("  curve rises monotonically past its minimum -- active space looks adequate")


def _save(target: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    json_path, csv_path = save_scan(target, metadata, rows)
    print(f"saved {display_path(json_path)}")
    print(f"saved {display_path(csv_path)}")
    print(f"reopen it later without recomputing:  "
          f"python {display_path(Path(__file__))} --load {json_path.name}")


if __name__ == "__main__":
    main()
