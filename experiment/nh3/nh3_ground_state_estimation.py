"""
nh3_ground_state_estimation.py -- ammonia's ground-state energy by VQE along
its symmetric internal coordinates, plotted live next to a picture of the
molecule flexing.

Simulator only. No IBM account, no queue, no quota.

DEGREES OF FREEDOM -- WHY THIS SCRIPT HAS A SURFACE AND LiH DOES NOT
----------------------------------------------------------------------
NH3 has 4 atoms and is not linear (it is a C3v pyramid), so it has
3N - 6 = 3(4) - 6 = **6** internal coordinates: three N-H bond lengths and
three H-N-H angles.

Six is far too many to scan. Symmetry cuts it down: forcing all three N-H
bonds to stay equal and all three H-N-H angles to stay equal -- keeping the
molecule C3v throughout -- collapses six coordinates into **2**:

    r       the symmetric N-H stretch (all three bonds together)
    theta   the H-N-H angle (the umbrella bend, all three together)

Two free coordinates is exactly what it takes to make a surface, so this
script produces **two curves and one surface**, the same shape of job as
BeH2. Contrast LiH and O2, which are diatomic, have one coordinate each, and
cannot have a surface at all.

THE BEND HAS A HARD CEILING AT 120 DEGREES
---------------------------------------------
This is the one geometric trap specific to NH3 and it is worth stating
before you pass ``--angle-stop 180`` out of habit from the BeH2 script.

Placing three equivalent H atoms at angle theta from each other requires

    sin^2(beta) = (1 - cos theta) / 1.5

where beta is the angle between an N-H bond and the C3 axis. For that to
have a solution, cos(theta) >= -0.5, i.e. **theta <= 120 degrees**. At
exactly 120 degrees beta is 90 degrees and the molecule is **planar** --
that is the inversion transition state, and it is the geometric limit.
Beyond it there is no C3v geometry at all; the formula asks for the square
root of a negative number.

So BeH2's bend runs to 180 degrees (linear) while NH3's stops at 120
(planar). Going "past" 120 is not a wider scan, it is the mirror-image
molecule coming back down the other side of the inversion. The CLI refuses
angles above 120 rather than silently producing nonsense.

WHAT THE THREE SCANS SHOW
---------------------------
  --coordinate stretch   All three N-H bonds pulled together, angle held at
                         --angle. A standard dissociation curve: repulsive
                         wall, minimum near 1.04 A, then a climb as three
                         bonds break at once. Breaking three bonds
                         simultaneously is much harder to describe than
                         breaking one, which is what makes the active-space
                         question below sharp.

  --coordinate bend      The umbrella. theta swept from pyramidal up to
                         planar at 120 degrees, bonds held at --bond. The
                         observable here is not the minimum but the
                         **inversion barrier**: the energy cost of flattening
                         the molecule, which is what lets ammonia tunnel
                         between its two mirror forms.

  --coordinate surface   Both together on a grid, walked in serpentine order
                         with a contour map filling in live and the real 3D
                         surface rendered at the end.

TWO HONEST WARNINGS ABOUT WHAT THESE NUMBERS MEAN
----------------------------------------------------
**1. STO-3G gets the inversion barrier badly wrong.** Measured on the
full-valence CAS(8,7) exact column at r = 1.0124 A:

    E(planar, 120 deg) - E(minimum)  =  22.4 mHa  =  4914 cm-1

The experimental barrier is about **1786 cm-1 (8.1 mHa)** [2]. So the model
is roughly **2.8x too high**. That is a basis-set failure -- STO-3G
over-pyramidalises ammonia -- and it is present before VQE is involved at
all. The shape of the bend curve is qualitatively right and worth
interpolating; the barrier height is not a number to quote as a prediction.
Report it as "the model's barrier", not "ammonia's barrier".

The same caution applies to the geometry. On the default CAS(6,5):

    quantity            model      experiment [2]
    r(N-H)              1.0401 A   1.0124 A
    theta(H-N-H)        104.27 deg 106.67 deg  (at the experimental r)

Both are the basis set's error, not the interpolant's. The interpolation
folders score themselves against the **model** values above for exactly this
reason -- see the note there.

**2. For NH3 the monotonicity check is NOT sufficient to catch a bad active
space.** This differs from LiH, where CAS(2,3) announced itself by dipping
downward past dissociation. Measured on the symmetric stretch at
theta = 106.67, exact column:

    r(N-H)    CAS(4,4)     CAS(6,5)     CAS(6,6)     CAS(8,7)
    1.10     -55.453313   -55.466408   -55.493731   -55.520704
    1.55     -55.143144   -55.190412   -55.279727   -55.304853
    2.00     -54.864875   -54.946850   -55.144606   -55.158760
    2.45     -54.719775   -54.810908   -55.120404   -55.125587

**Every one of those is monotonic.** None of them dips. Yet CAS(4,4) is
406 mHa away from full valence at its worst and CAS(6,5) is 315 mHa away.
The small spaces are smoothly, quietly wrong rather than visibly broken.

The reason is chemical: the symmetric stretch breaks three N-H bonds at
once, which needs three bonding and three antibonding orbitals to describe
-- CAS(6,6) at minimum. CAS(6,6) is within 30 mHa of full valence
everywhere; anything smaller is not.

The monotonicity check still runs, because it catches a different failure
and costs nothing. But do not read "curve rises monotonically" as "active
space is fine" here. If you are stretching far, spot-check against a larger
space. The script prints a reminder when the stretch goes past 1.6 A.

ACTIVE SPACE AND COST
-----------------------
NH3 in STO-3G is 10 electrons in 8 spatial orbitals (N gives 1s/2s/2p, each
H gives 1s) -- 16 qubits untruncated, which is not happening here.

    active space   qubits   UCCSD params   optimizer   ~seconds/point
    CAS(4,4)          6          26          SLSQP          9
    CAS(6,5)          8          54          SLSQP         43     <-- default
    CAS(6,6)         10         117          COBYLA       slow
    CAS(8,7)         12         204          COBYLA       very slow

The default is **CAS(6,5)** -- 8 qubits, 54 UCCSD parameters, the same size
as the H2O script's active space. It puts the minimum at the right angle
(105 deg on a coarse grid, against CAS(4,4)'s 100) and it is affordable.

Those timings are from a container CPU; yours will differ, and every scan
prints seconds per point as it runs. Rough guide at CAS(6,5): the default
stretch scan (11 points) and bend scan (9 points) are a few minutes each,
the default 7x7 surface (49 points) is around half an hour.

**If the surface is too slow, pass ``--active-electrons 4 --active-orbitals
4``.** That is 6 qubits and 26 parameters, roughly five times faster, and
good enough for the *shape* of the valley -- but it shifts the minimum by
about 5 degrees and sits ~100 mHa lower-quality in absolute terms. Fine for
a picture, not for a quoted geometry.

Optimizer is chosen from the parameter count: SLSQP at or below 60
parameters, COBYLA above, since SLSQP's finite-difference gradient costs
about one energy evaluation per parameter per step. ``--optimizer``
overrides. COBYLA spends its whole ``--max-iterations`` budget rather than
converging early, so above 60 parameters the cap sets the cost per point.

Ammonia's ground state is a closed-shell singlet (X 1-A-1, 10 electrons), so
the driver's default ``spin=0`` and RHF are correct. That was checked, not
assumed -- see the O2 script for what happens when it is assumed wrongly.

REFERENCES
-----------
[1] ActiveSpaceTransformer, and its default of taking orbitals around the
    Fermi level:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html
[2] NIST CCCBDB, experimental geometry and vibrational data for NH3
    (r = 1.0124 A, theta = 106.67 deg; inversion barrier ~1786 cm-1):
    https://cccbdb.nist.gov/exp2x.asp?casno=7664417&charge=0
[3] PySCFDriver:
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.drivers.PySCFDriver.html

USAGE
------
  STRETCH -- all three N-H bonds together, angle fixed
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate stretch
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate stretch --start 0.85 --stop 1.6 --step 0.075
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate stretch --angle 120   # planar slice
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate stretch --start 1.0 --stop 1.1 --step 0.05 --no-save

  BEND -- the umbrella, bonds fixed. 120 degrees is planar and is the ceiling.
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate bend
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate bend --angle-start 96 --angle-stop 120 --angle-step 3
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate bend --bond 1.04

  SURFACE -- both coordinates on a grid, serpentine walk, live contour
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate surface
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate surface --active-electrons 4 --active-orbitals 4
    python experiment/nh3/nh3_ground_state_estimation.py --coordinate surface --step 0.25 --angle-step 8 --no-save   # quick 4x4

  RELOAD -- no VQE re-run. Either format.
    python experiment/nh3/nh3_ground_state_estimation.py --load nh3_stretch_scan.json
    python experiment/nh3/nh3_ground_state_estimation.py --load nh3_surface_scan.csv

  OTHER KNOBS
    --active-electrons / --active-orbitals    default CAS(6,5)
    --optimizer auto|slsqp|cobyla             default auto, by parameter count
    --max-iterations N                        optimizer cap
    --cold-start                              no warm starting
    --save NAME / --no-save / --no-show
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
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

# Experimental geometry [2].
EXPERIMENTAL_BOND = 1.0124    # Angstrom
EXPERIMENTAL_ANGLE = 106.67   # degrees
EXPERIMENTAL_BARRIER_MHA = 8.14   # inversion barrier, ~1786 cm-1

# What the default CAS(6,5)/STO-3G model actually produces, from fine
# reference scans of the exact column. These -- not the experimental values
# -- are what an interpolant built on this data can be scored against.
#
# THERE ARE THREE OF THESE, NOT TWO, AND THE DISTINCTION IS NOT PEDANTIC.
# A curve scan holds one coordinate fixed, and the minimum of the other
# depends on where you fixed it. An earlier version of this file used a
# single pair of numbers for all three cases and was wrong by 1.6 degrees on
# the bend and 0.01 A on the surface -- errors comparable to what the
# interpolation is being asked to measure, which would have quietly
# corrupted the result.
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

# Hard geometric limit: three equivalent H atoms cannot subtend more than
# 120 degrees. At exactly 120 the molecule is planar. See the docstring.
PLANAR_ANGLE = 120.0

# Above this many UCCSD parameters SLSQP's finite-difference gradient stops
# paying for itself and COBYLA is cheaper.
SLSQP_PARAMETER_LIMIT = 60

MHA_PER_WAVENUMBER = 1000.0 / 219474.63   # mHa per cm-1, for barrier reporting


# ===========================================================================
#  Geometry
# ===========================================================================

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


def geometry_string(bond: float, angle_deg: float) -> str:
    """C3v ammonia: N at the origin, three H at radius ``bond``, each pair
    subtending ``angle_deg``.

    The three H sit on a cone about the z axis at azimuths 0, 120, 240
    degrees. If beta is the angle between an N-H bond and that axis, the
    H-N-H angle satisfies

        cos(theta) = cos^2(beta) - 0.5 sin^2(beta) = 1 - 1.5 sin^2(beta)

    so sin^2(beta) = (1 - cos theta) / 1.5. That is only solvable for
    theta <= 120 degrees; at 120 exactly, beta = 90 and the molecule is
    planar. See the module docstring.
    """
    if angle_deg > PLANAR_ANGLE + 1e-9:
        raise ValueError(
            f"H-N-H angle {angle_deg:.2f} deg is impossible: three equivalent H atoms "
            f"cannot subtend more than {PLANAR_ANGLE} degrees (that is the planar "
            f"geometry). See the module docstring."
        )
    sin2 = (1.0 - math.cos(math.radians(angle_deg))) / 1.5
    sin2 = min(max(sin2, 0.0), 1.0)      # clamp float drift exactly at 120
    sin_b, cos_b = math.sqrt(sin2), math.sqrt(1.0 - sin2)

    atoms = ["N 0 0 0"]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        atoms.append(f"H {bond * sin_b * math.cos(phi)} "
                     f"{bond * sin_b * math.sin(phi)} {bond * cos_b}")
    return "; ".join(atoms)


def projected_atoms(bond: float, angle_deg: float) -> List[Tuple[float, float, float]]:
    """The same geometry flattened to 2D for the viewer panel.

    A straight x-z projection would stack two of the hydrogens exactly on top
    of each other, so this uses a light axonometric shear (x - 0.5y, z + 0.3y)
    which separates them and reads as a pyramid. Returns (X, Y, depth), with
    depth used only to decide draw order.
    """
    sin2 = min(max((1.0 - math.cos(math.radians(angle_deg))) / 1.5, 0.0), 1.0)
    sin_b, cos_b = math.sqrt(sin2), math.sqrt(1.0 - sin2)
    out = [(0.0, 0.0, 0.0)]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        x, y, z = (bond * sin_b * math.cos(phi),
                   bond * sin_b * math.sin(phi),
                   bond * cos_b)
        out.append((x - 0.5 * y, z + 0.3 * y, y))
    return out


# ===========================================================================
#  The quantum chemistry -- one geometry at a time
# ===========================================================================

def build_problem(bond: float, angle_deg: float, active_electrons: int,
                  active_orbitals: int) -> ElectronicStructureProblem:
    """NH3 at this geometry, restricted to a CAS active space.

    No ``spin`` or ``method`` argument: ammonia's ground state is a
    closed-shell singlet, so the driver defaults (spin=0, RHF) are right.
    Without an explicit orbital list the transformer takes the orbitals
    around the Fermi level [1].
    """
    problem = PySCFDriver(atom=geometry_string(bond, angle_deg), basis=BASIS).run()
    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy.

    Adds every constant qiskit-nature tracks: nuclear repulsion plus the
    inactive-orbital energy the active-space transformer folded away. For
    NH3 the frozen orbitals are worth tens of Hartree, so omitting this is
    not a subtle error.
    """
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def choose_optimizer(requested: str, n_params: int) -> str:
    """Resolve ``--optimizer auto`` against the parameter count."""
    if requested != "auto":
        return requested
    return "slsqp" if n_params <= SLSQP_PARAMETER_LIMIT else "cobyla"


def run_one_geometry(bond: float, angle_deg: float, active_electrons: int,
                     active_orbitals: int, optimizer: str = "auto",
                     max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one NH3 geometry.

    ``theta0`` warm-starts from the previous geometry's solution. "exact" is
    the lowest eigenvalue of the mapped, active-space-reduced Hamiltonian --
    exact within this model, not a full-basis FCI energy.
    """
    problem = build_problem(bond, angle_deg, active_electrons, active_orbitals)
    mapper = ParityMapper(num_particles=problem.num_particles)
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                   initial_state=hf_circuit)
    # UCCSD's excitations are PauliEvolutionGate black boxes that would be
    # matrix-exponentiated on every energy evaluation. Transpile once, here,
    # outside the optimizer loop.
    ansatz_t = transpile(ansatz, basis_gates=["cx", "rz", "sx", "x", "h"],
                         optimization_level=1)

    estimator = StatevectorEstimator()
    exact_electronic = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
        hamiltonian).eigenvalue.real
    hf_electronic = float(estimator.run([(hf_circuit, hamiltonian, [])]).result()[0].data.evs)

    def cost(theta: np.ndarray) -> float:
        return float(estimator.run([(ansatz_t, hamiltonian, theta)]).result()[0].data.evs)

    start = np.zeros(ansatz.num_parameters) if theta0 is None else np.asarray(theta0)
    if start.shape != (ansatz.num_parameters,):
        start = np.zeros(ansatz.num_parameters)

    resolved = choose_optimizer(optimizer, ansatz.num_parameters)
    method = {"cobyla": "COBYLA", "slsqp": "SLSQP"}[resolved]
    result = minimize(cost, start, method=method, options={"maxiter": max_iterations})

    # Both coordinates are recorded on every row even though only one moves
    # in a curve scan -- it keeps the CSV shape identical across all three
    # modes, and scan_io derives its columns from the first row.
    return {
        "bond": float(bond),
        "angle": float(angle_deg),
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
        "n_qubits": hamiltonian.num_qubits,
        "n_params": ansatz.num_parameters,
        "n_evaluations": int(getattr(result, "nfev", 0)),
        "vqe_params": np.asarray(result.x),
    }


# ===========================================================================
#  The window -- curve mode
# ===========================================================================

ATOM_RADII = {"N": 0.34, "H": 0.20}
ATOM_COLORS = {"N": "#3050f8", "H": "#f2f2f2"}   # CPK: nitrogen blue


class NH3ScanViewer:
    """Live curve on the left, live molecule on the right; scrubbing after."""

    def __init__(self, coordinate: str, fixed_label: str, max_extent: float) -> None:
        plt.ion()
        self.coordinate = coordinate
        self.fig, (self.ax_energy, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        (self.line_hf,) = self.ax_energy.plot([], [], "-", color="#d73027", label="Hartree-Fock")
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2,
                                             label="current")
        self.ax_energy.set_xlabel("N-H bond length (Angstrom)" if coordinate == "stretch"
                                  else "H-N-H angle (degrees)")
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title(f"NH$_3$ {coordinate} scan   ({fixed_label})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        span = max_extent + 0.5
        self.ax_mol.set_xlim(-span, span)
        self.ax_mol.set_ylim(-span * 0.75, span * 0.95)
        self.ax_mol.set_aspect("equal")
        self.ax_mol.set_xticks([])
        self.ax_mol.set_yticks([])
        self.ax_mol.set_title("Geometry (viewed from the side)")
        self.bonds = [self.ax_mol.plot([], [], "-", color="#888888", linewidth=3, zorder=1)[0]
                      for _ in range(3)]
        self.n_atom = Circle((0, 0), ATOM_RADII["N"], facecolor=ATOM_COLORS["N"],
                             edgecolor="black", linewidth=1.5, zorder=3)
        self.ax_mol.add_patch(self.n_atom)
        self.h_atoms = [Circle((0, 0), ATOM_RADII["H"], facecolor=ATOM_COLORS["H"],
                               edgecolor="black", linewidth=1.5, zorder=2)
                        for _ in range(3)]
        for atom in self.h_atoms:
            self.ax_mol.add_patch(atom)
        self.mol_label = self.ax_mol.text(0, -span * 0.75 + 0.15, "", ha="center", fontsize=11)
        self._redraw()

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
        row = self.rows[index]
        self.marker.set_data([self._x_of(row)], [row["vqe"]])
        pts = projected_atoms(row["bond"], row["angle"])
        nx, ny, _ = pts[0]
        self.n_atom.center = (nx, ny)
        # Draw the H behind the nitrogen first so the pyramid reads correctly.
        order = sorted(range(3), key=lambda k: -pts[k + 1][2])
        for slot, k in enumerate(order):
            hx, hy, _ = pts[k + 1]
            self.h_atoms[slot].center = (hx, hy)
            self.bonds[slot].set_data([nx, hx], [ny, hy])
        self.mol_label.set_text(
            f"r = {row['bond']:.3f} A     theta = {row['angle']:.1f} deg"
            f"     E = {row['vqe']:.4f} Ha"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_figure(self, path: Path) -> None:
        self.fig.savefig(path, dpi=150, bbox_inches="tight")

    def enable_review(self) -> None:
        # A single-point scan has no range to scrub: valmin would equal
        # valmax and matplotlib warns.
        if len(self.rows) > 1:
            slider_ax = self.fig.add_axes([0.12, 0.06, 0.5, 0.04])
            self.slider = Slider(slider_ax, "point", 0, len(self.rows) - 1,
                                 valinit=self.index, valstep=1)
            self.slider.on_changed(lambda v: self._show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.ax_energy.set_title(f"{self.ax_energy.get_title()}  --  LEFT/RIGHT or slider")
        print("\nScan complete. The window is now interactive:")
        print("  LEFT / RIGHT arrow keys  -- step along the curve")
        print("  slider                   -- jump anywhere")
        print("  close the window         -- quit")
        plt.ioff()
        self._show(self.index)
        plt.show()

    def _on_key(self, event) -> None:
        if len(self.rows) <= 1:
            return
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
#  The window -- surface mode
# ===========================================================================

class NH3SurfaceViewer:
    """A contour map that fills in as the grid is walked.

    Deliberately a *contour* during the scan rather than a 3D surface:
    re-rendering ``plot_surface`` on every point is slow and, with most of
    the grid still empty, unreadable. The real 3D surface is drawn once at
    the end by :meth:`show_surface`.
    """

    def __init__(self, bond_values: np.ndarray, angle_values: np.ndarray) -> None:
        plt.ion()
        self.bond_values = bond_values
        self.angle_values = angle_values
        self.energies = np.full((len(angle_values), len(bond_values)), np.nan)
        self.rows: List[Dict[str, Any]] = []

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.ax.set_xlabel("N-H bond length (Angstrom)")
        self.ax.set_ylabel("H-N-H angle (degrees)")
        self.ax.set_title("NH$_3$ potential-energy surface (filling in)")
        self.mesh = None
        self.ax.set_xlim(bond_values[0], bond_values[-1])
        self.ax.set_ylim(angle_values[0], angle_values[-1])
        (self.current,) = self.ax.plot([], [], "wo", markersize=10, markeredgecolor="black",
                                       markeredgewidth=1.5, zorder=5)
        self._redraw()

    def add_point(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        i = int(np.argmin(np.abs(self.angle_values - row["angle"])))
        j = int(np.argmin(np.abs(self.bond_values - row["bond"])))
        self.energies[i, j] = row["vqe"]

        if self.mesh is not None:
            self.mesh.remove()
        masked = np.ma.masked_invalid(self.energies)
        self.mesh = self.ax.pcolormesh(self.bond_values, self.angle_values, masked,
                                       cmap="viridis", shading="nearest")
        self.current.set_data([row["bond"]], [row["angle"]])
        done = int(np.isfinite(self.energies).sum())
        self.ax.set_title(f"NH$_3$ potential-energy surface  "
                          f"({done}/{self.energies.size} points)")
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_surface(self, png_path: Path | None = None, show: bool = True) -> None:
        """The finished grid as a real 3D surface next to a contour map."""
        grid_bond, grid_angle = np.meshgrid(self.bond_values, self.angle_values)
        masked = np.ma.masked_invalid(self.energies)

        fig = plt.figure(figsize=(13, 5.5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot_surface(grid_bond, grid_angle, masked, cmap="viridis",
                          edgecolor="none", alpha=0.95)
        ax3d.set_xlabel("r(N-H) (A)")
        ax3d.set_ylabel("theta (deg)")
        ax3d.set_zlabel("Energy (Ha)")
        ax3d.set_title("NH$_3$ ground-state surface")

        ax2d = fig.add_subplot(1, 2, 2)
        levels = 24
        contour = ax2d.contourf(grid_bond, grid_angle, masked, levels=levels, cmap="viridis")
        ax2d.contour(grid_bond, grid_angle, masked, levels=levels,
                     colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
        if np.isfinite(self.energies).any():
            i, j = np.unravel_index(np.nanargmin(self.energies), self.energies.shape)
            ax2d.plot([self.bond_values[j]], [self.angle_values[i]], "r*", markersize=16,
                      label=f"lowest sample\n{self.bond_values[j]:.3f} A, "
                            f"{self.angle_values[i]:.1f} deg")
            ax2d.legend(loc="upper right", fontsize=8)
        ax2d.set_xlabel("r(N-H) (A)")
        ax2d.set_ylabel("theta (deg)")
        ax2d.set_title("Contours")
        fig.tight_layout()

        if png_path is not None:
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"saved {display_path(png_path)}")
        if show:
            plt.ioff()
            plt.show()


def serpentine(bond_values: np.ndarray,
               angle_values: np.ndarray) -> List[Tuple[float, float]]:
    """Grid walk that never jumps: left to right, then right to left.

    Warm starting is only useful if the previous geometry is a neighbour of
    this one. A naive row-major walk teleports from the right edge back to
    the left at every row boundary, throwing the warm start away once per
    row. Reversing alternate rows keeps every step adjacent.
    """
    order: List[Tuple[float, float]] = []
    for i, angle in enumerate(angle_values):
        bonds = bond_values if i % 2 == 0 else bond_values[::-1]
        for bond in bonds:
            order.append((float(bond), float(angle)))
    return order


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NH3 ground-state energy by VQE. Ammonia has 6 internal coordinates, "
                    "reduced to 2 by C3v symmetry (symmetric N-H stretch, H-N-H angle), "
                    "so it supports two curves and one surface."
    )
    parser.add_argument("--coordinate", choices=["stretch", "bend", "surface"],
                        default="stretch",
                        help="which geometry to vary (default stretch)")
    parser.add_argument("--start", type=float, default=0.85,
                        help="shortest N-H bond in Angstrom (default 0.85)")
    parser.add_argument("--stop", type=float, default=1.60,
                        help="longest N-H bond in Angstrom, inclusive (default 1.60)")
    parser.add_argument("--step", type=float, default=0.075,
                        help="N-H bond spacing (default 0.075; surface default 0.125). "
                             "Deliberately misses the 1.04 A minimum so the interpolation "
                             "folders have something to recover.")
    parser.add_argument("--angle-start", type=float, default=96.0,
                        help="smallest H-N-H angle in degrees (default 96)")
    parser.add_argument("--angle-stop", type=float, default=PLANAR_ANGLE,
                        help=f"largest H-N-H angle, inclusive (default {PLANAR_ANGLE}, which "
                             f"is planar and is a hard geometric ceiling -- see the docstring)")
    parser.add_argument("--angle-step", type=float, default=3.0,
                        help="angle spacing in degrees (default 3; surface default 4)")
    parser.add_argument("--bond", type=float, default=EXPERIMENTAL_BOND,
                        help=f"N-H bond held fixed during a bend scan "
                             f"(default {EXPERIMENTAL_BOND}, the experimental value)")
    parser.add_argument("--angle", type=float, default=EXPERIMENTAL_ANGLE,
                        help=f"H-N-H angle held fixed during a stretch scan "
                             f"(default {EXPERIMENTAL_ANGLE}, the experimental value)")
    parser.add_argument("--active-electrons", type=int, default=6,
                        help="electrons in the active space (default 6)")
    parser.add_argument("--active-orbitals", type=int, default=5,
                        help="spatial orbitals in the active space (default 5 -> 8 qubits, "
                             "54 UCCSD parameters). Pass 4 and 4 for a ~5x faster surface at "
                             "reduced accuracy; see the docstring's table.")
    parser.add_argument("--optimizer", choices=["auto", "slsqp", "cobyla"], default="auto",
                        help=f"'auto' (default) picks SLSQP at or below "
                             f"{SLSQP_PARAMETER_LIMIT} parameters, COBYLA above")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="optimizer iteration cap (default 1000). Under COBYLA this sets "
                             "the cost per point, because COBYLA spends its whole budget.")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart every geometry from Hartree-Fock instead of warm-"
                             "starting from the previous one")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="reopen a saved scan (.json or .csv) without running any VQE")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save (default nh3_<coordinate>_scan.json/.csv)")
    parser.add_argument("--no-save", action="store_true", help="don't save this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="don't block on the window at the end; figures are still written")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.no_show:
        matplotlib.use("Agg")

    if args.load:
        _replay(Path(args.load), here, args.no_show)
        return

    # -- validation ----------------------------------------------------
    if args.step <= 0 or args.angle_step <= 0:
        parser.error("--step and --angle-step must be positive")
    if args.stop < args.start:
        parser.error("--stop must be >= --start")
    if args.angle_stop < args.angle_start:
        parser.error("--angle-stop must be >= --angle-start")
    for name, value in (("--angle", args.angle), ("--angle-stop", args.angle_stop)):
        if value > PLANAR_ANGLE + 1e-9:
            parser.error(
                f"{name}={value} exceeds {PLANAR_ANGLE} degrees. Three equivalent hydrogens "
                f"cannot subtend more than that -- {PLANAR_ANGLE} is the planar geometry and "
                f"the hard ceiling. Going further is the mirror-image molecule, not a wider "
                f"scan. (BeH2's bend runs to 180; NH3's does not.)"
            )
    if args.active_electrons > 2 * args.active_orbitals:
        parser.error(f"{args.active_electrons} electrons will not fit in "
                     f"{args.active_orbitals} spatial orbitals")

    if args.coordinate == "surface":
        _run_surface(args, here)
    else:
        _run_curve(args, here)


def _run_curve(args, here: Path) -> None:
    """Stretch or bend: one coordinate moves, the other is held."""
    if args.coordinate == "stretch":
        values = np.arange(args.start, args.stop + args.step / 2, args.step)
        geometries = [(float(v), args.angle) for v in values]
        fixed_label = f"angle fixed at {args.angle:.2f} deg"
        max_extent = float(values[-1])
    else:
        values = np.arange(args.angle_start,
                           args.angle_stop + args.angle_step / 2, args.angle_step)
        values = values[values <= PLANAR_ANGLE + 1e-9]
        geometries = [(args.bond, float(v)) for v in values]
        fixed_label = f"bonds fixed at {args.bond:.3f} A"
        max_extent = float(args.bond)

    print(f"NH3 {args.coordinate} scan: {len(values)} geometries, {fixed_label}")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals}), "
          f"basis {BASIS}, closed-shell singlet (RHF)")

    viewer = NH3ScanViewer(args.coordinate, fixed_label, max_extent)
    rows: List[Dict[str, Any]] = []
    previous = None
    for i, (bond, angle) in enumerate(geometries):
        t0 = time.perf_counter()
        row = run_one_geometry(bond, angle, args.active_electrons, args.active_orbitals,
                               optimizer=args.optimizer,
                               max_iterations=args.max_iterations, theta0=previous)
        if not args.cold_start:
            previous = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            resolved = choose_optimizer(args.optimizer, row["n_params"])
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters, "
                  f"optimizer {resolved.upper()}"
                  f"{' [auto]' if args.optimizer == 'auto' else ''})")
        label = f"r={bond:.3f} A" if args.coordinate == "stretch" else f"theta={angle:.1f} deg"
        print(f"  [{i + 1:3d}/{len(geometries)}] {label}   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")

    _report_curve(rows, args)

    if not args.no_save:
        target = resolve_target(args.save, here, f"nh3_{args.coordinate}_scan")
        metadata = _metadata(args, rows, fixed_label, max_extent)
        json_path, csv_path = save_scan(target, metadata, rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        png = here / f"nh3_{args.coordinate}_curve.png"
        viewer.save_figure(png)
        print(f"saved {display_path(png)}")
        print(f"interpolate it:  python experiment/nh3_spline/nh3_spline_interpolation.py "
              f"--scan {json_path.name}")

    if args.no_show:
        print("\nScan complete. --no-show was passed, so exiting instead of waiting.")
        return
    viewer.enable_review()


def _run_surface(args, here: Path) -> None:
    """Both coordinates on a grid, walked serpentine."""
    # The surface gets coarser defaults than the curves, because its cost is
    # the product of the two axes rather than the sum.
    step = args.step if args.step != 0.075 else 0.125
    angle_step = args.angle_step if args.angle_step != 3.0 else 4.0

    bond_values = np.arange(args.start, args.stop + step / 2, step)
    angle_values = np.arange(args.angle_start, args.angle_stop + angle_step / 2, angle_step)
    angle_values = angle_values[angle_values <= PLANAR_ANGLE + 1e-9]

    order = serpentine(bond_values, angle_values)
    print(f"NH3 surface scan: {len(bond_values)} bonds x {len(angle_values)} angles "
          f"= {len(order)} geometries")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals}), "
          f"basis {BASIS}, closed-shell singlet (RHF)")
    if len(bond_values) < 7 or len(angle_values) < 7:
        print("  note: the half-grid holdout used by the interpolation folders keeps every")
        print("  other node, and a cubic bivariate spline needs 4 per axis. A grid smaller")
        print("  than 7x7 cannot be cross-validated that way.")

    viewer = NH3SurfaceViewer(bond_values, angle_values)
    rows: List[Dict[str, Any]] = []
    previous = None
    for i, (bond, angle) in enumerate(order):
        t0 = time.perf_counter()
        row = run_one_geometry(bond, angle, args.active_electrons, args.active_orbitals,
                               optimizer=args.optimizer,
                               max_iterations=args.max_iterations, theta0=previous)
        if not args.cold_start:
            previous = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            resolved = choose_optimizer(args.optimizer, row["n_params"])
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters, "
                  f"optimizer {resolved.upper()})")
            elapsed = time.perf_counter() - t0
            print(f"  first point took {elapsed:.1f}s -> the full grid is roughly "
                  f"{elapsed * len(order) / 60:.0f} minutes")
        print(f"  [{i + 1:3d}/{len(order)}] r={bond:.3f} A theta={angle:.1f} deg   "
              f"VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")

    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nlowest sampled point: {best['vqe']:.6f} Ha at "
          f"r = {best['bond']:.3f} A, theta = {best['angle']:.1f} deg")
    print(f"  model reference (true 2D minimum): r = {MODEL_SURFACE_BOND} A, "
          f"theta = {MODEL_SURFACE_ANGLE} deg")
    print(f"  experiment [2]:                       r = {EXPERIMENTAL_BOND} A, "
          f"theta = {EXPERIMENTAL_ANGLE} deg")
    print("  the grid is coarse, so the lowest *sample* is not the minimum of the")
    print("  surface. Fit an interpolant and minimize that -- see experiment/nh3_spline/.")
    _report_errors(rows)

    png = None
    if not args.no_save:
        target = resolve_target(args.save, here, "nh3_surface_scan")
        metadata = _metadata(args, rows, "surface", float(bond_values[-1]))
        metadata.update({
            "bond_values": [float(v) for v in bond_values],
            "angle_values": [float(v) for v in angle_values],
        })
        json_path, csv_path = save_scan(target, metadata, rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        png = here / "nh3_surface.png"

    viewer.show_surface(png_path=png, show=not args.no_show)


def _metadata(args, rows: List[Dict[str, Any]], fixed_label: str,
              max_extent: float) -> Dict[str, Any]:
    return {
        "molecule": "NH3", "basis": BASIS,
        "scan_type": "surface" if args.coordinate == "surface" else "curve",
        "coordinate": args.coordinate,
        "degrees_of_freedom": 6,
        "free_coordinates": 2,
        "symmetry": "C3v",
        "multiplicity": "singlet",
        "spin_2s": 0,
        "scf_method": "RHF",
        "fixed_label": fixed_label,
        "max_extent": float(max_extent),
        "bond": float(args.bond),
        "angle": float(args.angle),
        "start": float(args.start), "stop": float(args.stop), "step": float(args.step),
        "angle_start": float(args.angle_start), "angle_stop": float(args.angle_stop),
        "angle_step": float(args.angle_step),
        "active_electrons": int(args.active_electrons),
        "active_orbitals": int(args.active_orbitals),
        "optimizer": choose_optimizer(args.optimizer, rows[0]["n_params"]),
        "max_iterations": int(args.max_iterations),
        "planar_angle": PLANAR_ANGLE,
        "experimental_bond": EXPERIMENTAL_BOND,
        "experimental_angle": EXPERIMENTAL_ANGLE,
        "model_bond_stretch": MODEL_BOND_STRETCH,
        "model_angle_bend": MODEL_ANGLE_BEND,
        "model_surface_bond": MODEL_SURFACE_BOND,
        "model_surface_angle": MODEL_SURFACE_ANGLE,
    }


# ===========================================================================
#  Reloading
# ===========================================================================

def _load_csv(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read an NH3 scan from its ``.csv``.

    The JSON is canonical; a CSV has no metadata, so scan_type is inferred
    from whether both coordinates actually vary.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")
    numeric = ("bond", "angle", "hf", "vqe", "exact")
    rows = []
    for r in raw:
        row = {k: float(v) for k, v in r.items() if k in numeric and v != ""}
        for k in ("n_qubits", "n_params", "n_evaluations"):
            if r.get(k):
                row[k] = int(float(r[k]))
        rows.append(row)

    missing = {"bond", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(f"{display_path(path)} is missing column(s) {sorted(missing)} -- "
                         f"not an NH3 scan CSV.")

    bonds = {round(r["bond"], 6) for r in rows}
    angles = {round(r["angle"], 6) for r in rows}
    if len(bonds) > 1 and len(angles) > 1:
        scan_type, coordinate = "surface", "surface"
    elif len(bonds) > 1:
        scan_type, coordinate = "curve", "stretch"
    else:
        scan_type, coordinate = "curve", "bend"

    return {
        "scan_type": scan_type,
        "coordinate": coordinate,
        "fixed_label": "reloaded from csv",
        "max_extent": max(r["bond"] for r in rows),
        "bond_values": sorted(bonds),
        "angle_values": sorted(angles),
    }, rows


def _replay(load_path: Path, here: Path, no_show: bool) -> None:
    """Reopen a saved scan without running any chemistry."""
    if load_path.suffix.lower() == ".csv":
        found = next((c for c in (load_path, here / load_path.name) if c.exists()), None)
        if found is None:
            looked = "\n  ".join(display_path(c) for c in (load_path, here / load_path.name))
            raise SystemExit(f"no scan CSV found. Looked in:\n  {looked}")
        metadata, rows = _load_csv(found)
    else:
        metadata, rows = load_scan(load_path, default_dir=here)

    scan_type = metadata.get("scan_type", "curve")
    if metadata.get("active_electrons") is not None:
        space = f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
    else:
        space = "active space not recorded in the csv"
    print(f"loaded {len(rows)} points from {load_path.name} "
          f"({metadata.get('coordinate')} scan, {space}) -- no VQE re-run")

    if scan_type == "surface":
        bond_values = np.array(metadata.get("bond_values")
                               or sorted({r["bond"] for r in rows}), dtype=float)
        angle_values = np.array(metadata.get("angle_values")
                                or sorted({r["angle"] for r in rows}), dtype=float)
        viewer = NH3SurfaceViewer(bond_values, angle_values)
        for row in rows:
            viewer.add_point(row)
        viewer.show_surface(show=not no_show)
        return

    viewer = NH3ScanViewer(metadata.get("coordinate", "stretch"),
                           metadata.get("fixed_label", ""),
                           float(metadata.get("max_extent", 1.6)))
    for row in rows:
        viewer.add_point(row)
    if no_show:
        print("\n--no-show was passed, so exiting instead of waiting.")
        return
    viewer.enable_review()


# ===========================================================================
#  Reporting
# ===========================================================================

def _report_curve(rows: List[Dict[str, Any]], args) -> None:
    key = "bond" if args.coordinate == "stretch" else "angle"
    unit = "A" if args.coordinate == "stretch" else "deg"
    fixed = args.angle if args.coordinate == "stretch" else args.bond
    model, _, _ = curve_reference(args.coordinate, fixed)
    experiment = EXPERIMENTAL_BOND if args.coordinate == "stretch" else EXPERIMENTAL_ANGLE

    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at {key} = {best[key]:.4f} {unit}")
    print(f"  CAS(6,5)/STO-3G model minimum : {model} {unit}")
    warn_if_slice_differs(args.coordinate, fixed)
    print(f"  experiment [2]                : {experiment} {unit} "
          f"(the model misses it by {abs(model - experiment):.4f} {unit} on its own)")
    print(f"  the grid is coarse, so that is the lowest *sample*, not the curve's minimum.")
    print(f"  Fit an interpolant and minimize that -- see experiment/nh3_spline/. Score the")
    print(f"  result against the model value, not experiment: the basis-set error is much")
    print(f"  larger than the interpolation error and would swamp the comparison.")

    if args.coordinate == "bend":
        _report_barrier(rows)

    _report_errors(rows)

    if args.coordinate == "stretch":
        _report_monotonicity(rows)
        if max(r["bond"] for r in rows) > 1.6:
            print("\n  reminder: this scan goes past 1.6 A, where the active space matters a")
            print("  lot and the monotonicity check above does NOT catch the problem. Every")
            print("  space from CAS(4,4) to CAS(8,7) is monotonic on this coordinate, yet")
            print("  CAS(4,4) sits 406 mHa from full valence. Breaking three N-H bonds needs")
            print("  CAS(6,6) at minimum. Spot-check with --active-electrons 6")
            print("  --active-orbitals 6 before trusting long-stretch numbers.")


def _report_barrier(rows: List[Dict[str, Any]]) -> None:
    """The inversion barrier: the point of scanning the umbrella at all."""
    ordered = sorted(rows, key=lambda r: r["angle"])
    planar = [r for r in ordered if abs(r["angle"] - PLANAR_ANGLE) < 1e-6]
    if not planar:
        print("\n  (no planar point at 120 deg in this scan, so no inversion barrier to "
              "report -- rerun with --angle-stop 120)")
        return
    floor = min(r["vqe"] for r in ordered)
    barrier_mha = (planar[0]["vqe"] - floor) * 1000
    print(f"\nINVERSION BARRIER  E(planar 120 deg) - E(lowest sample)")
    print(f"  model : {barrier_mha:7.3f} mHa  ({barrier_mha / MHA_PER_WAVENUMBER:6.0f} cm-1)")
    print(f"  experiment [2]: {EXPERIMENTAL_BARRIER_MHA:.2f} mHa (~1786 cm-1)")
    ratio = barrier_mha / EXPERIMENTAL_BARRIER_MHA if EXPERIMENTAL_BARRIER_MHA else float("nan")
    print(f"  the model is {ratio:.1f}x too high. STO-3G over-pyramidalises ammonia; this is")
    print(f"  a basis-set failure, present before VQE is involved. Quote it as the model's")
    print(f"  barrier, not as a prediction for ammonia.")
    print(f"  (the barrier off a coarse grid is also an underestimate of the interpolated")
    print(f"  one, since the lowest sample is above the true minimum -- the spline folder")
    print(f"  recomputes it properly.)")


def _report_errors(rows: List[Dict[str, Any]]) -> None:
    """Classical-vs-quantum error, plus the fraction of correlation recovered.

    |HF - exact| is physics: what a single Slater determinant cannot
    represent. |VQE - exact| is numerics: how close the optimizer got. The
    fraction recovered is the more defensible number when the correlation
    energy itself varies by an order of magnitude across the scan.
    """
    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    print(f"\nlargest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")

    worst = max(rows, key=lambda r: abs(r["hf"] - r["exact"]))
    available = worst["hf"] - worst["exact"]
    if abs(available) > 1e-9:
        recovered = (worst["hf"] - worst["vqe"]) / available
        print(f"correlation energy recovered where there is most of it "
              f"(r={worst['bond']:.3f} A, theta={worst['angle']:.1f} deg): "
              f"{recovered * 100:.2f}%")
        if recovered < 0.93:
            print("  below ~93% means UCCSD is struggling, not the optimizer. UCCSD is")
            print("  single-reference; stretched multi-bond geometries are not. More")
            print("  optimizer budget will not fix that -- a larger active space might.")


def _report_monotonicity(rows: List[Dict[str, Any]]) -> None:
    """Past its minimum a dissociation curve must rise.

    Necessary but, for NH3, emphatically not sufficient -- see the module
    docstring. A dip proves the active space is broken; the absence of a dip
    proves nothing.
    """
    ordered = sorted(rows, key=lambda r: r["bond"])
    if len(ordered) < 3:
        return
    best_at = min(range(len(ordered)), key=lambda i: ordered[i]["exact"])
    dips = [(ordered[i]["bond"], ordered[i + 1]["bond"])
            for i in range(best_at, len(ordered) - 1)
            if ordered[i + 1]["exact"] < ordered[i]["exact"]]
    if dips:
        print("\n  WARNING: the exact curve dips downward past its minimum, between "
              + ", ".join(f"{a:.3f}-{b:.3f} A" for a, b in dips))
        print("  A ground-state curve must rise monotonically toward dissociation. The")
        print("  active space is too small to describe three bonds breaking. VQE has")
        print("  reproduced the (wrong) active-space energy faithfully.")
    else:
        print("  curve rises monotonically past its minimum "
              "(necessary, but NOT sufficient for NH3 -- see the docstring)")


if __name__ == "__main__":
    main()