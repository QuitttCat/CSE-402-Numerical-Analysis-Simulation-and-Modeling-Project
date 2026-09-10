"""
h2o_ground_state_estimation.py -- water's ground-state energy by VQE, scanned
along one internal coordinate at a time, plotted live next to a picture of the
molecule deforming.

Simulator only. No IBM account, no queue, no quota.

WHY TWO COORDINATES (AND WHY YOU GET TO PICK)
-----------------------------------------------
H2 had exactly one thing you could vary: the distance between its two atoms.
Water is bent and triatomic, so it has three internal degrees of freedom, and
the two interesting ones behave completely differently:

  --coordinate stretch   Both O-H bonds are stretched together, with the
                         H-O-H angle held fixed (default 104.5 deg, water's
                         equilibrium angle). This pulls the molecule apart
                         toward O + 2H. It is the standard benchmark for
                         correlated methods precisely because Hartree-Fock
                         *fails badly* out in the stretched region -- a
                         single Slater determinant simply cannot describe
                         two breaking bonds. Expect the HF curve to peel
                         away from VQE/exact dramatically, the same way it
                         did for H2.

  --coordinate bend      The H-O-H angle is varied with both O-H bond
                         lengths held fixed (default 0.958 A). Bending is a
                         much softer motion than stretching, the molecule
                         never actually comes apart, and the electronic
                         structure stays comfortably single-reference --
                         so HF tracks the correlated curves closely and the
                         energy well is shallow. Less dramatic physics, but
                         a much nicer animation, and a useful contrast:
                         here is a coordinate where HF is fine, there is one
                         where it isn't.

Run both and compare -- that contrast is the experiment.

ACTIVE SPACE -- AND A TRAP WORTH KNOWING ABOUT
------------------------------------------------
Water in STO-3G is 7 spatial orbitals and 10 electrons -> 14 qubits if you
take everything, which is far too slow to scan. ``--active-electrons`` and
``--active-orbitals`` restrict the calculation to a CAS(n_elec, n_orb)
active space, the same knob the GQE paper used for its molecules.

**The defaults differ by mode, and that is deliberate.** Bending defaults to
CAS(4,4) (6 qubits, 26 parameters, ~20 s/point). Stretching defaults to the
larger CAS(6,5), because CAS(4,4) is *too small to describe two bonds
breaking* and produces a visibly wrong curve. Measured exact energies along
the stretch, in Hartree:

    r(O-H)    CAS(4,4)     CAS(6,5)     CAS(8,6)
    1.50 A   -74.750839   -74.863898   -74.873401
    1.70 A   -74.805199   -74.805735   -74.812681   <-- CAS(4,4) drops!

A ground-state energy curve must rise monotonically toward the dissociation
limit as you pull the atoms apart. CAS(4,4) instead goes *down* from 1.50 to
1.70 A -- an unphysical kink caused by truncating the active space too
aggressively, not by any failure of VQE (VQE reproduces the exact
active-space energy to under 0.1 mHa at every one of these points). CAS(6,5)
and CAS(8,6) are both smooth. Note also that CAS(8,6) is exactly equivalent
to freezing only the oxygen 1s core, so it is the largest space worth asking
for in this basis.

The moral, which is worth a line in any write-up: an "exact" number is only
exact *within the model you chose*, and a badly chosen active space breaks
the physics before the quantum algorithm ever gets a chance to.

COST
-----
What dominates runtime is not circuit simulation (a single energy evaluation
is well under a second) but the *number* of evaluations. SLSQP estimates its
gradient by finite differences, so each optimizer step costs about one
evaluation per parameter -- and UCCSD's parameter count grows fast with the
active space (26 at CAS(4,4), 54 at CAS(6,5)).

Each geometry warm-starts its optimizer from the previous geometry's
solution, since neighbouring points on a scan have similar wavefunctions.
Measured at CAS(6,5): 333 evaluations cold vs 278 warm, so about 17% off --
a real but modest saving, not a game changer. Pass ``--cold-start`` to
disable it. (The reason it is only modest: SLSQP spends most of its budget
estimating finite-difference gradients, which costs one evaluation per
parameter regardless of where you start.)

Rough timings on a laptop CPU, per scan point:

    CAS(4,4)   6 qubits, 26 parameters    ~20 s
    CAS(6,5)   8 qubits, 54 parameters    ~130 s

So the default stretch scan (18 points at CAS(6,5)) takes around 40 minutes,
while the default bend scan (19 points at CAS(4,4)) takes about 6. Use a
coarser ``--step`` if you want a quicker look, and note the script plots
live -- you can watch the curve take shape rather than waiting blind.

SAVING AND REOPENING
----------------------
Given a stretch scan can run 40 minutes, every finished scan is saved next to
this script -- ``h2o_stretch_scan.json`` / ``h2o_bend_scan.json`` (full data
plus the metadata needed to rebuild the plot) and a matching ``.csv`` (flat
table for reports and spreadsheets). Reopening with ``--load`` skips the VQE
completely and drops you straight into review mode, so a long scan is a
one-time cost.

USAGE
------
    python experiment/h2o/h2o_ground_state_estimation.py
    python experiment/h2o/h2o_ground_state_estimation.py --coordinate stretch --start 0.7 --stop 2.4 --step 0.1
    python experiment/h2o/h2o_ground_state_estimation.py --coordinate bend --start 70 --stop 160 --step 5
    python experiment/h2o/h2o_ground_state_estimation.py --active-electrons 6 --active-orbitals 5
    python experiment/h2o/h2o_ground_state_estimation.py --load h2o_bend_scan.json   # no recompute
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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

# Water's experimental equilibrium geometry -- used as the fixed coordinate
# in whichever mode isn't being scanned.
EQUILIBRIUM_BOND = 0.958   # Angstrom
EQUILIBRIUM_ANGLE = 104.5  # degrees


# ===========================================================================
#  Geometry
# ===========================================================================

def water_positions(bond: float, angle_deg: float) -> Dict[str, tuple]:
    """Atom positions for water, in the x-z plane.

    Oxygen sits at the origin; the two hydrogens are placed symmetrically
    about the +z axis, each at `bond` Angstrom from O, with `angle_deg`
    between the two O-H bonds. Keeping the bisector along +z means the
    molecule stays centred and upright no matter which coordinate is being
    scanned, which makes the animation easy to read.
    """
    half = math.radians(angle_deg) / 2
    dx, dz = bond * math.sin(half), bond * math.cos(half)
    return {"O": (0.0, 0.0), "H1": (dx, dz), "H2": (-dx, dz)}


def geometry_string(bond: float, angle_deg: float) -> str:
    pos = water_positions(bond, angle_deg)
    return (f"O 0 0 0; "
            f"H {pos['H1'][0]} 0 {pos['H1'][1]}; "
            f"H {pos['H2'][0]} 0 {pos['H2'][1]}")


# ===========================================================================
#  The quantum chemistry -- one geometry at a time
# ===========================================================================

def build_problem(bond: float, angle_deg: float,
                  active_electrons: int, active_orbitals: int) -> ElectronicStructureProblem:
    """Water at this geometry, restricted to a CAS(active_electrons, active_orbitals)
    active space."""
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
    """Hartree-Fock, VQE and exact ground-state energy at one water geometry.

    ``theta0`` warm-starts the optimizer from a previous geometry's solution.
    Neighbouring points on a scan have very similar wavefunctions, so
    starting from the previous answer instead of from scratch cuts the
    number of energy evaluations substantially -- which matters here,
    because evaluation count (not simulation speed) is what dominates the
    runtime.
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
#  The window -- live curve on the left, live molecule on the right
# ===========================================================================

ATOM_RADII = {"O": 0.34, "H": 0.22}   # drawn radii, Angstrom
ATOM_COLORS = {"O": "#ff4d4d", "H": "#f2f2f2"}


class WaterScanViewer:
    """Live scan plot plus, once finished, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called per geometry as its energies come
    in -- the curve grows and the molecule redraws at the geometry just
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
        # Hollow VQE markers so the green exact crosses stay visible underneath.
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2, label="current")
        self.ax_energy.set_xlabel(
            "O-H bond length (Angstrom)" if coordinate == "stretch" else "H-O-H angle (degrees)"
        )
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title(f"H$_2$O {coordinate} scan   ({fixed_label})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        # -- right panel: the molecule ---------------------------------
        span = max_extent + 0.5
        self.ax_mol.set_xlim(-span, span)
        self.ax_mol.set_ylim(-span * 0.75, span * 1.05)
        self.ax_mol.set_aspect("equal")
        self.ax_mol.set_xticks([])
        self.ax_mol.set_yticks([])
        self.ax_mol.set_title("Geometry")
        self.bonds = [
            self.ax_mol.plot([], [], "-", color="#888888", linewidth=3, zorder=1)[0]
            for _ in range(2)
        ]
        self.atom_patches = {
            name: Circle((0, 0), ATOM_RADII[name[0]], facecolor=ATOM_COLORS[name[0]],
                         edgecolor="black", linewidth=1.5, zorder=2)
            for name in ("O", "H1", "H2")
        }
        for patch in self.atom_patches.values():
            self.ax_mol.add_patch(patch)
        self.mol_label = self.ax_mol.text(0, -span * 0.62, "", ha="center", fontsize=11)

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

        pos = water_positions(row["bond"], row["angle"])
        for name, (x, z) in pos.items():
            self.atom_patches[name].center = (x, z)
        for bond_line, h in zip(self.bonds, ("H1", "H2")):
            bond_line.set_data([pos["O"][0], pos[h][0]], [pos["O"][1], pos[h][1]])

        self.mol_label.set_text(
            f"r(O-H) = {row['bond']:.2f} A     angle = {row['angle']:.1f}$\\degree$\n"
            f"E = {row['vqe']:.4f} Ha"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -- phase 2: review the finished scan ------------------------------

    def enable_review(self) -> None:
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
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2O ground-state energy by VQE (local simulator), scanned along "
                    "either the symmetric O-H stretch or the H-O-H bend."
    )
    parser.add_argument("--coordinate", choices=["stretch", "bend"], default="stretch",
                        help="'stretch': vary both O-H bonds together at fixed angle (default). "
                             "'bend': vary the H-O-H angle at fixed bond length.")
    parser.add_argument("--start", type=float, default=None,
                        help="scan start -- Angstrom for stretch (default 0.7), degrees for bend (default 70)")
    parser.add_argument("--stop", type=float, default=None,
                        help="scan stop, inclusive -- default 2.4 A for stretch, 160 deg for bend")
    parser.add_argument("--step", type=float, default=None,
                        help="scan step -- default 0.1 A for stretch, 5 deg for bend")
    parser.add_argument("--angle", type=float, default=EQUILIBRIUM_ANGLE,
                        help=f"[stretch mode] fixed H-O-H angle in degrees (default {EQUILIBRIUM_ANGLE})")
    parser.add_argument("--bond", type=float, default=EQUILIBRIUM_BOND,
                        help=f"[bend mode] fixed O-H bond length in Angstrom (default {EQUILIBRIUM_BOND})")
    parser.add_argument("--active-electrons", type=int, default=None,
                        help="electrons in the active space (default: 6 for stretch, 4 for bend)")
    parser.add_argument("--active-orbitals", type=int, default=None,
                        help="spatial orbitals in the active space (default: 5 for stretch, 4 for "
                             "bend). Larger is more accurate but much slower -- see the module "
                             "docstring.")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each geometry's optimization from the Hartree-Fock state "
                             "instead of warm-starting from the previous geometry's solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json written by an earlier run) straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: h2o_<coordinate>_scan.json/.csv "
                             "next to this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save the results of this scan")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # -- replay a finished scan, no VQE at all -------------------------
    if args.load:
        metadata, rows = load_scan(Path(args.load), default_dir=here)
        print(f"loaded {len(rows)} points from {args.load} "
              f"({metadata.get('coordinate')} scan, "
              f"CAS({metadata.get('active_electrons')}, {metadata.get('active_orbitals')})) "
              f"-- no VQE re-run")
        viewer = WaterScanViewer(metadata["coordinate"], metadata["fixed_label"],
                                 float(metadata["max_extent"]))
        for row in rows:
            viewer.add_point(row)
        viewer.enable_review()
        return

    # Active-space defaults differ by mode for a real physical reason, see
    # the module docstring: CAS(4,4) is fine for bending but too small to
    # describe two bonds breaking, where it produces a visibly kinked curve.
    if args.active_electrons is None:
        args.active_electrons = 6 if args.coordinate == "stretch" else 4
    if args.active_orbitals is None:
        args.active_orbitals = 5 if args.coordinate == "stretch" else 4

    # Per-mode defaults, applied only where the user didn't say otherwise.
    if args.coordinate == "stretch":
        start = 0.7 if args.start is None else args.start
        stop = 2.4 if args.stop is None else args.stop
        step = 0.1 if args.step is None else args.step
        unit = "A"
    else:
        start = 70.0 if args.start is None else args.start
        stop = 160.0 if args.stop is None else args.stop
        step = 5.0 if args.step is None else args.step
        unit = "deg"

    if step <= 0:
        parser.error("--step must be positive")
    if stop < start:
        parser.error("--stop must be >= --start")

    # + step/2 so --stop is included despite floating-point drift
    values = np.arange(start, stop + step / 2, step)

    if args.coordinate == "stretch":
        geometries = [(v, args.angle) for v in values]
        fixed_label = f"angle fixed at {args.angle}$\\degree$"
        max_extent = float(values[-1])
    else:
        geometries = [(args.bond, v) for v in values]
        fixed_label = f"r(O-H) fixed at {args.bond} A"
        max_extent = float(args.bond)

    print(f"H2O {args.coordinate} scan: {len(values)} points from {values[0]:.2f} to "
          f"{values[-1]:.2f} {unit} (step {step} {unit})")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    # Very rough, measured on a laptop CPU -- just so a 40-minute run isn't a
    # surprise five points in.
    per_point = 20 if args.active_orbitals <= 4 else 130
    print(f"estimated runtime: ~{len(values) * per_point / 60:.0f} min "
          f"(~{per_point}s per point; the plot updates as each point lands)")

    viewer = WaterScanViewer(args.coordinate, fixed_label, max_extent)
    previous_params = None
    for i, (bond, angle) in enumerate(geometries):
        t0 = time.perf_counter()
        row = run_one_geometry(bond, angle, args.active_electrons, args.active_orbitals,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        viewer.add_point(row)
        scanned = row["bond"] if args.coordinate == "stretch" else row["angle"]
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:3d}/{len(values)}] {scanned:6.2f} {unit}   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({time.perf_counter() - t0:.1f}s)")

    best = min(viewer.rows, key=lambda r: r["vqe"])
    best_x = best["bond"] if args.coordinate == "stretch" else best["angle"]
    reference = (f"experimental r(O-H) is ~{EQUILIBRIUM_BOND} A" if args.coordinate == "stretch"
                 else f"experimental H-O-H angle is ~{EQUILIBRIUM_ANGLE} deg")
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at {best_x:.2f} {unit}  ({reference})")

    max_hf_gap = max(r["hf"] - r["exact"] for r in viewer.rows)
    print(f"largest HF error vs exact across the scan: {max_hf_gap * 1000:.1f} mHa "
          f"(this is the correlation energy VQE recovers and HF misses)")

    if not args.no_save:
        target = resolve_target(args.save, here, f"h2o_{args.coordinate}_scan")
        metadata = {
            "molecule": "H2O", "basis": BASIS,
            "coordinate": args.coordinate,
            "start": float(values[0]), "stop": float(values[-1]), "step": float(step),
            "unit": unit,
            "fixed_angle": float(args.angle), "fixed_bond": float(args.bond),
            "fixed_label": fixed_label,
            "max_extent": float(max_extent),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
        }
        json_path, csv_path = save_scan(target, metadata, viewer.rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        print(f"reopen it later without recomputing:  "
              f"python {display_path(Path(__file__))} --load {json_path.name}")

    viewer.enable_review()


if __name__ == "__main__":
    main()
