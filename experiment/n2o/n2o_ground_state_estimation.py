"""
n2o_ground_state_estimation.py -- nitrous oxide's (N2O) ground-state energy
by VQE, scanned along one internal coordinate at a time or as a full 2D
surface, plotted live next to a picture of the molecule deforming.

Simulator only. No IBM account, no queue, no quota.

N2O IS LINEAR AND *ASYMMETRIC* -- WHY THAT CHANGES THE SETUP
----------------------------------------------------------------
N2O is triatomic, so like water and BeH2 it has 3N-6 = 3 internal degrees of
freedom. But its atom order is N-N-O -- a terminal nitrogen, a central
nitrogen, and oxygen -- and the two bonds are chemically different (the N-N
bond has partial triple-bond character, N-O partial double-bond character;
N2O is usually drawn as the resonance structure :N=N=O: or N-=N+=O-). Unlike
water (two identical O-H bonds) or BeH2 in the sibling folder (two identical
Be-H bonds), **there is no symmetry to exploit here.** The two bond lengths
genuinely are two independent coordinates, not one coordinate doubled by a
mirror constraint.

That has a direct consequence for how a "surface" is defined in this script.
BeH2's surface scans its one shared bond length against the shared angle,
because collapsing to two coordinates was the whole trick. N2O already has
only two coordinates worth scanning as a pair -- **r(N-N) and r(N-O)** -- so
the natural surface here is bond-length x bond-length, at the angle fixed
linear, rather than bond x angle. That is a genuinely different kind of
surface from BeH2's, and the two are worth comparing side by side in a
write-up: one shows bond/angle coupling, this one shows bond/bond coupling
-- do the two stretches move independently, or is there cross-talk between
them?

FOUR MODES
-----------
  --coordinate nn_stretch   The N-N bond varied, N-O and the angle held
                           fixed (linear, 180 deg). This is the bond with
                           more double/triple-bond character, so expect the
                           deeper, stiffer well of the two stretches.

  --coordinate no_stretch   The N-O bond varied, N-N and the angle held
                           fixed. Expect a shallower well than N-N: this
                           bond is closer to a single/double bond and
                           dissociates into O + N2 rather than into atoms,
                           which is a different (and lower-energy) breakup
                           than pulling every bond apart at once.

  --coordinate bend         The N-N-O angle varied, both bonds held at their
                           equilibrium lengths. **The minimum is at 180 deg,
                           not in the middle of the range** -- like BeH2 and
                           unlike water, N2O is linear at equilibrium (the
                           central nitrogen carries no lone pairs pushing
                           the ends apart from 180 deg). So this curve is not
                           a symmetric well either: it falls toward 180 deg
                           from both sides and is flat exactly there by
                           symmetry -- bending either way costs the same to
                           first order. Compare directly against BeH2's own
                           bend curve; the shape should look the same for
                           the same reason.

  --coordinate surface      Both bond lengths varied on a grid, angle fixed
                           linear. A 2D valley bounded by two different
                           stiffnesses -- steep along N-N, gentler along
                           N-O -- so the contours come out elongated along
                           the N-O axis. Whether the floor of that valley
                           drifts (does the optimal N-O length change as
                           N-N stretches, the way BeH2's optimal bond drifts
                           with angle) is the interesting question a single
                           1D slice cannot answer -- see the sibling
                           ``n2o_spline``/``n2o_polynomial`` folders for
                           pulling that number out of a coarse grid, once
                           they exist for this molecule.

Run nn_stretch, no_stretch and bend, and compare all three -- three
differently-shaped curves from breaking or bending different parts of the
same triple-atom chain is the experiment.

ACTIVE SPACE -- SAME TRAP AS WATER'S, WORSE
----------------------------------------------
N2O in STO-3G is 22 electrons in 15 spatial orbitals -> 30 qubits taking
everything, hopelessly large for a statevector scan, let alone a surface.
``--active-electrons`` / ``--active-orbitals`` restrict to a
CAS(n_elec, n_orb) active space, exactly as in the H2O and BeH2 scripts.

**The defaults differ by mode, for the same reason they differ in the H2O
script.** Bending and the surface default to the cheaper CAS(4,4); the two
stretch modes default to the larger CAS(6,5). H2O's own docstring documents
measured evidence that CAS(4,4) is too small to describe a bond breaking --
it produces a curve that dips back down past its minimum instead of rising
monotonically toward dissociation, a physically impossible shape that is a
failure of the active space, not of VQE. N2O has *two* different bonds that
could each fail this way, so the same caution applies to both stretch modes,
doubled. :func:`_report_monotonicity` checks for it automatically on every
stretch scan.

Bear in mind a surface pays for whatever active space you choose N^2 times
over, which is why it defaults to the cheap CAS(4,4) rather than the safer
CAS(6,5) the stretch modes use -- treat the default surface as a first look
at the *shape* of the coupling, not as a converged energy, and raise
``--active-electrons``/``--active-orbitals`` for anything you plan to quote.

COST
-----
UCCSD's parameter count grows quickly with the active space, and SLSQP's
finite-difference gradient costs roughly one energy evaluation per
parameter per optimizer step, so cost is set far more by that count than by
the size of any single circuit evaluation. For reference, the sibling H2O
and BeH2 scripts measured (on a laptop CPU) roughly 20 s/point at CAS(4,4)
and 130 s/point at CAS(6,5) for a smaller, more symmetric molecule.
**N2O has more electrons in more orbitals even after freezing the same-size
core, so treat those numbers as an optimistic floor, not a benchmark of this
script** -- they have not been re-measured here. Start with a coarse
``--step`` (or a small ``--start``/``--stop`` window) the first time you run
a new mode, and let the live plot tell you whether a finer scan is worth the
wait.

Each geometry warm-starts its optimizer from the previous geometry's
solution (disable with ``--cold-start``), and the surface walks its grid in
serpentine order for the same reason BeH2's does -- so that warm start
always comes from an adjacent point.

REFERENCES
-----------
[1] Equilibrium bond lengths for N2O from a rovibrational analysis across
    twelve isotopologues: r(N-N) = 1.12729 A, r(N-O) = 1.18509 A, linear at
    180 deg: "Internuclear potential and equilibrium structure of N2O",
    https://core.ac.uk/works/38171884
[2] ActiveSpaceTransformer:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html

USAGE
------
  RUN A SCAN -- N-N stretch (N-O and angle fixed, linear)
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate nn_stretch
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate nn_stretch --start 0.9 --stop 2.2 --step 0.1
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate nn_stretch --start 0.9 --stop 1.4 --step 0.25 --no-save

  RUN A SCAN -- N-O stretch (N-N and angle fixed, linear)
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate no_stretch
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate no_stretch --start 0.95 --stop 2.3 --step 0.1

  RUN A SCAN -- bend (both bonds fixed at equilibrium)
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate bend
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate bend --start 90 --stop 180 --step 5

  RUN A SCAN -- the surface (r(N-N) x r(N-O), angle fixed linear)
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate surface
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate surface --step 0.15 --rno-step 0.15
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate surface --step 0.3 --rno-step 0.3 --no-save   # quick look

  RELOAD A FINISHED SCAN -- no VQE re-run. Either format works.
    python experiment/n2o/n2o_ground_state_estimation.py --load n2o_nn_stretch_scan.json
    python experiment/n2o/n2o_ground_state_estimation.py --load n2o_bend_scan.json
    python experiment/n2o/n2o_ground_state_estimation.py --load n2o_surface_scan.json

  A LARGER ACTIVE SPACE -- for a convergence spot-check
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate bend --active-electrons 6 --active-orbitals 5
    python experiment/n2o/n2o_ground_state_estimation.py --coordinate surface --active-electrons 6 --active-orbitals 5 --step 0.3 --rno-step 0.3

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

# N2O's experimental equilibrium geometry [1] -- used as the fixed coordinate(s)
# in whichever mode isn't being scanned.
EQUILIBRIUM_BOND_NN = 1.128    # Angstrom, r(N-N)
EQUILIBRIUM_BOND_NO = 1.185    # Angstrom, r(N-O)
EQUILIBRIUM_ANGLE = 180.0      # degrees -- N2O is linear


# ===========================================================================
#  Geometry -- asymmetric by construction; no mirror trick available
# ===========================================================================

def n2o_positions(r_nn: float, r_no: float, angle_deg: float) -> Dict[str, tuple]:
    """Atom positions for N2O (N1-N2-O), in the x-z plane.

    The central nitrogen, N2, sits at the origin. The terminal nitrogen, N1,
    sits `r_nn` Angstrom away along +z. Oxygen sits `r_no` Angstrom from N2,
    at `angle_deg` from the N2-N1 direction, rotated into the x-z plane.

    Unlike :func:`water_positions` or :func:`beh2_positions` in the sibling
    scripts, this does **not** split the angle symmetrically about an axis --
    there is nothing to be symmetric about, since N1 and O are different
    elements at different distances. At angle_deg = 180 the three atoms are
    exactly collinear with N2 in the middle: O lands at (0, -r_no), directly
    opposite N1 through the origin, which is N2O's real linear equilibrium
    shape.
    """
    rad = math.radians(angle_deg)
    n1 = (0.0, r_nn)
    o = (r_no * math.sin(rad), r_no * math.cos(rad))
    return {"N1": n1, "N2": (0.0, 0.0), "O": o}


def geometry_string(r_nn: float, r_no: float, angle_deg: float) -> str:
    pos = n2o_positions(r_nn, r_no, angle_deg)
    return (f"N {pos['N2'][0]} 0 {pos['N2'][1]}; "
            f"N {pos['N1'][0]} 0 {pos['N1'][1]}; "
            f"O {pos['O'][0]} 0 {pos['O'][1]}")


# ===========================================================================
#  The quantum chemistry -- one geometry at a time
# ===========================================================================

def build_problem(r_nn: float, r_no: float, angle_deg: float,
                  active_electrons: int, active_orbitals: int) -> ElectronicStructureProblem:
    """N2O at this geometry, restricted to a CAS(active_electrons, active_orbitals)
    active space. Ground state is an ordinary closed-shell singlet (X 1-Sigma+),
    so the default ``PySCFDriver`` call -- spin=0, RHF -- is already correct;
    see the O2 script in the sibling folder for the molecule where that is
    *not* true."""
    problem = PySCFDriver(atom=geometry_string(r_nn, r_no, angle_deg), basis=BASIS).run()
    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). N2O's inactive
    part is large -- three cores' worth at minimum -- so miss this and the
    energy is wrong by well over a hundred Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_geometry(r_nn: float, r_no: float, angle_deg: float, active_electrons: int,
                     active_orbitals: int, max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one N2O geometry.

    ``theta0`` warm-starts the optimizer from a previous geometry's solution.
    "exact" is the lowest eigenvalue of the mapped, active-space-reduced
    qubit Hamiltonian -- exact within the chosen model, not a full-basis FCI
    energy. "hf" is the Hartree-Fock circuit's expectation value on that same
    Hamiltonian, so all three numbers are directly comparable with each
    other, though none is a benchmark-quality absolute energy.
    """
    problem = build_problem(r_nn, r_no, angle_deg, active_electrons, active_orbitals)
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

    # Same row shape as the H2O/BeH2 scripts, extended with a second bond
    # length since N2O has no symmetry to fold r_nn and r_no into one number.
    return {
        "r_nn": float(r_nn),
        "r_no": float(r_no),
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

ATOM_RADII = {"N": 0.36, "O": 0.34}
ATOM_COLORS = {"N": "#3050f8", "O": "#ff4d4d"}   # CPK-ish: nitrogen blue, oxygen red


def _atom_element(name: str) -> str:
    """'N1' -> 'N', 'O' -> 'O'. Strips the trailing index off a label."""
    return name.rstrip("0123456789")


def _draw_molecule(ax, patches, bond_lines, row) -> None:
    """Move the atom circles and bond lines to match one scanned geometry."""
    pos = n2o_positions(row["r_nn"], row["r_no"], row["angle"])
    for name, (x, z) in pos.items():
        patches[name].center = (x, z)
    bond_lines[0].set_data([pos["N2"][0], pos["N1"][0]], [pos["N2"][1], pos["N1"][1]])
    bond_lines[1].set_data([pos["N2"][0], pos["O"][0]], [pos["N2"][1], pos["O"][1]])


def _build_molecule_panel(ax, span: float):
    """Set up the right-hand geometry panel. Returns (patches, bond_lines, label)."""
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
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
        for name in ("N1", "N2", "O")
    }
    for patch in patches.values():
        ax.add_patch(patch)
    label = ax.text(0, -span * 0.9, "", ha="center", fontsize=10)
    return patches, bond_lines, label


def _mol_label_text(row: Dict[str, Any]) -> str:
    return (f"r(N-N) = {row['r_nn']:.2f} A   r(N-O) = {row['r_no']:.2f} A   "
            f"angle = {row['angle']:.1f}$\\degree$\nE = {row['vqe']:.4f} Ha")


# ===========================================================================
#  Viewer 1 -- the 1D modes: live curve left, live molecule right
# ===========================================================================

class N2OScanViewer:
    """Live scan plot plus, once finished, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called per geometry as its energies come in
    -- the curve grows and the molecule redraws at the geometry just
    computed. Phase 2: :meth:`enable_review` turns the window into a
    scrubber over the finished scan.
    """

    _AXIS_LABELS = {
        "nn_stretch": "N-N bond length (Angstrom)",
        "no_stretch": "N-O bond length (Angstrom)",
        "bend": "N-N-O angle (degrees)",
    }

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

        (self.line_hf,) = self.ax_energy.plot([], [], "-", color="#d73027", label="Hartree-Fock")
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2, label="current")
        self.ax_energy.set_xlabel(self._AXIS_LABELS[coordinate])
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title(f"N$_2$O {coordinate} scan   ({fixed_label})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        span = max_extent + 0.5
        self.atom_patches, self.bonds, self.mol_label = _build_molecule_panel(self.ax_mol, span)

        self._redraw()

    def _x_of(self, row: Dict[str, Any]) -> float:
        return {"nn_stretch": row["r_nn"], "no_stretch": row["r_no"],
                "bend": row["angle"]}[self.coordinate]

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
        _draw_molecule(self.ax_mol, self.atom_patches, self.bonds, row)
        self.mol_label.set_text(_mol_label_text(row))
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def enable_review(self) -> None:
        if len(self.rows) > 1:
            slider_ax = self.fig.add_axes([0.12, 0.06, 0.5, 0.04])
            self.slider = Slider(slider_ax, "point", 0, len(self.rows) - 1,
                                 valinit=self.index, valstep=1)
            self.slider.on_changed(lambda v: self._show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax_energy.set_title(f"{self.ax_energy.get_title()}  --  LEFT/RIGHT arrows or slider")
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
        if self.slider is not None:
            self.slider.set_val(new_index)
        else:
            self._show(new_index)


# ===========================================================================
#  Viewer 2 -- the surface mode: r(N-N) x r(N-O)
# ===========================================================================

class N2OSurfaceViewer:
    """Live filling-in energy map on the left, live molecule on the right.

    Unlike BeH2's surface (bond x angle), both axes here are bond lengths --
    the two coordinates that N2O's lack of symmetry leaves genuinely
    independent. The angle is fixed (linear, by default) for the whole grid.
    """

    def __init__(self, rnn_values: np.ndarray, rno_values: np.ndarray, angle_deg: float) -> None:
        self.rnn_values = np.asarray(rnn_values, dtype=float)
        self.rno_values = np.asarray(rno_values, dtype=float)
        self.angle_deg = float(angle_deg)
        # Z is indexed [r_no, r_nn] so that imshow's rows run up the y axis.
        self.Z = np.full((len(self.rno_values), len(self.rnn_values)), np.nan)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        plt.ion()
        self.fig, (self.ax_map, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        d_nn = self._half_step(self.rnn_values)
        d_no = self._half_step(self.rno_values)
        self.extent = (self.rnn_values[0] - d_nn, self.rnn_values[-1] + d_nn,
                       self.rno_values[0] - d_no, self.rno_values[-1] + d_no)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)          # uncomputed cells stay transparent
        self.im = self.ax_map.imshow(self.Z, origin="lower", extent=self.extent,
                                     aspect="auto", cmap=cmap, interpolation="nearest")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_map, label="Energy (Hartree)")
        (self.map_marker,) = self.ax_map.plot([], [], "D", color="white", markersize=9,
                                              fillstyle="none", markeredgewidth=2)
        self.ax_map.set_xlabel("r(N-N) (Angstrom)")
        self.ax_map.set_ylabel("r(N-O) (Angstrom)")
        self.ax_map.set_title(f"N$_2$O energy surface   (angle fixed at {angle_deg:.0f}$\\degree$)")

        span = max(float(self.rnn_values[-1]), float(self.rno_values[-1])) + 0.5
        self.atom_patches, self.bonds, self.mol_label = _build_molecule_panel(self.ax_mol, span)

        self._redraw()

    @staticmethod
    def _half_step(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.5
        return float(values[1] - values[0]) / 2

    def _cell_of(self, row: Dict[str, Any]) -> Tuple[int, int]:
        """Which grid cell a row belongs in, matched by nearest value (a
        float round-trip through JSON makes exact equality unreliable)."""
        j = int(np.argmin(np.abs(self.rnn_values - row["r_nn"])))
        i = int(np.argmin(np.abs(self.rno_values - row["r_no"])))
        return i, j

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
        self.map_marker.set_data([row["r_nn"]], [row["r_no"]])
        _draw_molecule(self.ax_mol, self.atom_patches, self.bonds, row)
        self.mol_label.set_text(_mol_label_text(row))
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_surface(self, png_path: Path | None = None) -> None:
        """Render the finished grid as a real 3D surface beside a contour map."""
        grid_nn, grid_no = np.meshgrid(self.rnn_values, self.rno_values)
        masked = np.ma.masked_invalid(self.Z)

        fig = plt.figure(figsize=(13, 5.5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot_surface(grid_nn, grid_no, masked, cmap="viridis",
                          linewidth=0, antialiased=True, alpha=0.95)
        ax3d.set_xlabel("r(N-N) (A)")
        ax3d.set_ylabel("r(N-O) (A)")
        ax3d.set_zlabel("Energy (Ha)")
        ax3d.set_title("N$_2$O ground-state surface (VQE)")
        ax3d.view_init(elev=28, azim=-135)

        ax2d = fig.add_subplot(1, 2, 2)
        levels = 25
        contour = ax2d.contourf(grid_nn, grid_no, masked, levels=levels, cmap="viridis")
        ax2d.contour(grid_nn, grid_no, masked, levels=levels,
                     colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
        if self.rows:
            best = min(self.rows, key=lambda r: r["vqe"])
            ax2d.plot([best["r_nn"]], [best["r_no"]], "r*", markersize=16,
                      label=f"min {best['vqe']:.4f} Ha")
            ax2d.legend(loc="lower left")
        ax2d.set_xlabel("r(N-N) (Angstrom)")
        ax2d.set_ylabel("r(N-O) (Angstrom)")
        ax2d.set_title("Contours")

        fig.tight_layout()
        if png_path is not None:
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"saved {display_path(png_path)}")

    def enable_review(self) -> None:
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
    """``start`` to ``stop`` inclusive (the ``+ step/2`` absorbs float drift)."""
    return np.arange(start, stop + step / 2, step)


def serpentine(rnn_values: np.ndarray, rno_values: np.ndarray) -> List[Tuple[float, float]]:
    """Grid geometries ordered so consecutive points are always adjacent --
    see BeH2's twin for why this matters for the warm start."""
    geometries: List[Tuple[float, float]] = []
    for i, rno in enumerate(rno_values):
        row = rnn_values if i % 2 == 0 else rnn_values[::-1]
        geometries.extend((float(rnn), float(rno)) for rnn in row)
    return geometries


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N2O ground-state energy by VQE (local simulator), scanned along the "
                    "N-N stretch, the N-O stretch, the bend, or the N-N/N-O bond-length "
                    "surface. N2O has no symmetry between its two bonds, unlike H2O or BeH2."
    )
    parser.add_argument("--coordinate", choices=["nn_stretch", "no_stretch", "bend", "surface"],
                        default="nn_stretch",
                        help="'nn_stretch' (default): vary r(N-N), N-O and angle fixed. "
                             "'no_stretch': vary r(N-O), N-N and angle fixed. "
                             "'bend': vary the N-N-O angle, both bonds fixed. "
                             "'surface': vary r(N-N) and r(N-O) together on a grid, angle fixed.")
    parser.add_argument("--start", type=float, default=None,
                        help="scan start -- Angstrom for a stretch or the surface's r(N-N) "
                             "axis (default 0.9), degrees for bend (default 90)")
    parser.add_argument("--stop", type=float, default=None,
                        help="scan stop, inclusive -- default 2.2 A for nn_stretch/surface, "
                             "2.3 A for no_stretch, 180 deg for bend")
    parser.add_argument("--step", type=float, default=None,
                        help="scan step -- default 0.1 A for a stretch, 5 deg for bend, "
                             "0.15 A for the surface's r(N-N) axis")
    parser.add_argument("--rno-start", type=float, default=0.95,
                        help="[surface mode] first r(N-O) in Angstrom (default 0.95)")
    parser.add_argument("--rno-stop", type=float, default=1.65,
                        help="[surface mode] last r(N-O) in Angstrom (default 1.65)")
    parser.add_argument("--rno-step", type=float, default=0.15,
                        help="[surface mode] r(N-O) step in Angstrom (default 0.15)")
    parser.add_argument("--angle", type=float, default=EQUILIBRIUM_ANGLE,
                        help=f"[nn_stretch/no_stretch/surface] fixed N-N-O angle in degrees "
                             f"(default {EQUILIBRIUM_ANGLE}, i.e. linear)")
    parser.add_argument("--r-nn", type=float, default=EQUILIBRIUM_BOND_NN,
                        help=f"[no_stretch/bend] fixed r(N-N) in Angstrom "
                             f"(default {EQUILIBRIUM_BOND_NN})")
    parser.add_argument("--r-no", type=float, default=EQUILIBRIUM_BOND_NO,
                        help=f"[nn_stretch/bend] fixed r(N-O) in Angstrom "
                             f"(default {EQUILIBRIUM_BOND_NO})")
    parser.add_argument("--active-electrons", type=int, default=None,
                        help="electrons in the active space (default: 6 for a stretch, "
                             "4 for bend/surface -- see the module docstring)")
    parser.add_argument("--active-orbitals", type=int, default=None,
                        help="spatial orbitals in the active space (default: 5 for a "
                             "stretch -> 8 qubits, 4 for bend/surface -> 6 qubits)")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each geometry's optimization from the Hartree-Fock "
                             "state instead of warm-starting from the previous geometry.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json written by an earlier run) straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: "
                             "n2o_<coordinate>_scan.json/.csv next to this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save the results of this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="exit when the scan finishes instead of staying open in "
                             "review mode. The live plot still draws as the scan runs.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    if args.load:
        _replay(Path(args.load), here, args.no_show)
        return

    if args.active_electrons is None:
        args.active_electrons = 4 if args.coordinate in ("bend", "surface") else 6
    if args.active_orbitals is None:
        args.active_orbitals = 4 if args.coordinate in ("bend", "surface") else 5

    if args.coordinate == "surface":
        _run_surface(args, parser, here)
    else:
        _run_curve(args, parser, here)


def _finish(viewer, no_show: bool) -> None:
    if no_show:
        print("\nScan complete. --no-show was passed, so exiting instead of waiting.")
        return
    viewer.enable_review()


def _load_csv(path: Path) -> tuple:
    """Read a scan back from its ``.csv``. Metadata (mode, fixed values, grid
    axes) is inferred from which columns actually vary -- see BeH2's twin for
    the same trick. What is lost is the active space and ``vqe_params``,
    neither of which matters for replay.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")

    numeric = ("r_nn", "r_no", "angle", "hf", "vqe", "exact")
    rows = []
    for r in raw:
        row = {k: float(v) for k, v in r.items() if k in numeric and v != ""}
        for k in ("n_qubits", "n_params"):
            if r.get(k):
                row[k] = int(float(r[k]))
        rows.append(row)

    missing = {"r_nn", "r_no", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(
            f"{display_path(path)} is missing the column(s) {sorted(missing)}, so it is "
            f"not an N2O scan CSV. Expected columns: r_nn, r_no, angle, hf, vqe, exact."
        )

    rnns = sorted({r["r_nn"] for r in rows})
    rnos = sorted({r["r_no"] for r in rows})
    angles = sorted({r["angle"] for r in rows})

    if len(rnns) > 1 and len(rnos) > 1:
        metadata = {"scan_type": "surface", "coordinate": "surface",
                   "rnn_values": rnns, "rno_values": rnos, "fixed_angle": angles[0]}
    elif len(rnns) > 1:
        metadata = {"scan_type": "curve", "coordinate": "nn_stretch",
                   "fixed_label": f"r(N-O) fixed at {rnos[0]} A, angle {angles[0]}$\\degree$",
                   "max_extent": rnns[-1]}
    elif len(rnos) > 1:
        metadata = {"scan_type": "curve", "coordinate": "no_stretch",
                   "fixed_label": f"r(N-N) fixed at {rnns[0]} A, angle {angles[0]}$\\degree$",
                   "max_extent": rnos[-1]}
    else:
        metadata = {"scan_type": "curve", "coordinate": "bend",
                   "fixed_label": f"r(N-N)={rnns[0]} A, r(N-O)={rnos[0]} A",
                   "max_extent": max(rnns[0], rnos[0])}
    return metadata, rows


def _load_any(load_path: Path, here: Path) -> tuple:
    if load_path.suffix.lower() == ".csv":
        for candidate in (load_path, here / load_path.name):
            if candidate.exists():
                print("reading the CSV (metadata inferred from the columns; "
                      "the .json carries it explicitly)")
                return _load_csv(candidate)
        looked = "\n  ".join(display_path(c) for c in (load_path, here / load_path.name))
        raise SystemExit(f"no scan CSV found. Looked in:\n  {looked}")
    return load_scan(load_path, default_dir=here)


def _replay(load_path: Path, here: Path, no_show: bool = False) -> None:
    metadata, rows = _load_any(load_path, here)
    scan_type = metadata.get("scan_type", "curve")
    if metadata.get("active_electrons") is not None:
        space = f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
    elif rows and rows[0].get("n_qubits"):
        space = f"{rows[0]['n_qubits']} qubits, active space not recorded in the csv"
    else:
        space = "active space not recorded"
    print(f"loaded {len(rows)} points from {load_path} "
          f"({metadata.get('coordinate')} scan, {space}) -- no VQE re-run")

    if scan_type == "surface":
        viewer = N2OSurfaceViewer(np.asarray(metadata["rnn_values"], dtype=float),
                                  np.asarray(metadata["rno_values"], dtype=float),
                                  float(metadata.get("fixed_angle", EQUILIBRIUM_ANGLE)))
        for row in rows:
            viewer.add_point(row)
        viewer.show_surface()
        _finish(viewer, no_show)
    else:
        viewer = N2OScanViewer(metadata["coordinate"], metadata["fixed_label"],
                               float(metadata["max_extent"]))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, no_show)


def _run_curve(args, parser, here: Path) -> None:
    """The 1D modes: nn_stretch, no_stretch, or bend."""
    if args.coordinate == "bend":
        start = 90.0 if args.start is None else args.start
        stop = 180.0 if args.stop is None else args.stop
        step = 5.0 if args.step is None else args.step
        unit = "deg"
    else:
        start = 0.9 if args.start is None else args.start
        default_stop = 2.2 if args.coordinate == "nn_stretch" else 2.3
        stop = default_stop if args.stop is None else args.stop
        step = 0.1 if args.step is None else args.step
        unit = "A"

    if step <= 0:
        parser.error("--step must be positive")
    if stop < start:
        parser.error("--stop must be >= --start")

    values = inclusive_range(start, stop, step)

    if args.coordinate == "nn_stretch":
        geometries = [(float(v), args.r_no, args.angle) for v in values]
        fixed_label = f"r(N-O) fixed at {args.r_no} A, angle {args.angle}$\\degree$"
        max_extent = float(values[-1])
    elif args.coordinate == "no_stretch":
        geometries = [(args.r_nn, float(v), args.angle) for v in values]
        fixed_label = f"r(N-N) fixed at {args.r_nn} A, angle {args.angle}$\\degree$"
        max_extent = float(values[-1])
    else:
        geometries = [(args.r_nn, args.r_no, float(v)) for v in values]
        fixed_label = f"r(N-N)={args.r_nn} A, r(N-O)={args.r_no} A"
        max_extent = max(args.r_nn, args.r_no)

    print(f"N2O {args.coordinate} scan: {len(values)} points from {values[0]:.2f} to "
          f"{values[-1]:.2f} {unit} (step {step} {unit})")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    print("cost has not been benchmarked for N2O specifically -- see the module docstring's "
          "COST section for the (smaller-molecule) numbers this is extrapolated from. "
          "The plot updates as each point lands, so you can judge for yourself.")

    viewer = N2OScanViewer(args.coordinate, fixed_label, max_extent)
    rows = _scan(geometries, args, viewer, unit)

    best = min(rows, key=lambda r: r["vqe"])
    best_x = {"nn_stretch": best["r_nn"], "no_stretch": best["r_no"],
              "bend": best["angle"]}[args.coordinate]
    reference = ("literature r(N-N) is ~%.3f A" % EQUILIBRIUM_BOND_NN
                if args.coordinate == "nn_stretch"
                else "literature r(N-O) is ~%.3f A" % EQUILIBRIUM_BOND_NO
                if args.coordinate == "no_stretch"
                else "N2O is linear, so the minimum belongs at 180 deg")
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at {best_x:.2f} {unit}  ({reference})")
    _report_errors(rows)

    if args.coordinate in ("nn_stretch", "no_stretch"):
        _report_monotonicity(rows, args.coordinate)

    if not args.no_save:
        target = resolve_target(args.save, here, f"n2o_{args.coordinate}_scan")
        metadata = {
            "molecule": "N2O", "basis": BASIS,
            "scan_type": "curve",
            "coordinate": args.coordinate,
            "start": float(values[0]), "stop": float(values[-1]), "step": float(step),
            "unit": unit,
            "fixed_angle": float(args.angle), "fixed_r_nn": float(args.r_nn),
            "fixed_r_no": float(args.r_no),
            "fixed_label": fixed_label,
            "max_extent": float(max_extent),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
        }
        _save(target, metadata, rows)

    _finish(viewer, args.no_show)


def _run_surface(args, parser, here: Path) -> None:
    """The 2D mode: r(N-N) and r(N-O) both varied, angle fixed linear."""
    rnn_start = 0.9 if args.start is None else args.start
    rnn_stop = 2.2 if args.stop is None else args.stop
    rnn_step = 0.15 if args.step is None else args.step

    if rnn_step <= 0 or args.rno_step <= 0:
        parser.error("--step and --rno-step must be positive")
    if rnn_stop < rnn_start:
        parser.error("--stop must be >= --start")
    if args.rno_stop < args.rno_start:
        parser.error("--rno-stop must be >= --rno-start")

    rnn_values = inclusive_range(rnn_start, rnn_stop, rnn_step)
    rno_values = inclusive_range(args.rno_start, args.rno_stop, args.rno_step)
    geometries = serpentine(rnn_values, rno_values)

    print(f"N2O surface scan: {len(rnn_values)} r(N-N) x {len(rno_values)} r(N-O) "
          f"= {len(geometries)} points")
    print(f"  r(N-N) {rnn_values[0]:.2f} to {rnn_values[-1]:.2f} A (step {rnn_step} A)")
    print(f"  r(N-O) {rno_values[0]:.2f} to {rno_values[-1]:.2f} A (step {args.rno_step} A)")
    print(f"  angle fixed at {args.angle} deg")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    print("cost has not been benchmarked for N2O specifically -- a surface pays for "
          "whatever the active space costs per point, N times over. See the module "
          "docstring's COST section. The map fills in as each point lands.")

    viewer = N2OSurfaceViewer(rnn_values, rno_values, args.angle)
    rows = _scan([(rnn, rno, args.angle) for rnn, rno in geometries], args, viewer, "A")

    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at "
          f"r(N-N) = {best['r_nn']:.2f} A, r(N-O) = {best['r_no']:.2f} A")
    print(f"  (literature: r(N-N) ~{EQUILIBRIUM_BOND_NN} A, r(N-O) ~{EQUILIBRIUM_BOND_NO} A, linear)")
    _report_errors(rows)

    png_path = None
    if not args.no_save:
        target = resolve_target(args.save, here, "n2o_surface_scan")
        metadata = {
            "molecule": "N2O", "basis": BASIS,
            "scan_type": "surface",
            "coordinate": "surface",
            "rnn_values": [float(v) for v in rnn_values],
            "rno_values": [float(v) for v in rno_values],
            "rnn_step": float(rnn_step), "rno_step": float(args.rno_step),
            "fixed_angle": float(args.angle),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
        }
        _save(target, metadata, rows)
        png_path = target.with_name(target.stem + "_3d.png")

    viewer.show_surface(png_path)
    _finish(viewer, args.no_show)


def _scan(geometries, args, viewer, unit: str) -> List[Dict[str, Any]]:
    """Run VQE over a list of (r_nn, r_no, angle) geometries, feeding the viewer."""
    rows: List[Dict[str, Any]] = []
    previous_params = None
    total = len(geometries)
    for i, (r_nn, r_no, angle) in enumerate(geometries):
        t0 = time.perf_counter()
        row = run_one_geometry(r_nn, r_no, angle, args.active_electrons, args.active_orbitals,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:4d}/{total}] r_nn={r_nn:5.2f} r_no={r_no:5.2f} a={angle:6.1f}   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({time.perf_counter() - t0:.1f}s)")
    return rows


def _report_errors(rows: List[Dict[str, Any]]) -> None:
    """The classical-vs-quantum error summary, in milli-Hartree. See the
    BeH2/H2O scripts for why these two are kept separate."""
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


def _report_monotonicity(rows: List[Dict[str, Any]], coordinate: str) -> None:
    """Flag the active-space failure mode documented in the H2O script's
    docstring -- past its minimum, a bond-breaking curve must rise, never
    dip. N2O has two bonds that can each fail this way independently."""
    key = "r_nn" if coordinate == "nn_stretch" else "r_no"
    ordered = sorted(rows, key=lambda r: r[key])
    best_at = min(range(len(ordered)), key=lambda i: ordered[i]["exact"])
    dips = [
        (ordered[i][key], ordered[i + 1][key])
        for i in range(best_at, len(ordered) - 1)
        if ordered[i + 1]["exact"] < ordered[i]["exact"]
    ]
    if dips:
        print(f"\n  WARNING: the exact curve dips downward past its minimum, between "
              + ", ".join(f"{a:.2f}-{b:.2f} A" for a, b in dips))
        print("  A ground-state curve must rise monotonically toward dissociation. This is")
        print("  an active space too small for this bond, not a VQE failure. Try a larger")
        print("  --active-orbitals / --active-electrons.")
    else:
        print("  curve rises monotonically past its minimum -- active space looks adequate "
              f"for the {('N-N' if coordinate == 'nn_stretch' else 'N-O')} bond")


def _save(target: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    json_path, csv_path = save_scan(target, metadata, rows)
    print(f"saved {display_path(json_path)}")
    print(f"saved {display_path(csv_path)}")
    print(f"reopen it later without recomputing:  "
          f"python {display_path(Path(__file__))} --load {json_path.name}")


if __name__ == "__main__":
    main()