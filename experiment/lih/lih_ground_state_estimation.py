"""
lih_ground_state_estimation.py -- LiH's ground-state energy by VQE across a
range of bond lengths, plotted live next to a picture of the molecule pulling
itself apart.

Simulator only. No IBM account, no queue, no quota. (The hardware-capable
LiH workflow lives in ``../../VQE demo on IBM Quantum Computer/qiskit_vqe.py``;
this script is the interactive, savable, interpolatable version of the same
molecule.)

LiH HAS EXACTLY ONE DEGREE OF FREEDOM
---------------------------------------
LiH has two atoms. A linear molecule with N atoms has 3N-5 internal
coordinates, so LiH has 3(2) - 5 = **1**: the Li-H bond length. There is no
bond angle, because an angle needs three atoms to define it. A potential-energy
*surface* is therefore geometrically impossible for LiH -- there is no second
axis to plot against. One curve is not a shortcut here, it is the complete
description of the geometry.

So this script has no ``--coordinate`` flag and no surface mode. Compare O2 in
the sibling folder (also diatomic, also one curve) against BeH2 (three atoms,
3N-6 = 3 coordinates, reduced to 2 by symmetry, which is what it takes to make
a surface).

THE MULTIPLICITY QUESTION, ASKED AND ANSWERED
-----------------------------------------------
The O2 script documents a trap: an open-shell molecule handed to a default
``PySCFDriver`` call comes back as the wrong spin state, silently, with no
error. That trap is worth checking for every new molecule rather than assuming.

For LiH the answer is that it does not apply. LiH has 4 electrons and a
closed-shell **singlet** ground state, X 1-Sigma-plus [1]. Both electron pairs
are paired, so the default ``spin=0`` and RHF path is correct here, and this
script uses it deliberately rather than by omission.

Do not read that as "the default is always fine". It is fine *because it was
checked*. Any radical, any triplet, any odd-electron species needs the O2
treatment instead.

WHAT THE CURVE SHOULD LOOK LIKE, AND THE ONE INTERESTING THING ABOUT IT
-------------------------------------------------------------------------
A textbook Morse shape: a repulsive wall at short bond length, a minimum near
1.60 A, then a rise that flattens as the two atoms separate.

The interesting part is the **Hartree-Fock error**, which behaves quite
differently from H2's. LiH is an ionic bond -- near equilibrium it is roughly
Li(+)H(-) -- and restricted Hartree-Fock, being a single Slater determinant
with doubly occupied orbitals, cannot dissociate it correctly. RHF insists on
splitting the electron pair evenly between the two atoms all the way out,
which is the wrong dissociation limit. So the HF error does not merely grow as
you pull the atoms apart, it grows *without flattening*.

Measured on this script's default grid, at CAS(2,5):

    r(Li-H)    |HF - exact|
    1.40 A        17.7 mHa
    1.65 A        21.1 mHa
    2.40 A        47.0 mHa
    3.15 A        99.5 mHa
    3.90 A       153.2 mHa

That is a factor of nine across the scan, and it is still climbing at the
right-hand edge. It is the cleanest picture of "a correlated method is not
optional" in this project short of O2, and unlike O2 it costs minutes rather
than tens of minutes to produce.

ACTIVE SPACE -- AND THE MEASUREMENT THAT PICKED IT
----------------------------------------------------
LiH in STO-3G is 4 electrons in 6 spatial orbitals (Li contributes 1s, 2s and
three 2p; H contributes 1s) -> 12 qubits if you take everything.

The default here is **CAS(2,5)**: 2 electrons in 5 spatial orbitals, which
freezes the Li 1s core and keeps everything else. After the parity mapper's
two-qubit reduction that is **8 qubits and 24 UCCSD parameters**. It is
exactly the space ``FreezeCoreTransformer`` produces for LiH in this basis, so
it matches the IBM demo elsewhere in this repo.

Three candidate spaces were measured on the exact column before choosing, and
the result decided the default:

    r(Li-H)    CAS(2,3)     CAS(2,5)     CAS(4,6)
    1.65 A    -7.860916    -7.880947    -7.881178
    2.90 A    -7.733899    -7.802147    -7.802478
    3.40 A    -7.719002    -7.789145    -7.789499
    3.65 A    -7.731355    -7.786160    -7.786513   <-- CAS(2,3) turns back up
    3.90 A    -7.744616    -7.784406    -7.784755

**CAS(2,3) is broken**, and broken in precisely the way the H2O and BeH2
scripts warn about. Its curve dips back *downward* past 3.40 A instead of
rising monotonically toward dissociation -- three orbitals cannot describe the
bond breaking -- and VQE would still reproduce that wrong active-space energy
to under 0.001 mHa. The failure is in the model, not the algorithm.

CAS(2,5) is monotonic and sits within **0.35 mHa of CAS(4,6) at every point**
on the grid. Since CAS(4,6) costs 10 qubits and 92 UCCSD parameters -- roughly
four times the parameters for a third of a milli-Hartree -- CAS(2,5) is the
defensible default and CAS(4,6) is available for a convergence spot-check.

``--active-electrons`` / ``--active-orbitals`` expose all of this. The
automatic monotonicity check at the end of every run will tell you if the
space you picked has broken the physics.

COST
-----
Runtime is dominated not by circuit simulation -- one energy evaluation on 8
qubits is well under a second -- but by the *number* of evaluations.

Measured on a container CPU at the default CAS(2,5) with SLSQP: **176 energy
evaluations and about 21 s per point**, so the default 13-point scan is
roughly **4-5 minutes**. Time your own first point rather than trusting that
number; the scan prints seconds per point as it goes.

The optimizer is chosen automatically from the parameter count, because the
right answer changes with the active space:

    <= 60 parameters   SLSQP    finite-difference gradients, ~1 evaluation
                                per parameter per step. Thorough and cheap
                                enough at this size. CAS(2,5) is 24.
    >  60 parameters   COBYLA   gradient-free. CAS(4,6) is 92 parameters,
                                where SLSQP's gradient cost turns punishing.

``--optimizer`` overrides the choice. Note that COBYLA typically spends its
entire ``--max-iterations`` budget rather than converging early, so under
COBYLA the cost per point is set by the cap, not by how hard the geometry is.

Each geometry warm-starts its optimizer from the previous geometry's solution,
since neighbouring points have similar wavefunctions. ``--cold-start``
disables it.

A NOTE ON WHAT "EXACT" MEANS HERE
-----------------------------------
"exact" is the lowest eigenvalue of the *mapped, active-space-reduced* qubit
Hamiltonian. It is exact within the chosen model, not a full-basis FCI energy,
and STO-3G is a small basis. Concretely: the CAS(2,5)/STO-3G curve has its
minimum at **1.547 A**, while the experimental bond length is 1.5949 A [1].
That 0.048 A gap is a basis-set limitation and nothing to do with VQE.

This matters when reading the interpolation folders, where it would be easy to
credit or blame the interpolant for an error that belongs to the basis set.
See the note there.

REFERENCES
-----------
[1] NIST Chemistry WebBook, constants of diatomic molecules -- LiH's
    X 1-Sigma-plus ground state and r_e = 1.5949 A:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7580678&Units=SI&Mask=1000
[2] ActiveSpaceTransformer and its orbital-selection rules -- without an
    explicit orbital list it takes the orbitals around the Fermi level, which
    for LiH means freezing the Li 1s core:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html
[3] PySCFDriver, including the spin convention (spin = 2S) and MethodType --
    not needed for LiH, which is closed-shell, but the reason that was checked:
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.drivers.PySCFDriver.html

USAGE
------
Every case, explicitly. LiH is diatomic, so bond length is the only geometric
coordinate -- one curve, no surface. See the top of this docstring.

  RUN A SCAN
    python experiment/lih/lih_ground_state_estimation.py
    python experiment/lih/lih_ground_state_estimation.py --start 0.9 --stop 3.9 --step 0.25
    python experiment/lih/lih_ground_state_estimation.py --start 1.0 --stop 2.5 --step 0.1   # finer, ~2.5x the time
    python experiment/lih/lih_ground_state_estimation.py --start 1.6 --stop 1.6 --step 0.25 --no-save   # one point

  RELOAD A FINISHED SCAN -- no VQE re-run. Either format works.
    python experiment/lih/lih_ground_state_estimation.py --load lih_scan.json
    python experiment/lih/lih_ground_state_estimation.py --load lih_scan.csv

  ACTIVE SPACE -- the default is CAS(2,5); see the docstring for why
    python experiment/lih/lih_ground_state_estimation.py --active-electrons 4 --active-orbitals 6
    python experiment/lih/lih_ground_state_estimation.py --active-electrons 2 --active-orbitals 3   # deliberately broken, see above

  OPTIMIZER -- chosen from the parameter count unless you say otherwise
    python experiment/lih/lih_ground_state_estimation.py --optimizer slsqp
    python experiment/lih/lih_ground_state_estimation.py --optimizer cobyla --max-iterations 3000
    python experiment/lih/lih_ground_state_estimation.py --cold-start

  OTHER KNOBS
    --save myname / --no-save    where to write, or don't
    --no-show                    exit when the scan finishes instead of waiting on the window
"""

from __future__ import annotations

import argparse
import csv
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

# LiH's experimental equilibrium bond length [1]. Printed as a reference next
# to whatever minimum the scan finds -- but see MODEL_BOND below before
# reading any comparison against it too closely.
EQUILIBRIUM_BOND = 1.5949   # Angstrom

# Where the CAS(2,5)/STO-3G model actually puts its minimum, measured on a
# 0.02 A reference scan of the exact column. The 0.048 A gap between this and
# EQUILIBRIUM_BOND is the basis set's error, not VQE's and not the grid's.
# Reported separately so a coarse-grid result can be judged against the model
# it came from rather than against an accuracy the model never had.
MODEL_BOND = 1.5474         # Angstrom

# Above this many UCCSD parameters, SLSQP's finite-difference gradient (about
# one energy evaluation per parameter per step) stops being worth it and
# gradient-free COBYLA is cheaper. CAS(2,5) is 24 parameters; CAS(4,6) is 92.
SLSQP_PARAMETER_LIMIT = 60


# ===========================================================================
#  Geometry
# ===========================================================================

def geometry_string(distance: float) -> str:
    """LiH with the two atoms `distance` Angstrom apart, centred on the origin.

    Li on the left, H on the right, matching the order the viewer draws them.
    Centring on the origin keeps the molecule from drifting across the panel
    as the bond stretches.
    """
    return f"Li 0 0 {-distance / 2}; H 0 0 {distance / 2}"


# ===========================================================================
#  The quantum chemistry -- one bond length at a time
# ===========================================================================

def build_problem(distance: float, active_electrons: int,
                  active_orbitals: int) -> ElectronicStructureProblem:
    """LiH at this bond length, restricted to a CAS(active_electrons,
    active_orbitals) active space.

    No ``spin`` or ``method`` argument: LiH's ground state is a closed-shell
    singlet, so the driver's defaults (spin=0, RHF) are correct -- see the
    module docstring, which explains why that was verified rather than
    assumed. Without an explicit orbital list the transformer takes the
    orbitals around the Fermi level [2], which at CAS(2,5) means freezing the
    Li 1s core and keeping the rest.
    """
    problem = PySCFDriver(atom=geometry_string(distance), basis=BASIS).run()
    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). For LiH the frozen
    Li 1s pair alone is worth several Hartree, so miss this and the energy is
    wrong by more than the entire binding energy."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def choose_optimizer(requested: str, n_params: int) -> str:
    """Resolve ``--optimizer auto`` against the parameter count.

    The right optimizer is a function of the active space, not of the
    molecule, so hard-coding one would be wrong the moment someone passes
    ``--active-orbitals 6``. See SLSQP_PARAMETER_LIMIT.
    """
    if requested != "auto":
        return requested
    return "slsqp" if n_params <= SLSQP_PARAMETER_LIMIT else "cobyla"


def run_one_distance(distance: float, active_electrons: int, active_orbitals: int,
                     optimizer: str = "auto", max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one Li-H distance.

    ``theta0`` warm-starts the optimizer from the previous bond length's
    solution, since neighbouring points have similar wavefunctions.

    "exact" here is the lowest eigenvalue of the mapped, active-space-reduced
    qubit Hamiltonian -- exact within the chosen model, not a full-basis FCI
    energy. "hf" is the Hartree-Fock circuit's expectation value on that same
    Hamiltonian rather than PySCF's SCF energy, so the three numbers are
    directly comparable with each other.
    """
    problem = build_problem(distance, active_electrons, active_orbitals)
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

    resolved = choose_optimizer(optimizer, ansatz.num_parameters)
    method = {"cobyla": "COBYLA", "slsqp": "SLSQP"}[resolved]
    result = minimize(cost, start, method=method, options={"maxiter": max_iterations})

    # Same row shape as the H2 and O2 scripts. The classical and quantum
    # errors are not stored as columns -- they are differences of these three
    # numbers, so they are derived where they are reported rather than
    # duplicated into every row.
    return {
        "distance": float(distance),
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
        "n_qubits": hamiltonian.num_qubits,
        "n_params": ansatz.num_parameters,
        "n_evaluations": int(getattr(result, "nfev", 0)),
        "vqe_params": np.asarray(result.x),
    }


# ===========================================================================
#  The window -- live curve on the left, live molecule on the right
# ===========================================================================

# Drawn radii in Angstrom. Unlike H2 and O2, LiH is heteronuclear, so the two
# circles differ -- roughly the covalent radii (Li 1.28, H 0.31) scaled down
# so the panel stays readable out at 3.9 A without the lithium swallowing the
# hydrogen whole.
ATOM_RADII = {"Li": 0.45, "H": 0.22}
ATOM_COLORS = {"Li": "#cc80ff", "H": "#f2f2f2"}   # CPK: lithium is violet


class LiHDissociationViewer:
    """Two live panels plus, once the scan is done, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called once per bond length as its energies
    come in -- the curve grows and the molecule redraws at the geometry just
    computed. Phase 2: :meth:`enable_review` turns the window into a scrubber
    over the finished curve.
    """

    def __init__(self, max_distance: float) -> None:
        plt.ion()
        self.fig, (self.ax_energy, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        # -- left panel: the dissociation curve ------------------------
        (self.line_hf,) = self.ax_energy.plot([], [], "-", color="#d73027", label="Hartree-Fock")
        # Hollow VQE markers so the green exact crosses stay visible
        # underneath them.
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2, label="current")
        self.ax_energy.set_xlabel("Li-H bond length (Angstrom)")
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title("LiH dissociation curve")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        # -- right panel: the molecule itself --------------------------
        span = max_distance / 2 + max(ATOM_RADII.values()) + 0.3
        self.ax_mol.set_xlim(-span, span)
        self.ax_mol.set_ylim(-span / 2.6, span / 2.6)
        self.ax_mol.set_aspect("equal")
        self.ax_mol.set_xticks([])
        self.ax_mol.set_yticks([])
        self.ax_mol.set_title("Geometry")
        (self.bond,) = self.ax_mol.plot([], [], "-", color="#888888", linewidth=3, zorder=1)
        self.atoms = {
            element: Circle((0, 0), ATOM_RADII[element], facecolor=ATOM_COLORS[element],
                            edgecolor="black", linewidth=1.5, zorder=2)
            for element in ("Li", "H")
        }
        for atom in self.atoms.values():
            self.ax_mol.add_patch(atom)
        self.atom_labels = {
            element: self.ax_mol.text(0, 0, element, ha="center", va="center",
                                      fontsize=9, zorder=3)
            for element in ("Li", "H")
        }
        self.mol_label = self.ax_mol.text(0, -span / 2.6 + 0.12, "", ha="center", fontsize=11)

        self._redraw()

    # -- phase 1: live during the scan ---------------------------------

    def add_point(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        ds = [r["distance"] for r in self.rows]
        self.line_hf.set_data(ds, [r["hf"] for r in self.rows])
        self.line_vqe.set_data(ds, [r["vqe"] for r in self.rows])
        self.line_exact.set_data(ds, [r["exact"] for r in self.rows])
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        self.index = len(self.rows) - 1
        self._show(self.index)

    def _show(self, index: int) -> None:
        row = self.rows[index]
        d = row["distance"]
        self.marker.set_data([d], [row["vqe"]])
        self.atoms["Li"].center = (-d / 2, 0.0)
        self.atoms["H"].center = (d / 2, 0.0)
        self.atom_labels["Li"].set_position((-d / 2, 0.0))
        self.atom_labels["H"].set_position((d / 2, 0.0))
        self.bond.set_data([-d / 2, d / 2], [0.0, 0.0])
        self.mol_label.set_text(f"d = {d:.2f} A     E = {row['vqe']:.4f} Ha")
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -- phase 2: review the finished curve -----------------------------

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
        # Driving the slider re-enters _show() through on_changed, keeping the
        # slider handle and the plotted marker in sync.
        if self.slider is not None:
            self.slider.set_val(new_index)
        else:
            self._show(new_index)


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiH dissociation curve by VQE (local simulator). LiH is diatomic, so "
                    "bond length is its only geometric degree of freedom -- one curve, "
                    "no surface."
    )
    parser.add_argument("--start", type=float, default=0.9,
                        help="shortest Li-H bond length in Angstrom (default 0.9)")
    parser.add_argument("--stop", type=float, default=3.9,
                        help="longest Li-H bond length in Angstrom, inclusive (default 3.9). "
                             "Runs well past the 1.59 A minimum on purpose -- the "
                             "Hartree-Fock failure is the interesting part and it only "
                             "shows up out here.")
    parser.add_argument("--step", type=float, default=0.25,
                        help="spacing between bond lengths in Angstrom (default 0.25). "
                             "Deliberately coarse, and deliberately not a divisor of the "
                             "equilibrium bond length -- no grid point lands on the "
                             "minimum, which is what gives the interpolation folders "
                             "something real to recover.")
    parser.add_argument("--active-electrons", type=int, default=2,
                        help="electrons in the active space (default 2)")
    parser.add_argument("--active-orbitals", type=int, default=5,
                        help="spatial orbitals in the active space (default 5 -> 8 qubits). "
                             "CAS(2,5) freezes the Li 1s core and is within 0.35 mHa of the "
                             "full CAS(4,6) everywhere on the default grid -- see the module "
                             "docstring. CAS(2,3) is measurably broken; try it and watch the "
                             "monotonicity check fire.")
    parser.add_argument("--optimizer", choices=["auto", "slsqp", "cobyla"], default="auto",
                        help="'auto' (default) picks SLSQP at or below "
                             f"{SLSQP_PARAMETER_LIMIT} UCCSD parameters and COBYLA above it. "
                             "SLSQP estimates its gradient by finite differences, costing "
                             "about one energy evaluation per parameter per step, which is "
                             "fine at CAS(2,5)'s 24 parameters and punishing at CAS(4,6)'s 92. "
                             "COBYLA is gradient-free but spends its whole budget.")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="optimizer iteration cap (default 1000). SLSQP converges well "
                             "inside this at CAS(2,5) (~176 evaluations/point), so the cap "
                             "does nothing there. Under COBYLA the cap is what sets the cost "
                             "per point, because COBYLA spends the entire budget.")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each bond length's optimization from the Hartree-Fock "
                             "state instead of warm-starting from the previous solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json or .csv written by an earlier run) in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: lih_scan.json/.csv next to "
                             "this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save the results of this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="exit when the scan finishes instead of staying open in review "
                             "mode. The live plot still draws as the scan runs; it just "
                             "doesn't block at the end. Everything is still saved, and "
                             "--load reopens it interactively later.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # -- replay a finished scan, no VQE at all -------------------------
    if args.load:
        metadata, rows = _load_any(Path(args.load), here)

        # LiH has one degree of freedom, so every LiH scan is a curve. Check
        # anyway: a mis-pointed --load at a BeH2 surface file would otherwise
        # fail much later with something unhelpful about missing columns.
        scan_type = metadata.get("scan_type", "curve")
        if scan_type != "curve":
            raise SystemExit(
                f"{display_path(Path(args.load))} records scan_type={scan_type!r}, but LiH is "
                f"diatomic and can only ever produce a curve. That file is from another "
                f"molecule."
            )

        if metadata.get("active_electrons") is not None:
            space = f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
        elif rows and rows[0].get("n_qubits"):
            space = f"{rows[0]['n_qubits']} qubits, active space not recorded in the csv"
        else:
            space = "active space not recorded"
        print(f"loaded {len(rows)} points from {args.load} ({space}) -- no VQE re-run")
        viewer = LiHDissociationViewer(float(metadata["max_distance"]))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, args.no_show)
        return

    if args.step <= 0:
        parser.error("--step must be positive")
    if args.stop < args.start:
        parser.error("--stop must be >= --start")
    if args.active_electrons > 2 * args.active_orbitals:
        parser.error(f"{args.active_electrons} electrons will not fit in "
                     f"{args.active_orbitals} spatial orbitals (max "
                     f"{2 * args.active_orbitals})")

    # + step/2 so that --stop itself is included despite floating-point drift
    distances = np.arange(args.start, args.stop + args.step / 2, args.step)

    print(f"LiH scan: {len(distances)} bond lengths from {distances[0]:.2f} to "
          f"{distances[-1]:.2f} A (step {args.step} A)")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals}), "
          f"basis {BASIS}, closed-shell singlet (RHF)")

    viewer = LiHDissociationViewer(float(distances[-1]))
    rows: List[Dict[str, Any]] = []
    previous_params = None
    for i, d in enumerate(distances):
        t0 = time.perf_counter()
        row = run_one_distance(d, args.active_electrons, args.active_orbitals,
                               optimizer=args.optimizer,
                               max_iterations=args.max_iterations,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            resolved = choose_optimizer(args.optimizer, row["n_params"])
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters, "
                  f"optimizer {resolved.upper()}"
                  f"{' [auto]' if args.optimizer == 'auto' else ''})")
        print(f"  [{i + 1:3d}/{len(distances)}] d={d:.2f} A   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")

    _report(rows)

    if not args.no_save:
        target = resolve_target(args.save, here, "lih_scan")
        metadata = {
            "molecule": "LiH", "basis": BASIS,
            "scan_type": "curve",
            "coordinate": "bond",
            "degrees_of_freedom": 1,
            "multiplicity": "singlet",
            "spin_2s": 0,
            "scf_method": "RHF",
            "start": float(distances[0]), "stop": float(distances[-1]),
            "step": float(args.step),
            "unit": "A",
            "max_distance": float(distances[-1]),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
            "optimizer": choose_optimizer(args.optimizer, rows[0]["n_params"]),
            "max_iterations": int(args.max_iterations),
            "experimental_bond": EQUILIBRIUM_BOND,
            "model_bond": MODEL_BOND,
        }
        json_path, csv_path = save_scan(target, metadata, rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        print(f"reopen it later without recomputing:  "
              f"python {display_path(Path(__file__))} --load {json_path.name}")
        print(f"interpolate it:  python experiment/lih_spline/lih_spline_interpolation.py")

    _finish(viewer, args.no_show)


def _load_csv(path: Path) -> tuple:
    """Read a LiH scan back from its ``.csv`` instead of its ``.json``.

    The JSON is the canonical save -- it carries the metadata needed to
    rebuild the plot. A CSV is a flat table with none, so ``max_distance`` is
    taken from the data itself. Unlike O2, nothing important is lost: LiH has
    only ever one spin state and one coordinate, so a CSV reload is faithful.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")

    numeric = ("distance", "hf", "vqe", "exact")
    rows = []
    for r in raw:
        row = {k: float(v) for k, v in r.items() if k in numeric and v != ""}
        for k in ("n_qubits", "n_params", "n_evaluations"):
            if r.get(k):
                row[k] = int(float(r[k]))
        rows.append(row)

    missing = {"distance", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(
            f"{display_path(path)} is missing the column(s) {sorted(missing)}, so it is "
            f"not a LiH scan CSV. Expected columns: distance, hf, vqe, exact."
        )

    metadata = {
        "scan_type": "curve",
        "max_distance": max(r["distance"] for r in rows),
    }
    return metadata, rows


def _load_any(load_path: Path, here: Path) -> tuple:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    if load_path.suffix.lower() == ".csv":
        for candidate in (load_path, here / load_path.name):
            if candidate.exists():
                return _load_csv(candidate)
        looked = "\n  ".join(display_path(c) for c in (load_path, here / load_path.name))
        raise SystemExit(f"no scan CSV found. Looked in:\n  {looked}")
    return load_scan(load_path, default_dir=here)


def _finish(viewer: LiHDissociationViewer, no_show: bool) -> None:
    """Either hand the window over for review, or return so the process exits.

    ``enable_review`` calls ``plt.show()``, which blocks until the window is
    closed -- fine when you are sitting there, useless when you have queued
    several scans and gone to bed.
    """
    if no_show:
        print("\nScan complete. --no-show was passed, so exiting instead of waiting.")
        return
    viewer.enable_review()


def _report(rows: List[Dict[str, Any]]) -> None:
    """Final summary: the minimum, the errors, and the sanity checks."""
    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at d = {best['distance']:.2f} A")
    print(f"  experimental Li-H bond length : {EQUILIBRIUM_BOND} A")
    print(f"  CAS(2,5)/STO-3G model minimum : {MODEL_BOND} A "
          f"(the basis set's own answer, {abs(MODEL_BOND - EQUILIBRIUM_BOND):.4f} A "
          f"short of experiment)")

    step = rows[1]["distance"] - rows[0]["distance"] if len(rows) > 1 else 0.0
    if step > 0.05:
        print(f"  the grid spacing is {step:.2f} A, so the number above is the lowest")
        print(f"  *sample*, not the minimum of the curve. Fit an interpolant through these")
        print(f"  points and minimize that instead (see experiment/lih_spline/).")
        print(f"  Judge that result against the model minimum, not against experiment --")
        print(f"  the basis-set error is far larger than the interpolation error and")
        print(f"  would swamp the comparison.")

    _report_errors(rows)
    _report_monotonicity(rows)


def _report_errors(rows: List[Dict[str, Any]]) -> None:
    """The classical-vs-quantum error summary, in milli-Hartree.

    These two numbers measure different things and it is worth keeping them
    apart. |HF - exact| is *physics*: the correlation energy a single Slater
    determinant cannot represent. |VQE - exact| is *numerics*: how close the
    optimizer got to the exact answer for this Hamiltonian.

    Both are derived here rather than stored per row -- they are just
    differences of the hf/vqe/exact columns already in the file.

    The fraction recovered is reported alongside the absolute error because it
    is the more defensible number on a curve where the correlation energy
    itself varies by an order of magnitude. For LiH at CAS(2,5) it stays
    pinned at essentially 100%: UCCSD is a single-reference ansatz and LiH
    stays single-reference even when stretched, so there is nothing here for
    it to struggle with. That is a real (if undramatic) result -- contrast O2,
    where the fraction is the only error measure that behaves sensibly.
    """
    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))

    print(f"\nlargest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")

    # Fraction of the correlation energy VQE recovered, at the point where
    # there is the most of it to recover. Guard against a vanishing
    # denominator near any geometry where HF happens to be near-exact.
    worst = max(rows, key=lambda r: abs(r["hf"] - r["exact"]))
    available = worst["hf"] - worst["exact"]
    if abs(available) > 1e-9:
        recovered = (worst["hf"] - worst["vqe"]) / available
        print(f"correlation energy recovered at d = {worst['distance']:.2f} A "
              f"(where there is most of it): {recovered * 100:.2f}%")

    if max_vqe * 1000 > 1.0:
        print("  note: |VQE - exact| above 1 mHa suggests the optimizer did not fully")
        print("  converge somewhere. At CAS(2,5) this should be ~1e-4 mHa, so anything")
        print("  near 1 mHa means something is wrong -- try --cold-start, or")
        print("  --optimizer slsqp if you overrode it to COBYLA.")


def _report_monotonicity(rows: List[Dict[str, Any]]) -> None:
    """Flag the active-space failure mode described in the module docstring.

    Past its minimum a dissociation curve must rise as the atoms separate. A
    curve that turns back downward is the signature of an active space too
    small to describe the bond breaking. Worth catching automatically: it is
    easy to miss by eye on a coarse scan, VQE will reproduce the wrong energy
    to full precision without complaint, and it invalidates everything
    downstream in the interpolation folders.

    For LiH this is not hypothetical -- CAS(2,3) trips it past 3.40 A. Run
    ``--active-electrons 2 --active-orbitals 3`` and watch.
    """
    ordered = sorted(rows, key=lambda r: r["distance"])
    if len(ordered) < 3:
        return
    best_at = min(range(len(ordered)), key=lambda i: ordered[i]["exact"])
    dips = [
        (ordered[i]["distance"], ordered[i + 1]["distance"])
        for i in range(best_at, len(ordered) - 1)
        if ordered[i + 1]["exact"] < ordered[i]["exact"]
    ]
    if dips:
        print("\n  WARNING: the exact curve dips downward past its minimum, between "
              + ", ".join(f"{a:.2f}-{b:.2f} A" for a, b in dips))
        print("  A ground-state curve must rise monotonically toward dissociation. This is")
        print("  an active space too small to describe the bond breaking, not a VQE failure")
        print("  -- VQE has reproduced the (wrong) active-space energy faithfully.")
        print("  Try --active-orbitals 5 (the default) or larger.")
    else:
        print("  curve rises monotonically past its minimum -- active space looks adequate")


if __name__ == "__main__":
    main()