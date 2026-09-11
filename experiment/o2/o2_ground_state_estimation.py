"""
o2_ground_state_estimation.py -- O2's ground-state energy by VQE across a
range of bond lengths, plotted live next to a picture of the molecule pulling
itself apart.

Simulator only. No IBM account, no queue, no quota.

O2 HAS EXACTLY ONE DEGREE OF FREEDOM
--------------------------------------
Worth stating plainly, because it is the first question anyone asks when the
other molecules in this project get surfaces and this one does not.

O2 has two atoms. A linear molecule with N atoms has 3N-5 internal
coordinates, so O2 has 3(2) - 5 = **1**: the O-O bond length. There is no
bond angle, because an angle needs three atoms to define it. So a genuine
potential-energy *surface* is geometrically impossible for O2 -- there is
nothing to plot against the bond length. One curve is not a shortcut here, it
is the complete description of the geometry.

(Compare BeH2 in the sibling folder: three atoms, 3N-6 = 3 coordinates,
which the symmetry constraint reduces to two -- and two is what it takes to
make a surface.)

THE TRAP: O2's GROUND STATE IS A TRIPLET
------------------------------------------
This is the one place where the H2 and H2O templates in this project do not
carry over, and it fails *silently*, so it is worth understanding.

O2 has 16 electrons. Filling its molecular orbitals leaves the last two in
the doubly degenerate antibonding pi* orbitals, and by Hund's rule they go in
with parallel spins rather than pairing up. So O2's ground state has two
unpaired electrons: it is a **triplet**, spectroscopically X 3-Sigma-g-minus.
O2 is one of only two diatomics in the NIST compilation with a 3-Sigma
electronic ground state [1]. This is also why liquid oxygen is paramagnetic
and sticks to a magnet.

Every ``PySCFDriver`` call elsewhere in this repository is closed-shell -- it
takes the default ``spin=0`` and ``MethodType.RHF``, which assumes every
electron is paired. Point that at O2 and it will not complain. It will
quietly hand back the **singlet**, and you will get a smooth, plausible,
completely wrong curve with no error message anywhere.

So this script passes ``spin=2`` and ``method=MethodType.ROHF``. In PySCF's
convention -- which qiskit-nature adopts -- ``spin`` equals 2*S, twice the
total spin quantum number [2]. A triplet has S = 1, hence spin = 2, hence two
more alpha electrons than beta: (9 alpha, 7 beta) for neutral O2. ROHF is the
restricted *open-shell* Hartree-Fock method, which can represent that.

This is a correctness requirement, not a feature. Without it the numbers are
simply wrong.

WHAT THE CURVE SHOULD LOOK LIKE
---------------------------------
A textbook Morse shape: a hard repulsive wall at short bond length where the
nuclei push each other apart, a minimum near 1.208 A, then a rise that
flattens into a plateau as the two oxygens separate into free atoms.

O2 is the hardest molecule in this project, and that is exactly why it earns
a place. Its formal bond order is 2, with those two unpaired pi* electrons,
and it is strongly correlated *even at its equilibrium geometry* -- not just
out in the dissociated region the way H2 and BeH2 are. So expect the
**largest Hartree-Fock error of the whole project**, bigger than water's, and
growing dramatically as the bond stretches. If you want one plot that argues
for why correlated methods are necessary, this is the one.

A HONEST LIMITATION, WHICH IS ALSO THE POINT
----------------------------------------------
The default scan is 9 points at 0.2 A spacing. That is deliberately coarse,
because this is the slowest system here -- but it means the grid barely
samples the bottom of the well, and you **cannot** read the 1.208 A minimum
off the raw points with any accuracy. The lowest of nine coarse samples is
not the minimum of the curve.

That is the motivation for the sibling ``o2_spline/`` and ``o2_polynomial/``
folders. Fitting an interpolant through these nine expensive points and
minimizing *that* recovers the equilibrium bond length far better than
picking the lowest sample -- which is interpolation doing real numerical work,
rather than just smoothing a plot for looks.

ACTIVE SPACE
-------------
O2 in STO-3G is 16 electrons in 10 spatial orbitals -> 20 qubits if you take
everything, which is far beyond what a statevector scan can do. The default
is CAS(8,6): 8 electrons in the 6 orbitals derived from the 2p shell --
sigma(2p), the two pi(2p), the two pi*(2p), and sigma*(2p). That is the
standard active space for O2, and it is the smallest one that contains the
pi* orbitals where the two unpaired electrons live. Truncate below it and you
throw away the physics that makes O2 a triplet in the first place.

After the parity mapper's two-qubit reduction that is 10 qubits.

One subtlety to be aware of when reading the "exact" column. It is the lowest
eigenvalue of the mapped qubit Hamiltonian, found by direct diagonalization.
The parity mapping's two-qubit reduction pins the *parity* of the alpha and
beta electron counts, not the counts themselves, so in principle the lowest
eigenvalue could belong to a different electron-number sector than the one we
meant to study. The script prints the HF-to-exact gap at the end partly as a
check on this: a gap of a few hundred mHa is the expected strong correlation,
whereas a gap of several Hartree means the diagonalization wandered into
another sector and the comparison is meaningless.

COST
-----
This is the slowest script in the project, though not by as much as its qubit
count suggests. Runtime is dominated not by circuit simulation -- one energy
evaluation is well under a second -- but by the *number* of evaluations.

Measured on a laptop CPU at the defaults, CAS(8,6) is 10 qubits and 68 UCCSD
parameters, and one point takes about **143 s**, so the default 9-point scan
runs in roughly **20 minutes**. (An earlier estimate here said "hours"; that
was wrong, and the measurement is what matters.)

The reason COBYLA is the default: SLSQP estimates its gradient by finite
differences, costing about one evaluation per parameter per step, which is
tolerable at H2O's 54 parameters and punishing at 68 on a 10-qubit
Hamiltonian. COBYLA is gradient-free.

Worth knowing about that 143 s, though: at the default cap COBYLA uses its
**entire** budget of 1000 evaluations rather than converging and stopping
early, so the run is budget-limited, not convergence-limited. It still lands
within about 0.3 mHa of exact, which is fine. But it means the cost per point
is essentially fixed by ``--max-iterations`` rather than by how hard the
geometry is, and raising that cap buys accuracy at proportionally more time.

It saves itself when it finishes, and ``--load`` reopens it instantly, so you
only pay once.

REFERENCES
-----------
[1] NIST Diatomic Spectral Database, 3-Sigma ground-state molecules -- O2's
    X 3-Sigma-g-minus ground state and its internuclear distance:
    https://physics.nist.gov/PhysRefData/MolSpec/Diatomic/Html/sec3.html
[2] PySCFDriver: "the spin equals 2*S, where S is the total spin number of
    the molecule", plus the MethodType options including ROHF:
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.drivers.PySCFDriver.html
[3] ActiveSpaceTransformer, including its note that alpha and beta subspaces
    may not span the same space for unrestricted orbitals:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html

USAGE
------
Every case, explicitly. O2 is diatomic, so bond length is the only geometric
coordinate -- one curve, no surface. See the top of this docstring.

  RUN A SCAN -- the triplet, O2's real ground state (default)
    python experiment/o2/o2_ground_state_estimation.py
    python experiment/o2/o2_ground_state_estimation.py --start 0.9 --stop 2.5 --step 0.2
    python experiment/o2/o2_ground_state_estimation.py --start 0.9 --stop 2.5 --step 0.1     # finer, ~2x the time
    python experiment/o2/o2_ground_state_estimation.py --start 1.2 --stop 1.2 --step 0.1 --no-save   # one point

  RUN A SCAN -- the singlet, for contrast. NOT the ground state.
    python experiment/o2/o2_ground_state_estimation.py --multiplicity singlet
    python experiment/o2/o2_ground_state_estimation.py --multiplicity singlet --start 0.9 --stop 2.5 --step 0.2

  RELOAD A FINISHED SCAN -- no VQE re-run. Either format works.
    python experiment/o2/o2_ground_state_estimation.py --load o2_scan.json
    python experiment/o2/o2_ground_state_estimation.py --load o2_scan.csv
    python experiment/o2/o2_ground_state_estimation.py --load o2_singlet_scan.json
    python experiment/o2/o2_ground_state_estimation.py --load o2_singlet_scan.csv
    # note: multiplicity is not a CSV column, so a CSV reload always labels the
    # plot "triplet". Use the .json when reloading a singlet scan.

  OPTIMIZER -- COBYLA uses its whole budget rather than converging early, so
  cost per point is set by --max-iterations, not by how hard the geometry is
    python experiment/o2/o2_ground_state_estimation.py --max-iterations 3000
    python experiment/o2/o2_ground_state_estimation.py --optimizer slsqp
    python experiment/o2/o2_ground_state_estimation.py --cold-start

  A SMALLER ACTIVE SPACE -- much faster, but drops the pi* orbitals where O2's
  unpaired electrons live, so the physics degrades. For plumbing checks only.
    python experiment/o2/o2_ground_state_estimation.py --active-electrons 4 --active-orbitals 4
    python experiment/o2/o2_ground_state_estimation.py --start 1.2 --stop 1.2 --step 0.1 --active-electrons 4 --active-orbitals 4 --no-save

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
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

BASIS = "sto3g"

# O2's experimental equilibrium bond length [1]. Used only as the reference
# printed next to whatever minimum the scan actually finds.
EQUILIBRIUM_BOND = 1.208   # Angstrom

# spin = 2*S in PySCF's convention [2]. The triplet (S=1) is O2's true ground
# state; the singlet is offered only so you can scan it and see for yourself
# that it sits above the triplet everywhere.
MULTIPLICITY_SPIN = {"triplet": 2, "singlet": 0}


# ===========================================================================
#  The quantum chemistry -- one bond length at a time
# ===========================================================================

def active_particles(active_electrons: int, spin: int) -> Tuple[int, int]:
    """Split ``active_electrons`` into (alpha, beta) for a given ``spin`` = 2*S.

    ``ActiveSpaceTransformer`` accepts either a plain electron count or an
    explicit ``(alpha, beta)`` tuple. For an open-shell system the plain count
    is not enough information -- 8 electrons could be (4,4) or (5,3) -- and
    getting it wrong silently changes which electronic state you are
    computing. So this is always passed explicitly.

    spin = alpha - beta, and alpha + beta = active_electrons, which gives
    beta = (active_electrons - spin) / 2. For O2's default CAS(8,6) triplet
    that is (5, 3).
    """
    if active_electrons < spin:
        raise ValueError(
            f"cannot put {active_electrons} electrons in a spin-{spin} state: "
            f"need at least {spin} electrons to have {spin} of them unpaired"
        )
    if (active_electrons - spin) % 2 != 0:
        raise ValueError(
            f"active electrons ({active_electrons}) and spin ({spin}) must have the "
            f"same parity -- alpha and beta counts have to come out as whole numbers"
        )
    n_beta = (active_electrons - spin) // 2
    return active_electrons - n_beta, n_beta


def build_problem(distance: float, spin: int, active_electrons: int,
                  active_orbitals: int) -> ElectronicStructureProblem:
    """O2 with the two atoms `distance` Angstrom apart, centred on the origin.

    ``spin`` and ``method`` are the whole point of this function. The default
    closed-shell path used everywhere else in this repo would return the
    singlet without complaint -- see the module docstring. ROHF is the
    restricted open-shell method that can actually represent two unpaired
    electrons [2].
    """
    geometry = f"O 0 0 {-distance / 2}; O 0 0 {distance / 2}"
    method = MethodType.ROHF if spin else MethodType.RHF
    problem = PySCFDriver(atom=geometry, basis=BASIS, spin=spin, method=method).run()
    return ActiveSpaceTransformer(
        active_particles(active_electrons, spin), active_orbitals
    ).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). For O2 the inactive
    part is large -- eight core and 2s electrons' worth -- so miss this and
    the energy is wrong by well over a hundred Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_distance(distance: float, spin: int, active_electrons: int, active_orbitals: int,
                     optimizer: str = "cobyla", max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one O-O distance.

    ``theta0`` warm-starts the optimizer from the previous bond length's
    solution, since neighbouring points have similar wavefunctions.

    "exact" here is the lowest eigenvalue of the mapped, active-space-reduced
    qubit Hamiltonian -- exact within the chosen model, not a full-basis FCI
    energy. "hf" is the Hartree-Fock circuit's expectation value on that same
    Hamiltonian rather than PySCF's SCF energy, so the three numbers are
    directly comparable with each other.
    """
    problem = build_problem(distance, spin, active_electrons, active_orbitals)
    mapper = ParityMapper(num_particles=problem.num_particles)
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                   initial_state=hf_circuit)
    # UCCSD's excitations arrive as PauliEvolutionGate black boxes, which the
    # estimator would otherwise matrix-exponentiate on every call. Transpile
    # once here, outside the optimization loop. This matters far more at 10
    # qubits than it did at 2.
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
    method = {"cobyla": "COBYLA", "slsqp": "SLSQP"}[optimizer]
    result = minimize(cost, start, method=method, options={"maxiter": max_iterations})

    # Same row shape as the H2 and H2O scripts. The classical and quantum
    # errors are not stored as columns -- they are differences of these three
    # numbers, so they are derived where they are reported rather than
    # duplicated into every row.
    #
    # ``n_evaluations`` is the one addition, on the same grounds that H2O
    # added n_qubits/n_params: for O2 the optimizer's evaluation count is the
    # thing that decides whether a scan finishes tonight, so it is worth
    # having in the saved table.
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

# Drawn radius of an oxygen atom, in Angstrom -- roughly O's covalent radius
# (0.66 A), scaled down a little so the two circles separate visibly as the
# molecule dissociates instead of staying merged across the whole scan.
ATOM_RADIUS = 0.34
ATOM_COLOR = "#ff4d4d"


class O2DissociationViewer:
    """Two live panels plus, once the scan is done, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called once per bond length as its energies
    come in -- the curve grows and the molecule redraws at the geometry just
    computed. Phase 2: :meth:`enable_review` turns the window into a scrubber
    over the finished curve.
    """

    def __init__(self, max_distance: float, multiplicity: str) -> None:
        plt.ion()
        self.fig, (self.ax_energy, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None
        self.multiplicity = multiplicity

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
        self.ax_energy.set_xlabel("O-O bond length (Angstrom)")
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title(f"O$_2$ dissociation curve  ({multiplicity})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        # -- right panel: the molecule itself --------------------------
        span = max_distance / 2 + ATOM_RADIUS + 0.3
        self.ax_mol.set_xlim(-span, span)
        self.ax_mol.set_ylim(-span / 2.6, span / 2.6)
        self.ax_mol.set_aspect("equal")
        self.ax_mol.set_xticks([])
        self.ax_mol.set_yticks([])
        self.ax_mol.set_title("Geometry")
        (self.bond,) = self.ax_mol.plot([], [], "-", color="#888888", linewidth=3, zorder=1)
        self.atoms = [
            Circle((0, 0), ATOM_RADIUS, facecolor=ATOM_COLOR,
                   edgecolor="black", linewidth=1.5, zorder=2)
            for _ in range(2)
        ]
        for atom in self.atoms:
            self.ax_mol.add_patch(atom)
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
        self.atoms[0].center = (-d / 2, 0.0)
        self.atoms[1].center = (d / 2, 0.0)
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
        description="O2 dissociation curve by VQE (local simulator). O2 is diatomic, so "
                    "bond length is its only geometric degree of freedom -- one curve, "
                    "no surface. Defaults to O2's true triplet ground state."
    )
    parser.add_argument("--start", type=float, default=0.9,
                        help="shortest O-O bond length in Angstrom (default 0.9)")
    parser.add_argument("--stop", type=float, default=2.5,
                        help="longest O-O bond length in Angstrom, inclusive (default 2.5)")
    parser.add_argument("--step", type=float, default=0.2,
                        help="spacing between bond lengths in Angstrom (default 0.2). "
                             "Deliberately coarse -- this is the slowest system in the "
                             "project, and these points feed the interpolation folders.")
    parser.add_argument("--multiplicity", choices=["triplet", "singlet"], default="triplet",
                        help="'triplet' (default) is O2's true ground state, X 3-Sigma-g-minus. "
                             "'singlet' is the closed-shell state a naive setup would give you "
                             "by accident -- scan it to see that it lies above the triplet.")
    parser.add_argument("--active-electrons", type=int, default=8,
                        help="electrons in the active space (default 8)")
    parser.add_argument("--active-orbitals", type=int, default=6,
                        help="spatial orbitals in the active space (default 6 -> 10 qubits). "
                             "6 is the smallest space containing the pi* orbitals where O2's "
                             "unpaired electrons live -- see the module docstring.")
    parser.add_argument("--optimizer", choices=["cobyla", "slsqp"], default="cobyla",
                        help="'cobyla' (default) is gradient-free and much cheaper at this "
                             "parameter count. 'slsqp' is more thorough but estimates its "
                             "gradient by finite differences, costing one energy evaluation "
                             "per parameter per step.")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="optimizer iteration cap (default 1000). At CAS(8,6) COBYLA "
                             "reaches this cap rather than converging on its own, so the "
                             "run is budget-limited: raising it buys accuracy at "
                             "proportionally more time, lowering it does the reverse.")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each bond length's optimization from the Hartree-Fock "
                             "state instead of warm-starting from the previous solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json written by an earlier run) straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: o2_scan.json/.csv next to "
                             "this script, or o2_singlet_scan for --multiplicity singlet)")
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
        metadata, rows = _load_any(Path(args.load), here)
        # The active space is metadata, so a CSV-sourced scan has none. Report
        # the qubit count from the rows rather than printing "CAS(None, None)".
        if metadata.get("active_electrons") is not None:
            space = f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
        elif rows and rows[0].get("n_qubits"):
            space = f"{rows[0]['n_qubits']} qubits, active space not recorded in the csv"
        else:
            space = "active space not recorded"
        print(f"loaded {len(rows)} points from {args.load} "
              f"({metadata.get('multiplicity', 'triplet')}, {space}) -- no VQE re-run")
        viewer = O2DissociationViewer(float(metadata["max_distance"]),
                                      metadata.get("multiplicity", "triplet"))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, args.no_show)
        return

    if args.step <= 0:
        parser.error("--step must be positive")
    if args.stop < args.start:
        parser.error("--stop must be >= --start")

    spin = MULTIPLICITY_SPIN[args.multiplicity]
    try:
        n_alpha, n_beta = active_particles(args.active_electrons, spin)
    except ValueError as exc:
        parser.error(str(exc))

    distances = np.arange(args.start, args.stop + args.step / 2, args.step)

    print(f"O2 {args.multiplicity} scan: {len(distances)} bond lengths from "
          f"{distances[0]:.2f} to {distances[-1]:.2f} A (step {args.step} A)")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})  "
          f"-> {n_alpha} alpha + {n_beta} beta electrons, spin = {spin} (= 2S)")
    print(f"method: {'ROHF' if spin else 'RHF'}, optimizer: {args.optimizer.upper()}")
    if args.multiplicity == "singlet":
        print("  NOTE: the singlet is NOT O2's ground state. You are scanning it on purpose.")
    print("this is the slowest scan in the project -- budget hours, not minutes. "
          "The plot updates as each point lands.")

    viewer = O2DissociationViewer(float(distances[-1]), args.multiplicity)
    rows: List[Dict[str, Any]] = []
    previous_params = None
    for i, d in enumerate(distances):
        t0 = time.perf_counter()
        row = run_one_distance(d, spin, args.active_electrons, args.active_orbitals,
                               optimizer=args.optimizer,
                               max_iterations=args.max_iterations,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:3d}/{len(distances)}] d={d:.2f} A   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")

    _report(rows)

    if not args.no_save:
        stem = "o2_scan" if args.multiplicity == "triplet" else f"o2_{args.multiplicity}_scan"
        target = resolve_target(args.save, here, stem)
        metadata = {
            "molecule": "O2", "basis": BASIS,
            "scan_type": "curve",
            "coordinate": "bond",
            "degrees_of_freedom": 1,
            "multiplicity": args.multiplicity,
            "spin_2s": int(spin),
            "scf_method": "ROHF" if spin else "RHF",
            "start": float(distances[0]), "stop": float(distances[-1]),
            "step": float(args.step),
            "unit": "A",
            "max_distance": float(distances[-1]),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
            "active_alpha": int(n_alpha), "active_beta": int(n_beta),
            "optimizer": args.optimizer,
            "max_iterations": int(args.max_iterations),
        }
        json_path, csv_path = save_scan(target, metadata, rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        print(f"reopen it later without recomputing:  "
              f"python {display_path(Path(__file__))} --load {json_path.name}")
        print(f"interpolate it:  python experiment/o2_spline/o2_spline_interpolation.py")

    _finish(viewer, args.no_show)


def _load_csv(path: Path) -> tuple:
    """Read an O2 scan back from its ``.csv`` instead of its ``.json``.

    The JSON is the canonical save -- it carries the metadata needed to rebuild
    the plot. A CSV is a flat table with none, so ``max_distance`` is taken
    from the data itself.

    One thing genuinely cannot be recovered: **which spin state was scanned.**
    Multiplicity is metadata, not a column, so a triplet and a singlet scan
    produce indistinguishable CSVs. This assumes triplet, since that is the
    default and O2's real ground state, and says so rather than labelling the
    plot silently. If you are reloading a singlet scan from CSV, the title
    will be wrong -- use the .json for that one.
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
            f"not an O2 scan CSV. Expected columns: distance, hf, vqe, exact."
        )

    metadata = {
        "scan_type": "curve",
        "multiplicity": "triplet",
        "max_distance": max(r["distance"] for r in rows),
    }
    return metadata, rows


def _load_any(load_path: Path, here: Path) -> tuple:
    """Load a saved scan from either its ``.json`` or its ``.csv``."""
    if load_path.suffix.lower() == ".csv":
        for candidate in (load_path, here / load_path.name):
            if candidate.exists():
                print("reading the CSV -- multiplicity is not a column, so the title "
                      "assumes triplet; use the .json if it was a singlet scan")
                return _load_csv(candidate)
        looked = "\n  ".join(display_path(c) for c in (load_path, here / load_path.name))
        raise SystemExit(f"no scan CSV found. Looked in:\n  {looked}")
    return load_scan(load_path, default_dir=here)


def _finish(viewer: O2DissociationViewer, no_show: bool) -> None:
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
    """Final summary: where the minimum is, and how big each error is."""
    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at d = {best['distance']:.2f} A "
          f"(experimental O2 bond length is ~{EQUILIBRIUM_BOND} A)")

    # With a coarse grid the best sample is a poor estimate of the true
    # minimum, and saying so here is more useful than letting the number be
    # over-read. This is what the interpolation folders are for.
    step = rows[1]["distance"] - rows[0]["distance"] if len(rows) > 1 else 0.0
    if step > 0.05:
        print(f"  the grid spacing is {step:.2f} A, so this is the lowest *sample*, not the")
        print(f"  minimum of the curve -- fit an interpolant through these points and")
        print(f"  minimize that instead (see experiment/o2_spline/).")

    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    at_eq = min(rows, key=lambda r: abs(r["distance"] - EQUILIBRIUM_BOND))
    print(f"largest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"  ... and {abs(at_eq['hf'] - at_eq['exact']) * 1000:.3f} mHa "
          f"at d = {at_eq['distance']:.2f} A, "
          f"near equilibrium -- O2 is strongly correlated even at its minimum, "
          f"which is what sets it apart from H2 and BeH2")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")
    if max_vqe * 1000 > 1.0:
        print("  note: above 1 mHa somewhere -- COBYLA may have stopped early. "
              "Try --optimizer slsqp on a few points to check.")
    if max_hf > 1.0:
        print("  WARNING: an HF-to-exact gap above 1 Hartree is far larger than strong")
        print("  correlation explains. The diagonalization may have found a state in a")
        print("  different electron-number sector -- see the module docstring.")


if __name__ == "__main__":
    main()
