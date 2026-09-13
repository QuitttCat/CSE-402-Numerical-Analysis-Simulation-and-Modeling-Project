"""
n2_ground_state_estimation.py -- N2's ground-state energy by VQE across a
range of bond lengths, plotted live next to a picture of the molecule pulling
itself apart.

Simulator only. No IBM account, no queue, no quota.

N2 HAS EXACTLY ONE DEGREE OF FREEDOM
--------------------------------------
Same story as O2 in the sibling folder: a linear molecule with N atoms has
3N-5 internal coordinates, so N2's two atoms give 3(2) - 5 = **1**, the N-N
bond length. No angle exists to plot against it, because an angle needs
three atoms to define one. So there is one dissociation curve, not a
surface -- and that curve is the complete geometric description of N2.

N2's GROUND STATE IS AN ORDINARY SINGLET -- BUT DISSOCIATION IS NOT
----------------------------------------------------------------------
Unlike O2, N2's ground state at equilibrium is exactly the closed-shell
singlet the default ``PySCFDriver`` call gives you: X 1-Sigma-g+, 14
electrons, all paired, spin = 2S = 0, plain RHF. There is no ROHF trap here.

The trap is further out on the curve instead. N2's triple bond is one sigma
bond plus two pi bonds -- six bonding electrons -- and pulling the two
nitrogens apart means breaking all three at once. Each separated N atom's
own ground state is a quartet (three unpaired 2p electrons, S = 3/2), so the
*true* dissociation limit is nothing like a single closed-shell determinant.
A plain Hartree-Fock calculation, restricted to one Slater determinant the
whole way out, cannot represent that: this is the textbook RHF dissociation
catastrophe, the same failure mode as H2's, just for three bonds instead of
one, and correspondingly worse. It is *why* N2's bond is so strong
(945 kJ/mol, among the highest of any diatomic) and why breaking it in the
Haber process needs a catalyst and real heat -- the curve you are about to
plot is the numerical face of that chemistry.

The active space below exists specifically to blunt this. Correlating the
UCCSD ansatz over the bonding *and* antibonding orbitals of the triple bond
lets VQE mix in the multi-determinant character that a single Hartree-Fock
reference misses, so the HF-vs-VQE gap you will see growing with bond length
is exactly this effect becoming visible.

ACTIVE SPACE
-------------
N2 in STO-3G is 14 electrons in 10 spatial orbitals per atom-pair (1s, 2s,
2px, 2py, 2pz on each N) -> 20 qubits taking everything, far beyond a
statevector scan. The default here is CAS(6,6): 6 electrons in the 6
orbitals built from the valence 2p shell -- sigma(2p), the two pi(2p)
(bonding), and their three antibonding partners sigma*(2p) and the two
pi*(2p). That is the triple bond and nothing else: six electrons, six
orbitals, exactly matching the "one sigma bond + two pi bonds" picture of
N2 taught in every general chemistry course.

Left out of the active space, frozen at their Hartree-Fock occupation, are
the two 1s cores and the 2s-derived sigma/sigma* pair. The 2s orbitals *do*
carry some bonding character in a full CASSCF treatment, and a more complete
(and much more expensive) description would use the full-valence CAS(10,8)
instead -- this script supports that directly:

    python experiment/n2/n2_ground_state_estimation.py --active-electrons 10 --active-orbitals 8

but CAS(6,6) is the smallest space that still contains every orbital
directly involved in the triple bond, and keeps the qubit count in the same
range as O2's scan (10 qubits after the parity mapper's two-qubit
reduction) rather than the 14-16 qubits CAS(10,8) would need.

WHAT THE CURVE SHOULD LOOK LIKE
---------------------------------
A deep, narrow Morse well -- N2's bond is short (~1.098 A) and very stiff,
so the minimum is sharper than H2's or O2's -- followed by a rise that
should, within the active space, flatten out rather than keep climbing
without bound the way an uncorrelated RHF curve would. Watch the HF and VQE
lines: they sit close together near equilibrium and pull apart as the bond
stretches, the same qualitative signature as H2's dissociation, just with a
deeper well and a wider gap because three bonds are breaking instead of one.

COST
-----
Measured cost scales with CAS(6,6): 12 qubits before the parity reduction,
10 after -- the same qubit count as O2's default scan, so expect a similar
per-point cost (on the order of a couple of minutes on a laptop CPU with
COBYLA's default 1000-iteration budget). The default 9-point scan is a
similar time investment to O2's, i.e. budget tens of minutes, not seconds.
COBYLA is the default optimizer for the same reason it is for O2: SLSQP's
finite-difference gradient costs about one evaluation per parameter per
step, which adds up fast once the ansatz has dozens of parameters.

It saves itself when it finishes, and ``--load`` reopens it instantly, so
you only pay once.

REFERENCES
-----------
[1] NIST Diatomic Spectral Database, N2's X 1-Sigma-g+ ground state and its
    equilibrium bond length:
    https://physics.nist.gov/PhysRefData/MolSpec/Diatomic/Html/sec3.html
[2] ActiveSpaceTransformer:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html

USAGE
------
  RUN A SCAN -- N2's real ground state (default), one curve, no surface
    python experiment/n2/n2_ground_state_estimation.py
    python experiment/n2/n2_ground_state_estimation.py --start 0.8 --stop 2.5 --step 0.2
    python experiment/n2/n2_ground_state_estimation.py --start 0.8 --stop 2.5 --step 0.1   # finer, ~2x the time
    python experiment/n2/n2_ground_state_estimation.py --start 1.1 --stop 1.1 --step 0.1 --no-save   # one point

  RELOAD A FINISHED SCAN -- no VQE re-run
    python experiment/n2/n2_ground_state_estimation.py --load n2_scan.json
    python experiment/n2/n2_ground_state_estimation.py --load n2_scan.csv

  OPTIMIZER -- same trade-off as O2: COBYLA is cheap and gradient-free
    python experiment/n2/n2_ground_state_estimation.py --max-iterations 3000
    python experiment/n2/n2_ground_state_estimation.py --optimizer slsqp
    python experiment/n2/n2_ground_state_estimation.py --cold-start

  A LARGER ACTIVE SPACE -- the full valence space, more accurate, slower
    python experiment/n2/n2_ground_state_estimation.py --active-electrons 10 --active-orbitals 8

  A SMALLER ACTIVE SPACE -- much faster, but drops orbitals of the triple
  bond, so the physics degrades. For plumbing checks only.
    python experiment/n2/n2_ground_state_estimation.py --active-electrons 2 --active-orbitals 2
    python experiment/n2/n2_ground_state_estimation.py --start 1.1 --stop 1.1 --step 0.1 --active-electrons 2 --active-orbitals 2 --no-save

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

# N2's experimental equilibrium bond length [1]. Used only as the reference
# printed next to whatever minimum the scan actually finds.
EQUILIBRIUM_BOND = 1.098   # Angstrom


# ===========================================================================
#  The quantum chemistry -- one bond length at a time
# ===========================================================================

def build_problem(distance: float, active_electrons: int,
                  active_orbitals: int) -> ElectronicStructureProblem:
    """N2 with the two atoms `distance` Angstrom apart, centred on the origin.

    N2's ground state is an ordinary closed-shell singlet, so -- unlike
    O2's script -- there is no spin/method argument to get right here. The
    default ``PySCFDriver`` call is already RHF with spin=0. What N2 does
    need, that H2 does not, is the active-space reduction: 14 electrons in
    10 STO-3G orbitals is too large to scan directly. See the module
    docstring for why CAS(6,6) is the default.
    """
    geometry = f"N 0 0 {-distance / 2}; N 0 0 {distance / 2}"
    problem = PySCFDriver(atom=geometry, basis=BASIS).run()
    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). For N2 the
    inactive part is large -- two cores plus the 2s pair's worth -- so miss
    this and the energy is wrong by tens of Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_distance(distance: float, active_electrons: int, active_orbitals: int,
                     optimizer: str = "cobyla", max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one N-N distance.

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
    method = {"cobyla": "COBYLA", "slsqp": "SLSQP"}[optimizer]
    result = minimize(cost, start, method=method, options={"maxiter": max_iterations})

    # Same row shape as the H2 and O2 scripts. The classical and quantum
    # errors are not stored as columns -- they are differences of these
    # three numbers, so they are derived where they are reported rather
    # than duplicated into every row.
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

# Drawn radius of a nitrogen atom, in Angstrom -- roughly N's covalent radius
# (0.71 A), scaled down a little (matching the scale factor used for O in
# the O2 script) so the two circles separate visibly as the molecule
# dissociates instead of staying merged across the whole scan.
ATOM_RADIUS = 0.36
ATOM_COLOR = "#3050f8"   # standard CPK blue for nitrogen


class N2DissociationViewer:
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
        self.ax_energy.set_xlabel("N-N bond length (Angstrom)")
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title("N$_2$ dissociation curve")
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
        description="N2 dissociation curve by VQE (local simulator). N2 is diatomic, so "
                    "bond length is its only geometric degree of freedom -- one curve, "
                    "no surface. Ground state is an ordinary closed-shell singlet; the "
                    "interesting physics is the RHF dissociation catastrophe as the "
                    "triple bond breaks, see the module docstring."
    )
    parser.add_argument("--start", type=float, default=0.8,
                        help="shortest N-N bond length in Angstrom (default 0.8)")
    parser.add_argument("--stop", type=float, default=2.5,
                        help="longest N-N bond length in Angstrom, inclusive (default 2.5)")
    parser.add_argument("--step", type=float, default=0.2,
                        help="spacing between bond lengths in Angstrom (default 0.2)")
    parser.add_argument("--active-electrons", type=int, default=6,
                        help="electrons in the active space (default 6)")
    parser.add_argument("--active-orbitals", type=int, default=6,
                        help="spatial orbitals in the active space (default 6 -> 10 qubits "
                             "after the parity mapper's reduction). CAS(6,6) is the smallest "
                             "space containing the full triple bond -- sigma(2p), the two "
                             "pi(2p), and their three antibonding partners -- see the module "
                             "docstring.")
    parser.add_argument("--optimizer", choices=["cobyla", "slsqp"], default="cobyla",
                        help="'cobyla' (default) is gradient-free and much cheaper at this "
                             "parameter count. 'slsqp' is more thorough but estimates its "
                             "gradient by finite differences, costing one energy evaluation "
                             "per parameter per step.")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="optimizer iteration cap (default 1000).")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each bond length's optimization from the Hartree-Fock "
                             "state instead of warm-starting from the previous solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json written by an earlier run) straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: n2_scan.json/.csv next to "
                             "this script)")
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
        if metadata.get("active_electrons") is not None:
            space = f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
        elif rows and rows[0].get("n_qubits"):
            space = f"{rows[0]['n_qubits']} qubits, active space not recorded in the csv"
        else:
            space = "active space not recorded"
        print(f"loaded {len(rows)} points from {args.load} ({space}) -- no VQE re-run")
        viewer = N2DissociationViewer(float(metadata["max_distance"]))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, args.no_show)
        return

    if args.step <= 0:
        parser.error("--step must be positive")
    if args.stop < args.start:
        parser.error("--stop must be >= --start")
    if args.active_electrons % 2 != 0:
        parser.error("--active-electrons must be even: N2's ground state is a closed-shell "
                     "singlet, so active electrons split evenly into alpha and beta")

    distances = np.arange(args.start, args.stop + args.step / 2, args.step)

    print(f"N2 scan: {len(distances)} bond lengths from "
          f"{distances[0]:.2f} to {distances[-1]:.2f} A (step {args.step} A)")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    print(f"method: RHF, optimizer: {args.optimizer.upper()}")
    print("this is roughly as slow per point as the O2 scan -- the plot updates as each "
          "point lands.")

    viewer = N2DissociationViewer(float(distances[-1]))

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
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:3d}/{len(distances)}] d={d:.2f} A   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")

    _report(rows)

    if not args.no_save:
        target = resolve_target(args.save, here, "n2_scan")
        metadata = {
            "molecule": "N2", "basis": BASIS,
            "scan_type": "curve",
            "coordinate": "bond",
            "degrees_of_freedom": 1,
            "spin_2s": 0,
            "scf_method": "RHF",
            "start": float(distances[0]), "stop": float(distances[-1]),
            "step": float(args.step),
            "unit": "A",
            "max_distance": float(distances[-1]),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
            "active_alpha": int(args.active_electrons // 2),
            "active_beta": int(args.active_electrons // 2),
            "optimizer": args.optimizer,
            "max_iterations": int(args.max_iterations),
        }
        json_path, csv_path = save_scan(target, metadata, rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        print(f"reopen it later without recomputing:  "
              f"python {display_path(Path(__file__))} --load {json_path.name}")

    _finish(viewer, args.no_show)


def _load_csv(path: Path) -> tuple:
    """Read an N2 scan back from its ``.csv`` instead of its ``.json``.

    The JSON is the canonical save -- it carries the metadata needed to
    rebuild the plot. A CSV is a flat table with none, so ``max_distance``
    is taken from the data itself.
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
            f"not an N2 scan CSV. Expected columns: distance, hf, vqe, exact."
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


def _finish(viewer: N2DissociationViewer, no_show: bool) -> None:
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
          f"(experimental N2 bond length is ~{EQUILIBRIUM_BOND} A)")

    step = rows[1]["distance"] - rows[0]["distance"] if len(rows) > 1 else 0.0
    if step > 0.05:
        print(f"  the grid spacing is {step:.2f} A, so this is the lowest *sample*, not the")
        print(f"  minimum of the curve -- rerun with a smaller --step near the well if you "
              f"need the minimum more precisely.")

    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    at_eq = min(rows, key=lambda r: abs(r["distance"] - EQUILIBRIUM_BOND))
    print(f"largest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"  ... and {abs(at_eq['hf'] - at_eq['exact']) * 1000:.3f} mHa "
          f"at d = {at_eq['distance']:.2f} A, near equilibrium")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")
    if max_vqe * 1000 > 1.0:
        print("  note: above 1 mHa somewhere -- COBYLA may have stopped early. "
              "Try --optimizer slsqp on a few points to check.")
    if max_hf > 1.0:
        print("  WARNING: an HF-to-exact gap above 1 Hartree is far larger than strong")
        print("  correlation explains. The diagonalization may have found a state in a")
        print("  different electron-number sector.")


if __name__ == "__main__":
    main()