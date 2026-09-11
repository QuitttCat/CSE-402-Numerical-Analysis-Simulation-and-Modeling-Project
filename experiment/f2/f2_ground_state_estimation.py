"""
f2_ground_state_estimation.py -- F2's ground-state energy by VQE across a
range of bond lengths, plotted live next to a picture of the molecule pulling
itself apart.

Simulator only. No IBM account, no queue, no quota.

F2 HAS EXACTLY ONE DEGREE OF FREEDOM
--------------------------------------
Worth stating plainly, the same way the O2 script in the sibling folder does.

F2 has two atoms. A linear molecule with N atoms has 3N-5 internal
coordinates, so F2 has 3(2) - 5 = **1**: the F-F bond length. There is no
bond angle -- an angle needs three atoms to define it -- so a genuine
potential-energy *surface* is geometrically impossible here. One curve is the
complete description of the geometry, not a shortcut.

F2's GROUND STATE IS A CLOSED-SHELL SINGLET -- UNLIKE ITS NEIGHBOUR O2
-------------------------------------------------------------------------
This is the one place F2 does *not* inherit O2's trap, and it is worth being
explicit about why, since the two sit right next to each other in the
periodic table and it would be easy to assume they behave the same way.

F2 has 18 electrons. Filling its molecular orbitals in the usual diatomic
order (sigma_2s, sigma_2s*, sigma_2p, the degenerate pi_2p pair, the
degenerate pi_2p* pair, sigma_2p*) puts 2 electrons in every orbital up to
and including *both* pi_2p* orbitals, and leaves sigma_2p* empty. Compare O2,
which has two fewer electrons and so only half-fills the pi_2p* pair --
that's the pair of unpaired electrons that makes O2 a triplet. F2's extra two
electrons finish filling that pair, so every electron is paired: F2's ground
state is the closed-shell singlet X <sup>1</sup>Sigma_g^+ [1]. Plain
``PySCFDriver(atom=..., basis=...)`` with the default ``spin=0``/RHF is
therefore already correct here -- there is no ROHF step to add, unlike O2.

That does not make F2 an easy molecule, though -- see the next section.

THE ACTUAL TRAP: F2's BOND IS SURPRISINGLY HARD FOR HARTREE-FOCK
--------------------------------------------------------------------
Formal bond order 1, one shared electron pair, closed shell -- by that
description F2 should look as simple as H2. It measurably is not. Each
fluorine carries three tightly-packed lone pairs right next to the bond, and
their mutual repulsion destabilizes the sigma_2p bonding orbital relative to
what a naive picture would predict; the true F-F bond is far weaker than its
formula suggests (the experimental bond dissociation energy, ~157 kJ/mol, is
smaller than several element-element single bonds most people expect to be
comparable or weaker [2]). Hartree-Fock, being a single-determinant method,
underestimates the bonding still further, so expect a *substantial*
Hartree-Fock error even right at the equilibrium bond length -- measured on
this script's default active space, about 74 mHa at 1.412 A (run the smoke
test below and compare against the O2 script's own equilibrium-geometry
numbers to see how the two compare). That is the actual reason F2 earns a
place in this project: a closed-shell, single-bonded, textbook-simple-looking
molecule that Hartree-Fock still gets meaningfully wrong.

ACTIVE SPACE
-------------
F2 in STO-3G is 18 electrons in 10 spatial orbitals -> 20 qubits if you take
everything, same order of cost as O2's untruncated problem. The default here
is CAS(10,6): 10 electrons in the 6 orbitals derived from the 2p shell --
sigma(2p), the two pi(2p), the two pi*(2p), and sigma*(2p). This is the exact
analogue of the O2 script's CAS(8,6) -- the same six valence-shell orbitals,
just with F2's two extra electrons finishing the pi*(2p) shell instead of
leaving it half full. After the parity mapper's two-qubit reduction that is
**10 qubits**, matching O2's qubit count, but with only **35 UCCSD
parameters** rather than O2's 68 -- F2's closed-shell (5 alpha, 5 beta)
occupation admits fewer distinct excitations than O2's open-shell (5 alpha, 3
beta) one at the same orbital count. That keeps SLSQP's finite-difference
gradient cheap enough to use directly (see COST below), unlike O2.

COST
-----
SLSQP costs roughly one energy evaluation per parameter per optimizer step,
which is why H2, H2O and BeH2 all use it directly rather than reaching for a
gradient-free optimizer: it is fine anywhere from about 26 to 54 parameters,
and only starts to bite past roughly 60. F2's 35 parameters sit comfortably
inside that range, so this script follows H2/H2O/BeH2 and uses SLSQP, not
O2's COBYLA.

Measured on a laptop CPU (warm-started, i.e. the realistic per-point cost
after the first point): individual points ranged from **45 s to 150 s**,
averaging roughly **70-80 s**, with cost rising as the bond stretches (more
optimizer iterations near dissociation, where the landscape is flatter). An
earlier estimate here said "~5 s per point" -- that was a guess made before
measuring, and it was wrong by more than an order of magnitude; this is the
corrected, measured number. The default 17-point scan (0.9-2.5 A, step 0.1 A)
should therefore take on the order of **15-20 minutes**, not the "under two
minutes" the guess implied. It saves itself when it finishes, and ``--load``
reopens it instantly, so you only pay once.

A SECOND MEASURED FINDING: THE EXACT DIAGONALIZER CAN FAIL AT LONG RANGE.
Scanning out past about 4 A, ``NumPyMinimumEigensolver``'s sparse ARPACK
solver failed to converge at all in testing (``ArpackNoConvergence``) at
d = 5.0 A, while it succeeded at every point from 1.1 to 3.5 A, including
past the point where the curve should already be flattening out. This is a
property of the sparse eigensolver at near-degenerate geometries, not a VQE
problem -- the default scan range (0.9-2.5 A) never approaches it, so it is
mentioned here rather than guarded against.

SAVING AND REOPENING
----------------------
Every finished scan is saved next to this script as ``f2_scan.json`` (full
data plus the metadata needed to rebuild the plot) and ``f2_scan.csv`` (a flat
table for reports and spreadsheets). Reopening with ``--load`` skips the VQE
completely and drops you straight into review mode.

REFERENCES
-----------
[1] NIST Chemistry WebBook, F2 (CID 7782-41-4) diatomic constants -- ground
    state X 1-Sigma-g+, equilibrium bond length r_e = 1.41193 A (Raman
    spectroscopy):
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782414&Mask=1000
[2] CODATA / standard thermochemical tables list F2's bond dissociation
    energy at ~157 kJ/mol (~1.63 eV), notably weaker than Cl2's ~243 kJ/mol
    despite fluorine's much greater electronegativity and smaller size --
    the lone-pair-repulsion effect this script's docstring describes.

USAGE
------
Every case, explicitly. F2 is diatomic, so bond length is the only geometric
coordinate -- one curve, no surface.

  RUN A SCAN -- the default active space, CAS(10,6)
    python experiment/f2/f2_ground_state_estimation.py
    python experiment/f2/f2_ground_state_estimation.py --start 1.0 --stop 3.0 --step 0.1

  SMOKE TEST -- a handful of points, nothing saved
    python experiment/f2/f2_ground_state_estimation.py --start 1.2 --stop 1.6 --step 0.2 --no-save

  RELOAD A FINISHED SCAN -- no VQE re-run
    python experiment/f2/f2_ground_state_estimation.py --load f2_scan.json

  A SMALLER ACTIVE SPACE -- much faster, but drops orbitals from the space
  that carries F2's bonding physics. For plumbing checks only.
    python experiment/f2/f2_ground_state_estimation.py --active-electrons 6 --active-orbitals 4

  OTHER KNOBS
    --cold-start                 restart each point from the HF state instead of warm-starting
    --save myname / --no-save    where to write, or don't
    --no-show                    exit when the scan finishes instead of waiting on the window
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

# F2's experimental equilibrium bond length [1]. Used only as the reference
# printed next to whatever minimum the scan actually finds.
EQUILIBRIUM_BOND = 1.412   # Angstrom


# ===========================================================================
#  The quantum chemistry -- one bond length at a time
# ===========================================================================

def build_problem(distance: float, active_electrons: int,
                  active_orbitals: int) -> ElectronicStructureProblem:
    """F2 with the two atoms `distance` Angstrom apart, centred on the origin,
    restricted to a CAS(active_electrons, active_orbitals) active space.

    Closed-shell by construction -- F2's ground state has no unpaired
    electrons (see the module docstring), so this is a plain default-spin
    ``PySCFDriver`` call, the same shape as the H2O and BeH2 scripts. No
    ``spin=``/``MethodType.ROHF`` is needed here the way it is for O2.
    """
    geometry = f"F 0 0 {-distance / 2}; F 0 0 {distance / 2}"
    problem = PySCFDriver(atom=geometry, basis=BASIS).run()
    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). For F2 the inactive
    part is large -- the frozen 1s cores plus the 2s-derived orbitals -- so
    miss this and the energy is wrong by well over a hundred Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_distance(distance: float, active_electrons: int, active_orbitals: int,
                     max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one F-F distance.

    ``theta0`` warm-starts the optimizer from the previous bond length's
    solution, since neighbouring points have similar wavefunctions.
    """
    problem = build_problem(distance, active_electrons, active_orbitals)
    mapper = ParityMapper(num_particles=problem.num_particles)
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                   initial_state=hf_circuit)
    # UCCSD's excitations arrive as PauliEvolutionGate black boxes, which the
    # estimator would otherwise matrix-exponentiate on every single call.
    # Transpiling once here, outside the optimization loop, is much faster.
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
        "distance": float(distance),
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

# Drawn radius of a fluorine atom, in Angstrom -- close to F's covalent
# radius (0.64 A), scaled down like the O2 script so the two circles
# separate visibly as the molecule dissociates.
ATOM_RADIUS = 0.32
ATOM_COLOR = "#8ce050"   # conventional CPK-style light green for fluorine


class F2DissociationViewer:
    """Two live panels plus, once the scan is done, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called once per bond length as its energies
    come in -- the curve grows and the molecule redraws at the geometry just
    computed. Phase 2: :meth:`enable_review` turns the window into a
    scrubber over the finished curve.
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
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2, label="current")
        self.ax_energy.set_xlabel("F-F bond length (Angstrom)")
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title("F$_2$ dissociation curve")
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
        # A single-point scan (a smoke test) has an identical slider low/high
        # bound, which matplotlib warns about and cannot render -- skip it.
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
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="F2 dissociation curve by VQE (local simulator). F2 is diatomic, so "
                    "bond length is its only geometric degree of freedom -- one curve, "
                    "no surface. F2's ground state is a closed-shell singlet."
    )
    parser.add_argument("--start", type=float, default=0.9,
                        help="shortest F-F bond length in Angstrom (default 0.9)")
    parser.add_argument("--stop", type=float, default=2.5,
                        help="longest F-F bond length in Angstrom, inclusive (default 2.5)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="spacing between bond lengths in Angstrom (default 0.1)")
    parser.add_argument("--active-electrons", type=int, default=10,
                        help="electrons in the active space (default 10)")
    parser.add_argument("--active-orbitals", type=int, default=6,
                        help="spatial orbitals in the active space (default 6 -> 10 qubits). "
                             "6 is the smallest space containing the 2p-derived orbitals where "
                             "F2's bonding and antibonding electrons live -- see the module "
                             "docstring.")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each geometry's optimization from the Hartree-Fock "
                             "state instead of warm-starting from the previous geometry's "
                             "solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "(a .json written by an earlier run) straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: f2_scan.json/.csv next to "
                             "this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save the results of this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="exit when the scan finishes instead of staying open in "
                             "review mode. The live plot still draws as the scan runs; it "
                             "just doesn't block at the end waiting for you to close the "
                             "window.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # -- replay a finished scan, no VQE at all -------------------------
    if args.load:
        metadata, rows = load_scan(Path(args.load), default_dir=here)
        space = (f"CAS({metadata['active_electrons']}, {metadata['active_orbitals']})"
                 if metadata.get("active_electrons") is not None else "active space not recorded")
        print(f"loaded {len(rows)} points from {args.load} "
              f"(scanned {metadata['start']:.2f}-{metadata['stop']:.2f} A, "
              f"step {metadata['step']:.2f} A, {space}) -- no VQE re-run")
        viewer = F2DissociationViewer(max_distance=float(metadata["max_distance"]))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, args.no_show)
        return

    if args.step <= 0:
        parser.error("--step must be positive")
    if args.stop < args.start:
        parser.error("--stop must be >= --start")

    distances = np.arange(args.start, args.stop + args.step / 2, args.step)

    print(f"F2 scan: {len(distances)} bond lengths from {distances[0]:.2f} to "
          f"{distances[-1]:.2f} A (step {args.step} A)")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})")
    # Measured on a laptop CPU at the default CAS(10,6), warm-started: 45-150s
    # per point, averaging ~75s. Just so a 15-20 minute run isn't a surprise
    # a few points in -- see the module docstring for the full measurement.
    if args.active_electrons == 10 and args.active_orbitals == 6:
        print(f"estimated runtime: ~{len(distances) * 75 / 60:.0f} min "
              f"(~75s/point measured at this active space; the plot updates as "
              f"each point lands)")

    viewer = F2DissociationViewer(max_distance=float(distances[-1]))
    rows: List[Dict[str, Any]] = []
    previous_params = None
    for i, d in enumerate(distances):
        t0 = time.perf_counter()
        row = run_one_distance(d, args.active_electrons, args.active_orbitals,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:3d}/{len(distances)}] d={d:.2f} A   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({time.perf_counter() - t0:.2f}s)")

    _report(rows)
    _report_monotonicity(rows)

    if not args.no_save:
        target = resolve_target(args.save, here, "f2_scan")
        metadata = {
            "molecule": "F2", "basis": BASIS,
            "scan_type": "curve",
            "coordinate": "bond",
            "degrees_of_freedom": 1,
            "start": float(distances[0]), "stop": float(distances[-1]), "step": float(args.step),
            "unit": "A",
            "max_distance": float(distances[-1]),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
        }
        json_path, csv_path = save_scan(target, metadata, rows)
        print(f"saved {display_path(json_path)}")
        print(f"saved {display_path(csv_path)}")
        print(f"reopen it later without recomputing:  "
              f"python {display_path(Path(__file__))} --load {json_path.name}")
        print(f"interpolate it:  python experiment/f2_spline/f2_spline_interpolation.py")

    _finish(viewer, args.no_show)


def _finish(viewer: F2DissociationViewer, no_show: bool) -> None:
    if no_show:
        print("\nScan complete. --no-show was passed, so exiting instead of waiting.")
        return
    viewer.enable_review()


def _report(rows: List[Dict[str, Any]]) -> None:
    """Final summary: where the minimum is, and how big each error is."""
    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at d = {best['distance']:.2f} A "
          f"(experimental F2 bond length is ~{EQUILIBRIUM_BOND} A)")

    step = rows[1]["distance"] - rows[0]["distance"] if len(rows) > 1 else 0.0
    if step > 0.05:
        print(f"  the grid spacing is {step:.2f} A, so this is the lowest *sample*, not the")
        print(f"  minimum of the curve -- fit an interpolant through these points and")
        print(f"  minimize that instead (see experiment/f2_spline/).")

    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    at_eq = min(rows, key=lambda r: abs(r["distance"] - EQUILIBRIUM_BOND))
    print(f"largest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"  ... and {abs(at_eq['hf'] - at_eq['exact']) * 1000:.3f} mHa "
          f"at d = {at_eq['distance']:.2f} A, near equilibrium -- F2's bond is "
          f"harder for Hartree-Fock than its simple single-bond formula suggests, "
          f"see the module docstring")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")
    if max_vqe * 1000 > 1.0:
        print("  note: above 1 mHa somewhere -- the optimizer may not have fully converged. "
              "Try --cold-start on a few points to check.")


def _report_monotonicity(rows: List[Dict[str, Any]]) -> None:
    """Flag the active-space failure mode described in the sibling scripts'
    docstrings: past its minimum a ground-state curve must rise as the atoms
    separate. A curve that turns back downward is the signature of an active
    space too small to describe the bond breaking, not a VQE failure.
    """
    ordered = sorted(rows, key=lambda r: r["distance"])
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
        print("  an active space too small to describe the breaking bond, not a VQE failure.")
        print("  Try a larger --active-orbitals / --active-electrons.")
    else:
        print("  curve rises monotonically past its minimum -- active space looks adequate")


if __name__ == "__main__":
    main()
