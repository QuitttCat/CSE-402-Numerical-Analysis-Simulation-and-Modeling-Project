"""
h2_ground_state_estimation.py -- H2 ground-state energy by VQE across a range
of bond lengths, plotted live, next to a picture of the molecule pulling
itself apart.

Simulator only. No IBM account, no queue, no quota -- every energy here is
computed locally with Qiskit's exact ``StatevectorEstimator``. (The
hardware-capable version of this workflow lives in
``../../VQE demo on IBM Quantum Computer/qiskit_vqe.py``.)

WHAT YOU GET
-------------
One window, two panels, side by side:

  LEFT   the dissociation curve -- Hartree-Fock, VQE and exact energy versus
         H-H bond length. It is drawn *as the scan runs*: each bond length
         finishes its own little VQE optimization, then immediately appears
         on the curve.
  RIGHT  the two hydrogen atoms themselves, drawn at the bond length
         currently being computed, so you can watch the geometry stretch
         while the left panel fills in.

When the scan finishes, the window stays open in **review mode**: press the
LEFT / RIGHT arrow keys (or drag the slider) to walk back and forth along
the finished curve. The molecule on the right re-draws to match whichever
point you land on, so you can step through "what does the molecule actually
look like at this energy?" one bond length at a time.

THE PHYSICS, BRIEFLY (see ../../src/vqe.py for the full explanation)
----------------------------------------------------------------------
At each bond length we build H2's electronic Hamiltonian with PySCF, map it
onto qubits, and minimize <psi(theta)|H|psi(theta)> over a UCCSD ansatz --
that minimum is the molecule's ground-state energy at that geometry. Doing
that at many bond lengths traces out the dissociation curve: energy drops as
the atoms approach their equilibrium separation (~0.74 A for H2), then rises
again as they are squeezed closer, and flattens out as they are pulled apart
into two free atoms.

H2 in a minimal (STO-3G) basis is small -- 2 qubits and 3 parameters after
the parity mapping's two-qubit reduction -- so each point takes well under a
second and a fine scan is cheap.

USAGE
------
    python experiment/h2/h2_ground_state_estimation.py
    python experiment/h2/h2_ground_state_estimation.py --start 0.3 --stop 3.0 --step 0.05
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem

BASIS = "sto3g"


# ===========================================================================
#  The quantum chemistry -- one bond length at a time
# ===========================================================================

def build_problem(distance: float) -> ElectronicStructureProblem:
    """H2 with the two atoms `distance` Angstrom apart, centred on the origin.

    Note there is no ``FreezeCoreTransformer`` here, unlike the LiH version
    of this script: hydrogen has no core orbitals to freeze. Its only
    orbital *is* its valence orbital, so freezing "the core" would throw
    away the entire molecule.
    """
    geometry = f"H 0 0 {-distance / 2}; H 0 0 {distance / 2}"
    return PySCFDriver(atom=geometry, basis=BASIS).run()


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature is tracking (for H2 that is just the nuclear repulsion,
    which grows as the atoms get closer)."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_distance(distance: float, max_iterations: int = 1000) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one bond length."""
    problem = build_problem(distance)
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

    theta0 = np.zeros(ansatz.num_parameters)  # theta=0 is exactly the HF state
    result = minimize(cost, theta0, method="SLSQP", options={"maxiter": max_iterations})

    return {
        "distance": float(distance),
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
    }


# ===========================================================================
#  The window -- live curve on the left, live molecule on the right
# ===========================================================================

# Drawn radius of a hydrogen atom, in Angstrom. Roughly H's covalent radius
# (0.31 A) -- big enough that the two circles visibly overlap at short bond
# lengths and clearly separate as the molecule dissociates.
ATOM_RADIUS = 0.30


class DissociationViewer:
    """Two live panels plus, once the scan is done, arrow-key/slider review.

    Used in two phases:
      1. During the scan, :meth:`add_point` is called once per bond length
         as its energies come in -- the curve grows and the molecule on the
         right redraws at the bond length just computed.
      2. After the scan, :meth:`enable_review` turns the same window into a
         scrubber: left/right arrows or the slider move a marker along the
         finished curve, and the molecule follows.
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
        # VQE markers are hollow so the green "exact" crosses stay visible
        # underneath them -- for H2 the two agree to ~1e-10, so a filled
        # marker would completely hide the exact curve and make it look
        # like it was never plotted.
        (self.line_vqe,) = self.ax_energy.plot([], [], "o-", color="#4c5fd5", label="VQE",
                                               markerfacecolor="none", markeredgewidth=1.5)
        (self.line_exact,) = self.ax_energy.plot([], [], "x", color="#1a9850", label="Exact",
                                                 markersize=5, zorder=3)
        (self.marker,) = self.ax_energy.plot([], [], "D", color="black", markersize=11,
                                             fillstyle="none", markeredgewidth=2, label="current")
        self.ax_energy.set_xlabel("H-H bond length (Angstrom)")
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title("H$_2$ dissociation curve")
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
            Circle((0, 0), ATOM_RADIUS, facecolor="#f2f2f2", edgecolor="black", linewidth=1.5, zorder=2)
            for _ in range(2)
        ]
        for atom in self.atoms:
            self.ax_mol.add_patch(atom)
        self.mol_label = self.ax_mol.text(0, -span / 2.6 + 0.12, "", ha="center", fontsize=11)

        self._redraw()

    # -- phase 1: live during the scan ---------------------------------

    def add_point(self, row: Dict[str, Any]) -> None:
        """Add one finished bond length to the curve and move the molecule."""
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
        """Point the marker and the molecule at ``rows[index]``."""
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
        """Turn the window into a scrubber over the finished scan."""
        slider_ax = self.fig.add_axes([0.12, 0.06, 0.5, 0.04])
        self.slider = Slider(
            slider_ax, "point", 0, len(self.rows) - 1,
            valinit=self.index, valstep=1,
        )
        self.slider.on_changed(lambda v: self._show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax_energy.set_title("H$_2$ dissociation curve  --  use LEFT/RIGHT arrows or the slider")
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
        # Driving the slider re-enters _show() through on_changed, which
        # keeps the slider handle and the plotted marker in sync.
        if self.slider is not None:
            self.slider.set_val(new_index)
        else:
            self._show(new_index)


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2 dissociation curve by VQE (local simulator), plotted live "
                    "alongside the molecular geometry."
    )
    parser.add_argument("--start", type=float, default=0.3,
                        help="shortest H-H bond length in Angstrom (default 0.3)")
    parser.add_argument("--stop", type=float, default=2.5,
                        help="longest H-H bond length in Angstrom, inclusive (default 2.5)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="spacing between bond lengths in Angstrom (default 0.1)")
    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be positive")
    if args.stop < args.start:
        parser.error("--stop must be >= --start")

    # + step/2 so that --stop itself is included despite floating-point drift
    distances = np.arange(args.start, args.stop + args.step / 2, args.step)
    print(f"scanning {len(distances)} bond lengths from {distances[0]:.2f} to "
          f"{distances[-1]:.2f} A (step {args.step} A)")

    viewer = DissociationViewer(max_distance=float(distances[-1]))
    for i, d in enumerate(distances):
        t0 = time.perf_counter()
        row = run_one_distance(d)
        viewer.add_point(row)
        print(f"  [{i + 1:3d}/{len(distances)}] d={d:.2f} A   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({time.perf_counter() - t0:.2f}s)")

    best = min(viewer.rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at d = {best['distance']:.2f} A "
          f"(experimental H2 bond length is ~0.74 A)")
    viewer.enable_review()


if __name__ == "__main__":
    main()
