"""
vqe_demo.py -- a tour of src/vqe.py

Run from the project root:

    python examples/vqe_demo.py

What this does, in three parts, one after another:

  1. Build H2 (see molecule.py) and hand it to VQE with the UCCSD ansatz and
     the COBYLA optimizer -- the standard, reliable VQE recipe.
  2. Run the same molecule through the *hand-written* gradient-descent
     optimizer (parameter-shift rule) instead of COBYLA, to see the same
     answer reached by an optimizer whose math is fully spelled out in
     vqe.py.
  3. Run the generic hardware-efficient ansatz on the same molecule and
     compare: fewer chemistry assumptions, but (as vqe.py's docstring
     explains) no guarantee of hitting the right answer.

Each ``VQE(...).run()`` call below pops up its own Matplotlib window and
updates the loss curve live, point by point, as the optimizer works.
Nothing is written to disk -- when a run finishes, its window freezes and
the script waits for you to close it before moving on to the next part
(exactly what ``live_plot=True``, the default, does).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import molecule as mol  # noqa: E402
from vqe import VQE  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    h2 = mol.Molecule("h2", basis="sto-3g")
    print(f"Molecule: {h2.name}  ({h2.n_qubits} qubits, {h2.n_electrons} electrons)")
    print(f"Hartree-Fock energy : {h2.hf_energy:.10f} Ha  (VQE's classical starting point)")
    print(f"Exact ground energy : {h2.ground_state_energy_exact():.10f} Ha  (what VQE is trying to find)")

    # -- 1. the standard recipe: UCCSD + COBYLA -----------------------
    section("1. UCCSD ansatz + COBYLA optimizer  (the standard VQE recipe)")
    print("(a live loss-curve window will pop up now -- close it when you're done looking)")
    result = VQE(h2, ansatz="uccsd", optimizer="cobyla").run(max_iterations=200, verbose=False)
    result.print_steps(every=5)

    # -- 2. same ansatz, but the optimizer is our own gradient descent -
    section("2. UCCSD ansatz + hand-written gradient descent (parameter-shift rule)")
    result_gd = VQE(h2, ansatz="uccsd", optimizer="gradient_descent").run(
        max_iterations=200, learning_rate=0.3, verbose=False
    )
    result_gd.print_steps(every=5)

    # -- 3. a generic ansatz, for comparison ---------------------------
    section("3. Hardware-efficient ansatz + COBYLA  (generic, chemistry-agnostic)")
    result_hea = VQE(h2, ansatz="hardware_efficient", optimizer="cobyla", depth=2).run(
        max_iterations=200, seed=0, verbose=False
    )
    result_hea.print_steps(every=10)

    # -- summary table --------------------------------------------------
    section("Summary")
    rows = [
        ("UCCSD + COBYLA", result),
        ("UCCSD + gradient descent", result_gd),
        ("hardware-efficient + COBYLA", result_hea),
    ]
    print(f"{'run':<30} {'E (Ha)':>16} {'|error| (Ha)':>16} {'# evaluations':>14}")
    print("-" * 78)
    for name, r in rows:
        print(f"{name:<30} {r.energy:>16.8f} {r.error_vs_exact:>16.3e} {r.n_evaluations:>14d}")
    print("\ndone.")


if __name__ == "__main__":
    main()
