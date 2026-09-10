"""
qiskit_vqe.py -- VQE on LiH: Hartree-Fock vs. VQE vs. exact energy across a
scan of Li-H bond lengths, built on Qiskit so it can (eventually) run on a
real IBM Quantum computer.

This reproduces the classic "PySCF driver -> qubit Hamiltonian -> UCCSD
ansatz -> SLSQP-optimized VQE -> compare to exact diagonalization" workflow
from the old ``qiskit.aqua`` / ``qiskit.chemistry`` tutorials. Those two
packages were retired years ago (Aqua's algorithms moved into
``qiskit_algorithms``; chemistry moved into ``qiskit-nature`` and then its
own API was overhauled into ``qiskit_nature.second_q``), so a notebook
written against them no longer runs -- the imports themselves fail. This
file is the same algorithm and the same comparison, rewritten against the
packages that actually install and run today:

    retired package/class                          this file uses instead
    -------------------------------------------------------------------------
    qiskit.aqua.QuantumInstance                     qiskit.primitives.StatevectorEstimator
    qiskit.aqua.algorithms.VQE                      scipy.optimize / qiskit_algorithms.optimizers
                                                     driving the estimator directly
    qiskit.aqua.algorithms.NumPyMinimumEigensolver  qiskit_algorithms.NumPyMinimumEigensolver
    qiskit.aqua.components.optimizers.SLSQP         qiskit_algorithms.optimizers.SLSQP
    qiskit.chemistry.drivers.PySCFDriver            qiskit_nature.second_q.drivers.PySCFDriver
    qiskit.chemistry.components.initial_states
        .HartreeFock                                qiskit_nature.second_q.circuit.library.HartreeFock
    qiskit.chemistry.components.variational_forms
        .UCCSD                                      qiskit_nature.second_q.circuit.library.UCCSD
    qiskit.chemistry.core.Hamiltonian(freeze_core=True,
        two_qubit_reduction=True, ...)               FreezeCoreTransformer
                                                      + ParityMapper(num_particles=...)
                                                      (parity mapping is what makes the
                                                      "two-qubit reduction" possible)

WHY THIS FILE (AND NOT src/vqe.py) CAN TARGET IBM HARDWARE
-------------------------------------------------------------
``src/vqe.py`` is built on NVIDIA CUDA-Q, which has no IBM backend -- run
``cudaq.get_targets()`` and IBM is simply not in the list (CUDA-Q's hardware
partners are Quantinuum, IonQ, OQC, Amazon Braket, Pasqal, QuEra, IQM, ...).
IBM only exposes its quantum computers through its own SDK, Qiskit, via the
Qiskit Runtime cloud service. This file is built on Qiskit for exactly that
reason. It runs on a local, exact simulator by default (no account needed);
see ``run_hardware_scan()`` near the bottom for the actual path onto real
hardware, using the ``CRN`` / ``IBM_API_KEY`` credentials from ``.env``.

ONE IMPORTANT GOTCHA THAT WAS SILENTLY WRONG THE FIRST TIME
----------------------------------------------------------------
After ``FreezeCoreTransformer``, the *total* molecular energy is not just
"eigenvalue of the qubit Hamiltonian + nuclear repulsion energy". Freezing
the core orbitals adds its *own* constant energy shift (the frozen
electrons' contribution), stored separately in
``problem.hamiltonian.constants``. Forgetting it gives an energy off by
several Hartree (LiH came out near 0 instead of near -7.88 the first time
this was tested) -- ``total_energy()`` below adds up *all* of
``problem.hamiltonian.constants.values()``, not just nuclear repulsion.

Run from the project root:

    python "VQE demo on IBM Quantum Computer/qiskit_vqe.py"

A full 15-point bond-length scan takes roughly 5-6 minutes on a laptop CPU
(each point runs its own from-scratch VQE); a progress line prints as each
step finishes.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

# Geometry template: H at -d/2, Li at +d/2 on the z-axis, so the two atoms
# are exactly `d` Angstrom apart (same convention as the original notebook).
GEOMETRY_TEMPLATE = "H 0 0 {h_z}; Li 0 0 {li_z}"
BASIS = "sto3g"


def build_problem(distance: float) -> ElectronicStructureProblem:
    """Run PySCF (via qiskit-nature's driver) for LiH at the given Li-H
    distance (Angstrom), then freeze the Li 1s core orbital -- the same
    `freeze_core=True` step as the original notebook's ``Hamiltonian(...)``
    call."""
    geometry = GEOMETRY_TEMPLATE.format(h_z=-distance / 2, li_z=distance / 2)
    driver = PySCFDriver(atom=geometry, basis=BASIS)
    problem = driver.run()
    return FreezeCoreTransformer().transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """electronic eigenvalue -> total molecular energy, including *every*
    constant shift qiskit-nature is tracking (nuclear repulsion AND the
    frozen-core correction) -- see the module docstring's gotcha above."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_distance(distance: float, max_iterations: int = 1000) -> Dict[str, Any]:
    """Hartree-Fock, VQE (UCCSD + SLSQP) and exact energies for one Li-H
    bond length -- one point of the dissociation curve."""
    problem = build_problem(distance)
    mapper = ParityMapper(num_particles=problem.num_particles)
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                    initial_state=hf_circuit)
    # UCCSD's excitation operators come out as PauliEvolutionGate black
    # boxes; StatevectorEstimator simulates those by matrix-exponentiating
    # them on every single call (~20x slower). Transpiling once, up front,
    # to concrete 1-/2-qubit gates fixes that -- do this before the
    # optimization loop, not inside `cost()`.
    ansatz_t = transpile(ansatz, basis_gates=["cx", "rz", "sx", "x", "h"], optimization_level=1)

    estimator = StatevectorEstimator()

    exact_electronic = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian).eigenvalue.real
    hf_electronic = float(estimator.run([(hf_circuit, hamiltonian, [])]).result()[0].data.evs)

    def cost(theta: np.ndarray) -> float:
        return float(estimator.run([(ansatz_t, hamiltonian, theta)]).result()[0].data.evs)

    theta0 = np.zeros(ansatz.num_parameters)  # theta=0 --> exactly the HF state
    result = minimize(cost, theta0, method="SLSQP", options={"maxiter": max_iterations})

    return {
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
        "vqe_params": np.asarray(result.x),
        # Handed back so a hardware run can reuse them without recomputing:
        # every distance in this scan shares the same active space (same
        # frozen core, same num_spatial_orbitals/num_particles), so the
        # *shape* of the ansatz circuit is identical across the whole scan
        # -- only the Hamiltonian's coefficients and the fitted theta differ.
        "problem": problem,
        "hamiltonian": hamiltonian,
        "ansatz": ansatz,
    }


def run_local_scan(distances: np.ndarray, out_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Hartree-Fock / local-VQE / exact energies across `distances`, on the
    free local simulator -- no account, no quota, nothing queued.
    """
    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(distances):
        t0 = time.perf_counter()
        energies = run_one_distance(d)
        rows.append({"distance": float(d), "hf": energies["hf"],
                     "vqe_local": energies["vqe"], "exact": energies["exact"]})
        print(f"step {i:2d}  distance={d:.2f} A   "
              f"HF={energies['hf']:.6f}  VQE={energies['vqe']:.6f}  "
              f"exact={energies['exact']:.6f}  "
              f"(|VQE-exact|={abs(energies['vqe'] - energies['exact']):.2e} Ha, "
              f"{time.perf_counter() - t0:.1f}s)")

    plt.figure()
    plt.plot(distances, [r["hf"] for r in rows], label="Hartree-Fock")
    plt.plot(distances, [r["vqe_local"] for r in rows], "o", label="VQE")
    plt.plot(distances, [r["exact"] for r in rows], "x", label="Exact")
    plt.xlabel("Interatomic distance")
    plt.ylabel("Energy")
    plt.title("LiH Ground State Energy")
    plt.legend(loc="upper right")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "lih_energies_simulator.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["distance", "hf", "vqe_local", "exact"])
            writer.writeheader()
            writer.writerows(rows)
        plt.savefig(out_dir / "lih_energy_scan_simulator.png", dpi=150, bbox_inches="tight")
        print(f"wrote {out_dir / 'lih_energies_simulator.csv'}")
        print(f"wrote {out_dir / 'lih_energy_scan_simulator.png'}")
    plt.show()
    return rows


# ===========================================================================
#  Real IBM Quantum hardware
# ===========================================================================
#
# Needs an IBM Quantum Platform account and ../.env filled in from
# ../.env.example (CRN + IBM_API_KEY). Not wired into a bare `python
# qiskit_vqe.py` run -- call `run_hardware_scan()` yourself (see the bottom
# of this file) when you're ready to spend real device time.
#
# All distance points share the same active space (same frozen core, same
# num_spatial_orbitals/num_particles -- only the bond length changes), so
# they share one ansatz *shape*. That means the whole scan can go out as
# ONE Runtime job with one pub (circuit + observable + parameters) per
# distance, instead of queuing separately N times -- you wait in line once,
# not N times, and only spend one job's worth of overhead.

def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (KEY=value per line) -- avoids adding a new
    dependency just to read two environment variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _ibm_credentials() -> tuple[str, str]:
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    crn = os.environ.get("CRN")
    api_key = os.environ.get("IBM_API_KEY")
    if not crn or not api_key or "your-ibm-quantum-api-key-here" in api_key:
        raise RuntimeError(
            "Fill in CRN and IBM_API_KEY in .env (copy .env.example -> .env) first."
        )
    return crn, api_key


def run_hardware_scan(
    distances: np.ndarray,
    out_dir: Path,
    csv_name: str = "lih_energies.csv",
    figure_name: str = "lih_energy_scan.png",
    shots: int = 2000,
    confirm: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full Hartree-Fock / local-VQE / exact / real-hardware-VQE
    comparison across `distances`, then save a CSV and a PNG into `out_dir`.

    Local optimization (SLSQP against the simulator) still happens per
    point -- that part is cheap and there's no reason to burn real device
    time on hundreds of optimizer evaluations. Only the *final*, already-
    optimized circuit for each distance gets sent to real hardware, all in
    one batched job.

    ``shots`` is set *explicitly* rather than left at the service's default:
    a previous run of this exact function, with shots left unset, burned
    about 4 of a 10-minute account balance on a single 7-point job before
    it had to be cancelled. An explicit, small shot count keeps the cost
    predictable -- more shots means less statistical noise per energy but
    also more QPU time (and therefore quota) per circuit.

    ``confirm=True`` (the default) prints the plan -- backend, number of
    points, shots -- and requires you to type ``yes`` before anything is
    submitted. Pass ``confirm=False`` only once you already know the cost
    is acceptable (e.g. scripting a rerun you've sized before).
    """
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    crn, api_key = _ibm_credentials()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key, instance=crn)
    backend = service.least_busy(operational=True, simulator=False)

    if confirm:
        print(f"About to submit ONE batched job to a REAL IBM backend:")
        print(f"  backend       : {backend.name}")
        print(f"  distance points: {len(distances)}")
        print(f"  shots/circuit : {shots}  (more shots = less noise, more quota used)")
        print(f"This uses your account's usage quota and cannot be un-spent once it runs.")
        answer = input("Type 'yes' to submit, anything else to cancel: ").strip().lower()
        if answer != "yes":
            print("cancelled -- nothing was submitted.")
            return []

    rows: List[Dict[str, Any]] = []
    pubs = []
    isa_circuit = None
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)

    for i, d in enumerate(distances):
        t0 = time.perf_counter()
        local = run_one_distance(d)
        if isa_circuit is None:
            # Same ansatz shape at every distance in this scan (see the
            # comment above) -- transpile it for this backend exactly once.
            isa_circuit = pass_manager.run(local["ansatz"])
        isa_observable = local["hamiltonian"].apply_layout(isa_circuit.layout)
        pubs.append((isa_circuit, isa_observable, local["vqe_params"]))
        rows.append({
            "distance": float(d),
            "hf": local["hf"],
            "vqe_local": local["vqe"],
            "exact": local["exact"],
            "problem": local["problem"],  # dropped before writing the CSV
        })
        print(f"  [{i + 1}/{len(distances)}] local VQE done for d={d:.2f} A "
              f"({time.perf_counter() - t0:.1f}s) -- queued for hardware")

    print(f"submitting one batched job ({shots} shots/circuit) with {len(pubs)} points "
          f"to {backend.name} (queue time varies -- this can take a while)")
    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = shots
    estimator.options.resilience_level = 0  # skip the (expensive) default error mitigation
    job = estimator.run(pubs)
    print(f"job id: {job.job_id()}  (track it at https://quantum.cloud.ibm.com/)")
    job_result = job.result()

    for row, pub_result in zip(rows, job_result):
        electronic = float(pub_result.data.evs)
        row["vqe_hardware"] = total_energy(row.pop("problem"), electronic)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["distance", "hf", "vqe_local", "exact", "vqe_hardware"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    plt.figure()
    ds = [r["distance"] for r in rows]
    plt.plot(ds, [r["hf"] for r in rows], label="Hartree-Fock")
    plt.plot(ds, [r["vqe_local"] for r in rows], "o", label="VQE (simulator)")
    plt.plot(ds, [r["exact"] for r in rows], "x", label="Exact")
    plt.plot(ds, [r["vqe_hardware"] for r in rows], "^", label=f"VQE ({backend.name})")
    plt.xlabel("Interatomic distance")
    plt.ylabel("Energy")
    plt.title("LiH Ground State Energy -- simulator vs. real IBM hardware")
    plt.legend(loc="upper right")
    figure_path = out_dir / figure_name
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"wrote {figure_path}")

    return rows


# ===========================================================================
#  CLI -- choose simulator (free, default) or real hardware (uses quota)
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQE on LiH: Hartree-Fock vs. VQE vs. exact energy across a bond-length scan."
    )
    parser.add_argument(
        "--backend", choices=["simulator", "hardware"], default="simulator",
        help="'simulator' (default): free, local, exact StatevectorEstimator. "
             "'hardware': submits one real job to your IBM Quantum account -- uses quota.",
    )
    parser.add_argument(
        "--points", type=int, default=7,
        help="number of bond-length points to scan, starting at 0.5 A in 0.25 A steps "
             "(default 7 -> 0.5 to 2.0 A). Keep this small for --backend hardware.",
    )
    parser.add_argument(
        "--shots", type=int, default=2000,
        help="[hardware only] shots per circuit -- lower is cheaper/faster but noisier (default 2000).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="[hardware only] skip the confirmation prompt before submitting.",
    )
    args = parser.parse_args()

    distances = 0.5 + 0.25 * np.arange(args.points)
    out_dir = Path(__file__).resolve().parent

    if args.backend == "simulator":
        run_local_scan(distances, out_dir=out_dir)
    else:
        run_hardware_scan(distances, out_dir, shots=args.shots, confirm=not args.yes)


if __name__ == "__main__":
    main()
