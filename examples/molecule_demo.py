"""
molecule_demo.py -- a tour of src/molecule.py

Run from the project root:

    python examples/molecule_demo.py            # visualization only
    python examples/molecule_demo.py --quantum  # + qubit Hamiltonian (needs cudaq-solvers)

Outputs land in ./out/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import molecule as mol  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def visualization_demo() -> None:
    print("\n=== visualization ===")
    for name, style in [("caffeine", "ball_and_stick"),
                        ("benzene", "spacefill"),
                        ("h2o", "ball_and_stick")]:
        geom = mol.load_geometry(name)
        html = OUT / f"{name}_{style}.html"
        mol.save_html(geom, html, style=style, title=name.title())
        print(f"  {name:<10} {mol.chemical_formula(geom):<10} "
              f"{len(mol.covalent_bonds(geom))} bonds  ->  {html.relative_to(ROOT)}")

    # plotly figure, e.g. to embed in a report
    fig = mol.view_plotly("nh3", style="ball_and_stick", show_labels=True, title="Ammonia")
    fig.write_html(str(OUT / "nh3_plotly.html"), include_plotlyjs="cdn")
    print(f"  nh3 (plotly, labelled)  ->  {(OUT / 'nh3_plotly.html').relative_to(ROOT)}")


def quantum_demo() -> None:
    print("\n=== qubit Hamiltonian (CUDA-Q Solvers) ===")
    if not mol.cudaq_available():
        print("  cudaq-solvers not installed -- skipping.")
        print("  Linux/WSL:  pip install cudaq-solvers")
        return

    h2 = mol.Molecule("h2", basis="sto-3g")
    h2.print_summary()
    print(f"  exact ground-state energy : {h2.ground_state_energy_exact():.10f} Ha")
    print(f"  Hartree-Fock reference    : |{h2.hartree_fock_bitstring()}>")
    print(f"  # UCCSD parameters        : {h2.num_uccsd_parameters()}")

    # H2 dissociation curve: HF vs FCI vs exact diagonalization
    print("\n  scanning H-H bond length ...")
    scan = mol.bond_scan("h2", (0, 1), np.linspace(0.4, 2.6, 23),
                         energies=("hf_energy", "fci_energy"), exact=True, progress=False)
    r_min = scan["distance"][np.nanargmin(scan["fci_energy"])]
    print(f"  FCI equilibrium bond length ~ {r_min:.3f} A")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        for key, label in [("hf_energy", "Hartree-Fock"), ("fci_energy", "FCI"),
                           ("exact", "exact diag.")]:
            plt.plot(scan["distance"], scan[key], marker=".", label=label)
        plt.xlabel("H-H distance  (Angstrom)")
        plt.ylabel("energy  (Hartree)")
        plt.title("H$_2$ / STO-3G dissociation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "h2_dissociation.png", dpi=150)
        print(f"  wrote {(OUT / 'h2_dissociation.png').relative_to(ROOT)}")
    except ImportError:
        print("  (install matplotlib to plot the dissociation curve)")


if __name__ == "__main__":
    visualization_demo()
    if "--quantum" in sys.argv:
        quantum_demo()
    print("\ndone.")
