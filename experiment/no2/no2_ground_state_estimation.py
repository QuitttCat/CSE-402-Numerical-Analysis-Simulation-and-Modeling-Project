"""
no2_ground_state_estimation.py -- nitrogen dioxide's (NO2) ground-state
energy by VQE, scanned along its symmetric stretch, its bend, or the full
bond/angle surface, plotted live next to a picture of the molecule deforming.

Simulator only. No IBM account, no queue, no quota.

NO2 COMBINES BOTH TRAPS THIS PROJECT HAS SEEN SEPARATELY
--------------------------------------------------------------
BeH2 in the sibling folder is symmetric (two identical Be-H bonds) but
closed-shell, which is what makes its surface a two-coordinate scan of an
ordinary singlet. O2 is open-shell (a triplet) but diatomic, so it never
needs a surface at all. **NO2 is both**: it is bent and symmetric like BeH2
-- one N-O bond length shared by both bonds, one O-N-O angle -- and it is a
genuine radical like O2, with an odd number of electrons and one of them
unpaired. Getting NO2 right means combining the two lessons those scripts
teach separately, in one molecule.

THE TRAP: NO2 IS A DOUBLET
------------------------------
Nitrogen contributes 7 electrons, each oxygen 8, for 23 total -- an odd
number, so pairing every electron is not merely wrong, it is *impossible*.
One electron is necessarily unpaired: NO2's ground state is a doublet
(S = 1/2), sitting in a nonbonding orbital on nitrogen that VSEPR theory
treats as taking up less room than a full lone pair would. That is why the
O-N-O angle (~134 deg) sits between water's tightly bent 104.5 deg (a full
lone pair) and BeH2's linear 180 deg (no lone pair at all) -- NO2's single
electron pushes the two N-O bonds apart, but less forcefully than a pair
would.

Exactly as in the O2 script: every default ``PySCFDriver`` call elsewhere in
this repo assumes ``spin=0`` and closed-shell RHF. Point that at NO2 and it
will not complain -- it will silently return some other electronic state and
hand back a smooth, plausible, wrong curve. So this script passes ``spin=1``
(PySCF's convention: spin = 2*S) and ``MethodType.ROHF``, the restricted
open-shell method that can actually represent one unpaired electron. This is
a correctness requirement, not a stylistic choice.

One consequence worth being explicit about: with one electron unpaired,
active-space electron counts must be **odd**, and alpha/beta must be passed
as an explicit ``(alpha, beta)`` tuple rather than a plain count -- an odd
total does not split evenly, and letting the transformer guess would risk
silently picking the wrong state. See :func:`active_particles`.

THREE MODES -- SAME SHAPE AS BeH2's
----------------------------------------
Because the two N-O bonds are chemically identical (both bonds are the same
resonance-averaged bond order, roughly one and a half), NO2 enjoys the same
symmetry constraint BeH2 does: one bond length shared by both legs, one
angle. So the same three modes make sense here, with the same meaning:

  --coordinate stretch   Both N-O bonds lengthened together, angle held at
                         its equilibrium value. Watch the correlation energy
                         grow with distance, same story as every stretch
                         curve in this project -- except here it stacks on
                         top of the doublet's already-large static
                         correlation at equilibrium, the way O2's does.

  --coordinate bend      The O-N-O angle varied, both bonds held fixed. The
                         minimum sits at ~134 deg, not at either end of a
                         typical scan range -- a shallower, wider-angle well
                         than water's, because the unpaired electron's
                         opening-out effect is weaker than a full lone pair.

  --coordinate surface   Both bond length and angle varied together on a
                         grid -- directly comparable to BeH2's own surface.
                         The interesting question is the same one: does the
                         optimal bond length drift as the angle changes?

Run all three and compare against BeH2's equivalents: same geometric setup,
open-shell physics instead of closed-shell.

ACTIVE SPACE
-------------
NO2 in STO-3G is 23 electrons in 15 spatial orbitals -> 30 qubits taking
everything, hopelessly large for a statevector scan. As with every other
polyatomic in this project, ``--active-electrons``/``--active-orbitals``
restrict to a CAS(n_elec, n_orb) active space -- with the odd-electron
subtlety from :func:`active_particles` applied throughout.

The defaults differ by mode for the same reason H2O's and N2O's do: bend
and surface default to the cheaper CAS(5,4) (3 alpha, 2 beta -- doublet);
stretch defaults to the larger CAS(7,5) (4 alpha, 3 beta). Both are small
relative to the full valence space that a rigorous treatment of NO2's
three-center, partially-delocalized bonding would want, and both are
smaller than O2's default CAS(8,6) despite NO2 having more electrons --
treat this script's numbers as demonstrating the *method* and the
*qualitative* shape of the curves, not as converged energies. Raise both
flags together for a convergence check, same as everywhere else in this
project; :func:`_report_monotonicity` still checks the stretch curve for
the active-space failure mode documented in the H2O script.

COST
-----
Cost is set mostly by UCCSD's parameter count and, for the default COBYLA
optimizer, its iteration cap -- not by circuit-simulation speed, per the O2
script's measurements at a similar-sized active space. **This has not been
benchmarked for NO2 specifically; the open-shell ROHF/UCCSD path is not
necessarily the same cost as a closed-shell one at the same qubit count,**
so treat the O2 and N2O scripts' numbers as a rough starting point, not a
promise, and let the live plot's per-point timing tell you what to expect
after the first point or two.

Each geometry warm-starts its optimizer from the previous geometry's
solution (disable with ``--cold-start``), and the surface walks its grid in
serpentine order for the same reason BeH2's does.

REFERENCES
-----------
[1] NO2's equilibrium geometry -- bond length ~1.194 A, O-N-O angle
    ~134.3 deg, from gas-phase electron diffraction / rotational
    spectroscopy, as tabulated by NIST's Computational Chemistry Comparison
    and Benchmark Database (CCCBDB): https://cccbdb.nist.gov/
[2] PySCFDriver's spin convention (spin = 2*S) and MethodType.ROHF:
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.drivers.PySCFDriver.html
[3] ActiveSpaceTransformer, including its note on alpha/beta subspaces for
    open-shell references:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html

USAGE
------
  RUN A SCAN -- symmetric stretch (angle fixed at equilibrium)
    python experiment/no2/no2_ground_state_estimation.py --coordinate stretch
    python experiment/no2/no2_ground_state_estimation.py --coordinate stretch --start 0.9 --stop 2.4 --step 0.1
    python experiment/no2/no2_ground_state_estimation.py --coordinate stretch --start 1.0 --stop 1.6 --step 0.25 --no-save

  RUN A SCAN -- bend (bonds fixed at equilibrium)
    python experiment/no2/no2_ground_state_estimation.py --coordinate bend
    python experiment/no2/no2_ground_state_estimation.py --coordinate bend --start 90 --stop 180 --step 5

  RUN A SCAN -- the surface (bond x angle)
    python experiment/no2/no2_ground_state_estimation.py --coordinate surface
    python experiment/no2/no2_ground_state_estimation.py --coordinate surface --step 0.15 --angle-step 10
    python experiment/no2/no2_ground_state_estimation.py --coordinate surface --step 0.5 --angle-step 30 --no-save   # quick look

  RELOAD A FINISHED SCAN -- no VQE re-run
    python experiment/no2/no2_ground_state_estimation.py --load no2_stretch_scan.json
    python experiment/no2/no2_ground_state_estimation.py --load no2_bend_scan.json
    python experiment/no2/no2_ground_state_estimation.py --load no2_surface_scan.json

  A LARGER ACTIVE SPACE -- for a convergence spot-check
    python experiment/no2/no2_ground_state_estimation.py --coordinate stretch --active-electrons 9 --active-orbitals 7
    python experiment/no2/no2_ground_state_estimation.py --coordinate surface --active-electrons 7 --active-orbitals 5 --step 0.5 --angle-step 30

  OTHER KNOBS
    --optimizer slsqp / --cold-start
    --save myname / --no-save
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

# NO2's experimental equilibrium geometry [1].
EQUILIBRIUM_BOND = 1.194     # Angstrom, r(N-O)
EQUILIBRIUM_ANGLE = 134.3    # degrees, O-N-O

# spin = 2*S in PySCF's convention [2]. NO2's doublet ground state (one
# unpaired electron) is not optional the way O2's triplet-vs-singlet choice
# was -- there is no closed-shell state to compare against, since 23
# electrons cannot be paired at all. So there is no --multiplicity flag here.
SPIN = 1


# ===========================================================================
#  Geometry -- symmetric by construction, same convention as BeH2
# ===========================================================================

def no2_positions(bond: float, angle_deg: float) -> Dict[str, tuple]:
    """Atom positions for NO2, in the x-z plane.

    Nitrogen sits at the origin. The two oxygens are placed symmetrically
    about the +z axis, both at `bond` Angstrom from N, with `angle_deg`
    between the two N-O bonds -- the same symmetric construction BeH2 uses,
    since NO2's two N-O bonds are chemically identical (both are the same
    resonance-averaged one-and-a-half bond order).
    """
    half = math.radians(angle_deg) / 2
    dx, dz = bond * math.sin(half), bond * math.cos(half)
    return {"N": (0.0, 0.0), "O1": (dx, dz), "O2": (-dx, dz)}


def geometry_string(bond: float, angle_deg: float) -> str:
    pos = no2_positions(bond, angle_deg)
    return (f"N 0 0 0; "
            f"O {pos['O1'][0]} 0 {pos['O1'][1]}; "
            f"O {pos['O2'][0]} 0 {pos['O2'][1]}")


# ===========================================================================
#  The quantum chemistry -- one geometry at a time
# ===========================================================================

def active_particles(active_electrons: int, spin: int) -> Tuple[int, int]:
    """Split ``active_electrons`` into (alpha, beta) for ``spin`` = 2*S.

    Identical reasoning to the O2 script: ``ActiveSpaceTransformer`` accepts
    either a plain count or an explicit ``(alpha, beta)`` tuple, and for an
    open-shell system the plain count is ambiguous -- 5 electrons could be
    (3, 2) or (4, 1). Always pass it explicitly.
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


def build_problem(bond: float, angle_deg: float, active_electrons: int,
                  active_orbitals: int) -> ElectronicStructureProblem:
    """NO2 at this geometry, restricted to a CAS(active_electrons, active_orbitals)
    active space, computed with ROHF at spin=1 -- see the module docstring
    for why the default closed-shell path used elsewhere in this repo would
    silently give the wrong electronic state here."""
    problem = PySCFDriver(atom=geometry_string(bond, angle_deg), basis=BASIS,
                          spin=SPIN, method=MethodType.ROHF).run()
    return ActiveSpaceTransformer(
        active_particles(active_electrons, SPIN), active_orbitals
    ).transform(problem)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). NO2's inactive
    part is large -- three cores' worth at minimum -- so miss this and the
    energy is wrong by well over a hundred Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_geometry(bond: float, angle_deg: float, active_electrons: int,
                     active_orbitals: int, optimizer: str = "cobyla",
                     max_iterations: int = 1000,
                     theta0: np.ndarray | None = None) -> Dict[str, Any]:
    """Hartree-Fock, VQE and exact ground-state energy at one NO2 geometry.

    ``theta0`` warm-starts the optimizer from a previous geometry's solution.
    "exact" is the lowest eigenvalue of the mapped, active-space-reduced
    qubit Hamiltonian -- exact within the chosen model, not a full-basis FCI
    energy. "hf" is the (RO)Hartree-Fock circuit's expectation value on that
    same Hamiltonian, so all three numbers are directly comparable with each
    other, though none is a benchmark-quality absolute energy.
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
    method = {"cobyla": "COBYLA", "slsqp": "SLSQP"}[optimizer]
    result = minimize(cost, start, method=method, options={"maxiter": max_iterations})

    return {
        "bond": float(bond),
        "angle": float(angle_deg),
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
        "n_qubits": hamiltonian.num_qubits,
        "n_params": ansatz.num_parameters,
        "n_evaluations": int(getattr(result, "nfev", 0)),
        "vqe_params": np.asarray(result.x),
    }


# ===========================================================================
#  Shared drawing constants
# ===========================================================================

ATOM_RADII = {"N": 0.36, "O": 0.34}
ATOM_COLORS = {"N": "#3050f8", "O": "#ff4d4d"}   # CPK-ish: nitrogen blue, oxygen red


def _atom_element(name: str) -> str:
    return name.rstrip("0123456789")


def _draw_molecule(ax, patches, bond_lines, row) -> None:
    pos = no2_positions(row["bond"], row["angle"])
    for name, (x, z) in pos.items():
        patches[name].center = (x, z)
    bond_lines[0].set_data([pos["N"][0], pos["O1"][0]], [pos["N"][1], pos["O1"][1]])
    bond_lines[1].set_data([pos["N"][0], pos["O2"][0]], [pos["N"][1], pos["O2"][1]])


def _build_molecule_panel(ax, span: float):
    ax.set_xlim(-span, span)
    ax.set_ylim(-span * 0.5, span)
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
        for name in ("N", "O1", "O2")
    }
    for patch in patches.values():
        ax.add_patch(patch)
    label = ax.text(0, -span * 0.4, "", ha="center", fontsize=10)
    return patches, bond_lines, label


def _mol_label_text(row: Dict[str, Any]) -> str:
    return (f"r(N-O) = {row['bond']:.2f} A   angle = {row['angle']:.1f}$\\degree$\n"
            f"E = {row['vqe']:.4f} Ha")


# ===========================================================================
#  Viewer 1 -- the 1D modes: live curve left, live molecule right
# ===========================================================================

class NO2ScanViewer:
    _AXIS_LABELS = {
        "stretch": "N-O bond length (Angstrom)",
        "bend": "O-N-O angle (degrees)",
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
        self.ax_energy.set_title(f"NO$_2$ {coordinate} scan   ({fixed_label})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        span = max_extent + 0.5
        self.atom_patches, self.bonds, self.mol_label = _build_molecule_panel(self.ax_mol, span)
        self._redraw()

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
#  Viewer 2 -- the surface mode: bond x angle, same layout as BeH2's
# ===========================================================================

class NO2SurfaceViewer:
    def __init__(self, bond_values: np.ndarray, angle_values: np.ndarray) -> None:
        self.bond_values = np.asarray(bond_values, dtype=float)
        self.angle_values = np.asarray(angle_values, dtype=float)
        # Z is indexed [angle, bond] so imshow's rows run up the y axis.
        self.Z = np.full((len(self.angle_values), len(self.bond_values)), np.nan)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        plt.ion()
        self.fig, (self.ax_map, self.ax_mol) = plt.subplots(
            1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        d_bond = self._half_step(self.bond_values)
        d_angle = self._half_step(self.angle_values)
        self.extent = (self.bond_values[0] - d_bond, self.bond_values[-1] + d_bond,
                       self.angle_values[0] - d_angle, self.angle_values[-1] + d_angle)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)
        self.im = self.ax_map.imshow(self.Z, origin="lower", extent=self.extent,
                                     aspect="auto", cmap=cmap, interpolation="nearest")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_map, label="Energy (Hartree)")
        (self.map_marker,) = self.ax_map.plot([], [], "D", color="white", markersize=9,
                                              fillstyle="none", markeredgewidth=2)
        self.ax_map.set_xlabel("N-O bond length (Angstrom)")
        self.ax_map.set_ylabel("O-N-O angle (degrees)")
        self.ax_map.set_title("NO$_2$ energy surface")

        span = float(self.bond_values[-1]) + 0.5
        self.atom_patches, self.bonds, self.mol_label = _build_molecule_panel(self.ax_mol, span)
        self._redraw()

    @staticmethod
    def _half_step(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.5
        return float(values[1] - values[0]) / 2

    def _cell_of(self, row: Dict[str, Any]) -> Tuple[int, int]:
        j = int(np.argmin(np.abs(self.bond_values - row["bond"])))
        i = int(np.argmin(np.abs(self.angle_values - row["angle"])))
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
        self.map_marker.set_data([row["bond"]], [row["angle"]])
        _draw_molecule(self.ax_mol, self.atom_patches, self.bonds, row)
        self.mol_label.set_text(_mol_label_text(row))
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_surface(self, png_path: Path | None = None) -> None:
        grid_bond, grid_angle = np.meshgrid(self.bond_values, self.angle_values)
        masked = np.ma.masked_invalid(self.Z)

        fig = plt.figure(figsize=(13, 5.5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot_surface(grid_bond, grid_angle, masked, cmap="viridis",
                          linewidth=0, antialiased=True, alpha=0.95)
        ax3d.set_xlabel("bond (A)")
        ax3d.set_ylabel("angle (deg)")
        ax3d.set_zlabel("Energy (Ha)")
        ax3d.set_title("NO$_2$ ground-state surface (VQE)")
        ax3d.view_init(elev=28, azim=-135)

        ax2d = fig.add_subplot(1, 2, 2)
        levels = 25
        contour = ax2d.contourf(grid_bond, grid_angle, masked, levels=levels, cmap="viridis")
        ax2d.contour(grid_bond, grid_angle, masked, levels=levels,
                     colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
        if self.rows:
            best = min(self.rows, key=lambda r: r["vqe"])
            ax2d.plot([best["bond"]], [best["angle"]], "r*", markersize=16,
                      label=f"min {best['vqe']:.4f} Ha")
            ax2d.legend(loc="lower left")
        ax2d.set_xlabel("N-O bond length (Angstrom)")
        ax2d.set_ylabel("O-N-O angle (degrees)")
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
    return np.arange(start, stop + step / 2, step)


def serpentine(bond_values: np.ndarray, angle_values: np.ndarray) -> List[Tuple[float, float]]:
    geometries: List[Tuple[float, float]] = []
    for i, angle in enumerate(angle_values):
        row = bond_values if i % 2 == 0 else bond_values[::-1]
        geometries.extend((float(b), float(angle)) for b in row)
    return geometries


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NO2 ground-state energy by VQE (local simulator), scanned along the "
                    "symmetric stretch, the bend, or the bond/angle surface. NO2 is an "
                    "open-shell doublet radical -- see the module docstring."
    )
    parser.add_argument("--coordinate", choices=["stretch", "bend", "surface"],
                        default="stretch",
                        help="'stretch' (default): both N-O bonds, angle fixed. "
                             "'bend': the O-N-O angle, bonds fixed. "
                             "'surface': bond and angle together on a grid.")
    parser.add_argument("--start", type=float, default=None,
                        help="scan start -- Angstrom for stretch/surface's bond axis "
                             "(default 0.9), degrees for bend (default 90)")
    parser.add_argument("--stop", type=float, default=None,
                        help="scan stop, inclusive -- default 2.4 A for stretch/surface, "
                             "180 deg for bend")
    parser.add_argument("--step", type=float, default=None,
                        help="scan step -- default 0.1 A for stretch, 5 deg for bend, "
                             "0.15 A for the surface's bond axis")
    parser.add_argument("--angle-start", type=float, default=90.0,
                        help="[surface mode] first angle in degrees (default 90)")
    parser.add_argument("--angle-stop", type=float, default=180.0,
                        help="[surface mode] last angle in degrees (default 180)")
    parser.add_argument("--angle-step", type=float, default=10.0,
                        help="[surface mode] angle step in degrees (default 10)")
    parser.add_argument("--bond", type=float, default=EQUILIBRIUM_BOND,
                        help=f"[bend mode] fixed N-O bond length in Angstrom "
                             f"(default {EQUILIBRIUM_BOND})")
    parser.add_argument("--angle", type=float, default=EQUILIBRIUM_ANGLE,
                        help=f"[stretch mode] fixed O-N-O angle in degrees "
                             f"(default {EQUILIBRIUM_ANGLE})")
    parser.add_argument("--active-electrons", type=int, default=None,
                        help="electrons in the active space, must be odd (default: 7 for "
                             "stretch, 5 for bend/surface -- see the module docstring)")
    parser.add_argument("--active-orbitals", type=int, default=None,
                        help="spatial orbitals in the active space (default: 5 for "
                             "stretch -> 8 qubits, 4 for bend/surface -> 6 qubits)")
    parser.add_argument("--optimizer", choices=["cobyla", "slsqp"], default="cobyla",
                        help="'cobyla' (default) is gradient-free. 'slsqp' estimates its "
                             "gradient by finite differences, costing one extra evaluation "
                             "per parameter per step.")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="optimizer iteration cap (default 1000)")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each geometry's optimization from the Hartree-Fock "
                             "state instead of warm-starting from the previous geometry.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan "
                             "straight in review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: "
                             "no2_<coordinate>_scan.json/.csv next to this script)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save the results of this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="exit when the scan finishes instead of staying open in "
                             "review mode.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    if args.load:
        _replay(Path(args.load), here, args.no_show)
        return

    if args.active_electrons is None:
        args.active_electrons = 5 if args.coordinate in ("bend", "surface") else 7
    if args.active_orbitals is None:
        args.active_orbitals = 4 if args.coordinate in ("bend", "surface") else 5

    if (args.active_electrons - SPIN) % 2 != 0:
        parser.error(f"--active-electrons must be odd (NO2 is a doublet: spin={SPIN}), "
                     f"got {args.active_electrons}")

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
    """Read a scan back from its ``.csv``, inferring mode from which columns vary."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")

    numeric = ("bond", "angle", "hf", "vqe", "exact")
    rows = []
    for r in raw:
        row = {k: float(v) for k, v in r.items() if k in numeric and v != ""}
        for k in ("n_qubits", "n_params", "n_evaluations"):
            if r.get(k):
                row[k] = int(float(r[k]))
        rows.append(row)

    missing = {"bond", "angle", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(
            f"{display_path(path)} is missing the column(s) {sorted(missing)}, so it is "
            f"not an NO2 scan CSV. Expected columns: bond, angle, hf, vqe, exact."
        )

    bonds = sorted({r["bond"] for r in rows})
    angles = sorted({r["angle"] for r in rows})

    if len(bonds) > 1 and len(angles) > 1:
        metadata = {"scan_type": "surface", "coordinate": "surface",
                   "bond_values": bonds, "angle_values": angles}
    elif len(bonds) > 1:
        metadata = {"scan_type": "curve", "coordinate": "stretch",
                   "fixed_label": f"angle fixed at {angles[0]}$\\degree$",
                   "max_extent": bonds[-1]}
    else:
        metadata = {"scan_type": "curve", "coordinate": "bend",
                   "fixed_label": f"bond fixed at {bonds[0]} A",
                   "max_extent": bonds[0]}
    return metadata, rows


def _load_any(load_path: Path, here: Path) -> tuple:
    if load_path.suffix.lower() == ".csv":
        for candidate in (load_path, here / load_path.name):
            if candidate.exists():
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
        viewer = NO2SurfaceViewer(np.asarray(metadata["bond_values"], dtype=float),
                                  np.asarray(metadata["angle_values"], dtype=float))
        for row in rows:
            viewer.add_point(row)
        viewer.show_surface()
        _finish(viewer, no_show)
    else:
        viewer = NO2ScanViewer(metadata["coordinate"], metadata["fixed_label"],
                               float(metadata["max_extent"]))
        for row in rows:
            viewer.add_point(row)
        _finish(viewer, no_show)


def _run_curve(args, parser, here: Path) -> None:
    if args.coordinate == "bend":
        start = 90.0 if args.start is None else args.start
        stop = 180.0 if args.stop is None else args.stop
        step = 5.0 if args.step is None else args.step
        unit = "deg"
    else:
        start = 0.9 if args.start is None else args.start
        stop = 2.4 if args.stop is None else args.stop
        step = 0.1 if args.step is None else args.step
        unit = "A"

    if step <= 0:
        parser.error("--step must be positive")
    if stop < start:
        parser.error("--stop must be >= --start")

    values = inclusive_range(start, stop, step)

    if args.coordinate == "stretch":
        geometries = [(float(v), args.angle) for v in values]
        fixed_label = f"angle fixed at {args.angle}$\\degree$"
        max_extent = float(values[-1])
    else:
        geometries = [(args.bond, float(v)) for v in values]
        fixed_label = f"bond fixed at {args.bond} A"
        max_extent = args.bond

    print(f"NO2 {args.coordinate} scan: {len(values)} points from {values[0]:.2f} to "
          f"{values[-1]:.2f} {unit} (step {step} {unit})")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals}), "
          f"spin={SPIN} (doublet, ROHF)")
    print("cost has not been benchmarked for NO2 specifically -- see the module docstring's "
          "COST section. The plot updates as each point lands.")

    viewer = NO2ScanViewer(args.coordinate, fixed_label, max_extent)
    rows = _scan(geometries, args, viewer)

    best = min(rows, key=lambda r: r["vqe"])
    best_x = best["bond"] if args.coordinate == "stretch" else best["angle"]
    reference = (f"literature bond is ~{EQUILIBRIUM_BOND} A" if args.coordinate == "stretch"
                else f"literature angle is ~{EQUILIBRIUM_ANGLE} deg")
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at {best_x:.2f} {unit}  ({reference})")
    _report_errors(rows)
    if args.coordinate == "stretch":
        _report_monotonicity(rows)

    if not args.no_save:
        target = resolve_target(args.save, here, f"no2_{args.coordinate}_scan")
        metadata = {
            "molecule": "NO2", "basis": BASIS,
            "scan_type": "curve", "coordinate": args.coordinate,
            "spin_2s": SPIN, "scf_method": "ROHF",
            "start": float(values[0]), "stop": float(values[-1]), "step": float(step),
            "unit": unit,
            "fixed_bond": float(args.bond), "fixed_angle": float(args.angle),
            "fixed_label": fixed_label,
            "max_extent": float(max_extent),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
            "optimizer": args.optimizer, "max_iterations": int(args.max_iterations),
        }
        _save(target, metadata, rows)

    _finish(viewer, args.no_show)


def _run_surface(args, parser, here: Path) -> None:
    bond_start = 0.9 if args.start is None else args.start
    bond_stop = 2.4 if args.stop is None else args.stop
    bond_step = 0.15 if args.step is None else args.step

    if bond_step <= 0 or args.angle_step <= 0:
        parser.error("--step and --angle-step must be positive")
    if bond_stop < bond_start:
        parser.error("--stop must be >= --start")
    if args.angle_stop < args.angle_start:
        parser.error("--angle-stop must be >= --angle-start")

    bond_values = inclusive_range(bond_start, bond_stop, bond_step)
    angle_values = inclusive_range(args.angle_start, args.angle_stop, args.angle_step)
    geometries = serpentine(bond_values, angle_values)

    print(f"NO2 surface scan: {len(bond_values)} bond x {len(angle_values)} angle "
          f"= {len(geometries)} points")
    print(f"  bond {bond_values[0]:.2f} to {bond_values[-1]:.2f} A (step {bond_step} A)")
    print(f"  angle {angle_values[0]:.1f} to {angle_values[-1]:.1f} deg "
          f"(step {args.angle_step} deg)")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals}), "
          f"spin={SPIN} (doublet, ROHF)")
    print("cost has not been benchmarked for NO2 specifically -- a surface pays for "
          "whatever the active space costs per point, N times over. The map fills in "
          "as each point lands.")

    viewer = NO2SurfaceViewer(bond_values, angle_values)
    rows = _scan(geometries, args, viewer)

    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at bond={best['bond']:.2f} A, "
          f"angle={best['angle']:.1f} deg")
    print(f"  (literature: bond ~{EQUILIBRIUM_BOND} A, angle ~{EQUILIBRIUM_ANGLE} deg)")
    _report_errors(rows)

    png_path = None
    if not args.no_save:
        target = resolve_target(args.save, here, "no2_surface_scan")
        metadata = {
            "molecule": "NO2", "basis": BASIS,
            "scan_type": "surface", "coordinate": "surface",
            "spin_2s": SPIN, "scf_method": "ROHF",
            "bond_values": [float(v) for v in bond_values],
            "angle_values": [float(v) for v in angle_values],
            "bond_step": float(bond_step), "angle_step": float(args.angle_step),
            "active_electrons": int(args.active_electrons),
            "active_orbitals": int(args.active_orbitals),
            "optimizer": args.optimizer, "max_iterations": int(args.max_iterations),
        }
        _save(target, metadata, rows)
        png_path = target.with_name(target.stem + "_3d.png")

    viewer.show_surface(png_path)
    _finish(viewer, args.no_show)


def _scan(geometries, args, viewer) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    previous_params = None
    total = len(geometries)
    for i, (bond, angle) in enumerate(geometries):
        t0 = time.perf_counter()
        row = run_one_geometry(bond, angle, args.active_electrons, args.active_orbitals,
                               optimizer=args.optimizer, max_iterations=args.max_iterations,
                               theta0=previous_params)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:4d}/{total}] bond={bond:5.2f} angle={angle:6.1f}   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")
    return rows


def _report_errors(rows: List[Dict[str, Any]]) -> None:
    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    print(f"largest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses -- expect this to be large, per the docstring)")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")
    if max_vqe * 1000 > 1.0:
        print("  note: |VQE - exact| above 1 mHa suggests the optimizer did not fully "
              "converge somewhere -- try --cold-start, --optimizer slsqp, or a finer --step")


def _report_monotonicity(rows: List[Dict[str, Any]]) -> None:
    """Same active-space sanity check as the H2O/N2O/N2O scripts: past its
    minimum, a bond-breaking curve must rise, never dip."""
    ordered = sorted(rows, key=lambda r: r["bond"])
    best_at = min(range(len(ordered)), key=lambda i: ordered[i]["exact"])
    dips = [
        (ordered[i]["bond"], ordered[i + 1]["bond"])
        for i in range(best_at, len(ordered) - 1)
        if ordered[i + 1]["exact"] < ordered[i]["exact"]
    ]
    if dips:
        print(f"\n  WARNING: the exact curve dips downward past its minimum, between "
              + ", ".join(f"{a:.2f}-{b:.2f} A" for a, b in dips))
        print("  A ground-state curve must rise monotonically toward dissociation. This is")
        print("  an active space too small for this bond, not a VQE failure. Try a larger")
        print("  --active-orbitals / --active-electrons (keeping it odd).")
    else:
        print("  curve rises monotonically past its minimum -- active space looks adequate")


def _save(target: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    json_path, csv_path = save_scan(target, metadata, rows)
    print(f"saved {display_path(json_path)}")
    print(f"saved {display_path(csv_path)}")
    print(f"reopen it later without recomputing:  "
          f"python {display_path(Path(__file__))} --load {json_path.name}")


if __name__ == "__main__":
    main()