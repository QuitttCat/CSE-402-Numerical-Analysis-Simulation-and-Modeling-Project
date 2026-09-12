"""
h2o2_ground_state_estimation.py -- hydrogen peroxide's ground-state energy by
VQE, scanned along its symmetric internal coordinates, plotted live next to a
3D picture of the molecule stretching, bending or twisting.

Simulator only. No IBM account, no queue, no quota.

FOUR SYMMETRIC COORDINATES, NOT TWO
-------------------------------------
H2O2 is 4 atoms and non-linear, so it has 3N-6 = 6 internal degrees of
freedom: two O-H bonds, the O-O bond, two H-O-O angles, and the H-O-O-H
dihedral (the torsion). Forcing the molecule's C2 symmetry -- the two O-H
bonds equal, the two H-O-O angles equal, exactly the move BeH2 and H2O
already make on their own bonds -- collapses that to **four** independent
coordinates:

    oo         the O-O bond length
    oh         the shared O-H distance      (both O-H bonds move together)
    angle      the H-O-O angle              (both, symmetric)
    dihedral   the H-O-O-H torsion          -- new. Nothing else in this
                                               project has a coordinate that
                                               takes the molecule out of a
                                               single plane.

:func:`h2o2_positions` takes exactly these four numbers and returns full 3D
coordinates. There is no way to express an asymmetric geometry through this
interface -- the same design choice BeH2 and H2O already make.

This project's plotting only goes up to two dimensions (a curve or a
surface), so no single mode scans all four at once; two are always held
fixed at their equilibrium value. See each mode below.

WHY THIS MOLECULE, AND HOW IT DIFFERS FROM BeH2/H2O
-------------------------------------------------------
BeH2 and water stay perfectly planar through every scan they run -- their
"bend" is a 2D motion that happens to be drawn in 3D-looking axes. H2O2's
equilibrium structure is genuinely **non-planar**: the two O-H bonds sit at a
dihedral of ~111.5 deg [1], not 0 (cis, eclipsed) or 180 (trans, planar).
That torsion also makes the molecule chiral -- mirror-image "skewed"
structures exist at dihedral ~111.5 deg and ~248.5 deg (= 360 - 111.5),
connected by two *different* barriers: a higher one through the eclipsed cis
form (dihedral = 0 deg) and a lower one through the planar trans form
(dihedral = 180 deg) [2]. Scanning dihedral 0 -> 180 deg captures one minimum
and both barriers, which is the whole physically interesting content -- the
other half of the circle (180 -> 360) is its mirror image.

FOUR MODES
-----------
  --coordinate stretch   The O-O bond varied, oh/angle/dihedral held at their
                         equilibrium values. A dissociation curve, the same
                         shape as every other molecule here: repulsive wall,
                         a minimum near 1.46 A [1], then a monotonic climb.

  --coordinate bend      The H-O-O angle varied, everything else fixed. A
                         soft, single-well motion -- expect HF to track the
                         correlated curves closely, the same contrast BeH2's
                         and water's bends already show.

  --coordinate twist     THE HEADLINE MOTION. The H-O-O-H dihedral varied
                         0 -> 180 deg, oo/oh/angle held at equilibrium. NOT a
                         single well: a cis barrier at 0 deg, the true
                         minimum near 111.5 deg, a trans barrier at 180 deg.
                         Expect a genuinely lopsided double-sided well, not a
                         parabola.

                         A caution worth taking seriously: the literature
                         barriers are small. The cis barrier is ~2560 cm^-1
                         (= 7.3 kcal/mol = ~11.7 mHa); the **trans** barrier
                         is only ~386 cm^-1 (= 1.1 kcal/mol = ~1.8 mHa) [2].
                         That trans number is the same order of magnitude as
                         the VQE-vs-exact convergence noise measured
                         elsewhere in this project (order 0.1-1 mHa). Trust
                         the cis barrier; treat the trans barrier as
                         provisional until you have checked that
                         |VQE - exact| at those specific points is
                         comfortably smaller than the barrier itself --
                         :func:`_report_barriers` prints exactly that
                         comparison, it is not left for you to eyeball.

  --coordinate surface   dihedral x angle, both varied on a grid, oo/oh fixed
                         at equilibrium. This is the pair that actually
                         produces the coupled barrier physics -- the O-O
                         bond itself is comparatively decoupled, the same
                         way BeH2's bond axis was the stiff, less interesting
                         one next to its soft bend. Expect a genuinely
                         asymmetric-looking result: two ridges (the cis and
                         trans barriers, unequal height) framing a trough
                         that is *not* centred at dihedral = 90 deg but
                         sitting off to one side, at the ~111.5 deg skew
                         angle.

THE MOLECULE IS CLOSED-SHELL -- NO ROHF NEEDED
-------------------------------------------------
Unlike O2 and F2 in the sibling folders, H2O2 has 18 electrons filling
molecular orbitals with nothing left unpaired: it is a closed-shell singlet,
X ~1~A. Plain ``PySCFDriver(atom=..., basis=...)`` with the default
spin=0/RHF is already correct here, the same as for BeH2, H2O and H2.

ACTIVE SPACE -- A STARTING HYPOTHESIS, NOT A SETTLED CHOICE
-----------------------------------------------------------------
H2O2 in STO-3G is 18 electrons in 12 spatial orbitals -> 24 qubits taking
everything, hopeless for a scan, let alone a surface. The default here is
CAS(8,6) -- the same size as O2's active space in the sibling folder, and
chosen for the same reason: asking ``ActiveSpaceTransformer`` for orbitals
nearest the Fermi level, with 8 of the 18 electrons active, freezes both
oxygen 1s cores *and* both O-H bonding orbitals (which barely change through
any of these scans), leaving active more or less the O-O sigma/sigma* pair
and the oxygen lone-pair combinations -- whose mutual repulsion is the
textbook explanation for why H2O2 twists away from planarity at all [2][3].

That is a physically motivated guess, not a verified one. Unlike O2's
CAS(8,6), there is no single textbook-standard active space for H2O2's
torsion to point to here. **Verify it before trusting the barrier heights**:
rerun the twist scan at a larger --active-electrons/--active-orbitals and
check the cis/trans barriers are stable rather than drifting. If CAS(8,6)
turns out to be too small, expect the same failure signature the H2O and
BeH2 scripts document -- a *smooth but wrong* curve, not an error message.

ORBITAL CONTINUITY ACROSS THE SCAN -- A REAL BUG THAT COST US A FULL RE-RUN
--------------------------------------------------------------------------------
Solving each geometry from scratch, independently, has a trap this project's
other molecules never hit: two valence orbitals of H2O2 pass through a near
degeneracy around dihedral ~125 deg (their energies come within ~0.0002 Ha of
each other there). PySCF always returns orbitals sorted by energy, with no
memory of the previous geometry, so right at that near-degeneracy the
*character* of "orbital index 4" and "orbital index 5" can rotate into one
another within a couple of degrees -- Hartree-Fock's total energy stays
perfectly smooth through this (it only depends on the combined occupied
space, which does not change), but which *specific* orbital ends up frozen
versus active flips, and CAS(8,6)'s energy jumps by several mHa as a result.
We measured this directly on a real twist scan: a clean, monotonic curve from
0 to 120 deg, then a sudden ~7.7 mHa drop between 120 and 130 deg that stays
flat the rest of the way to 180 -- backwards from the literature physics,
where trans (180 deg) is a barrier above the skew minimum, not the global
minimum of the scan. We first suspected the auto active-space selector and
tested pinning explicit orbital indices; that did *not* fix it, which is what
proved the cause is an avoided crossing in orbital character, not orbital
*selection*.

The fix: :func:`_track_orbitals` reorders each new geometry's MO coefficients
to maximize overlap with the *previous* geometry's (a cross-molecule AO
overlap via ``pyscf.gto.mole.intor_cross``, then an optimal one-to-one match
via ``scipy.optimize.linear_sum_assignment``, done separately within the
occupied block and the virtual block so occupied orbitals can never be
matched into virtual slots or vice versa). This is standard orbital-tracking
/ maximum-overlap practice for following a single electronic state along a
coordinate scan. It requires reaching past ``PySCFDriver``'s public API into
its ``_mol``/``_calc`` attributes to inject the reordered orbitals before the
integral transform -- there is no public hook for this in qiskit-nature
0.8.0, so :func:`build_problem` depends on that driver's internal structure
rather than only its documented interface.

Practical consequence: geometries must be solved in an order where each one
is close to the previous (which every scan here already does -- plain
increasing order for the three curves, serpentine order for the surface),
and the *first* geometry of any scan has no previous orbitals to track
against, so it just keeps PySCF's own energy-sorted order and becomes the
reference for the next point.

COST -- MEASURED
-------------------
CAS(8,6) here is 10 qubits but **92** UCCSD parameters (closed-shell H2O2 has
more than O2's open-shell 68 at the same (electrons, orbitals) size -- more
occupied-virtual pairs are available when nothing is already singly occupied).
COBYLA spends its entire 1000-evaluation budget every point regardless of
warm-starting, exactly as O2's docstring warns -- so cost per point is
essentially fixed, not dependent on how hard a given geometry is.

Measured on a laptop CPU (twist mode, dihedral = 0/90/180 deg): **537-787 s
per point, ~690 s average** -- roughly 5x O2's 143 s/point, not the 1x this
docstring originally guessed (that estimate was extrapolated from O2's
parameter count, not measured, and undercounted H2O2's actual parameter
count badly). At ~690 s/point:

    stretch   21 points   ~4.0 hours
    bend      13 points   ~2.5 hours
    twist     19 points   ~3.6 hours
    surface   50 points   ~9.6 hours

These are real per-point measurements, not a guess, but still only 3 sample
points -- treat the hour totals as order-of-magnitude, and expect them to
shift once a full run's own timing prints replace them.

REFERENCES
-----------
[1] Gas-phase equilibrium structure -- r(O-O) = 1.458 A, r(O-H) = 0.988 A,
    angle(H-O-O) = 101.9 deg -- NIST CCCBDB experimental geometry for H2O2:
    https://cccbdb.nist.gov/exp2x.asp?casno=7722841
[2] Torsional barriers from far-infrared internal-rotation spectroscopy --
    trans barrier 386 cm^-1, cis barrier 2560 cm^-1, equilibrium (skew)
    dihedral 111.5 deg -- Hunt, Leacock, Peters & Hecht, "Internal Rotation
    in Hydrogen Peroxide: The Far-Infrared Spectrum and the Determination of
    the Hindering Potential", J. Chem. Phys. 42, 1931 (1965):
    https://pubs.aip.org/aip/jcp/article-abstract/42/6/1931/209866
[3] A computational account of *why* the two barriers differ (lone-pair
    repulsion versus hyperconjugation) -- "Origins of Rotational Barriers in
    Hydrogen Peroxide and Hydrazine", J. Chem. Theory Comput. 1, 394 (2005).
    Its computed barriers (8.34 / 5.57 kcal/mol) do not exactly match the
    experimental ones in [2] -- worth noting that the literature itself does
    not fully agree on this number:
    https://dasher.wustl.edu/chem430/labs/lab-08/jctc-1-394-05.pdf
[4] ActiveSpaceTransformer, and the "orbitals nearest the Fermi level"
    default this script relies on:
    https://qiskit-community.github.io/qiskit-nature/tutorials/05_problem_transformers.html

USAGE
------
  RUN A SCAN -- symmetric O-O stretch
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate stretch
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate stretch --start 1.0 --stop 1.6 --step 0.2 --no-save   # smoke test

  RUN A SCAN -- symmetric H-O-O bend
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate bend
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate bend --start 90 --stop 120 --step 15 --no-save        # smoke test

  RUN A SCAN -- the torsion. This is the point of the whole molecule.
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate twist
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate twist --start 0 --stop 180 --step 60 --no-save        # smoke test

  RUN A SCAN -- the surface (dihedral x angle)
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate surface
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate surface --step 90 --angle-step 30 --no-save          # quick smoke test

  RELOAD A FINISHED SCAN -- no VQE re-run. Either format works.
    python experiment/h2o2/h2o2_ground_state_estimation.py --load h2o2_twist_scan.json
    python experiment/h2o2/h2o2_ground_state_estimation.py --load h2o2_surface_scan.json

  A LARGER ACTIVE SPACE -- to check CAS(8,6) is actually big enough
    python experiment/h2o2/h2o2_ground_state_estimation.py --coordinate twist --active-electrons 10 --active-orbitals 8

  OTHER KNOBS
    --optimizer slsqp        SLSQP instead of the default COBYLA
    --max-iterations N       optimizer iteration cap (default 1000)
    --cold-start             no warm start from the previous geometry
    --oo / --oh / --angle / --dihedral   override a coordinate's fixed value
                             in whichever mode does not scan it
    --save myname / --no-save / --no-show
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
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the 3D projection
from pyscf import gto
from scipy.optimize import linear_sum_assignment, minimize

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

# H2O2's gas-phase equilibrium geometry [1][2]. Used as the fixed value for
# whichever coordinate a given mode isn't scanning.
EQUILIBRIUM_OO = 1.458        # Angstrom, O-O bond
EQUILIBRIUM_OH = 0.988        # Angstrom, O-H bond
EQUILIBRIUM_ANGLE = 101.9     # degrees, H-O-O
EQUILIBRIUM_DIHEDRAL = 111.5  # degrees, H-O-O-H (the chiral skew minimum)

HARTREE_TO_KCAL = 627.5094740631

# Literature torsional barriers [2], for comparison against whatever the
# VQE scan actually finds -- see _report_barriers().
LIT_CIS_BARRIER_KCAL = 7.32     # 2560 cm^-1
LIT_TRANS_BARRIER_KCAL = 1.10   # 386 cm^-1


# ===========================================================================
#  Geometry -- symmetric by construction, and genuinely 3D
# ===========================================================================

def h2o2_positions(oo: float, oh: float, angle_deg: float,
                    dihedral_deg: float) -> Dict[str, np.ndarray]:
    """Atom positions for H2O2 in 3D, placed analytically from the four
    symmetric coordinates.

    O1 sits at the origin, O2 at ``(oo, 0, 0)``. H1 hangs off O1 in the x-y
    plane at ``angle_deg`` from the O-O axis -- that plane is the dihedral
    reference (dihedral = 0). H2 hangs off O2 at the same angle from the O-O
    axis, but rotated by ``dihedral_deg`` about the O-O axis itself, so its
    offset picks up a z-component as the dihedral moves away from 0 or 180.

    Both O-H distances and both H-O-O angles come out exactly equal to `oh`
    and `angle_deg` for every value of `dihedral_deg` -- the offset vectors
    are unit vectors by construction, so rotating around the O-O axis changes
    the torsion without perturbing either bond length or either angle. That
    decoupling is what makes it possible to scan the dihedral on its own.

    dihedral_deg = 0    -> both O-H bonds in the same half-plane (cis,
                            eclipsed)
    dihedral_deg = 180  -> the whole molecule is planar (trans)
    dihedral_deg = 111.5 -> the real, non-planar equilibrium skew structure
    """
    theta = math.radians(angle_deg)
    phi = math.radians(dihedral_deg)
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([oo, 0.0, 0.0])
    h1 = o1 + oh * np.array([math.cos(theta), math.sin(theta), 0.0])
    h2 = o2 + oh * np.array([-math.cos(theta),
                              math.sin(theta) * math.cos(phi),
                              math.sin(theta) * math.sin(phi)])
    return {"O1": o1, "O2": o2, "H1": h1, "H2": h2}


def geometry_string(oo: float, oh: float, angle_deg: float, dihedral_deg: float) -> str:
    pos = h2o2_positions(oo, oh, angle_deg, dihedral_deg)
    atoms = (("O", pos["O1"]), ("O", pos["O2"]), ("H", pos["H1"]), ("H", pos["H2"]))
    return "; ".join(f"{el} {p[0]} {p[1]} {p[2]}" for el, p in atoms)


# ===========================================================================
#  The quantum chemistry -- one geometry at a time
# ===========================================================================

OrbitalState = Tuple[Any, np.ndarray]  # (pyscf Mole, tracked MO coefficients) from one geometry


def _track_orbitals(previous: OrbitalState | None, mol_new, mo_coeff_new: np.ndarray,
                     mo_energy_new: np.ndarray, mo_occ_new: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder ``mo_coeff_new``'s columns to maximize overlap with the
    *previous* geometry's tracked orbitals, so a given column index always
    refers to the same physical orbital across the scan. See the module
    docstring's "ORBITAL CONTINUITY" section for why this is necessary --
    without it, an avoided crossing between two near-degenerate valence
    orbitals around dihedral ~125 deg silently corrupts the twist curve.

    ``previous`` is ``None`` for the first geometry of a scan: nothing to
    track against yet, so PySCF's own energy-sorted order is kept as-is and
    becomes the reference for the next point.

    Matching is done separately within the occupied block and the virtual
    block (split by ``mo_occ_new``), so an occupied orbital can never be
    matched into a virtual slot or vice versa regardless of overlap noise.
    """
    if previous is None:
        return mo_coeff_new, mo_energy_new, mo_occ_new

    mol_prev, mo_coeff_prev = previous
    s_cross = gto.mole.intor_cross("int1e_ovlp", mol_prev, mol_new)
    overlap = mo_coeff_prev.T @ s_cross @ mo_coeff_new  # [previous_idx, new_idx]

    n = mo_coeff_new.shape[1]
    order = np.empty(n, dtype=int)
    occ_mask = mo_occ_new > 0
    for mask in (occ_mask, ~occ_mask):
        idx = np.where(mask)[0]
        sub = overlap[np.ix_(idx, idx)]
        prev_rows, new_cols = linear_sum_assignment(-np.abs(sub))
        order[idx[prev_rows]] = idx[new_cols]

    signs = np.sign([overlap[i, order[i]] for i in range(n)])
    signs[signs == 0] = 1.0
    mo_coeff_tracked = mo_coeff_new[:, order] * signs
    return mo_coeff_tracked, mo_energy_new[order], mo_occ_new[order]


def build_problem(oo: float, oh: float, angle_deg: float, dihedral_deg: float,
                   active_electrons: int, active_orbitals: int,
                   orbitals0: OrbitalState | None = None
                   ) -> Tuple[ElectronicStructureProblem, OrbitalState]:
    """H2O2 at this geometry, restricted to a CAS(active_electrons, active_orbitals)
    active space. Closed-shell by default (spin=0, RHF) -- see the module
    docstring for why that is correct here, unlike O2/F2.

    Reaches past ``PySCFDriver``'s public interface into its ``_mol``/
    ``_calc`` attributes to inject orbital-tracked MO coefficients before the
    integral transform -- see :func:`_track_orbitals` and the module
    docstring. There is no public hook for this in qiskit-nature 0.8.0.
    """
    driver = PySCFDriver(atom=geometry_string(oo, oh, angle_deg, dihedral_deg), basis=BASIS)
    driver.run_pyscf()
    mo_coeff, mo_energy, mo_occ = _track_orbitals(
        orbitals0, driver._mol, driver._calc.mo_coeff, driver._calc.mo_energy, driver._calc.mo_occ
    )
    driver._calc.mo_coeff = mo_coeff
    driver._calc.mo_energy = mo_energy
    driver._calc.mo_occ = mo_occ

    problem = driver.to_problem()
    reduced = ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)
    return reduced, (driver._mol, mo_coeff)


def total_energy(problem: ElectronicStructureProblem, electronic_energy: float) -> float:
    """Electronic eigenvalue -> total molecular energy, adding every constant
    qiskit-nature tracks (nuclear repulsion plus the inactive-orbital energy
    the active-space transformer folded into a constant). Miss these and the
    energy is wrong by tens to hundreds of Hartree."""
    return electronic_energy + sum(problem.hamiltonian.constants.values())


def run_one_geometry(oo: float, oh: float, angle_deg: float, dihedral_deg: float,
                      active_electrons: int, active_orbitals: int,
                      optimizer: str = "cobyla", max_iterations: int = 1000,
                      theta0: np.ndarray | None = None,
                      orbitals0: OrbitalState | None = None
                      ) -> Tuple[Dict[str, Any], OrbitalState]:
    """Hartree-Fock, VQE and exact ground-state energy at one H2O2 geometry.

    ``theta0`` warm-starts the VQE optimizer from a previous geometry's
    solution, since neighbouring points on a scan have similar wavefunctions
    -- the same warm start every other script in this project uses.

    ``orbitals0`` is the unrelated orbital-continuity state from the
    previous geometry (see :func:`_track_orbitals`) -- a correctness fix,
    not an optimization heuristic, so it is threaded through regardless of
    ``--cold-start``. Returns ``(row, orbitals_next)``; ``orbitals_next``
    is deliberately not part of the row dict since it holds a live PySCF
    object and a raw coefficient matrix, neither of which belongs in the
    saved JSON/CSV.
    """
    problem, orbitals_next = build_problem(oo, oh, angle_deg, dihedral_deg,
                                           active_electrons, active_orbitals, orbitals0)
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

    row = {
        "oo": float(oo), "oh": float(oh), "angle": float(angle_deg), "dihedral": float(dihedral_deg),
        "hf": total_energy(problem, hf_electronic),
        "vqe": total_energy(problem, float(result.fun)),
        "exact": total_energy(problem, exact_electronic),
        "n_qubits": hamiltonian.num_qubits,
        "n_params": ansatz.num_parameters,
        "n_evaluations": int(getattr(result, "nfev", 0)),
        "vqe_params": np.asarray(result.x),
    }
    return row, orbitals_next


# ===========================================================================
#  The window -- live curve/map on the left, a real 3D molecule on the right
# ===========================================================================

ATOM_COLORS = {"O": "#ff4d4d", "H": "#f2f2f2"}
ATOM_MARKER_SIZE = {"O": 16, "H": 10}   # matplotlib points -- Axes3D has no data-scaled patches


def _atom_element(name: str) -> str:
    return name[0]


def _build_molecule_panel_3d(ax, oo_max: float, oh: float):
    """Set up the 3D geometry panel. Returns (atom_lines, bond_lines, label)."""
    pad = oh + 0.5
    ax.set_xlim(-0.4, oo_max + pad)
    ax.set_ylim(-pad, pad)
    ax.set_zlim(-pad, pad)
    ax.set_box_aspect((oo_max + pad + 0.4, 2 * pad, 2 * pad))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("Geometry")
    # View chosen so the O-O axis runs roughly left-right and an out-of-plane
    # twist is visible as genuine depth rather than foreshortened to nothing.
    ax.view_init(elev=18, azim=-60)

    atom_lines = {}
    for name in ("O1", "O2", "H1", "H2"):
        el = _atom_element(name)
        (line,) = ax.plot([], [], [], "o", color=ATOM_COLORS[el],
                          markersize=ATOM_MARKER_SIZE[el], markeredgecolor="black",
                          markeredgewidth=1.0)
        atom_lines[name] = line
    bond_lines = [ax.plot([], [], [], "-", color="#888888", linewidth=3)[0] for _ in range(3)]
    label = ax.text2D(0.5, -0.08, "", transform=ax.transAxes, ha="center", fontsize=10.5)
    return atom_lines, bond_lines, label


def _draw_molecule_3d(atom_lines, bond_lines, row) -> None:
    pos = h2o2_positions(row["oo"], row["oh"], row["angle"], row["dihedral"])
    for name, p in pos.items():
        atom_lines[name].set_data_3d([p[0]], [p[1]], [p[2]])
    pairs = (("O1", "O2"), ("O1", "H1"), ("O2", "H2"))
    for bond_line, (a, b) in zip(bond_lines, pairs):
        pa, pb = pos[a], pos[b]
        bond_line.set_data_3d([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]])


def _mol_label_text(row: Dict[str, Any]) -> str:
    return (f"r(O-O)={row['oo']:.2f} A  r(O-H)={row['oh']:.2f} A\n"
            f"angle={row['angle']:.1f}$\\degree$  dihedral={row['dihedral']:.1f}$\\degree$"
            f"   E = {row['vqe']:.4f} Ha")


COORD_LABELS = {
    "stretch": "O-O bond length (Angstrom)",
    "bend": "H-O-O angle (degrees)",
    "twist": "H-O-O-H dihedral (degrees)",
}
COORD_KEY = {"stretch": "oo", "bend": "angle", "twist": "dihedral"}


class H2O2ScanViewer:
    """Live scan plot plus, once finished, arrow-key/slider review.

    Phase 1: :meth:`add_point` is called per geometry as its energies come in
    -- the curve grows and the molecule redraws at the geometry just
    computed. Phase 2: :meth:`enable_review` turns the window into a
    scrubber over the finished scan.
    """

    def __init__(self, coordinate: str, fixed_label: str, oo_span: float, oh_span: float) -> None:
        self.coordinate = coordinate
        plt.ion()
        self.fig = plt.figure(figsize=(12, 5.5))
        self.ax_energy = self.fig.add_subplot(1, 2, 1)
        self.ax_mol = self.fig.add_subplot(1, 2, 2, projection="3d")
        self.fig.subplots_adjust(bottom=0.22, wspace=0.3)

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
        self.ax_energy.set_xlabel(COORD_LABELS[coordinate])
        self.ax_energy.set_ylabel("Energy (Hartree)")
        self.ax_energy.set_title(f"H$_2$O$_2$ {coordinate} scan   ({fixed_label})")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.grid(alpha=0.3)

        self.atom_lines, self.bond_lines, self.mol_label = _build_molecule_panel_3d(
            self.ax_mol, oo_span, oh_span
        )
        self._redraw()

    def _x_of(self, row: Dict[str, Any]) -> float:
        return row[COORD_KEY[self.coordinate]]

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
        _draw_molecule_3d(self.atom_lines, self.bond_lines, row)
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
#  Viewer 2 -- the surface mode: dihedral x angle
# ===========================================================================

class H2O2SurfaceViewer:
    """Live filling-in energy map on the left, live 3D molecule on the right.

    Same design as BeH2's surface viewer: a 2D heat map while the scan runs
    (re-rendering a 3D surface every point would be slow and, with most of
    the grid still empty, unreadable), with the real 3D surface rendered once
    at the end by :meth:`show_surface`.
    """

    def __init__(self, dihedral_values: np.ndarray, angle_values: np.ndarray,
                 oo_fixed: float, oh_fixed: float) -> None:
        self.dihedral_values = np.asarray(dihedral_values, dtype=float)
        self.angle_values = np.asarray(angle_values, dtype=float)
        self.oo_fixed = float(oo_fixed)
        self.oh_fixed = float(oh_fixed)
        # Z indexed [angle, dihedral] so imshow's rows run up the y axis.
        self.Z = np.full((len(self.angle_values), len(self.dihedral_values)), np.nan)

        self.rows: List[Dict[str, Any]] = []
        self.index = 0
        self.slider: Slider | None = None

        plt.ion()
        self.fig = plt.figure(figsize=(12.5, 5.5))
        self.ax_map = self.fig.add_subplot(1, 2, 1)
        self.ax_mol = self.fig.add_subplot(1, 2, 2, projection="3d")
        self.fig.subplots_adjust(bottom=0.22, wspace=0.3)

        dd = self._half_step(self.dihedral_values)
        da = self._half_step(self.angle_values)
        self.extent = (self.dihedral_values[0] - dd, self.dihedral_values[-1] + dd,
                       self.angle_values[0] - da, self.angle_values[-1] + da)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)
        self.im = self.ax_map.imshow(self.Z, origin="lower", extent=self.extent,
                                     aspect="auto", cmap=cmap, interpolation="nearest")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_map, label="Energy (Hartree)")
        (self.map_marker,) = self.ax_map.plot([], [], "D", color="white", markersize=9,
                                              fillstyle="none", markeredgewidth=2)
        self.ax_map.set_xlabel("H-O-O-H dihedral (degrees)")
        self.ax_map.set_ylabel("H-O-O angle (degrees)")
        self.ax_map.set_title("H$_2$O$_2$ energy surface")

        self.atom_lines, self.bond_lines, self.mol_label = _build_molecule_panel_3d(
            self.ax_mol, self.oo_fixed, self.oh_fixed
        )
        self._redraw()

    @staticmethod
    def _half_step(values: np.ndarray) -> float:
        if len(values) < 2:
            return 5.0
        return float(values[1] - values[0]) / 2

    def _cell_of(self, row: Dict[str, Any]) -> Tuple[int, int]:
        j = int(np.argmin(np.abs(self.dihedral_values - row["dihedral"])))
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
        self.map_marker.set_data([row["dihedral"]], [row["angle"]])
        _draw_molecule_3d(self.atom_lines, self.bond_lines, row)
        self.mol_label.set_text(_mol_label_text(row))
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_surface(self, png_path: Path | None = None) -> None:
        grid_dihedral, grid_angle = np.meshgrid(self.dihedral_values, self.angle_values)
        masked = np.ma.masked_invalid(self.Z)

        fig = plt.figure(figsize=(13, 5.5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot_surface(grid_dihedral, grid_angle, masked, cmap="viridis",
                          linewidth=0, antialiased=True, alpha=0.95)
        ax3d.set_xlabel("dihedral (deg)")
        ax3d.set_ylabel("H-O-O angle (deg)")
        ax3d.set_zlabel("Energy (Ha)")
        ax3d.set_title("H$_2$O$_2$ ground-state surface (VQE)")
        ax3d.view_init(elev=25, azim=-50)

        ax2d = fig.add_subplot(1, 2, 2)
        levels = 25
        contour = ax2d.contourf(grid_dihedral, grid_angle, masked, levels=levels, cmap="viridis")
        ax2d.contour(grid_dihedral, grid_angle, masked, levels=levels,
                     colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(contour, ax=ax2d, label="Energy (Hartree)")
        if self.rows:
            best = min(self.rows, key=lambda r: r["vqe"])
            ax2d.plot([best["dihedral"]], [best["angle"]], "r*", markersize=16,
                      label=f"min {best['vqe']:.4f} Ha")
            ax2d.legend(loc="lower center")
        ax2d.set_xlabel("H-O-O-H dihedral (degrees)")
        ax2d.set_ylabel("H-O-O angle (degrees)")
        ax2d.set_title("Contours -- expect two unequal ridges (cis at 0, trans at 180)")

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
    """``start`` to ``stop`` inclusive, robust to float drift -- the same
    trick every other script in this project uses."""
    return np.arange(start, stop + step / 2, step)


def serpentine(xs: np.ndarray, ys: np.ndarray) -> List[Tuple[float, float]]:
    """Grid points ordered so consecutive points are always adjacent (a
    boustrophedon walk), so the warm start always comes from a neighbouring
    geometry -- same as BeH2's surface scan."""
    geometries: List[Tuple[float, float]] = []
    for i, y in enumerate(ys):
        row_xs = xs if i % 2 == 0 else xs[::-1]
        geometries.extend((float(x), float(y)) for x in row_xs)
    return geometries


# ===========================================================================
#  CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2O2 ground-state energy by VQE (local simulator), scanned along its "
                    "symmetric stretch, bend, torsion, or the angle x dihedral surface."
    )
    parser.add_argument("--coordinate", choices=["stretch", "bend", "twist", "surface"],
                        default="twist",
                        help="'stretch': O-O bond. 'bend': H-O-O angle. 'twist' (default): the "
                             "H-O-O-H dihedral -- the headline motion, see the module docstring. "
                             "'surface': dihedral x angle grid.")
    parser.add_argument("--start", type=float, default=None,
                        help="scan start -- Angstrom for stretch, degrees for bend/twist/surface")
    parser.add_argument("--stop", type=float, default=None, help="scan stop, inclusive")
    parser.add_argument("--step", type=float, default=None,
                        help="scan step (primary axis; the dihedral axis in surface mode)")
    parser.add_argument("--angle-start", type=float, default=80.0,
                        help="[surface mode] first H-O-O angle in degrees (default 80)")
    parser.add_argument("--angle-stop", type=float, default=140.0,
                        help="[surface mode] last H-O-O angle in degrees (default 140)")
    parser.add_argument("--angle-step", type=float, default=15.0,
                        help="[surface mode] angle step in degrees (default 15)")
    parser.add_argument("--oo", type=float, default=EQUILIBRIUM_OO,
                        help=f"fixed O-O bond length in Angstrom when oo is not the scanned "
                             f"coordinate (default {EQUILIBRIUM_OO})")
    parser.add_argument("--oh", type=float, default=EQUILIBRIUM_OH,
                        help=f"fixed O-H bond length in Angstrom (default {EQUILIBRIUM_OH}); "
                             f"never scanned by this script")
    parser.add_argument("--angle", type=float, default=EQUILIBRIUM_ANGLE,
                        help=f"fixed H-O-O angle in degrees when angle is not the scanned "
                             f"coordinate (default {EQUILIBRIUM_ANGLE})")
    parser.add_argument("--dihedral", type=float, default=EQUILIBRIUM_DIHEDRAL,
                        help=f"fixed H-O-O-H dihedral in degrees when dihedral is not the "
                             f"scanned coordinate (default {EQUILIBRIUM_DIHEDRAL}, the true "
                             f"non-planar equilibrium)")
    parser.add_argument("--active-electrons", type=int, default=8,
                        help="electrons in the active space (default 8)")
    parser.add_argument("--active-orbitals", type=int, default=6,
                        help="spatial orbitals in the active space (default 6 -> 10 qubits). "
                             "See the module docstring: this is a starting hypothesis, not a "
                             "verified choice -- check it before trusting the twist barriers.")
    parser.add_argument("--optimizer", choices=["cobyla", "slsqp"], default="cobyla",
                        help="'cobyla' (default), gradient-free -- expected parameter count at "
                             "the default active space favors it, the same reasoning as O2. "
                             "'slsqp' estimates gradients by finite differences.")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="optimizer iteration cap (default 1000)")
    parser.add_argument("--cold-start", action="store_true",
                        help="restart each geometry's optimization from the Hartree-Fock state "
                             "instead of warm-starting from the previous geometry's solution.")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="skip the VQE entirely and reopen a previously saved scan straight "
                             "into review mode")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="where to save this scan (default: h2o2_<coordinate>_scan.json/.csv "
                             "next to this script)")
    parser.add_argument("--no-save", action="store_true", help="don't save the results of this scan")
    parser.add_argument("--no-show", action="store_true",
                        help="exit when the scan finishes instead of staying open in review mode.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    if args.load:
        _replay(Path(args.load), here, args.no_show)
        return

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
    """Read a scan back from its ``.csv`` instead of its ``.json``.

    Which of oo/angle/dihedral actually varies tells us the scan type --
    `oh` is never scanned by this script, so it is excluded from that check.
    """
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{display_path(path)} has no data rows.")

    numeric = ("oo", "oh", "angle", "dihedral", "hf", "vqe", "exact")
    rows = []
    for r in raw:
        row = {k: float(v) for k, v in r.items() if k in numeric and v != ""}
        for k in ("n_qubits", "n_params", "n_evaluations"):
            if r.get(k):
                row[k] = int(float(r[k]))
        rows.append(row)

    missing = {"oo", "oh", "angle", "dihedral", "vqe"} - set(rows[0])
    if missing:
        raise SystemExit(
            f"{display_path(path)} is missing the column(s) {sorted(missing)}, so it is not "
            f"an H2O2 scan CSV."
        )

    oos = sorted({r["oo"] for r in rows})
    angles = sorted({r["angle"] for r in rows})
    dihedrals = sorted({r["dihedral"] for r in rows})
    oh_fixed = rows[0]["oh"]

    if len(angles) > 1 and len(dihedrals) > 1:
        metadata = {
            "scan_type": "surface", "coordinate": "surface",
            "dihedral_values": dihedrals, "angle_values": angles,
            "fixed_oo": oos[0], "fixed_oh": oh_fixed,
        }
    elif len(oos) > 1:
        metadata = {
            "scan_type": "curve", "coordinate": "stretch",
            "fixed_label": f"angle={angles[0]}$\\degree$, dihedral={dihedrals[0]}$\\degree$",
            "oo_span": oos[-1], "oh_span": oh_fixed,
        }
    elif len(angles) > 1:
        metadata = {
            "scan_type": "curve", "coordinate": "bend",
            "fixed_label": f"r(O-O)={oos[0]} A, dihedral={dihedrals[0]}$\\degree$",
            "oo_span": oos[0], "oh_span": oh_fixed,
        }
    else:
        metadata = {
            "scan_type": "curve", "coordinate": "twist",
            "fixed_label": f"r(O-O)={oos[0]} A, angle={angles[0]}$\\degree$",
            "oo_span": oos[0], "oh_span": oh_fixed,
        }
    return metadata, rows


def _load_any(load_path: Path, here: Path) -> tuple:
    candidates = [load_path, here / load_path.name]
    if load_path.suffix.lower() == ".csv":
        for candidate in candidates:
            if candidate.exists():
                print("reading the CSV (metadata inferred from the columns; "
                      "the .json carries it explicitly)")
                return _load_csv(candidate)
        looked = "\n  ".join(display_path(c) for c in candidates)
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
        viewer = H2O2SurfaceViewer(np.asarray(metadata["dihedral_values"], dtype=float),
                                   np.asarray(metadata["angle_values"], dtype=float),
                                   float(metadata["fixed_oo"]), float(metadata["fixed_oh"]))
        for row in rows:
            viewer.add_point(row)
        viewer.show_surface()
        _report_barriers(rows, is_surface=True)
        _finish(viewer, no_show)
    else:
        viewer = H2O2ScanViewer(metadata["coordinate"], metadata["fixed_label"],
                                float(metadata["oo_span"]), float(metadata["oh_span"]))
        for row in rows:
            viewer.add_point(row)
        if metadata["coordinate"] == "twist":
            _report_barriers(rows, is_surface=False)
        _finish(viewer, no_show)


def _run_curve(args, parser, here: Path) -> None:
    if args.coordinate == "stretch":
        start = 1.0 if args.start is None else args.start
        stop = 3.0 if args.stop is None else args.stop
        step = 0.1 if args.step is None else args.step
        unit = "A"
    elif args.coordinate == "bend":
        start = 80.0 if args.start is None else args.start
        stop = 140.0 if args.stop is None else args.stop
        step = 5.0 if args.step is None else args.step
        unit = "deg"
    else:  # twist
        start = 0.0 if args.start is None else args.start
        stop = 180.0 if args.stop is None else args.stop
        step = 10.0 if args.step is None else args.step
        unit = "deg"

    if step <= 0:
        parser.error("--step must be positive")
    if stop < start:
        parser.error("--stop must be >= --start")

    values = inclusive_range(start, stop, step)

    if args.coordinate == "stretch":
        geometries = [(float(v), args.oh, args.angle, args.dihedral) for v in values]
        fixed_label = f"angle={args.angle}$\\degree$, dihedral={args.dihedral}$\\degree$"
        oo_span, oh_span = float(values[-1]), args.oh
    elif args.coordinate == "bend":
        geometries = [(args.oo, args.oh, float(v), args.dihedral) for v in values]
        fixed_label = f"r(O-O)={args.oo} A, dihedral={args.dihedral}$\\degree$"
        oo_span, oh_span = args.oo, args.oh
    else:
        geometries = [(args.oo, args.oh, args.angle, float(v)) for v in values]
        fixed_label = f"r(O-O)={args.oo} A, angle={args.angle}$\\degree$"
        oo_span, oh_span = args.oo, args.oh

    print(f"H2O2 {args.coordinate} scan: {len(values)} points from {values[0]:.2f} to "
          f"{values[-1]:.2f} {unit} (step {step} {unit})")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})  "
          f"optimizer: {args.optimizer.upper()}")
    print("cost is an ESTIMATE, not yet measured for H2O2 -- see the module docstring; "
          "the plot updates as each point lands and prints its own timing")

    viewer = H2O2ScanViewer(args.coordinate, fixed_label, oo_span, oh_span)
    rows = _scan(geometries, args, viewer)

    best = min(rows, key=lambda r: r["vqe"])
    best_x = best[COORD_KEY[args.coordinate]]
    reference = {
        "stretch": f"literature r(O-O) is ~{EQUILIBRIUM_OO} A",
        "bend": f"literature H-O-O angle is ~{EQUILIBRIUM_ANGLE} deg",
        "twist": f"literature skew minimum is ~{EQUILIBRIUM_DIHEDRAL} deg",
    }[args.coordinate]
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at {best_x:.2f} {unit}  ({reference})")
    _report_errors(rows)

    if args.coordinate == "stretch":
        _report_monotonicity(rows)
    if args.coordinate == "twist":
        _report_barriers(rows, is_surface=False)

    if not args.no_save:
        target = resolve_target(args.save, here, f"h2o2_{args.coordinate}_scan")
        metadata = {
            "molecule": "H2O2", "basis": BASIS,
            "scan_type": "curve", "coordinate": args.coordinate, "symmetric": True,
            "start": float(values[0]), "stop": float(values[-1]), "step": float(step), "unit": unit,
            "fixed_oo": float(args.oo), "fixed_oh": float(args.oh),
            "fixed_angle": float(args.angle), "fixed_dihedral": float(args.dihedral),
            "fixed_label": fixed_label,
            "oo_span": float(oo_span), "oh_span": float(oh_span),
            "active_electrons": int(args.active_electrons), "active_orbitals": int(args.active_orbitals),
            "optimizer": args.optimizer, "max_iterations": int(args.max_iterations),
        }
        _save(target, metadata, rows)

    _finish(viewer, args.no_show)


def _run_surface(args, parser, here: Path) -> None:
    dihedral_start = 0.0 if args.start is None else args.start
    dihedral_stop = 180.0 if args.stop is None else args.stop
    dihedral_step = 20.0 if args.step is None else args.step

    if dihedral_step <= 0 or args.angle_step <= 0:
        parser.error("--step and --angle-step must be positive")
    if dihedral_stop < dihedral_start:
        parser.error("--stop must be >= --start")
    if args.angle_stop < args.angle_start:
        parser.error("--angle-stop must be >= --angle-start")

    dihedral_values = inclusive_range(dihedral_start, dihedral_stop, dihedral_step)
    angle_values = inclusive_range(args.angle_start, args.angle_stop, args.angle_step)
    grid = serpentine(dihedral_values, angle_values)
    geometries = [(args.oo, args.oh, angle, dihedral) for dihedral, angle in grid]

    print(f"H2O2 surface scan: {len(dihedral_values)} dihedral x {len(angle_values)} angle "
          f"= {len(geometries)} points")
    print(f"  dihedral {dihedral_values[0]:.1f} to {dihedral_values[-1]:.1f} deg "
          f"(step {dihedral_step} deg)")
    print(f"  angle    {angle_values[0]:.1f} to {angle_values[-1]:.1f} deg "
          f"(step {args.angle_step} deg)")
    print(f"  fixed: r(O-O)={args.oo} A, r(O-H)={args.oh} A")
    print(f"active space: CAS({args.active_electrons}, {args.active_orbitals})  "
          f"optimizer: {args.optimizer.upper()}")
    print("cost is an ESTIMATE, not yet measured for H2O2 -- see the module docstring")

    viewer = H2O2SurfaceViewer(dihedral_values, angle_values, args.oo, args.oh)
    rows = _scan(geometries, args, viewer)

    best = min(rows, key=lambda r: r["vqe"])
    print(f"\nminimum VQE energy {best['vqe']:.6f} Ha at "
          f"dihedral={best['dihedral']:.1f} deg, angle={best['angle']:.1f} deg")
    print(f"  (literature: skew minimum ~{EQUILIBRIUM_DIHEDRAL} deg, ~{EQUILIBRIUM_ANGLE} deg)")
    _report_errors(rows)
    _report_barriers(rows, is_surface=True)

    png_path = None
    if not args.no_save:
        target = resolve_target(args.save, here, "h2o2_surface_scan")
        metadata = {
            "molecule": "H2O2", "basis": BASIS,
            "scan_type": "surface", "coordinate": "surface", "symmetric": True,
            "dihedral_values": [float(v) for v in dihedral_values],
            "angle_values": [float(v) for v in angle_values],
            "dihedral_step": float(dihedral_step), "angle_step": float(args.angle_step),
            "fixed_oo": float(args.oo), "fixed_oh": float(args.oh),
            "active_electrons": int(args.active_electrons), "active_orbitals": int(args.active_orbitals),
            "optimizer": args.optimizer, "max_iterations": int(args.max_iterations),
        }
        _save(target, metadata, rows)
        png_path = target.with_name(target.stem + "_3d.png")

    viewer.show_surface(png_path)
    _finish(viewer, args.no_show)


def _scan(geometries, args, viewer) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    previous_params = None
    previous_orbitals = None  # orbital-continuity state -- see _track_orbitals; always
                              # carried forward regardless of --cold-start (correctness,
                              # not an optimizer warm-start heuristic)
    total = len(geometries)
    for i, (oo, oh, angle, dihedral) in enumerate(geometries):
        t0 = time.perf_counter()
        row, previous_orbitals = run_one_geometry(
            oo, oh, angle, dihedral, args.active_electrons, args.active_orbitals,
            optimizer=args.optimizer, max_iterations=args.max_iterations,
            theta0=previous_params, orbitals0=previous_orbitals)
        if not args.cold_start:
            previous_params = row["vqe_params"]
        rows.append(row)
        viewer.add_point(row)
        if i == 0:
            print(f"  ({row['n_qubits']} qubits, {row['n_params']} UCCSD parameters)")
        print(f"  [{i + 1:4d}/{total}] oo={oo:5.2f} angle={angle:6.1f} dihedral={dihedral:6.1f}   "
              f"HF={row['hf']:.6f}  VQE={row['vqe']:.6f}  exact={row['exact']:.6f}  "
              f"({row['n_evaluations']} evals, {time.perf_counter() - t0:.1f}s)")
    return rows


def _report_errors(rows: List[Dict[str, Any]]) -> None:
    """HF-vs-exact (physics: correlation energy) and VQE-vs-exact (numerics:
    optimizer convergence) -- the same split every other script here reports."""
    max_hf = max(abs(r["hf"] - r["exact"]) for r in rows)
    max_vqe = max(abs(r["vqe"] - r["exact"]) for r in rows)
    rms_vqe = math.sqrt(sum((r["vqe"] - r["exact"]) ** 2 for r in rows) / len(rows))
    print(f"largest |HF - exact|  across the scan: {max_hf * 1000:9.3f} mHa "
          f"(correlation energy HF misses)")
    print(f"largest |VQE - exact| across the scan: {max_vqe * 1000:9.3f} mHa "
          f"(RMS {rms_vqe * 1000:.3f} mHa -- VQE convergence)")
    if max_vqe * 1000 > 1.0:
        print("  note: |VQE - exact| above 1 mHa suggests the optimizer did not fully "
              "converge somewhere -- try --optimizer slsqp or a higher --max-iterations")


def _report_monotonicity(rows: List[Dict[str, Any]]) -> None:
    """Same active-space sanity check as BeH2/H2O/O2: past its minimum, a
    dissociation curve must rise monotonically."""
    ordered = sorted(rows, key=lambda r: r["oo"])
    best_at = min(range(len(ordered)), key=lambda i: ordered[i]["exact"])
    dips = [
        (ordered[i]["oo"], ordered[i + 1]["oo"])
        for i in range(best_at, len(ordered) - 1)
        if ordered[i + 1]["exact"] < ordered[i]["exact"]
    ]
    if dips:
        print("\n  WARNING: the exact curve dips downward past its minimum, between "
              + ", ".join(f"{a:.2f}-{b:.2f} A" for a, b in dips))
        print("  A ground-state curve must rise monotonically toward dissociation. This is")
        print("  an active space too small to describe the O-O bond breaking, not a VQE failure.")
        print("  Try a larger --active-orbitals / --active-electrons.")
    else:
        print("  curve rises monotonically past its minimum -- active space looks adequate")


def _report_barriers(rows: List[Dict[str, Any]], is_surface: bool) -> None:
    """Compare the measured cis/trans torsional barriers against literature,
    and flag explicitly if the trans barrier is not resolved above the
    optimizer's own convergence noise -- see the module docstring's warning
    that the trans barrier (~1.8 mHa) is the same order of magnitude as
    typical |VQE - exact| noise (~0.1-1 mHa).
    """
    dihedrals = sorted({r["dihedral"] for r in rows})
    if not dihedrals or dihedrals[0] > 15 or dihedrals[-1] < 165:
        return  # scan doesn't span cis (~0) to trans (~180); nothing to compare

    minimum = min(rows, key=lambda r: r["vqe"])
    cis = min(rows, key=lambda r: r["dihedral"])
    trans = max(rows, key=lambda r: r["dihedral"])

    cis_barrier_ha = cis["vqe"] - minimum["vqe"]
    trans_barrier_ha = trans["vqe"] - minimum["vqe"]
    cis_kcal = cis_barrier_ha * HARTREE_TO_KCAL
    trans_kcal = trans_barrier_ha * HARTREE_TO_KCAL
    noise_mha = max(abs(r["vqe"] - r["exact"]) for r in rows) * 1000

    print(f"\ntorsional barriers (measured relative to the VQE minimum at "
          f"dihedral={minimum['dihedral']:.1f} deg):")
    print(f"  cis   (dihedral={cis['dihedral']:.1f} deg): {cis_barrier_ha * 1000:7.3f} mHa "
          f"= {cis_kcal:5.2f} kcal/mol   (literature ~{LIT_CIS_BARRIER_KCAL} kcal/mol)")
    print(f"  trans (dihedral={trans['dihedral']:.1f} deg): {trans_barrier_ha * 1000:7.3f} mHa "
          f"= {trans_kcal:5.2f} kcal/mol   (literature ~{LIT_TRANS_BARRIER_KCAL} kcal/mol)")
    print(f"  largest |VQE - exact| noise floor in this scan: {noise_mha:.3f} mHa")
    if abs(trans_barrier_ha * 1000) < 3 * noise_mha:
        print("  WARNING: the trans barrier is within 3x the VQE convergence noise floor -- "
              "treat this number as unresolved, not as a measurement. Try --optimizer slsqp "
              "or a finer grid near dihedral=180 before trusting it.")


def _save(target: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    json_path, csv_path = save_scan(target, metadata, rows)
    print(f"saved {display_path(json_path)}")
    print(f"saved {display_path(csv_path)}")
    print(f"reopen it later without recomputing:  "
          f"python {display_path(Path(__file__))} --load {json_path.name}")


if __name__ == "__main__":
    main()
