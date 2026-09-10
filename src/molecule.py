"""
molecule.py
===========

A single-file toolkit for the *Molecular Simulation on Quantum Computing Systems*
project (CSE 402).  It does two jobs:

1. **Molecule generation** -- a well-documented wrapper around NVIDIA
   **CUDA-Q Solvers** (`cudaq_solvers.create_molecule`, see arXiv:2401.09253
   App. A.1).  It runs the classical pre-computation (Hartree-Fock / MP2 /
   CASCI / CASSCF / CCSD / FCI through PySCF) and exposes the qubit Hamiltonian
   together with a large set of convenience accessors: energies, one/two-body
   integrals, qubit count, Pauli decomposition, an *exact* diagonalization
   reference energy, the Hartree-Fock reference state, ADAPT/UCCSD operator
   pools, OpenFermion export, save/load, ...

2. **Visualization** -- best-practice interactive, rotatable 3D rendering of a
   molecular geometry with three backends that degrade gracefully:

   ================  ===================================================
   ``py3Dmol``       3Dmol.js in a Jupyter cell -- the community standard
   ``plotly``        interactive HTML, no JavaScript knowledge required
   ``html``          a self-contained 3Dmol.js page, *zero* Python deps
   ================  ===================================================

CUDA-Q Solvers only runs on Linux / WSL.  The whole visualization half works
everywhere (Windows included) and this module imports fine **without** cudaq
installed -- the heavy imports are deferred until you actually build a
:class:`Molecule`.

Quick start
-----------
>>> from molecule import Molecule, visualize, load_geometry
>>> visualize("caffeine", filename="caffeine.html")          # no cudaq needed
>>> h2o = Molecule("water", basis="sto-3g")                  # needs cudaq-solvers
>>> h2o.n_qubits, h2o.hf_energy, h2o.fci_energy
>>> h2o.ground_state_energy_exact()                          # numerical reference
>>> h2o.visualize(style="ball_and_stick", spin=True)

References
----------
* CUDA-Q Solvers docs .... https://nvidia.github.io/cudaqx/
* create_molecule ........ arXiv:2401.09253, Appendix A.1
* Covalent radii ......... Cordero et al., Dalton Trans. (2008) 2832
* van der Waals radii .... Bondi J. Phys. Chem. 68 (1964) 441; Mantina et al. (2009)
* CPK / Jmol colors ...... http://jmol.sourceforge.net/jscolors/
"""

from __future__ import annotations

import math
import os
import pickle
import re
import sys
import tempfile
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__version__ = "0.1.0"

# A geometry is a list of ``(element_symbol, (x, y, z))`` pairs, coordinates in Angstrom.
Vec3 = Tuple[float, float, float]
GeometryItem = Tuple[str, Vec3]
Geometry = List[GeometryItem]
GeometryLike = Union[str, os.PathLike, Sequence[GeometryItem], "Molecule"]

BOHR_PER_ANGSTROM = 1.8897261246257702
ANGSTROM_PER_BOHR = 1.0 / BOHR_PER_ANGSTROM


# ===========================================================================
#  Periodic-table data
# ===========================================================================

# Symbol indexed by atomic number - 1  (covers the whole periodic table).
_SYMBOLS: Tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

ATOMIC_NUMBER: Dict[str, int] = {s: i + 1 for i, s in enumerate(_SYMBOLS)}

# Standard atomic weights (CIAAW 2021); integers for elements without a stable isotope.
_MASSES: Tuple[float, ...] = (
    1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,
    22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.95, 39.098, 40.078,
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546,
    65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62,
    88.906, 91.224, 92.906, 95.95, 98.0, 101.07, 102.91, 106.42, 107.87, 112.41,
    114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91,
    140.12, 140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93, 162.50,
    164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21,
    190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.04, 231.04, 238.03, 237.0, 244.0, 243.0,
    247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 266.0, 267.0, 268.0, 269.0,
    270.0, 269.0, 278.0, 281.0, 282.0, 285.0, 286.0, 289.0, 290.0, 293.0, 294.0,
    294.0,
)
ATOMIC_MASS: Dict[str, float] = dict(zip(_SYMBOLS, _MASSES))

# Covalent radii, Angstrom (Cordero 2008, low-spin values where relevant), Z = 1..86.
_COVALENT: Tuple[float, ...] = (
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41,
    1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39,
    1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44,
    1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75,
    1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40,
    1.50, 1.50,
)
COVALENT_RADII: Dict[str, float] = dict(zip(_SYMBOLS, _COVALENT))
COVALENT_RADIUS_DEFAULT = 0.75

# van der Waals radii, Angstrom (Bondi 1964 + Mantina 2009 extensions).
VDW_RADII: Dict[str, float] = {
    "H": 1.10, "He": 1.40, "Li": 1.81, "Be": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39, "Ga": 1.87,
    "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02, "Pd": 1.63,
    "Ag": 1.72, "Cd": 1.58, "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06,
    "I": 1.98, "Xe": 2.16, "Pt": 1.75, "Au": 1.66, "Hg": 1.55, "Tl": 1.96,
    "Pb": 2.02, "Bi": 2.07, "U": 1.86,
}
VDW_RADIUS_DEFAULT = 2.00

# CPK / Jmol element colors (hex, no leading '#').
CPK_COLORS: Dict[str, str] = {
    "H": "FFFFFF", "He": "D9FFFF", "Li": "CC80FF", "Be": "C2FF00", "B": "FFB5B5",
    "C": "909090", "N": "3050F8", "O": "FF0D0D", "F": "90E050", "Ne": "B3E3F5",
    "Na": "AB5CF2", "Mg": "8AFF00", "Al": "BFA6A6", "Si": "F0C8A0", "P": "FF8000",
    "S": "FFFF30", "Cl": "1FF01F", "Ar": "80D1E3", "K": "8F40D4", "Ca": "3DFF00",
    "Sc": "E6E6E6", "Ti": "BFC2C7", "V": "A6A6AB", "Cr": "8A99C7", "Mn": "9C7AC7",
    "Fe": "E06633", "Co": "F090A0", "Ni": "50D050", "Cu": "C88033", "Zn": "7D80B0",
    "Ga": "C28F8F", "Ge": "668F8F", "As": "BD80E3", "Se": "FFA100", "Br": "A62929",
    "Kr": "5CB8D1", "Rb": "702EB0", "Sr": "00FF00", "Y": "94FFFF", "Zr": "94E0E0",
    "Nb": "73C2C9", "Mo": "54B5B5", "Tc": "3B9E9E", "Ru": "248F8F", "Rh": "0A7D8C",
    "Pd": "006985", "Ag": "C0C0C0", "Cd": "FFD98F", "In": "A67573", "Sn": "668080",
    "Sb": "9E63B5", "Te": "D47A00", "I": "940094", "Xe": "429EB0", "Cs": "57178F",
    "Ba": "00C900", "La": "70D4FF", "Ce": "FFFFC7", "Pr": "D9FFC7", "Nd": "C7FFC7",
    "Sm": "8FFFC7", "Eu": "61FFC7", "Gd": "45FFC7", "Tb": "30FFC7", "Dy": "1FFFC7",
    "Ho": "00FF9C", "Er": "00E675", "Tm": "00D452", "Yb": "00BF38", "Lu": "00AB24",
    "Hf": "4DC2FF", "Ta": "4DA6FF", "W": "2194D6", "Re": "267DAB", "Os": "266696",
    "Ir": "175487", "Pt": "D0D0E0", "Au": "FFD123", "Hg": "B8B8D0", "Tl": "A6544D",
    "Pb": "575961", "Bi": "9E4FB5", "Po": "AB5C00", "At": "754F45", "Rn": "428296",
    "Fr": "420066", "Ra": "007D00", "Ac": "70ABFA", "Th": "00BAFF", "Pa": "00A1FF",
    "U": "008FFF", "Np": "0080FF", "Pu": "006BFF",
}
CPK_COLOR_DEFAULT = "FF1493"  # Jmol "unknown element" pink


def normalize_symbol(symbol: str) -> str:
    """``'CL'`` / ``'cl'`` -> ``'Cl'`` (and validate)."""
    s = symbol.strip()
    s = s[:1].upper() + s[1:].lower()
    if s not in ATOMIC_NUMBER:
        raise ValueError(f"unknown element symbol: {symbol!r}")
    return s


def atomic_number(symbol: str) -> int:
    return ATOMIC_NUMBER[normalize_symbol(symbol)]


def atomic_mass(symbol: str) -> float:
    return ATOMIC_MASS[normalize_symbol(symbol)]


def covalent_radius(symbol: str) -> float:
    return COVALENT_RADII.get(normalize_symbol(symbol), COVALENT_RADIUS_DEFAULT)


def vdw_radius(symbol: str) -> float:
    return VDW_RADII.get(normalize_symbol(symbol), VDW_RADIUS_DEFAULT)


def cpk_color(symbol: str) -> str:
    """Return a ``'#rrggbb'`` CPK color for *symbol*."""
    return "#" + CPK_COLORS.get(normalize_symbol(symbol), CPK_COLOR_DEFAULT)


# ===========================================================================
#  Built-in geometries  (Angstrom, near-equilibrium)
# ===========================================================================

def _tetrahedral(center: str, ligand: str, bond: float) -> Geometry:
    d = bond / math.sqrt(3.0)
    return [
        (center, (0.0, 0.0, 0.0)),
        (ligand, (d, d, d)),
        (ligand, (d, -d, -d)),
        (ligand, (-d, d, -d)),
        (ligand, (-d, -d, d)),
    ]


def _linear_chain(symbol: str, n: int, spacing: float) -> Geometry:
    start = -0.5 * spacing * (n - 1)
    return [(symbol, (0.0, 0.0, start + i * spacing)) for i in range(n)]


BUILTIN_GEOMETRIES: Dict[str, Geometry] = {
    "h2": [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))],
    "h3+": [("H", (0.0, 0.0, 0.0)), ("H", (0.8, 0.0, 0.0)),
            ("H", (0.4, 0.6928, 0.0))],
    "h4": _linear_chain("H", 4, 1.23),
    "h6": _linear_chain("H", 6, 1.23),
    "lih": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))],
    "hf": [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.9168))],
    "hcl": [("H", (0.0, 0.0, 0.0)), ("Cl", (0.0, 0.0, 1.2746))],
    "n2": [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
    "o2": [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.2075))],
    "f2": [("F", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 1.4119))],
    "co": [("C", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.1282))],
    "beh2": [("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.3264)),
             ("H", (0.0, 0.0, -1.3264))],
    "h2o": [("O", (0.0, 0.0, 0.1173)), ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692))],
    "water": [("O", (0.0, 0.0, 0.1173)), ("H", (0.0, 0.7572, -0.4692)),
              ("H", (0.0, -0.7572, -0.4692))],
    "h2s": [("S", (0.0, 0.0, 0.1030)), ("H", (0.0, 0.9698, -0.8240)),
            ("H", (0.0, -0.9698, -0.8240))],
    "nh3": [("N", (0.0, 0.0, 0.1173)), ("H", (0.0, 0.9377, -0.2739)),
            ("H", (0.8121, -0.4689, -0.2739)), ("H", (-0.8121, -0.4689, -0.2739))],
    "ch4": _tetrahedral("C", "H", 1.0870),
    "sih4": _tetrahedral("Si", "H", 1.4798),
    "co2": [("C", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.1621)),
            ("O", (0.0, 0.0, -1.1621))],
    "hcn": [("H", (0.0, 0.0, -1.0640)), ("C", (0.0, 0.0, 0.0)),
            ("N", (0.0, 0.0, 1.1560))],
    "c2h2": [("C", (0.0, 0.0, 0.6013)), ("C", (0.0, 0.0, -0.6013)),
             ("H", (0.0, 0.0, 1.6624)), ("H", (0.0, 0.0, -1.6624))],
    "c2h4": [("C", (0.0, 0.0, 0.6695)), ("C", (0.0, 0.0, -0.6695)),
             ("H", (0.0, 0.9289, 1.2321)), ("H", (0.0, -0.9289, 1.2321)),
             ("H", (0.0, 0.9289, -1.2321)), ("H", (0.0, -0.9289, -1.2321))],
    "c2h6": [("C", (0.0, 0.0, 0.7680)), ("C", (0.0, 0.0, -0.7680)),
             ("H", (0.0, 1.0192, 1.1573)), ("H", (0.8826, -0.5096, 1.1573)),
             ("H", (-0.8826, -0.5096, 1.1573)), ("H", (0.0, -1.0192, -1.1573)),
             ("H", (-0.8826, 0.5096, -1.1573)), ("H", (0.8826, 0.5096, -1.1573))],
    "ch3oh": [("C", (-0.3689, 0.0, 0.0)), ("O", (1.0428, 0.0, 0.0)),
              ("H", (-0.7660, 1.0182, 0.0)), ("H", (-0.7660, -0.5091, 0.8818)),
              ("H", (-0.7660, -0.5091, -0.8818)), ("H", (1.3667, -0.8901, 0.0))],
    "benzene": [
        ("C", (0.0000, 1.3970, 0.0)), ("C", (1.2098, 0.6985, 0.0)),
        ("C", (1.2098, -0.6985, 0.0)), ("C", (0.0000, -1.3970, 0.0)),
        ("C", (-1.2098, -0.6985, 0.0)), ("C", (-1.2098, 0.6985, 0.0)),
        ("H", (0.0000, 2.4810, 0.0)), ("H", (2.1486, 1.2405, 0.0)),
        ("H", (2.1486, -1.2405, 0.0)), ("H", (0.0000, -2.4810, 0.0)),
        ("H", (-2.1486, -1.2405, 0.0)), ("H", (-2.1486, 1.2405, 0.0)),
    ],
    "caffeine": [
        ("O", (0.4700, 2.5688, 0.0)), ("O", (-3.1271, -0.4436, 0.0)),
        ("N", (-0.9686, -1.3125, 0.0)), ("N", (2.2182, 0.1412, 0.0)),
        ("N", (-1.3477, 1.0797, 0.0)), ("N", (1.4119, -1.9372, 0.0)),
        ("C", (0.8579, 0.2592, 0.0)), ("C", (0.3897, -1.0264, 0.0)),
        ("C", (0.0307, 1.4220, 0.0)), ("C", (-1.9061, -0.2495, 0.0)),
        ("C", (2.5032, -1.1998, 0.0)), ("C", (-1.4276, -2.6960, 0.0)),
        ("C", (3.1926, 1.2061, 0.0)), ("C", (-2.2969, 2.1881, 0.0)),
        ("H", (3.5163, -1.5787, 0.0)), ("H", (-1.0451, -3.1973, 0.8937)),
        ("H", (-2.5186, -2.7596, 0.0)), ("H", (-1.0447, -3.1973, -0.8937)),
        ("H", (4.1992, 0.7801, 0.0)), ("H", (3.0468, 1.8092, 0.8963)),
        ("H", (3.0468, 1.8092, -0.8963)), ("H", (-1.8087, 3.1651, 0.0)),
        ("H", (-2.9322, 2.1027, 0.8963)), ("H", (-2.9322, 2.1027, -0.8963)),
    ],
}
BUILTIN_GEOMETRIES["methane"] = BUILTIN_GEOMETRIES["ch4"]
BUILTIN_GEOMETRIES["ammonia"] = BUILTIN_GEOMETRIES["nh3"]
BUILTIN_GEOMETRIES["methanol"] = BUILTIN_GEOMETRIES["ch3oh"]
BUILTIN_GEOMETRIES["ethylene"] = BUILTIN_GEOMETRIES["c2h4"]
BUILTIN_GEOMETRIES["ethene"] = BUILTIN_GEOMETRIES["c2h4"]
BUILTIN_GEOMETRIES["ethane"] = BUILTIN_GEOMETRIES["c2h6"]
BUILTIN_GEOMETRIES["acetylene"] = BUILTIN_GEOMETRIES["c2h2"]


# ===========================================================================
#  Geometry: parsing, formatting, analysis, transforms
# ===========================================================================

def normalize_geometry(geometry: Sequence[GeometryItem]) -> Geometry:
    """Validate and canonicalize a geometry to ``[(Sym, (x, y, z)), ...]`` floats."""
    out: Geometry = []
    for i, item in enumerate(geometry):
        try:
            sym, xyz = item
            x, y, z = xyz
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"geometry item {i} must be (symbol, (x, y, z)); got {item!r}"
            ) from exc
        out.append((normalize_symbol(sym), (float(x), float(y), float(z))))
    if not out:
        raise ValueError("geometry is empty")
    return out


def parse_xyz(text_or_path: Union[str, os.PathLike]) -> Geometry:
    """Parse an XYZ file / string.

    Accepts the standard ``<count>`` + comment + atoms layout (comment line may
    be blank) as well as a bare list of ``<symbol> <x> <y> <z>`` lines.  Any
    line that is not a valid atom record (count line, comment, blank, ``#``
    comment) is skipped.
    """
    raw = text_or_path
    p = Path(str(text_or_path))
    if isinstance(text_or_path, os.PathLike) or ("\n" not in str(text_or_path) and p.exists()):
        raw = p.read_text()

    lines = str(raw).splitlines()
    declared: Optional[int] = None
    for k, ln in enumerate(lines):
        if ln.strip():
            if re.fullmatch(r"\d+", ln.strip()):
                declared = int(ln.strip())
                lines = lines[k + 2:]  # skip the count line and the comment line
            break

    geom: Geometry = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4 or ln.lstrip().startswith("#"):
            continue
        sym_tok = parts[0]
        try:
            sym = _SYMBOLS[int(sym_tok) - 1] if sym_tok.isdigit() else normalize_symbol(sym_tok)
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
        except (ValueError, IndexError):
            continue
        geom.append((sym, xyz))

    if not geom:
        raise ValueError("no atoms parsed from XYZ input")
    if declared is not None and declared != len(geom):
        raise ValueError(
            f"XYZ header declares {declared} atoms but {len(geom)} were parsed"
        )
    return normalize_geometry(geom)


def load_geometry(source: GeometryLike) -> Geometry:
    """Resolve *source* into a geometry.

    * a :class:`Molecule`                       -> its geometry
    * a list of ``(symbol, (x, y, z))``         -> validated as-is
    * a built-in name (``"h2o"``, ``"caffeine"``, ...; see
      :data:`BUILTIN_GEOMETRIES`)
    * ``"smiles:CCO"`` or a bare SMILES         -> 3D embed via RDKit (if installed)
    * a path to a ``.xyz`` file                 -> parsed
    * a multi-line XYZ string                   -> parsed
    """
    if isinstance(source, Molecule):
        return list(source.geometry)
    if not isinstance(source, (str, os.PathLike)):
        return normalize_geometry(source)  # type: ignore[arg-type]

    s = str(source).strip()
    key = s.lower()
    if key in BUILTIN_GEOMETRIES:
        return [(sym, tuple(map(float, xyz))) for sym, xyz in BUILTIN_GEOMETRIES[key]]

    if key.startswith("smiles:"):
        return _geometry_from_smiles(s.split(":", 1)[1].strip())

    p = Path(s)
    if p.suffix.lower() == ".xyz" or (p.exists() and p.is_file()):
        return parse_xyz(p)

    if "\n" in s:
        return parse_xyz(s)

    # A single token that is not a known name: try SMILES as a last resort.
    if re.fullmatch(r"[A-Za-z0-9@+\-\[\]()/\\=#$%.]+", s):
        try:
            return _geometry_from_smiles(s)
        except Exception:  # noqa: BLE001
            pass

    raise ValueError(
        f"could not interpret geometry source {source!r}. "
        f"Known names: {', '.join(sorted(BUILTIN_GEOMETRIES))}"
    )


def _geometry_from_smiles(smiles: str) -> Geometry:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "RDKit is required to build a geometry from SMILES "
            "(`pip install rdkit`)."
        ) from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        raise RuntimeError(f"RDKit could not embed 3D coordinates for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    geom: Geometry = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        geom.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    return normalize_geometry(geom)


def format_xyz(geometry: GeometryLike, comment: str = "") -> str:
    """Return an XYZ-format string (with the ``<count>`` + comment header)."""
    geom = load_geometry(geometry)
    lines = [str(len(geom)), comment]
    for sym, (x, y, z) in geom:
        lines.append(f"{sym:<2s} {x:>15.8f} {y:>15.8f} {z:>15.8f}")
    return "\n".join(lines) + "\n"


def chemical_formula(geometry: GeometryLike) -> str:
    """Hill-notation formula, e.g. ``'C8H10N4O2'``."""
    geom = load_geometry(geometry)
    counts: Dict[str, int] = {}
    for sym, _ in geom:
        counts[sym] = counts.get(sym, 0) + 1

    def _fmt(sym: str) -> str:
        return sym + (str(counts[sym]) if counts[sym] > 1 else "")

    ordered: List[str] = []
    if "C" in counts:
        ordered.append(_fmt("C"))
        if "H" in counts:
            ordered.append(_fmt("H"))
        ordered += [_fmt(s) for s in sorted(counts) if s not in ("C", "H")]
    else:
        ordered += [_fmt(s) for s in sorted(counts)]
    return "".join(ordered)


def _positions(geometry: Geometry) -> np.ndarray:
    return np.array([xyz for _, xyz in geometry], dtype=float)


def center_of_mass(geometry: GeometryLike) -> np.ndarray:
    geom = load_geometry(geometry)
    w = np.array([atomic_mass(s) for s, _ in geom])
    return (w[:, None] * _positions(geom)).sum(0) / w.sum()


def moments_of_inertia(geometry: GeometryLike) -> np.ndarray:
    """Principal moments of inertia (amu * Angstrom^2), ascending."""
    geom = load_geometry(geometry)
    w = np.array([atomic_mass(s) for s, _ in geom])
    r = _positions(geom) - center_of_mass(geom)
    tensor = np.zeros((3, 3))
    for wi, ri in zip(w, r):
        tensor += wi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    return np.sort(np.linalg.eigvalsh(tensor))


def bounding_box(geometry: GeometryLike) -> Tuple[np.ndarray, np.ndarray]:
    r = _positions(load_geometry(geometry))
    return r.min(0), r.max(0)


def distance_matrix(geometry: GeometryLike) -> np.ndarray:
    r = _positions(load_geometry(geometry))
    diff = r[:, None, :] - r[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def covalent_bonds(
    geometry: GeometryLike, tolerance: float = 1.3, min_distance: float = 0.4
) -> List[Tuple[int, int, float]]:
    """Guess bonds from interatomic distances.

    Two atoms *i*, *j* are bonded when
    ``d_ij < tolerance * (r_cov[i] + r_cov[j])``.

    Returns a list of ``(i, j, distance)`` with ``i < j``.
    """
    geom = load_geometry(geometry)
    r = _positions(geom)
    rcov = np.array([covalent_radius(s) for s, _ in geom])
    bonds: List[Tuple[int, int, float]] = []
    for i in range(len(geom)):
        for j in range(i + 1, len(geom)):
            d = float(np.linalg.norm(r[i] - r[j]))
            if min_distance < d < tolerance * (rcov[i] + rcov[j]):
                bonds.append((i, j, d))
    return bonds


def nuclear_repulsion_energy(geometry: GeometryLike) -> float:
    """Classical nuclear repulsion energy in Hartree (a cheap cross-check for
    ``Molecule.energies['nuclear_energy']``)."""
    geom = load_geometry(geometry)
    r = _positions(geom) * BOHR_PER_ANGSTROM
    z = np.array([atomic_number(s) for s, _ in geom], dtype=float)
    e = 0.0
    for i in range(len(geom)):
        for j in range(i + 1, len(geom)):
            e += z[i] * z[j] / np.linalg.norm(r[i] - r[j])
    return float(e)


def total_electrons(geometry: GeometryLike, charge: int = 0) -> int:
    geom = load_geometry(geometry)
    return int(sum(atomic_number(s) for s, _ in geom) - charge)


# --- transforms (return new geometries) ---------------------------------------

def translate(geometry: GeometryLike, vector: Sequence[float]) -> Geometry:
    geom = load_geometry(geometry)
    v = np.asarray(vector, float)
    return [(s, tuple((np.asarray(xyz) + v).tolist())) for s, xyz in geom]


def center(geometry: GeometryLike, on: str = "com") -> Geometry:
    geom = load_geometry(geometry)
    origin = center_of_mass(geom) if on == "com" else _positions(geom).mean(0)
    return translate(geom, -origin)


def rotate(geometry: GeometryLike, axis: Sequence[float], angle_deg: float) -> Geometry:
    """Rotate about *axis* (through the origin) by *angle_deg* degrees."""
    geom = load_geometry(geometry)
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    th = math.radians(angle_deg)
    kx, ky, kz = k
    kmat = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    rot = np.eye(3) + math.sin(th) * kmat + (1 - math.cos(th)) * (kmat @ kmat)
    return [(s, tuple((rot @ np.asarray(xyz)).tolist())) for s, xyz in geom]


def set_distance(
    geometry: GeometryLike, i: int, j: int, distance: float, move: str = "j"
) -> Geometry:
    """Return a copy with the *i*--*j* distance set to *distance* (Angstrom).

    ``move='j'`` shifts atom *j* along the bond axis, ``move='both'`` moves both
    symmetrically.  Handy for rigid potential-energy-surface scans -- see
    :func:`bond_scan`.
    """
    geom = load_geometry(geometry)
    pi, pj = np.asarray(geom[i][1]), np.asarray(geom[j][1])
    axis = pj - pi
    length = float(np.linalg.norm(axis))
    unit = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    out = list(geom)
    if move == "both":
        mid = 0.5 * (pi + pj)
        out[i] = (geom[i][0], tuple((mid - 0.5 * distance * unit).tolist()))
        out[j] = (geom[j][0], tuple((mid + 0.5 * distance * unit).tolist()))
    else:
        out[j] = (geom[j][0], tuple((pi + distance * unit).tolist()))
    return out


def interpolate(
    geom_a: GeometryLike, geom_b: GeometryLike, fractions: Sequence[float]
) -> List[Geometry]:
    """Linear interpolation between two aligned geometries (same atom order)."""
    a, b = load_geometry(geom_a), load_geometry(geom_b)
    if [s for s, _ in a] != [s for s, _ in b]:
        raise ValueError("interpolate() needs identical atom ordering")
    ra, rb = _positions(a), _positions(b)
    frames: List[Geometry] = []
    for f in fractions:
        r = (1 - f) * ra + f * rb
        frames.append([(s, tuple(row)) for (s, _), row in zip(a, r.tolist())])
    return frames


# ===========================================================================
#  CUDA-Q Solvers plumbing
# ===========================================================================

_SOLVERS_HELP = (
    "NVIDIA CUDA-Q Solvers is required for molecule generation.\n"
    "  Linux / WSL:  pip install cudaq-solvers\n"
    "  Docs:         https://nvidia.github.io/cudaqx/\n"
    "It is not available on native Windows -- use WSL, a container, or a Linux\n"
    "host.  (The visualization tools in this module work without it.)"
)


def _load_solvers():
    try:
        import cudaq_solvers as solvers  # type: ignore
        return solvers
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(_SOLVERS_HELP) from exc


def _load_cudaq():
    try:
        import cudaq  # type: ignore
        return cudaq
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(_SOLVERS_HELP) from exc


def cudaq_available() -> bool:
    """``True`` if ``cudaq_solvers`` can be imported in this environment."""
    try:
        import cudaq_solvers  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_PYSCF_SERVER_URL = "http://127.0.0.1:8000"


def _pyscf_server_running() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_PYSCF_SERVER_URL}/status", timeout=0.5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _ensure_pyscf_server(timeout: float = 60.0) -> None:
    """Make sure the ``cudaq-pyscf`` REST server create_molecule() talks to is up.

    In a plain pip venv (no conda base with cudaq's bin/ preinstalled on PATH),
    cudaq_solvers' own auto-launch of this server fails silently and
    create_molecule() raises immediately -- the wheel ships the script under
    ``cudaq_solvers/bin/`` but never registers it as a console-script, so the
    C++ layer can't find it on PATH.  Launch it ourselves, once; subsequent
    calls (this process or another) just find it already listening.
    """
    if _pyscf_server_running():
        return
    import subprocess
    import time

    solvers = _load_solvers()
    script = Path(solvers.__file__).parent / "bin" / "cudaq-pyscf"
    if not script.exists():
        return  # let create_molecule() fail with its own error message
    subprocess.Popen(
        [sys.executable, str(script), "--server-mode"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pyscf_server_running():
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"timed out waiting for the cudaq-pyscf server to come up at {_PYSCF_SERVER_URL}"
    )


# MoleculeOptions field  ->  cudaq_solvers.create_molecule kwarg
_OPTION_ALIASES = {
    "active_electrons": "nele_cas",
    "active_orbitals": "norb_cas",
    "ccsd": "ccsd",
    "casci": "casci",
    "casscf": "casscf",
    "mp2": "MP2",
    "natorb": "natorb",
    "integrals_natorb": "integrals_natorb",
    "integrals_casscf": "integrals_casscf",
    "unrestricted": "UR",
    "symmetry": "symmetry",
    "memory": "memory",
    "cycles": "cycles",
    "init_guess": "initguess",
    "verbose": "verbose",
}


@dataclass
class MoleculeOptions:
    """Everything :func:`cudaq_solvers.create_molecule` accepts, with readable names.

    The four positionals (``geometry``, ``basis``, ``spin``, ``charge``) live on
    :class:`Molecule`; this bundles the keyword options.
    """

    basis: str = "sto-3g"
    spin: int = 0
    charge: int = 0
    active_electrons: Optional[int] = None       # -> nele_cas
    active_orbitals: Optional[int] = None        # -> norb_cas
    casci: bool = True                           # compute the CASCI/FCI reference
    ccsd: bool = False
    casscf: bool = False
    mp2: bool = False
    natorb: bool = False
    integrals_natorb: bool = False
    integrals_casscf: bool = False
    unrestricted: bool = False
    symmetry: bool = False
    memory: int = 4000
    cycles: int = 100
    init_guess: str = "minao"
    verbose: bool = False
    #: 'jordan_wigner' or 'bravyi_kitaev' -- applied by :meth:`Molecule.qubit_hamiltonian`.
    mapping: str = "jordan_wigner"
    #: extra kwargs passed straight through to create_molecule
    extra: Dict[str, Any] = field(default_factory=dict)

    def create_molecule_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        for name, target in _OPTION_ALIASES.items():
            if name in ("basis", "spin", "charge"):
                continue
            val = getattr(self, name, None)
            if val is None or val is False:
                continue
            kw[target] = val
        kw.update(self.extra)
        return kw


class Molecule:
    """A molecular electronic-structure problem, ready for quantum simulation.

    Construction runs the classical pre-computation through
    ``cudaq_solvers.create_molecule`` (PySCF under the hood).

    Parameters
    ----------
    geometry
        Anything :func:`load_geometry` understands: a built-in name, a geometry
        list, an ``.xyz`` path/string, or ``"smiles:..."``.
    basis
        Gaussian basis set (``"sto-3g"``, ``"6-31g"``, ``"cc-pvdz"``, ...).
    spin
        ``2 * S`` -- the number of unpaired electrons (0 for closed shell).
    charge
        Total molecular charge.
    **options
        Any field of :class:`MoleculeOptions` (``active_electrons``,
        ``active_orbitals``, ``ccsd``, ``casci``, ``casscf``, ``mp2``,
        ``mapping``, ``verbose``, ...).

    Examples
    --------
    >>> m = Molecule("h2", basis="sto-3g")
    >>> m.n_qubits
    4
    >>> abs(m.ground_state_energy_exact() - m.fci_energy) < 1e-8
    True
    >>> lih = Molecule("lih", active_electrons=2, active_orbitals=3)
    >>> pool = lih.operator_pool("spin_complement_gsd")
    """

    def __init__(
        self,
        geometry: GeometryLike,
        basis: str = "sto-3g",
        *,
        spin: int = 0,
        charge: int = 0,
        name: Optional[str] = None,
        **options: Any,
    ) -> None:
        self.geometry: Geometry = load_geometry(geometry)
        self.name = name or (str(geometry) if isinstance(geometry, str) else chemical_formula(self.geometry))

        mapping = options.pop("mapping", "jordan_wigner")
        extra = options.pop("extra", {})
        unknown = set(options) - set(MoleculeOptions.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unexpected option(s): {sorted(unknown)}")
        self.options = MoleculeOptions(
            basis=basis, spin=spin, charge=charge, mapping=mapping, extra=extra, **options
        )

        self._raw: Any = None
        self._qubit_h_cache: Dict[str, Any] = {}
        self._build()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        solvers = _load_solvers()
        _ensure_pyscf_server()
        geom = [(s, tuple(map(float, xyz))) for s, xyz in self.geometry]
        kwargs = self.options.create_molecule_kwargs()
        self._raw = solvers.create_molecule(
            geom, self.options.basis, int(self.options.spin), int(self.options.charge), **kwargs
        )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> "Molecule":
        """Build from a :data:`BUILTIN_GEOMETRIES` key."""
        return cls(name, name=name, **kwargs)

    @classmethod
    def from_xyz(cls, path_or_text: Union[str, os.PathLike], **kwargs: Any) -> "Molecule":
        return cls(parse_xyz(path_or_text), **kwargs)

    @classmethod
    def from_geometry(cls, geometry: Sequence[GeometryItem], **kwargs: Any) -> "Molecule":
        return cls(normalize_geometry(geometry), **kwargs)

    # -- the underlying CUDA-Q object -------------------------------------

    @property
    def raw(self) -> Any:
        """The bare ``cudaq_solvers`` ``MolecularHamiltonian`` object."""
        return self._raw

    @property
    def hamiltonian(self) -> Any:
        """Qubit Hamiltonian as a ``cudaq.SpinOperator`` (Jordan-Wigner by default)."""
        if self.options.mapping in ("jordan_wigner", "jw", None, ""):
            return self._raw.hamiltonian
        return self.qubit_hamiltonian(self.options.mapping)

    # -- sizes -----------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.geometry)

    @property
    def n_electrons(self) -> int:
        """Electrons in the (active) problem passed to the quantum computer."""
        return int(self._raw.n_electrons)

    @property
    def n_orbitals(self) -> int:
        """Spatial molecular orbitals in the active space."""
        return int(self._raw.n_orbitals)

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    @property
    def n_qubits(self) -> int:
        """Qubits required for the JW/BK-mapped Hamiltonian (``2 * n_orbitals``)."""
        return 2 * self.n_orbitals

    @property
    def total_electrons(self) -> int:
        """All electrons in the molecule (ignores any active-space restriction)."""
        return total_electrons(self.geometry, self.options.charge)

    @property
    def n_frozen_electrons(self) -> int:
        return self.total_electrons - self.n_electrons

    # -- energies ------------------------------------------------------

    @property
    def energies(self) -> Dict[str, float]:
        """The raw ``energies`` dict from PySCF (keys depend on the options used:
        ``hf_energy``, ``nuclear_energy``, ``core_energy``, ``fci_energy``,
        ``MP2_energy``, ``R-CCSD``, ``R-CASCI``, ``R-CASSCF``, ...)."""
        e = self._raw.energies
        return dict(e) if not isinstance(e, dict) else e

    def _energy(self, *keys: str) -> Optional[float]:
        e = self.energies
        for k in keys:
            if k in e and e[k] is not None:
                return float(e[k])
        return None

    @property
    def hf_energy(self) -> Optional[float]:
        return self._energy("hf_energy", "R-HF", "UR-HF")

    @property
    def mp2_energy(self) -> Optional[float]:
        return self._energy("MP2_energy", "mp2_energy")

    @property
    def ccsd_energy(self) -> Optional[float]:
        return self._energy("R-CCSD", "UR-CCSD", "ccsd_energy")

    @property
    def casci_energy(self) -> Optional[float]:
        return self._energy("R-CASCI", "UR-CASCI", "casci_energy")

    @property
    def casscf_energy(self) -> Optional[float]:
        return self._energy("R-CASSCF", "UR-CASSCF", "casscf_energy")

    @property
    def fci_energy(self) -> Optional[float]:
        """FCI (== CASCI in the active space) energy, if it was computed."""
        return self._energy("fci_energy", "R-CASCI", "UR-CASCI", "casci_energy")

    @property
    def nuclear_repulsion(self) -> float:
        v = self._energy("nuclear_energy", "nuclear_repulsion")
        return v if v is not None else nuclear_repulsion_energy(self.geometry)

    @property
    def core_energy(self) -> float:
        """Constant term of the active-space Hamiltonian (nuclear repulsion +
        frozen-core contribution).  This is what :func:`jordan_wigner` wants."""
        v = self._energy("core_energy")
        return v if v is not None else self.nuclear_repulsion

    def correlation_energy(self, reference: str = "hf") -> Optional[float]:
        """``E_FCI - E_ref`` (Hartree); ``reference`` in ``{'hf', 'ccsd', 'mp2'}``."""
        ref = {"hf": self.hf_energy, "ccsd": self.ccsd_energy, "mp2": self.mp2_energy}[reference]
        target = self.fci_energy
        if ref is None or target is None:
            return None
        return target - ref

    # -- integrals --------------------------------------------------------

    @property
    def one_body_integrals(self) -> np.ndarray:
        r"""One-electron integrals :math:`h_{pq}` in the MO basis (``n_orb x n_orb``)."""
        return np.asarray(self._raw.hpq)

    @property
    def two_body_integrals(self) -> np.ndarray:
        r"""Two-electron integrals :math:`h_{pqrs}` (``n_orb^4``)."""
        return np.asarray(self._raw.hpqrs)

    hpq = one_body_integrals
    hpqrs = two_body_integrals

    # -- qubit Hamiltonian & spectra -----------------------------------

    def qubit_hamiltonian(self, mapping: str = "jordan_wigner", tol: float = 1e-13) -> Any:
        """(Re)build the qubit Hamiltonian from the integrals.

        ``mapping`` is ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        """
        mapping = {"jw": "jordan_wigner", "bk": "bravyi_kitaev"}.get(mapping, mapping)
        if mapping in self._qubit_h_cache:
            return self._qubit_h_cache[mapping]
        solvers = _load_solvers()
        try:
            fn = getattr(solvers, mapping)
        except AttributeError as exc:  # noqa: BLE001
            raise ValueError(
                f"unknown mapping {mapping!r} (expected 'jordan_wigner' or 'bravyi_kitaev')"
            ) from exc
        hpq, hpqrs, core = self.one_body_integrals, self.two_body_integrals, self.core_energy
        op = None
        for call in (lambda: fn(hpq, hpqrs, core, tol=tol),
                     lambda: fn(hpq, hpqrs, core, tolerance=tol),
                     lambda: fn(hpq, hpqrs, core),
                     lambda: fn(hpq, hpqrs)):
            try:
                op = call()
                break
            except TypeError:
                continue
        if op is None:
            raise TypeError(f"could not call cudaq_solvers.{mapping} with the integrals")
        self._qubit_h_cache[mapping] = op
        return op

    def pauli_terms(self) -> List[Tuple[str, complex]]:
        """Best-effort ``[(pauli_word, coefficient), ...]`` for the Hamiltonian.

        ``pauli_word`` is an ``n_qubits``-long string over ``{I, X, Y, Z}``.
        The exact iteration API of ``cudaq.SpinOperator`` shifts between CUDA-Q
        releases, so this tries a few strategies and finally parses the string
        form.
        """
        H = self.hamiltonian
        n = self.n_qubits

        def _clean(word: str) -> str:
            word = re.sub(r"[^IXYZ]", "", word.strip().upper())
            if not word:
                return "I" * n
            return word.ljust(n, "I") if len(word) <= n else word

        # Strategy 1: iterate terms.
        try:
            terms: List[Tuple[str, complex]] = []
            for term in H:
                coeff = complex(term.get_coefficient())
                word = None
                for getter, arg in (("get_pauli_word", (n,)), ("to_string", (False,)),
                                    ("get_pauli_word", ()), ("to_string", ())):
                    fn = getattr(term, getter, None)
                    if fn is None:
                        continue
                    try:
                        word = fn(*arg)
                        break
                    except Exception:  # noqa: BLE001
                        continue
                terms.append((_clean(word if word is not None else str(term)), coeff))
            if terms:
                return terms
        except Exception:  # noqa: BLE001
            pass

        # Strategy 2: parse H.to_string().
        text = H.to_string() if hasattr(H, "to_string") else str(H)
        out: List[Tuple[str, complex]] = []
        pattern = re.compile(
            r"\(?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,?\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?\s*\)?\s*([IXYZ]+)"
        )
        for m in pattern.finditer(text):
            re_part = float(m.group(1))
            im_part = float(m.group(2)) if m.group(2) else 0.0
            out.append((_clean(m.group(3)), complex(re_part, im_part)))
        if not out:
            raise RuntimeError(
                "could not extract Pauli terms from this cudaq.SpinOperator; "
                "inspect `self.hamiltonian` directly."
            )
        return out

    @property
    def n_pauli_terms(self) -> int:
        H = self.hamiltonian
        for attr in ("get_term_count", "term_count"):
            fn = getattr(H, attr, None)
            if fn is None:
                continue
            try:
                return int(fn() if callable(fn) else fn)
            except Exception:  # noqa: BLE001
                pass
        return len(self.pauli_terms())

    def hamiltonian_matrix(self, sparse: bool = False):
        """Dense ``ndarray`` (or SciPy CSR if ``sparse=True``) of the Hamiltonian.

        Dimension is ``2**n_qubits`` -- only sensible for small active spaces.
        """
        H = self.hamiltonian
        if not sparse:
            for meth in ("to_matrix", "to_dense", "matrix"):
                fn = getattr(H, meth, None)
                if fn is None:
                    continue
                try:
                    return np.array(fn(), dtype=complex, copy=False)
                except Exception:  # noqa: BLE001
                    continue
        return self._matrix_from_terms(sparse=sparse)

    def _matrix_from_terms(self, sparse: bool = True):
        import scipy.sparse as sp

        n = self.n_qubits
        if n > 24:
            raise ValueError(f"{n} qubits is too large for explicit matrix construction")
        paulis = {
            "I": sp.identity(2, format="csr", dtype=complex),
            "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
            "Y": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
            "Z": sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
        }
        dim = 1 << n
        total = sp.csr_matrix((dim, dim), dtype=complex)
        for word, coeff in self.pauli_terms():
            acc = paulis[word[0]]
            for ch in word[1:]:
                acc = sp.kron(acc, paulis[ch], format="csr")
            total = total + coeff * acc
        return total if sparse else total.toarray()

    def ground_state_energy_exact(
        self, method: str = "auto", max_dense_qubits: int = 12
    ) -> float:
        """Lowest eigenvalue of the qubit Hamiltonian -- the *numerical* reference
        that VQE / ADAPT-VQE / GQE should converge to.

        ``method``: ``'auto'`` (dense for small systems, sparse Lanczos otherwise),
        ``'dense'``, or ``'sparse'``.
        """
        n = self.n_qubits
        if method == "dense" or (method == "auto" and n <= max_dense_qubits):
            mat = self.hamiltonian_matrix(sparse=False)
            return float(np.linalg.eigvalsh(mat)[0].real)
        from scipy.sparse.linalg import eigsh

        mat = self.hamiltonian_matrix(sparse=True)
        val = eigsh(mat, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
        return float(np.real(val[0]))

    def spectrum(self, k: Optional[int] = None) -> np.ndarray:
        """Sorted (ascending) eigenvalues of the Hamiltonian; ``k`` lowest if given."""
        mat = self.hamiltonian_matrix(sparse=False)
        w = np.linalg.eigvalsh(mat)
        return np.real(w[:k] if k else w)

    # -- state preparation helpers -------------------------------------

    def hartree_fock_occupation(self) -> List[int]:
        """Jordan-Wigner occupation vector of the HF reference state
        (``[1] * n_electrons + [0] * ...``), length ``n_qubits``."""
        return [1] * self.n_electrons + [0] * (self.n_qubits - self.n_electrons)

    def hartree_fock_bitstring(self) -> str:
        return "".join(map(str, self.hartree_fock_occupation()))

    def num_uccsd_parameters(self) -> int:
        solvers = _load_solvers()
        return int(
            solvers.stateprep.get_num_uccsd_parameters(
                self.n_electrons, self.n_qubits, self.options.spin
            )
        )

    def uccsd_excitations(self):
        """``(singles_alpha, singles_beta, doubles_...)`` index lists from
        ``cudaq_solvers.stateprep.get_uccsd_excitations``."""
        solvers = _load_solvers()
        return solvers.stateprep.get_uccsd_excitations(
            self.n_electrons, self.n_qubits, self.options.spin
        )

    def operator_pool(self, name: str = "uccsd", **config: Any) -> List[Any]:
        """ADAPT-VQE / GQE operator pool from ``cudaq_solvers.get_operator_pool``.

        Common names: ``'uccsd'``, ``'spin_complement_gsd'``.  Sensible defaults
        for ``num_qubits`` / ``num_electrons`` / ``num_orbitals`` / ``spin`` are
        filled in from this molecule.
        """
        solvers = _load_solvers()
        defaults = {
            "num_qubits": self.n_qubits,
            "num_electrons": self.n_electrons,
            "num_orbitals": self.n_orbitals,
            "spin": self.options.spin,
        }
        for key, val in defaults.items():
            config.setdefault(key, val)
        try:
            return list(solvers.get_operator_pool(name, **config))
        except TypeError:
            # Older/newer signatures accept only a subset of the kwargs.
            for drop in ("spin", "num_orbitals", "num_electrons", "num_qubits"):
                config.pop(drop, None)
                try:
                    return list(solvers.get_operator_pool(name, **config))
                except TypeError:
                    continue
            raise

    # -- interop -------------------------------------------------------

    def to_openfermion(self):
        """Return the qubit Hamiltonian as an OpenFermion ``QubitOperator``."""
        try:
            from openfermion import QubitOperator
        except ImportError as exc:  # noqa: BLE001
            raise ImportError("`pip install openfermion` for OpenFermion export") from exc
        op = QubitOperator()
        for word, coeff in self.pauli_terms():
            term = tuple((i, p) for i, p in enumerate(word) if p != "I")
            op += QubitOperator(term, complex(coeff))
        return op

    def save(self, path: Union[str, os.PathLike]) -> Path:
        """Pickle the *portable* data (geometry, options, energies, integrals,
        Pauli terms).  The C++ ``MolecularHamiltonian`` itself is not pickled."""
        payload = {
            "version": __version__,
            "name": self.name,
            "geometry": self.geometry,
            "options": asdict(self.options),
            "n_electrons": self.n_electrons,
            "n_orbitals": self.n_orbitals,
            "energies": self.energies,
            "hpq": self.one_body_integrals,
            "hpqrs": self.two_body_integrals,
        }
        try:
            payload["pauli_terms"] = self.pauli_terms()
        except Exception:  # noqa: BLE001
            pass
        path = Path(path)
        path.write_bytes(pickle.dumps(payload))
        return path

    # -- reporting ----------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "formula": chemical_formula(self.geometry),
            "basis": self.options.basis,
            "charge": self.options.charge,
            "spin (2S)": self.options.spin,
            "atoms": self.n_atoms,
            "active electrons": self.n_electrons,
            "active orbitals": self.n_orbitals,
            "qubits": self.n_qubits,
            "Pauli terms": _safe(lambda: self.n_pauli_terms),
            "E(HF)": self.hf_energy,
            "E(MP2)": self.mp2_energy,
            "E(CCSD)": self.ccsd_energy,
            "E(CASCI/FCI)": self.fci_energy,
            "E(nuclear)": self.nuclear_repulsion,
        }

    def print_summary(self, file=None) -> None:
        rows = self.summary()
        width = max(len(k) for k in rows)
        print(f"Molecule: {self.name}", file=file or sys.stdout)
        print("-" * (width + 24), file=file or sys.stdout)
        for k, v in rows.items():
            if isinstance(v, float):
                v = f"{v:.10f}"
            print(f"  {k:<{width}} : {v}", file=file or sys.stdout)

    def visualize(self, **kwargs: Any):
        """Shortcut for :func:`visualize(self, **kwargs) <visualize>`."""
        return visualize(self.geometry, **kwargs)

    def __repr__(self) -> str:
        try:
            return (
                f"<Molecule {self.name!r} {chemical_formula(self.geometry)} "
                f"basis={self.options.basis!r} qubits={self.n_qubits} "
                f"e-={self.n_electrons}>"
            )
        except Exception:  # noqa: BLE001
            return f"<Molecule {self.name!r} (unbuilt)>"


def load_molecule(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Inverse of :meth:`Molecule.save` -- returns the stored dict."""
    return pickle.loads(Path(path).read_bytes())


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


# ===========================================================================
#  Potential-energy-surface scan
# ===========================================================================

def bond_scan(
    geometry: GeometryLike,
    pair: Tuple[int, int],
    distances: Iterable[float],
    *,
    basis: str = "sto-3g",
    spin: int = 0,
    charge: int = 0,
    energies: Sequence[str] = ("hf_energy", "fci_energy"),
    exact: bool = False,
    progress: bool = True,
    **molecule_options: Any,
) -> Dict[str, np.ndarray]:
    """Rigid 1D potential-energy-surface scan.

    Sets the distance between atoms ``pair = (i, j)`` to each value in
    *distances*, rebuilds the :class:`Molecule`, and collects energies.

    Returns a dict of equal-length arrays: ``'distance'``, one entry per name in
    *energies* (``'hf_energy'``, ``'fci_energy'``, ``'ccsd_energy'``, ...), and
    ``'exact'`` if ``exact=True``.

    >>> res = bond_scan("h2", (0, 1), np.linspace(0.4, 2.5, 22))
    >>> res["distance"][res["fci_energy"].argmin()]      # ~0.74 Angstrom
    """
    base = load_geometry(geometry)
    i, j = pair
    dists = [float(d) for d in distances]
    cols: Dict[str, List[Optional[float]]] = {k: [] for k in energies}
    if exact:
        cols["exact"] = []

    for n, d in enumerate(dists):
        if progress:
            print(f"  [{n + 1:>3}/{len(dists)}] r({base[i][0]}{i}-{base[j][0]}{j}) = {d:.4f} A",
                  file=sys.stderr)
        mol = Molecule(set_distance(base, i, j, d), basis=basis, spin=spin,
                       charge=charge, **molecule_options)
        emap = mol.energies
        for key in energies:
            if key in ("fci_energy", "casci_energy"):
                cols[key].append(mol.fci_energy)
            elif key in ("ccsd_energy",):
                cols[key].append(mol.ccsd_energy)
            elif key in ("hf_energy",):
                cols[key].append(mol.hf_energy)
            elif key in ("mp2_energy",):
                cols[key].append(mol.mp2_energy)
            else:
                cols[key].append(float(emap[key]) if key in emap else None)
        if exact:
            cols["exact"].append(_safe(mol.ground_state_energy_exact))

    out: Dict[str, np.ndarray] = {"distance": np.array(dists)}
    for key, vals in cols.items():
        out[key] = np.array([np.nan if v is None else v for v in vals], dtype=float)
    return out


# ===========================================================================
#  Visualization
# ===========================================================================

_STYLES = ("ball_and_stick", "stick", "spacefill", "wireframe", "sphere")


def _py3dmol_style(style: str) -> Dict[str, Any]:
    style = style.lower()
    scheme = {"colorscheme": "Jmol"}
    if style in ("ball_and_stick", "ballandstick", "bs"):
        return {"stick": {"radius": 0.12, **scheme}, "sphere": {"scale": 0.25, **scheme}}
    if style == "stick":
        return {"stick": {"radius": 0.15, **scheme}}
    if style in ("spacefill", "sphere", "cpk", "vdw"):
        return {"sphere": {**scheme}}
    if style in ("wireframe", "line"):
        return {"line": {**scheme}}
    raise ValueError(f"unknown style {style!r}; choose from {_STYLES}")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:  # noqa: BLE001
        return False


def view_py3dmol(
    geometry: GeometryLike,
    *,
    style: str = "ball_and_stick",
    width: int = 640,
    height: int = 480,
    background: str = "white",
    show_labels: bool = False,
    label_kind: str = "element",   # 'element' | 'index' | 'element_index'
    show_axes: bool = False,
    spin: bool = False,
    surface: Optional[str] = None,   # 'VDW' | 'SAS' | 'MS' | 'SES'
    title: Optional[str] = None,
):
    """Build an interactive ``py3Dmol`` view (3Dmol.js).  Returns the ``view``;
    call ``.show()`` in a notebook or ``.write_html(path)`` to save."""
    try:
        import py3Dmol  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("`pip install py3Dmol` for the py3Dmol backend") from exc

    geom = load_geometry(geometry)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(format_xyz(geom, title or ""), "xyz")
    view.setStyle(_py3dmol_style(style))
    view.setBackgroundColor(background)

    if show_labels:
        for idx, (sym, _) in enumerate(geom):
            text = {"element": sym, "index": str(idx),
                    "element_index": f"{sym}{idx}"}[label_kind]
            view.addLabel(text, {"fontSize": 11, "backgroundOpacity": 0.35,
                                 "inFront": True}, {"serial": idx})
    if surface:
        import py3Dmol  # noqa: F811

        view.addSurface(getattr(py3Dmol, surface, surface),
                        {"opacity": 0.7, "colorscheme": "whiteCarbon"})
    if show_axes:
        _add_axes_py3dmol(view, geom)
    view.zoomTo()
    if spin:
        view.spin(True)
    return view


def _add_axes_py3dmol(view, geom: Geometry) -> None:
    lo, hi = _positions(geom).min(0), _positions(geom).max(0)
    L = float(np.max(hi - lo)) * 0.6 + 1.0
    for vec, color in (((L, 0, 0), "red"), ((0, L, 0), "green"), ((0, 0, L), "blue")):
        view.addArrow({"start": {"x": 0, "y": 0, "z": 0},
                       "end": {"x": vec[0], "y": vec[1], "z": vec[2]},
                       "radius": 0.04, "color": color})


def view_plotly(
    geometry: GeometryLike,
    *,
    style: str = "ball_and_stick",
    width: int = 720,
    height: int = 560,
    background: Optional[str] = None,
    show_labels: bool = False,
    label_kind: str = "element_index",
    show_axes: bool = False,
    bond_tolerance: float = 1.3,
    title: Optional[str] = None,
):
    """Interactive 3D view as a Plotly ``Figure`` (rotate / zoom / pan; works in
    notebooks and as standalone HTML via ``fig.write_html``).

    Portable and dependency-light, but Plotly's marker sizes are in screen
    pixels and do not rescale on zoom -- for publication-quality figures prefer
    the ``py3dmol`` / ``html`` backends.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("`pip install plotly` for the plotly backend") from exc

    geom = load_geometry(geometry)
    r = _positions(geom)
    syms = [s for s, _ in geom]
    bonds = covalent_bonds(geom, tolerance=bond_tolerance)

    dark = _is_dark(background if background is not None else "#0f1117")
    bg = background if background is not None else "#0f1117"
    fg = "#e8e8f0" if dark else "#1a1a22"
    edge = "rgba(245,245,255,0.35)" if dark else "rgba(20,20,25,0.55)"

    if style in ("spacefill", "sphere", "cpk", "vdw"):
        def _size(s: str) -> float:
            return 8.0 + 26.0 * vdw_radius(s)
    elif style == "stick":
        def _size(s: str) -> float:
            return 10.0
    elif style in ("wireframe", "line"):
        def _size(s: str) -> float:
            return 4.0
    else:  # ball_and_stick
        def _size(s: str) -> float:
            return 6.0 + 20.0 * covalent_radius(s)

    bond_width = {"ball_and_stick": 8, "stick": 13, "wireframe": 4}.get(style, 8)

    traces = []

    # Bonds: split each bond at its midpoint so each half takes its atom's
    # color, then group half-bonds by color into one trace apiece.
    if style not in ("spacefill", "sphere"):
        by_color: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for i, j, _ in bonds:
            mid = (r[i] + r[j]) / 2.0
            by_color.setdefault(cpk_color(syms[i]), []).append((r[i], mid))
            by_color.setdefault(cpk_color(syms[j]), []).append((r[j], mid))
        for color, segs in by_color.items():
            xs, ys, zs = [], [], []
            for p, q in segs:
                xs += [p[0], q[0], None]
                ys += [p[1], q[1], None]
                zs += [p[2], q[2], None]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=color, width=bond_width),
                hoverinfo="skip", showlegend=False,
            ))

    # Atoms, one trace per element (doubles as a legend).
    for sym in sorted(set(syms), key=atomic_number):
        idx = [k for k, s in enumerate(syms) if s == sym]
        mode = "markers+text" if show_labels else "markers"
        text = None
        if show_labels:
            text = [{"element": sym, "index": str(k), "element_index": f"{sym}{k}"}[label_kind]
                    for k in idx]
        traces.append(go.Scatter3d(
            x=r[idx, 0], y=r[idx, 1], z=r[idx, 2],
            mode=mode, name=sym, text=text, textposition="top center",
            textfont=dict(color=fg, size=11),
            marker=dict(
                size=_size(sym),
                color=cpk_color(sym),
                line=dict(color=edge, width=1.5),
                opacity=1.0,
            ),
            hovertext=[f"{sym}{k}  ({r[k, 0]:.3f}, {r[k, 1]:.3f}, {r[k, 2]:.3f}) A"
                       for k in idx],
            hoverinfo="text",
        ))

    fig = go.Figure(data=traces)
    axis_kw = dict(showbackground=False, showgrid=show_axes, zeroline=show_axes,
                   showticklabels=show_axes, visible=show_axes,
                   color=fg, title=dict(text=""))
    fig.update_layout(
        width=width, height=height,
        title=dict(text=title or chemical_formula(geom), font=dict(color=fg)),
        showlegend=True,
        legend=dict(font=dict(color=fg)),
        scene=dict(xaxis=axis_kw, yaxis=axis_kw, zaxis=axis_kw,
                   aspectmode="data", dragmode="orbit", bgcolor=bg),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=bg,
    )
    return fig


def _is_dark(color: str) -> bool:
    """Rough luminance test for a hex / named background color."""
    c = color.strip().lower()
    named_dark = {"black", "navy", "midnightblue", "#000", "#000000"}
    if c in named_dark:
        return True
    m = re.fullmatch(r"#?([0-9a-f]{6})", c)
    if not m:
        return c not in ("white", "#fff", "#ffffff", "ivory", "snow")
    r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 128


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font: 14px/1.4 -apple-system, Segoe UI, Roboto, sans-serif; }
  #app { position: fixed; inset: 0; }
  #viewer { width: 100%; height: 100%; position: relative; }
  #panel {
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(20,22,34,.82); color: #eef; backdrop-filter: blur(6px);
    border-radius: 10px; padding: 10px 12px; max-width: 260px;
    box-shadow: 0 6px 24px rgba(0,0,0,.35);
  }
  #panel h1 { font-size: 15px; margin: 0 0 2px; }
  #panel .sub { opacity: .7; font-size: 12px; margin-bottom: 8px; }
  #panel button {
    font: inherit; color: #eef; background: #33365a; border: 1px solid #4a4e7a;
    border-radius: 7px; padding: 4px 8px; margin: 2px 2px 0 0; cursor: pointer;
  }
  #panel button:hover { background: #43477a; }
  #panel button.on { background: #5b62d6; border-color: #7b82ff; }
  #hint { position: absolute; bottom: 10px; left: 12px; z-index: 10;
          color: #99a; font-size: 11px; }
</style>
</head>
<body>
<div id="app">
  <div id="viewer"></div>
  <div id="panel">
    <h1>@@NAME@@</h1>
    <div class="sub">@@FORMULA@@ &middot; @@NATOMS@@ atoms</div>
    <div>
      <button data-style="ball_and_stick">Ball &amp; stick</button>
      <button data-style="stick">Stick</button>
      <button data-style="spacefill">Spacefill</button>
      <button data-style="wireframe">Wireframe</button>
    </div>
    <div style="margin-top:6px">
      <button id="spin">Spin</button>
      <button id="labels">Labels</button>
      <button id="reset">Reset view</button>
    </div>
  </div>
  <div id="hint">drag to rotate &nbsp;&middot;&nbsp; scroll to zoom &nbsp;&middot;&nbsp; right-drag to pan</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
        onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js'"></script>
<script>
  const XYZ = @@XYZ_JSON@@;
  const STYLES = {
    ball_and_stick: { stick: { radius: 0.12, colorscheme: "Jmol" }, sphere: { scale: 0.25, colorscheme: "Jmol" } },
    stick:          { stick: { radius: 0.15, colorscheme: "Jmol" } },
    spacefill:      { sphere: { colorscheme: "Jmol" } },
    wireframe:      { line:  { colorscheme: "Jmol" } }
  };
  function boot() {
    if (!window.$3Dmol) { return setTimeout(boot, 60); }
    const viewer = $3Dmol.createViewer("viewer", { backgroundColor: "@@BACKGROUND@@" });
    viewer.addModel(XYZ, "xyz");
    let current = STYLES["@@STYLE@@"] ? "@@STYLE@@" : "ball_and_stick";
    let labelsOn = false, spinning = false;
    document.querySelectorAll("#panel button[data-style]").forEach(
      b => b.classList.toggle("on", b.dataset.style === current));
    const apply = () => {
      viewer.setStyle({}, STYLES[current]);
      viewer.removeAllLabels();
      if (labelsOn) {
        viewer.getModel().selectedAtoms({}).forEach(a => viewer.addLabel(
          a.elem + a.serial,
          { fontSize: 11, backgroundOpacity: 0.35, inFront: true },
          { serial: a.serial }));
      }
      viewer.render();
    };
    apply(); viewer.zoomTo(); viewer.render();
    document.querySelectorAll("#panel button[data-style]").forEach(b => {
      b.onclick = () => {
        document.querySelectorAll("#panel button[data-style]").forEach(x => x.classList.remove("on"));
        b.classList.add("on"); current = b.dataset.style; apply();
      };
    });
    document.getElementById("spin").onclick = e => {
      spinning = !spinning; e.target.classList.toggle("on", spinning);
      viewer.spin(spinning ? "y" : false);
    };
    document.getElementById("labels").onclick = e => {
      labelsOn = !labelsOn; e.target.classList.toggle("on", labelsOn); apply();
    };
    document.getElementById("reset").onclick = () => { viewer.zoomTo(); viewer.render(); };
    addEventListener("resize", () => viewer.resize());
  }
  boot();
</script>
</body>
</html>
"""


_STYLE_CANON = {
    "ball_and_stick": "ball_and_stick", "ballandstick": "ball_and_stick", "bs": "ball_and_stick",
    "stick": "stick",
    "spacefill": "spacefill", "sphere": "spacefill", "cpk": "spacefill", "vdw": "spacefill",
    "wireframe": "wireframe", "line": "wireframe",
}


def to_3dmol_html(
    geometry: GeometryLike,
    *,
    name: Optional[str] = None,
    background: str = "white",
    title: Optional[str] = None,
    style: str = "ball_and_stick",
) -> str:
    """A **self-contained** interactive HTML page (3Dmol.js from a CDN, no Python
    deps).  Style switcher, spin, labels and reset are built in."""
    import html as _html
    import json

    geom = load_geometry(geometry)
    nm = name or (title or chemical_formula(geom))
    subs = {
        "@@TITLE@@": _html.escape(title or f"{nm} - 3D"),
        "@@NAME@@": _html.escape(str(nm)),
        "@@FORMULA@@": _html.escape(chemical_formula(geom)),
        "@@NATOMS@@": str(len(geom)),
        "@@XYZ_JSON@@": json.dumps(format_xyz(geom, str(nm))),
        "@@BACKGROUND@@": background,
        "@@STYLE@@": _STYLE_CANON.get(style.lower(), "ball_and_stick"),
    }
    page = _HTML_TEMPLATE
    for token, value in subs.items():
        page = page.replace(token, value)
    return page


def visualize(
    obj: GeometryLike,
    *,
    backend: str = "auto",
    style: str = "ball_and_stick",
    filename: Optional[Union[str, os.PathLike]] = None,
    width: int = 720,
    height: int = 560,
    background: Optional[str] = None,
    show_labels: bool = False,
    show_axes: bool = False,
    spin: bool = False,
    title: Optional[str] = None,
    auto_open: bool = True,
    **backend_kwargs: Any,
):
    """Render a molecule in interactive, rotatable 3D.

    Parameters
    ----------
    obj
        A :class:`Molecule`, a geometry list, a built-in name, or an XYZ
        path/string.
    backend
        ``'auto'`` (py3Dmol in notebooks, else plotly, else a standalone HTML
        page), or force ``'py3dmol'`` / ``'plotly'`` / ``'html'``.
    style
        ``'ball_and_stick'`` | ``'stick'`` | ``'spacefill'`` | ``'wireframe'``.
    filename
        If given, write an interactive HTML file there and return its path.
    background
        CSS color for the canvas.  Defaults to white for the 3Dmol backends and
        a dark slate for plotly (both look best that way).
    auto_open
        Outside notebooks, open the rendered HTML in a browser.

    Returns
    -------
    * in a notebook: the ``py3Dmol`` view or Plotly ``Figure`` (renders inline)
    * otherwise: the :class:`~pathlib.Path` to the written HTML file
    """
    bg_html = background if background is not None else "white"
    geom = load_geometry(obj)
    name = title
    if name is None and isinstance(obj, Molecule):
        name = obj.name
    if name is None and isinstance(obj, str):
        name = obj

    chosen = backend.lower()
    if chosen == "auto":
        if _in_notebook() and _module_exists("py3Dmol"):
            chosen = "py3dmol"
        elif _module_exists("plotly"):
            chosen = "plotly"
        elif _module_exists("py3Dmol"):
            chosen = "py3dmol"
        else:
            chosen = "html"

    # ---- notebook inline render -------------------------------------
    if _in_notebook() and filename is None:
        if chosen == "py3dmol":
            v = view_py3dmol(geom, style=style, width=width, height=height,
                             background=bg_html, show_labels=show_labels,
                             show_axes=show_axes, spin=spin, title=name, **backend_kwargs)
            v.show()
            return v
        if chosen == "plotly":
            fig = view_plotly(geom, style=style, width=width, height=height,
                              background=background, show_labels=show_labels,
                              show_axes=show_axes, title=name, **backend_kwargs)
            fig.show()
            return fig

    # ---- write an HTML file ---------------------------------------
    path = Path(filename) if filename else Path(
        tempfile.gettempdir()) / f"molecule_{chemical_formula(geom)}_{os.getpid()}.html"

    if chosen == "plotly":
        fig = view_plotly(geom, style=style, width=width, height=height,
                          background=background, show_labels=show_labels,
                          show_axes=show_axes, title=name, **backend_kwargs)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    elif chosen == "py3dmol":
        v = view_py3dmol(geom, style=style, width=width, height=height,
                         background=bg_html, show_labels=show_labels,
                         show_axes=show_axes, spin=spin, title=name, **backend_kwargs)
        try:
            v.write_html(str(path))
        except Exception:  # noqa: BLE001
            path.write_text(to_3dmol_html(geom, name=name, background=bg_html,
                                          title=name, style=style), encoding="utf-8")
    else:  # 'html'
        path.write_text(
            to_3dmol_html(geom, name=name, background=bg_html, title=name, style=style),
            encoding="utf-8")

    if auto_open and not _in_notebook():
        webbrowser.open(path.resolve().as_uri())
    return path


def save_html(obj: GeometryLike, path: Union[str, os.PathLike], **kwargs: Any) -> Path:
    """Write a self-contained interactive HTML page (3Dmol.js).  Convenience
    wrapper over :func:`visualize` with ``backend='html'``."""
    kwargs.setdefault("backend", "html")
    kwargs.setdefault("auto_open", False)
    return visualize(obj, filename=path, **kwargs)


def save_image(
    obj: GeometryLike,
    path: Union[str, os.PathLike],
    *,
    style: str = "ball_and_stick",
    width: int = 900,
    height: int = 700,
    scale: float = 2.0,
    **kwargs: Any,
) -> Path:
    """Write a static PNG/SVG/PDF via Plotly + Kaleido (``pip install kaleido``)."""
    fig = view_plotly(obj, style=style, width=width, height=height, **kwargs)
    try:
        fig.write_image(str(path), scale=scale)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "static image export needs kaleido (`pip install kaleido`)"
        ) from exc
    return Path(path)


def _module_exists(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# ===========================================================================
#  __all__
# ===========================================================================

__all__ = [
    # periodic table
    "ATOMIC_NUMBER", "ATOMIC_MASS", "COVALENT_RADII", "VDW_RADII", "CPK_COLORS",
    "atomic_number", "atomic_mass", "covalent_radius", "vdw_radius", "cpk_color",
    "normalize_symbol",
    # geometry
    "Geometry", "GeometryItem", "BUILTIN_GEOMETRIES",
    "load_geometry", "normalize_geometry", "parse_xyz", "format_xyz",
    "chemical_formula", "covalent_bonds", "distance_matrix",
    "nuclear_repulsion_energy", "total_electrons",
    "center_of_mass", "moments_of_inertia", "bounding_box",
    "translate", "center", "rotate", "set_distance", "interpolate",
    # molecule generation
    "Molecule", "MoleculeOptions", "load_molecule", "bond_scan",
    "cudaq_available",
    # visualization
    "visualize", "view_py3dmol", "view_plotly", "to_3dmol_html",
    "save_html", "save_image",
]


# ===========================================================================
#  CLI / smoke demo
# ===========================================================================

def _demo(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Molecule generation + 3D visualization (CSE 402).")
    parser.add_argument("target", nargs="?", default="caffeine",
                        help="built-in name, .xyz path, or 'smiles:...'")
    parser.add_argument("--style", default="ball_and_stick", choices=list(_STYLES))
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "py3dmol", "plotly", "html"])
    parser.add_argument("--out", default=None, help="write interactive HTML here")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--quantum", action="store_true",
                        help="also build the qubit Hamiltonian (needs cudaq-solvers)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    geom = load_geometry(args.target)
    print(f"target        : {args.target}")
    print(f"formula       : {chemical_formula(geom)}")
    print(f"atoms         : {len(geom)}")
    print(f"bonds (guess) : {len(covalent_bonds(geom))}")
    print(f"E_nuc (calc)  : {nuclear_repulsion_energy(geom):.8f} Hartree")
    print(f"electrons     : {total_electrons(geom)}")

    out = args.out or str(Path(tempfile.gettempdir()) / f"{chemical_formula(geom)}_3d.html")
    path = visualize(geom, backend=args.backend, style=args.style, filename=out,
                     title=args.target, auto_open=not args.no_open)
    print(f"viewer        : {path}")

    if args.quantum:
        if not cudaq_available():
            print("\ncudaq-solvers not installed -- skipping the quantum part.")
            print(_SOLVERS_HELP)
            return 0
        print("\nbuilding qubit Hamiltonian ...")
        mol = Molecule(geom, basis=args.basis)
        mol.print_summary()
        try:
            print(f"  exact E0 (numerical) : {mol.ground_state_energy_exact():.10f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  exact diagonalization skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo(sys.argv[1:]))
