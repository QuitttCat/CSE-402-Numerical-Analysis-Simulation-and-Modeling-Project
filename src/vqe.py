"""
vqe.py
======

The Variational Quantum Eigensolver (VQE), explained and implemented without
assuming you know any quantum computing.

WHAT PROBLEM IS THIS SOLVING?  (a numerical-methods view, not a physics one)
-----------------------------------------------------------------------------
`molecule.py` builds a Hamiltonian matrix H for a molecule (see
``Molecule.hamiltonian`` / ``Molecule.hamiltonian_matrix()``).  The quantity a
chemist actually wants is the molecule's ground-state energy, which is just
the *smallest eigenvalue* of H:

    E_0 = min spectrum(H)

For a small molecule this is a plain linear-algebra problem -- that is
literally what ``Molecule.ground_state_energy_exact()`` in ``molecule.py``
does (``numpy.linalg.eigvalsh`` / Lanczos).  It is exact, but the matrix has
size ``2**n_qubits x 2**n_qubits``, so it explodes exponentially and is only
feasible for small toy systems (a dozen or so qubits on a laptop).

VQE is a *different, approximate* way to estimate the same smallest
eigenvalue, one that scales far better and is the reason "quantum computers
for chemistry" is a research field at all.  It rests on one classical fact
you already know from linear algebra -- the **Rayleigh quotient** /
variational principle:

    For ANY unit vector |psi>,          <psi| H |psi>  >=  E_0.

Equality holds only when |psi> is (a) an eigenvector of H belonging to the
smallest eigenvalue.  So: if we can *cheaply prepare* a family of trial
vectors |psi(theta)> indexed by some tunable numbers theta = (theta_1, ...,
theta_p), and *cheaply evaluate* the expectation value

    E(theta) = <psi(theta)| H |psi(theta)>,

then minimizing E(theta) over theta with an ordinary numerical optimizer
gives us the best estimate of E_0 achievable with that family of vectors.
That is the entire algorithm:

    1.  ANSATZ      -- a recipe that turns numbers theta into a trial state
                        |psi(theta)>.  Here it is a small quantum circuit
                        (`build_hardware_efficient_ansatz` /
                        `build_uccsd_ansatz` below), simulated exactly on a
                        classical CPU by CUDA-Q -- no real quantum hardware
                        involved, this is a numerical simulation project.
    2.  ENERGY EVAL -- feed |psi(theta)> and H into ``cudaq.observe(...)``,
                        which returns the Rayleigh quotient <psi(theta)|H|psi(theta)>.
                        From the optimizer's point of view this is just an
                        ordinary (expensive, black-box) scalar function
                        theta -> E(theta), exactly like a function you'd
                        hand to ``scipy.optimize.minimize``.
    3.  OPTIMIZER   -- any numerical minimizer drives theta downhill:
                        COBYLA, Nelder-Mead, Powell, BFGS (all from
                        ``scipy.optimize``), or the hand-written gradient
                        descent in this file (`gradient_descent_optimizer`),
                        which differentiates E(theta) using the *parameter-shift
                        rule* -- an exact analogue of a central finite
                        difference that happens to be exact (not just
                        approximate) for the rotation gates used here.

So: VQE = "Rayleigh-quotient minimization by a classical numerical
optimizer, where the trial vectors come from a quantum circuit simulator
instead of a formula."  Nothing about the optimizer or the underlying math
is quantum -- only the (very efficient) way trial states are represented and
evaluated is.

WHAT THIS FILE GIVES YOU
-------------------------
    >>> from molecule import Molecule
    >>> from vqe import VQE
    >>> h2 = Molecule("h2", basis="sto-3g")
    >>> result = VQE(h2, ansatz="uccsd", optimizer="cobyla").run()
    ... # a window pops up here and updates live as the optimizer runs;
    ... # closing it (or letting it finish) lets the call above return.
    >>> result.energy                      # the lowest eigenvalue VQE found
    -1.1372701734498487
    >>> result.exact_energy                # numpy's exact answer, for comparison
    -1.1372701746609035
    >>> result.print_steps()               # the whole optimization, step by step

Everything below is deliberately verbose and commented like a lab notebook:
the target reader is someone who knows numerical methods (optimization,
linear algebra, convergence plots) but has never touched quantum computing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import molecule as mol

try:
    import cudaq
    import cudaq_solvers as solvers
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "vqe.py requires NVIDIA CUDA-Q Solvers (`pip install cudaq-solvers`; "
        "Linux/WSL only). See molecule.py's module docstring for details."
    ) from exc

__version__ = "0.1.0"

ParamVector = List[float]
# A "kernel" here is a CUDA-Q quantum circuit: a Python function, decorated
# with @cudaq.kernel, that takes the trial parameters theta and prepares the
# state |psi(theta)> on a fresh register of qubits. You never touch qubits
# directly when using this module -- only theta, a plain vector of floats.
Kernel = Callable[[ParamVector], None]


# ===========================================================================
#  Ansatzes -- the "shape" of the trial wavefunction |psi(theta)>
# ===========================================================================
#
# Both ansatzes below start by occupying the first n_electrons qubits (X
# gates), the same computational-basis-state encoding of "fill the lowest
# orbitals with electrons" used everywhere in this project (it is exactly
# ``Molecule.hartree_fock_occupation()``).  Where they differ is what
# happens next:
#
#   * "uccsd" (Unitary Coupled-Cluster Singles & Doubles) -- the
#     chemistry-motivated ansatz used in practically every real VQE paper.
#     Each theta_i is the amplitude of one "single" or "double" electron
#     excitation out of the Hartree-Fock state (e.g. "move one electron
#     from orbital 2 to orbital 5"), applied through a unitary that is
#     built to conserve electron number and spin. At theta = 0 every
#     excitation amplitude is zero, so the circuit does *nothing* extra and
#     you are left with exactly the Hartree-Fock state -- a safe, physically
#     meaningful starting point. You saw this already in the H2 example
#     above, where UCCSD's 3 parameters matched the exact energy to 9
#     decimal places.
#
#   * "hardware_efficient" -- a generic, chemistry-agnostic pattern of
#     single-qubit Ry rotations + a ladder of CNOT entanglers, repeated
#     `depth` times.  It is cheap and needs no molecule-specific structure,
#     but the CNOTs fire *unconditionally* -- regardless of theta -- so
#     unlike UCCSD, theta = 0 does **not** give back the Hartree-Fock state;
#     it gives some other, fixed computational-basis state instead. Worse,
#     because this ansatz has no built-in notion of "how many electrons",
#     the optimizer is free to wander into states that don't correspond to
#     the right electron count at all -- the variational principle
#     (E(theta) >= smallest eigenvalue of H) still holds, but that smallest
#     eigenvalue may belong to an unphysical sector, and a poorly-seeded
#     search can converge to it. This is a well-documented, real limitation
#     of generic hardware-efficient ansatzes (not a bug in this file), and
#     exactly why chemistry-aware ansatzes like UCCSD are usually preferred
#     for molecules -- run both on the same molecule and compare!


def build_hardware_efficient_ansatz(
    n_qubits: int, n_electrons: int, depth: int = 2
) -> Tuple[Kernel, int]:
    """A generic ansatz: HF reference + ``depth`` layers of (Ry rotations + a
    CNOT ladder).  Returns ``(kernel, n_parameters)``.

    Each layer applies one independent rotation angle to every qubit, so
    ``n_parameters == n_qubits * depth``.  Increasing ``depth`` makes the
    trial-state family strictly larger (more expressive) at the cost of more
    parameters for the optimizer to tune -- the classic numerical-methods
    trade-off between model flexibility and optimization difficulty.
    """
    n_params = n_qubits * depth

    @cudaq.kernel
    def kernel(thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        # (a) Hartree-Fock reference: occupy the first n_electrons qubits.
        for i in range(n_electrons):
            x(qubits[i])
        # (b) `depth` layers of single-qubit rotations + nearest-neighbour
        # entanglers.  This is the same "ansatz" idea as a truncated Fourier
        # series or a shallow neural net: a small number of tunable knobs
        # that can bend the state in many directions.
        idx = 0
        for _layer in range(depth):
            for q in range(n_qubits):
                ry(thetas[idx], qubits[q])
                idx += 1
            for q in range(n_qubits - 1):
                x.ctrl(qubits[q], qubits[q + 1])

    return kernel, n_params


def build_uccsd_ansatz(n_qubits: int, n_electrons: int, spin: int) -> Tuple[Kernel, int]:
    """The UCCSD ansatz: HF reference + CUDA-Q Solvers' built-in unitary
    coupled-cluster singles-and-doubles state preparation.  Returns
    ``(kernel, n_parameters)``.

    The parameter count is not something we choose -- it is fixed by the
    number of allowed single/double excitations out of the Hartree-Fock
    state, which ``cudaq_solvers`` works out for us
    (``get_num_uccsd_parameters``, also exposed as
    ``Molecule.num_uccsd_parameters()``).
    """
    n_params = int(solvers.stateprep.get_num_uccsd_parameters(n_electrons, n_qubits, spin))

    @cudaq.kernel
    def kernel(thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(qubits[i])
        solvers.stateprep.uccsd(qubits, thetas, n_electrons, spin)

    return kernel, n_params


_BUILTIN_ANSATZE = {
    "uccsd": lambda m, **kw: build_uccsd_ansatz(m.n_qubits, m.n_electrons, m.options.spin),
    "hardware_efficient": lambda m, **kw: build_hardware_efficient_ansatz(
        m.n_qubits, m.n_electrons, depth=kw.get("depth", 2)
    ),
}
_BUILTIN_ANSATZE["hea"] = _BUILTIN_ANSATZE["hardware_efficient"]
_BUILTIN_ANSATZE["hardware-efficient"] = _BUILTIN_ANSATZE["hardware_efficient"]

AnsatzLike = Union[str, Tuple[Kernel, int], Callable[..., Tuple[Kernel, int]]]


def _resolve_ansatz(molecule: "mol.Molecule", ansatz: AnsatzLike, **kwargs: Any) -> Tuple[Kernel, int, str]:
    """Turn the user's ``ansatz`` argument into ``(kernel, n_params, name)``.

    Accepts, in order of how most people will use it:
      * a name string        -- ``"uccsd"``, ``"hardware_efficient"`` (alias
                                 ``"hea"``), looked up in ``_BUILTIN_ANSATZE``.
      * a ``(kernel, n_params)`` tuple you built yourself, e.g. with
        :func:`build_hardware_efficient_ansatz`.
      * a factory function ``f(molecule, **kwargs) -> (kernel, n_params)``,
        for a fully custom ansatz.
    """
    if isinstance(ansatz, str):
        key = ansatz.strip().lower()
        if key not in _BUILTIN_ANSATZE:
            raise ValueError(
                f"unknown ansatz {ansatz!r}. Known names: {sorted(_BUILTIN_ANSATZE)}, "
                "or pass a (kernel, n_params) tuple / factory function."
            )
        kernel, n_params = _BUILTIN_ANSATZE[key](molecule, **kwargs)
        return kernel, n_params, key
    if isinstance(ansatz, tuple) and len(ansatz) == 2:
        kernel, n_params = ansatz
        return kernel, int(n_params), getattr(kernel, "__name__", "custom")
    if callable(ansatz):
        kernel, n_params = ansatz(molecule, **kwargs)
        return kernel, int(n_params), getattr(ansatz, "__name__", "custom")
    raise TypeError(
        "ansatz must be a name string, a (kernel, n_params) tuple, or a "
        f"factory function f(molecule) -> (kernel, n_params); got {type(ansatz)!r}"
    )


# ===========================================================================
#  Optimizers -- the classical numerical-methods half of VQE
# ===========================================================================
#
# From here on there is nothing quantum left: `cost` below is just a Python
# function float^p -> float, and every optimizer in this section treats it
# exactly like it would treat any other expensive black-box objective.
#
# An "optimizer" in this module is any callable with the signature
#
#     optimizer(cost, theta0, max_iterations=..., **kwargs) -> OptimizeResult
#
# where the return value only needs a ``.x`` (best parameters found) and
# ``.fun`` (the energy there) -- exactly the ``scipy.optimize.OptimizeResult``
# convention, which is why the SciPy-backed optimizers below are one-line
# wrappers around ``scipy.optimize.minimize``.

from scipy.optimize import OptimizeResult, minimize  # noqa: E402

# Optimizers that don't need a gradient -- they only ever *evaluate* E(theta)
# and compare values, similar in spirit to a bisection or golden-section
# search generalized to many dimensions.
_SCIPY_GRADIENT_FREE = {"cobyla", "nelder-mead", "powell"}
# Optimizers that use a gradient. SciPy will approximate it for us with
# finite differences (2-point stencil) since we don't supply one -- this
# costs p+1 extra energy evaluations per gradient, where p = len(theta).
_SCIPY_GRADIENT_BASED = {"bfgs", "l-bfgs-b", "cg", "slsqp"}


def _make_scipy_optimizer(method: str) -> Callable[..., OptimizeResult]:
    def optimizer(cost: Callable[[np.ndarray], float], theta0: np.ndarray,
                  max_iterations: int = 200, **kwargs: Any) -> OptimizeResult:
        options = {"maxiter": max_iterations}
        options.update(kwargs.pop("options", {}))
        return minimize(cost, theta0, method=method, options=options, **kwargs)
    return optimizer


def gradient_descent_optimizer(
    cost: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    max_iterations: int = 200,
    learning_rate: float = 0.2,
    shift: float = math.pi / 2,
    tol: float = 1e-7,
    **_ignored: Any,
) -> OptimizeResult:
    """Plain steepest-descent gradient descent, with the gradient computed by
    the **parameter-shift rule** instead of a finite difference.

    Every parameter theta_i in both ansatzes above enters the circuit as the
    argument of a rotation gate, i.e. the state depends on theta_i through a
    factor like ``exp(-i * theta_i / 2 * P)`` for some Hermitian operator P
    with eigenvalues +-1.  For *exactly* this functional form, one can show
    (this is a standard result, not specific to CUDA-Q) that the derivative
    of ``E(theta) = <psi(theta)|H|psi(theta)>`` with respect to theta_i is

        dE/dtheta_i = [ E(theta + (pi/2) e_i)  -  E(theta - (pi/2) e_i) ] / 2

    i.e. a **central difference with a fixed, exact step size** -- unlike a
    generic finite-difference approximation, this is exact for these gates
    (no truncation error from the step size, only the usual floating-point
    noise). It costs 2 energy evaluations per parameter per gradient, so one
    gradient-descent step costs ``2 * len(theta)`` circuit evaluations.

    This optimizer is here mainly so a numerical-methods reader can see the
    gradient actually being computed, rather than trusting a library's
    black box; for real work, COBYLA (`"cobyla"`) or L-BFGS-B usually
    converge in far fewer circuit evaluations.
    """
    theta = np.array(theta0, dtype=float)
    p = len(theta)
    prev_energy = cost(theta)
    k = -1
    for k in range(max_iterations):
        grad = np.empty(p)
        for i in range(p):
            shift_vec = np.zeros(p)
            shift_vec[i] = shift
            grad[i] = 0.5 * (cost(theta + shift_vec) - cost(theta - shift_vec))
        theta = theta - learning_rate * grad
        energy = cost(theta)
        if abs(prev_energy - energy) < tol:
            prev_energy = energy
            break
        prev_energy = energy
    return OptimizeResult(
        x=theta, fun=prev_energy, nit=k + 1, success=True,
        message="gradient norm/energy-change below tolerance" if k + 1 < max_iterations
        else "reached max_iterations",
    )


OptimizerLike = Union[str, Callable[..., OptimizeResult]]


def _resolve_optimizer(optimizer: OptimizerLike) -> Tuple[Callable[..., OptimizeResult], str]:
    """Turn the user's ``optimizer`` argument into a callable + display name."""
    if callable(optimizer) and not isinstance(optimizer, str):
        return optimizer, getattr(optimizer, "__name__", "custom")
    name = str(optimizer).strip().lower()
    if name in ("gradient_descent", "gd", "parameter_shift", "parameter-shift"):
        return gradient_descent_optimizer, "gradient_descent"
    if name in _SCIPY_GRADIENT_FREE or name in _SCIPY_GRADIENT_BASED:
        return _make_scipy_optimizer(name), name
    raise ValueError(
        f"unknown optimizer {optimizer!r}. Known names: "
        f"{sorted(_SCIPY_GRADIENT_FREE | _SCIPY_GRADIENT_BASED)} + "
        "'gradient_descent', or pass a callable optimizer(cost, theta0, "
        "max_iterations) -> OptimizeResult."
    )


# ===========================================================================
#  The result object -- "the steps" + "the graph"
# ===========================================================================

@dataclass
class VQEStep:
    """One evaluation of E(theta) during the optimization -- one row of the
    lab notebook."""
    iteration: int
    energy: float
    best_energy_so_far: float


@dataclass
class VQEResult:
    """Everything about one VQE run: the answer, the full trace of how the
    optimizer got there, and reference numbers to judge it by.

    Attributes
    ----------
    energy
        The lowest energy VQE found -- its estimate of the smallest
        eigenvalue of the Hamiltonian.
    params
        The parameters theta that achieve ``energy``.
    history
        One :class:`VQEStep` per energy evaluation, in order. This is the
        raw material behind :meth:`print_steps` and :meth:`plot`.
    exact_energy
        The true smallest eigenvalue, from ``Molecule.ground_state_energy_exact()``
        (exact diagonalization) -- the number VQE is trying to reproduce.
    hf_energy
        The Hartree-Fock energy (VQE's theta=0-ish starting point, roughly:
        the best *classical, non-entangled* guess).  ``energy`` should land
        between ``hf_energy`` and ``exact_energy``, closer to the latter.
    """

    energy: float
    params: np.ndarray
    history: List[VQEStep]
    ansatz_name: str
    optimizer_name: str
    molecule_name: str
    n_qubits: int
    n_params: int
    exact_energy: Optional[float]
    hf_energy: Optional[float]
    fci_energy: Optional[float]
    wall_time_seconds: float
    converged: bool
    message: str

    @property
    def n_evaluations(self) -> int:
        return len(self.history)

    @property
    def error_vs_exact(self) -> Optional[float]:
        """``|E_VQE - E_exact|`` in Hartree -- 0 would mean a perfect answer."""
        if self.exact_energy is None:
            return None
        return abs(self.energy - self.exact_energy)

    # -- "the steps" --------------------------------------------------

    def print_steps(self, every: int = 1, max_rows: int = 40) -> None:
        """Print the optimization trace as a numbered table, like you would
        for any other iterative numerical method (Newton's method, gradient
        descent on a textbook function, ...).

        ``every``    print only every N-th evaluation (they can number in the
                     hundreds; this keeps the table readable).
        ``max_rows`` hard cap on how many rows are printed, always including
                     the first and last evaluation.
        """
        rows = self.history[::every]
        if rows and rows[-1] is not self.history[-1]:
            rows.append(self.history[-1])
        if len(rows) > max_rows:
            head, tail = rows[: max_rows // 2], rows[-max_rows // 2 :]
            rows = head + [None] + tail  # type: ignore[list-item]

        print(f"VQE steps for {self.molecule_name}  "
              f"(ansatz={self.ansatz_name}, optimizer={self.optimizer_name}, "
              f"{self.n_params} parameters)")
        print(f"{'eval #':>8}  {'energy (Ha)':>18}  {'best so far (Ha)':>18}")
        print("-" * 50)
        for row in rows:
            if row is None:
                print("   ...")
                continue
            print(f"{row.iteration:>8}  {row.energy:>18.10f}  {row.best_energy_so_far:>18.10f}")
        print("-" * 50)
        print(f"result   : E = {self.energy:.10f} Ha  "
              f"after {self.n_evaluations} energy evaluations "
              f"({self.wall_time_seconds:.2f} s)")
        if self.exact_energy is not None:
            print(f"exact    : E = {self.exact_energy:.10f} Ha  "
                  f"(exact diagonalization of the qubit Hamiltonian)")
            print(f"error    : {self.error_vs_exact:.3e} Ha")
        if self.hf_energy is not None:
            print(f"reference: Hartree-Fock energy = {self.hf_energy:.10f} Ha "
                  "(the classical starting point VQE improves on)")

    def summary(self) -> Dict[str, Any]:
        return {
            "molecule": self.molecule_name,
            "ansatz": self.ansatz_name,
            "optimizer": self.optimizer_name,
            "n_qubits": self.n_qubits,
            "n_params": self.n_params,
            "n_evaluations": self.n_evaluations,
            "E(VQE)": self.energy,
            "E(HF)": self.hf_energy,
            "E(FCI)": self.fci_energy,
            "E(exact diag.)": self.exact_energy,
            "|E(VQE) - E(exact)|": self.error_vs_exact,
            "converged": self.converged,
            "wall_time_s": self.wall_time_seconds,
        }

    # -- "the graph" ----------------------------------------------------
    #
    # There is no ``result.plot()`` here on purpose: the graph is drawn
    # *while VQE runs*, live, by ``_LiveLossPlot`` below -- see
    # ``VQE.run(live_plot=True)``. Nothing gets written to disk; the plot is
    # the pop-up window itself, updated one point at a time through a
    # callback, the same way you would watch a training loss curve update
    # epoch by epoch.


class _LiveLossPlot:
    """A pop-up window showing the VQE loss curve as it happens.

    This is a *callable*: ``VQE.run()`` creates one instance, then calls
    ``live_plot(step)`` once per energy evaluation (exactly like a callback
    you'd pass to ``scipy.optimize.minimize(..., callback=...)``). Each call
    pushes one more point onto the curve and redraws -- no buffering, no
    file, just a Matplotlib window that updates in real time. Closing the
    window (or reaching the end of the run) is the only "save" step there
    is.
    """

    def __init__(
        self,
        molecule_name: str,
        ansatz_name: str,
        optimizer_name: str,
        exact_energy: Optional[float] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()  # interactive mode: draw() shows changes immediately

        self.fig, self.ax = plt.subplots()
        self._xs: List[int] = []
        self._ys: List[float] = []

        (self._line,) = self.ax.plot([], [], label="VQE")
        if exact_energy is not None:
            self.ax.axhline(exact_energy, color="C1", label="Target")

        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Cost")
        self.ax.set_title(f"VQE on {molecule_name}  --  {ansatz_name} + {optimizer_name}")
        self.ax.legend()
        self.fig.tight_layout()
        self._redraw()

    def __call__(self, step: VQEStep) -> None:
        """The callback: record one more evaluation and redraw immediately."""
        self._xs.append(step.iteration)
        self._ys.append(step.energy)
        self._line.set_data(self._xs, self._ys)
        self.ax.relim()
        self.ax.autoscale_view()
        self._redraw()

    def _redraw(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finish(self, block: bool = True) -> None:
        """Optimization is over -- freeze the window on screen.

        ``block=True`` (the default) pauses the script here until you close
        the window, exactly like a plain ``plt.show()`` at the end of any
        Matplotlib script: you get to look at the finished curve, and
        closing it is the only thing you need to do to move on.
        """
        self._plt.ioff()
        if block:
            self._plt.show()
        else:
            self._redraw()


# ===========================================================================
#  VQE -- ties the ansatz, the optimizer and the molecule together
# ===========================================================================

@dataclass
class VQE:
    """Configure one VQE run, then call :meth:`run`.

    Parameters
    ----------
    molecule
        A :class:`molecule.Molecule` (see ``molecule.py``) -- supplies the
        qubit Hamiltonian to minimize, plus reference numbers (Hartree-Fock,
        FCI, exact diagonalization) to grade the result against.
    ansatz
        ``"uccsd"`` (default, chemistry-aware, few parameters) or
        ``"hardware_efficient"``/``"hea"`` (generic, works on anything);
        see the module docstring.  You can also pass your own
        ``(kernel, n_params)`` tuple or a factory function.
    optimizer
        ``"cobyla"`` (default -- gradient-free, robust, the standard choice
        for VQE), ``"nelder-mead"``, ``"powell"``, ``"bfgs"``,
        ``"l-bfgs-b"``, ``"cg"``, ``"slsqp"`` (all from ``scipy.optimize``),
        ``"gradient_descent"`` (hand-written, parameter-shift gradient --
        see :func:`gradient_descent_optimizer`), or your own callable
        ``optimizer(cost, theta0, max_iterations) -> OptimizeResult``.
    depth
        Only used by ``"hardware_efficient"``: number of rotation+entangler
        layers (more layers = more parameters = more expressive, but harder
        to optimize).

    Examples
    --------
    >>> h2 = molecule.Molecule("h2", basis="sto-3g")
    >>> result = VQE(h2, ansatz="uccsd", optimizer="cobyla").run()   # pops up a live loss-curve window
    >>> result.energy, result.exact_energy
    (-1.1372701734498487, -1.1372701746609035)
    """

    molecule: "mol.Molecule"
    ansatz: AnsatzLike = "uccsd"
    optimizer: OptimizerLike = "cobyla"
    depth: int = 2

    def run(
        self,
        max_iterations: int = 200,
        initial_params: Optional[Union[Sequence[float], "np.ndarray"]] = None,
        seed: int = 0,
        verbose: bool = True,
        live_plot: bool = True,
        **optimizer_kwargs: Any,
    ) -> VQEResult:
        """Run the optimization and return a :class:`VQEResult`.

        ``initial_params``  starting theta; if omitted, UCCSD starts at
                             theta=0 (== the Hartree-Fock state exactly, a
                             standard and sensible VQE initialization).
                             hardware_efficient has no such safe zero point
                             (see the module docstring), so it starts at
                             small random angles instead -- a nonconvex
                             optimization landscape rewards a bit of luck in
                             the starting guess, exactly like starting
                             Newton's method or gradient descent from
                             different points on a bumpy function.
        ``seed``             random seed for the default random initial
                             guess (ignored if ``initial_params`` is given
                             or the ansatz starts at zero).
        ``verbose``          print one line per energy evaluation while
                             running.
        ``live_plot``        pop up a Matplotlib window and update it, one
                              point at a time, as each energy evaluation
                              comes in -- literally watching the loss curve
                              converge live, no file ever written. When the
                              optimizer finishes, the window freezes and the
                              call blocks until you close it (set
                              ``live_plot=False`` for a plain, silent run,
                              e.g. in a script that loops over many
                              molecules).
        ``**optimizer_kwargs`` forwarded to the optimizer (e.g.
                             ``learning_rate=`` for ``"gradient_descent"``).
        """
        kernel, n_params, ansatz_name = _resolve_ansatz(self.molecule, self.ansatz, depth=self.depth)
        optimizer_fn, optimizer_name = _resolve_optimizer(self.optimizer)

        if initial_params is not None:
            theta0 = np.asarray(initial_params, dtype=float)
            if theta0.shape != (n_params,):
                raise ValueError(f"initial_params must have length {n_params}, got {theta0.shape}")
        elif ansatz_name == "uccsd":
            theta0 = np.zeros(n_params)  # theta=0 --> exactly the HF state
        else:
            rng = np.random.default_rng(seed)
            theta0 = rng.uniform(-0.1, 0.1, n_params)

        exact_energy = _safe_call(self.molecule.ground_state_energy_exact)

        history: List[VQEStep] = []
        best_so_far = math.inf
        live_plotter = (
            _LiveLossPlot(self.molecule.name, ansatz_name, optimizer_name, exact_energy=exact_energy)
            if live_plot else None
        )

        def cost(theta: np.ndarray) -> float:
            nonlocal best_so_far
            energy = cudaq.observe(kernel, self.molecule.hamiltonian, list(theta)).expectation()
            best_so_far = min(best_so_far, energy)
            step = VQEStep(iteration=len(history) + 1, energy=energy, best_energy_so_far=best_so_far)
            history.append(step)
            if live_plotter is not None:
                live_plotter(step)
            if verbose:
                print(f"  eval {len(history):4d}:  E = {energy:.10f} Ha   (best so far: {best_so_far:.10f})")
            return energy

        if verbose:
            print(f"VQE on {self.molecule.name}: {n_params} parameters, "
                  f"{self.molecule.n_qubits} qubits, ansatz={ansatz_name}, optimizer={optimizer_name}")

        start = time.perf_counter()
        try:
            opt_result = optimizer_fn(cost, theta0, max_iterations=max_iterations, **optimizer_kwargs)
        finally:
            # Stop the clock before freezing/blocking on the plot window --
            # wall_time should measure the optimization, not how long you
            # spend looking at the result.
            wall_time = time.perf_counter() - start
            if live_plotter is not None:
                live_plotter.finish()

        final_params = np.asarray(opt_result.x, dtype=float)
        final_energy = float(opt_result.fun)

        result = VQEResult(
            energy=final_energy,
            params=final_params,
            history=history,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            molecule_name=self.molecule.name,
            n_qubits=self.molecule.n_qubits,
            n_params=n_params,
            exact_energy=exact_energy,
            hf_energy=self.molecule.hf_energy,
            fci_energy=self.molecule.fci_energy,
            wall_time_seconds=wall_time,
            converged=bool(getattr(opt_result, "success", True)),
            message=str(getattr(opt_result, "message", "")),
        )
        if verbose:
            print()
            result.print_steps(every=max(1, len(history) // 20))
        return result


def _safe_call(fn: Callable[[], float]) -> Optional[float]:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "VQE",
    "VQEResult",
    "VQEStep",
    "build_hardware_efficient_ansatz",
    "build_uccsd_ansatz",
    "gradient_descent_optimizer",
]
