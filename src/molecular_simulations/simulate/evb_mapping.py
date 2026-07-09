"""Energy-gap EVB free energy by lambda-mapping (Warshel EVB-FEP/US).

This is the two-topology / diabatic-free-energy formulation of EVB. Two diabatic
valence-bond states are described by two full classical force fields on a shared
coordinate frame (see :class:`molecular_simulations.build.EVBBuilder`):

  * V1 -- reactant diabat (transferring atom bonded to the donor)
  * V2 -- product  diabat (transferring atom bonded to the acceptor)

Sampling is driven across the barrier by a *mapping potential*

    V(lambda) = (1 - lambda) * V1 + lambda * V2 ,   lambda in [0, 1]

and the reaction coordinate is the *energy gap* dE = V2 - V1 evaluated on each
configuration. Because both diabatic energies are available for every frame, the
diabatic free-energy curves share a common origin -- so the reaction free energy
dG_rxn comes out directly, with no separate alignment step, and with no quantum
single-point calculations anywhere (V1, V2 are classical MM energies).

The mapping potential is integrated in OpenMM by carrying *both* force fields in
one System -- reactant forces in force group 1, product forces in force group 2 --
and a :class:`~openmm.CustomIntegrator` whose force is the lambda blend of the two
groups. Each state's own nonbonded exclusions and charges are preserved (they
live in separate NonbondedForce objects), which is exactly what makes the two
diabatic energies correct. Reading ``getState(groups={1})`` / ``{2}`` returns V1
and V2 directly.

The barrier obtained without an off-diagonal H12 coupling is the *diabatic*
crossing barrier -- a rigorous upper bound on the true adiabatic barrier. An
optional constant ``h12`` recovers the adiabatic ground state
E_g = 0.5(V1+V2) - sqrt(0.25 (V1-V2)^2 + H12^2) for analysis.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

PathLike = str | Path

#: Boltzmann constant in kJ/(mol*K), matching free_energy.KB.
KB = 0.00831446261815324

# Force-group ids for the two diabatic force fields in the combined system.
REACTANT_GROUP = 1
PRODUCT_GROUP = 2


def build_mapping_system(
    reactant_prmtop: PathLike,
    product_prmtop: PathLike,
    nonbonded_cutoff: float = 1.0,
) -> mm.System:
    """Combine two diabatic AMBER topologies into one dual-force-group System.

    The reactant force field is assigned force group ``REACTANT_GROUP`` and the
    product force field ``PRODUCT_GROUP``; both act on the same particles (the
    two prmtops must share atom order and coordinates -- use
    :class:`~molecular_simulations.build.EVBBuilder`). Only one center-of-mass
    remover and one particle set are kept (from the reactant); the product's
    forces are deep-copied in with the product force group.

    The systems are built with ``constraints=None`` so the reactive bond -- which
    differs between the states -- is never constrained; run with a 1 fs step.

    Args:
        reactant_prmtop: Reactant diabatic topology.
        product_prmtop: Product diabatic topology (same atoms/order).
        nonbonded_cutoff: PME real-space cutoff in nm.

    Returns:
        An OpenMM System whose group-1 energy is V1 and group-2 energy is V2.

    Raises:
        ValueError: If the two topologies differ in particle count.
    """
    cutoff = nonbonded_cutoff * unit.nanometer
    # Water is identical between the two states, so keeping it rigid is safe and
    # is essential for stability; the reactive solute bond (which differs between
    # states) is left unconstrained, so run with a 1 fs step.
    kwargs = dict(
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff,
        constraints=None,
        rigidWater=True,
    )
    react = app.AmberPrmtopFile(str(reactant_prmtop)).createSystem(**kwargs)
    prod = app.AmberPrmtopFile(str(product_prmtop)).createSystem(**kwargs)

    if react.getNumParticles() != prod.getNumParticles():
        raise ValueError(
            'reactant/product topologies differ in particle count '
            f'({react.getNumParticles()} vs {prod.getNumParticles()}); the two '
            'diabatic states must share atoms/coordinates.'
        )

    # Reactant forces -> group 1.
    for force in react.getForces():
        force.setForceGroup(REACTANT_GROUP)

    # Product forces -> group 2, deep-copied into the reactant system. Skip the
    # product's COM remover (the reactant already has one; two would double-count).
    for force in prod.getForces():
        if isinstance(force, mm.CMMotionRemover):
            continue
        clone = copy.deepcopy(force)
        clone.setForceGroup(PRODUCT_GROUP)
        react.addForce(clone)

    return react


def mapping_integrator(
    temperature: float,
    friction: float,
    dt: float,
    lam: float,
) -> mm.CustomIntegrator:
    """Langevin-middle integrator on the lambda-mapped force.

    Integrates V(lambda) = (1-lambda) V1 + lambda V2 by blending the per-group
    forces f1 (reactant) and f2 (product): F = (1-lam) f1 + lam f2. ``lam`` is a
    settable global variable so one integrator serves every window.

    Args:
        temperature: Temperature in K.
        friction: Collision rate in 1/ps.
        dt: Timestep in ps.
        lam: Initial mapping parameter in [0, 1].

    Returns:
        A configured CustomIntegrator (BAOAB splitting).
    """
    kT = KB * temperature
    integ = mm.CustomIntegrator(dt)
    integ.addGlobalVariable('lam', lam)
    integ.addGlobalVariable('kT', kT)
    integ.addGlobalVariable('friction', friction)
    # A CustomIntegrator step may reference only ONE force group, so the two
    # diabatic forces are captured into per-dof variables in separate steps and
    # then blended: Fmap = (1-lam) f1 + lam f2.
    integ.addPerDofVariable('f1v', 0)
    integ.addPerDofVariable('f2v', 0)
    integ.addPerDofVariable('x0', 0)  # pre-constraint position (RATTLE)
    integ.addUpdateContextState()

    def kick_half():
        integ.addComputePerDof('f1v', f'f{REACTANT_GROUP}')
        integ.addComputePerDof('f2v', f'f{PRODUCT_GROUP}')
        integ.addComputePerDof('v', 'v + 0.5*dt*((1-lam)*f1v + lam*f2v)/m')
        integ.addConstrainVelocities()

    def drift_half():
        # Half drift with constraint projection; the (x - x0)/dt term feeds the
        # constraint correction back into the velocities (RATTLE).
        integ.addComputePerDof('x', 'x + 0.5*dt*v')
        integ.addComputePerDof('x0', 'x')
        integ.addConstrainPositions()
        integ.addComputePerDof('v', 'v + (x - x0)/dt')
        integ.addConstrainVelocities()

    # Constrained BAOAB: B (half kick) A (half drift) O (friction+noise) A B.
    kick_half()
    drift_half()
    integ.addComputePerDof(
        'v',
        'z*v + sqrt(1 - z*z)*sqrt(kT/m)*gaussian; z = exp(-friction*dt)',
    )
    integ.addConstrainVelocities()
    drift_half()
    kick_half()
    return integ


def diabatic_energies(context: mm.Context) -> tuple[float, float]:
    """Return (V1, V2) in kJ/mol for the current configuration.

    Args:
        context: A Context on a system from :func:`build_mapping_system`.

    Returns:
        ``(V1, V2)`` -- the reactant and product diabatic potential energies.
    """
    v1 = (
        context.getState(getEnergy=True, groups={REACTANT_GROUP})
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    v2 = (
        context.getState(getEnergy=True, groups={PRODUCT_GROUP})
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    return v1, v2


def default_lambda_schedule(n_windows: int) -> np.ndarray:
    """Evenly spaced lambda windows on [0, 1] inclusive."""
    return np.linspace(0.0, 1.0, n_windows)


def ground_state_energy(v1: np.ndarray, v2: np.ndarray, h12: float) -> np.ndarray:
    """Adiabatic ground-state energy from two diabats and a constant coupling.

    E_g = 0.5 (V1 + V2) - sqrt(0.25 (V1 - V2)^2 + H12^2). With ``h12 = 0`` this is
    ``min(V1, V2)`` (the diabatic crossing, no coupling).

    Args:
        v1: Reactant diabatic energies.
        v2: Product diabatic energies.
        h12: Off-diagonal coupling in kJ/mol.

    Returns:
        Ground-state energies, elementwise.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return 0.5 * (v1 + v2) - np.sqrt(0.25 * (v1 - v2) ** 2 + h12**2)


# ---------------------------------------------------------------------------
# Sampling one lambda window
# ---------------------------------------------------------------------------


def run_lambda_window(
    system: mm.System,
    positions,
    box_vectors,
    lam: float,
    temperature: float = 300.0,
    friction: float = 1.0,
    dt: float = 0.001,
    n_equil: int = 20000,
    n_prod: int = 200000,
    sample_interval: int = 100,
    platform: str = 'CUDA',
    minimize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one mapping window and record the two diabatic energies.

    Minimizes, equilibrates and then collects V1/V2 every ``sample_interval``
    steps on the mapping potential V(lam). The energy gap V2 - V1 is the EVB
    reaction coordinate; recording V1 and V2 separately lets the analysis
    reweight to any target surface (diabatic or adiabatic).

    Args:
        system: A dual-force-group system from :func:`build_mapping_system`.
        positions: Initial coordinates (OpenMM positions / array with units).
        box_vectors: Periodic box vectors, or None.
        lam: Mapping parameter in [0, 1].
        temperature: Temperature in K.
        friction: Langevin collision rate in 1/ps.
        dt: Timestep in ps (keep at 1 fs; the reactive bond is unconstrained).
        n_equil: Equilibration steps (discarded).
        n_prod: Production steps.
        sample_interval: Steps between V1/V2 samples.
        platform: OpenMM platform name.
        minimize: Whether to energy-minimize before equilibration.

    Returns:
        ``(v1, v2)`` arrays of per-sample diabatic energies (kJ/mol).
    """
    integ = mapping_integrator(temperature, friction, dt, lam)
    context = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    if minimize:
        mm.LocalEnergyMinimizer.minimize(context, maxIterations=1000)
    context.setVelocitiesToTemperature(temperature)
    integ.step(n_equil)

    n_samples = max(1, n_prod // sample_interval)
    v1 = np.empty(n_samples)
    v2 = np.empty(n_samples)
    for i in range(n_samples):
        integ.step(sample_interval)
        v1[i], v2[i] = diabatic_energies(context)
    return v1, v2


# ---------------------------------------------------------------------------
# Free-energy analysis: BAR ladder + FEP/US gap profile
# ---------------------------------------------------------------------------


def bar(work_f: np.ndarray, work_r: np.ndarray, kT: float) -> float:
    """Bennett acceptance ratio free-energy difference dG = G_B - G_A.

    Args:
        work_f: Forward works U_B - U_A sampled in state A (kJ/mol).
        work_r: Reverse works U_A - U_B sampled in state B (kJ/mol).
        kT: Thermal energy kB*T (kJ/mol).

    Returns:
        The free-energy difference G_B - G_A in kJ/mol.
    """
    from scipy.optimize import brentq  # ty: ignore[unresolved-import]

    uf = np.asarray(work_f) / kT
    ur = np.asarray(work_r) / kT
    nf, nr = len(uf), len(ur)
    m = np.log(nf / nr)

    def objective(d: float) -> float:
        # Shirts & Chodera (2008) two-state BAR self-consistency:
        #   sum_A f(ln(nf/nr) + u_f - d) = sum_B f(ln(nr/nf) + u_r + d),
        # with u_f = beta(U_B - U_A) from A and u_r = beta(U_A - U_B) from B.
        lhs = np.sum(1.0 / (1.0 + np.exp(m + uf - d)))
        rhs = np.sum(1.0 / (1.0 + np.exp(-m + ur + d)))
        return lhs - rhs

    d = brentq(objective, -500.0, 500.0, xtol=1e-8)
    return d * kT


def ladder_free_energies(
    lambdas: np.ndarray, gaps: list[np.ndarray], kT: float
) -> np.ndarray:
    """Cumulative mapping-potential free energies dG_m(lambda) via BAR.

    Between adjacent windows the work is (lambda_{m+1}-lambda_m) * dE, since
    V(lambda_{m+1}) - V(lambda_m) = (lambda_{m+1}-lambda_m)(V2 - V1).

    Args:
        lambdas: Window lambda values (M,).
        gaps: Per-window energy-gap samples dE = V2 - V1 (list of M arrays).
        kT: Thermal energy (kJ/mol).

    Returns:
        Free energies dG_m relative to window 0 (M,), in kJ/mol.
    """
    dG = np.zeros(len(lambdas))
    for m in range(len(lambdas) - 1):
        dlam = lambdas[m + 1] - lambdas[m]
        w_f = dlam * gaps[m]  # forward work from window m
        w_r = -dlam * gaps[m + 1]  # reverse work from window m+1
        dG[m + 1] = dG[m] + bar(w_f, w_r, kT)
    return dG


@dataclass
class EVBGapResult:
    """Recovered energy-gap free-energy profile and observables."""

    gap_centers: np.ndarray  # reaction coordinate dE bin centers (kJ/mol)
    pmf: np.ndarray  # free energy along dE (kJ/mol), min-referenced
    dG_rxn: float  # product basin - reactant basin (kJ/mol)
    dG_barrier: float  # diabatic (or adiabatic) barrier (kJ/mol)
    ladder: np.ndarray  # dG_m(lambda) mapping free energies (kJ/mol)


def analyze_gap(
    lambdas: np.ndarray,
    v1: list[np.ndarray],
    v2: list[np.ndarray],
    temperature: float = 300.0,
    h12: float = 0.0,
    n_bins: int = 50,
) -> EVBGapResult:
    """Combine lambda windows into a free-energy profile along the energy gap.

    Implements the FEP/umbrella-sampling free-energy functional (Warshel): the
    ladder free energies dG_m(lambda) (BAR) set each window's reference, and the
    target-surface free energy in each energy-gap bin is reweighted from the
    mapping potential with exp(-(E_g - V_map,m)/kT). Windows are spliced by
    sample-count weighting. With ``h12 = 0`` the target E_g is min(V1, V2) (the
    diabatic crossing -> upper-bound barrier); a positive ``h12`` uses the
    adiabatic ground state.

    Args:
        lambdas: Window lambda values (M,).
        v1: Per-window reactant diabatic energies (list of M arrays).
        v2: Per-window product diabatic energies (list of M arrays).
        temperature: Temperature in K.
        h12: Off-diagonal coupling in kJ/mol (0 = diabatic).
        n_bins: Number of energy-gap bins.

    Returns:
        An :class:`EVBGapResult`.
    """
    kT = KB * temperature
    lambdas = np.asarray(lambdas)
    gaps = [np.asarray(b) - np.asarray(a) for a, b in zip(v1, v2, strict=True)]

    dG_m = ladder_free_energies(lambdas, gaps, kT)

    all_gap = np.concatenate(gaps)
    edges = np.linspace(all_gap.min(), all_gap.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Per-window, per-bin FEP/US estimate g_m(bin), and sample counts for the
    # weighting splice.
    g_est = np.full((len(lambdas), n_bins), np.nan)
    counts = np.zeros((len(lambdas), n_bins))
    for m, (a, b) in enumerate(zip(v1, v2, strict=True)):
        a = np.asarray(a)
        b = np.asarray(b)
        e_map = (1.0 - lambdas[m]) * a + lambdas[m] * b
        e_g = ground_state_energy(a, b, h12)
        dw = -(e_g - e_map) / kT  # log reweight to the target surface
        which = np.clip(np.digitize(b - a, edges) - 1, 0, n_bins - 1)
        for bin_idx in range(n_bins):
            sel = which == bin_idx
            n = int(sel.sum())
            if n == 0:
                continue
            # g_m(bin) = dG_m - kT ln <exp(dw)>_m,bin  (mean over ALL window
            # samples, so the bin population enters as n/N -> the umbrella term).
            lse = _logsumexp(dw[sel])
            g_est[m, bin_idx] = dG_m[m] - kT * (lse - np.log(len(a)))
            counts[m, bin_idx] = n

    # Splice windows: count-weighted average of the per-window estimates.
    pmf = np.full(n_bins, np.nan)
    for bin_idx in range(n_bins):
        w = counts[:, bin_idx]
        vals = g_est[:, bin_idx]
        ok = w > 0
        if ok.any():
            pmf[bin_idx] = np.average(vals[ok], weights=w[ok])

    finite = np.isfinite(pmf)
    pmf = pmf - np.nanmin(pmf)

    dG_rxn, dG_barrier = _gap_observables(centers, pmf, finite)
    return EVBGapResult(centers, pmf, dG_rxn, dG_barrier, dG_m)


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def _gap_observables(
    centers: np.ndarray, pmf: np.ndarray, finite: np.ndarray
) -> tuple[float, float]:
    """Reaction free energy and barrier from a gap PMF.

    The reactant basin is the reactant-favoured side (energy gap dE = V2 - V1 >
    0, V1 low) and the product basin dE < 0; the barrier sits near dE = 0.

    Returns:
        ``(dG_rxn, dG_barrier)`` in kJ/mol; NaN where a side is unsampled.
    """
    reactant = finite & (centers > 0)
    product = finite & (centers < 0)
    if not reactant.any() or not product.any():
        return float('nan'), float('nan')

    g_react = np.nanmin(pmf[reactant])
    g_prod = np.nanmin(pmf[product])
    # Barrier: highest point between the two basins (near the crossing).
    react_min_c = centers[reactant][np.nanargmin(pmf[reactant])]
    prod_min_c = centers[product][np.nanargmin(pmf[product])]
    lo, hi = sorted((react_min_c, prod_min_c))
    between = finite & (centers >= lo) & (centers <= hi)
    g_ts = np.nanmax(pmf[between]) if between.any() else float('nan')

    return float(g_prod - g_react), float(g_ts - g_react)
