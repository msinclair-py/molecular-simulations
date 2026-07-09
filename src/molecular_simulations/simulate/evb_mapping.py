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
from parsl import python_app

PathLike = str | Path

#: Boltzmann constant in kJ/(mol*K), matching free_energy.KB.
KB = 0.00831446261815324

# Force-group ids for the two diabatic force fields in the combined system.
REACTANT_GROUP = 1
PRODUCT_GROUP = 2


def build_mapping_system(
    reactant_prmtop: PathLike,
    product_prmtop: PathLike,
    donor: int | None = None,
    acceptor: int | None = None,
    reactive: int | None = None,
    D_e: float = 460.0,
    alpha: float = 22.0,
    r0: float = 0.097,
    nonbonded_cutoff: float = 1.0,
) -> mm.System:
    """Combine two diabatic AMBER topologies into one dual-force-group System.

    The reactant force field is assigned force group ``REACTANT_GROUP`` and the
    product force field ``PRODUCT_GROUP``; both act on the same particles (the
    two prmtops must share atom order and coordinates -- use
    :class:`~molecular_simulations.build.EVBBuilder`). Only one center-of-mass
    remover and one particle set are kept (from the reactant); the product's
    forces are deep-copied in with the product force group.

    If the transfer atoms are supplied, the reactive bond in each diabatic state
    (donor-reactive in the reactant, acceptor-reactive in the product) is
    replaced by a Morse bond. This is essential: a harmonic reactive bond
    diverges when the *other* state is evaluated at this state's geometry (the
    transferring atom is far from its partner), so the diabatic energy gap
    explodes and the lambda windows have no overlap. A Morse bond plateaus at
    ``D_e`` when stretched, keeping the two diabatic surfaces close enough to
    bridge.

    The systems are built with ``constraints=None`` so the reactive bond -- which
    differs between the states -- is never constrained; run with a 1 fs step.

    Args:
        reactant_prmtop: Reactant diabatic topology.
        product_prmtop: Product diabatic topology (same atoms/order).
        donor: Donor atom index (bonded to reactive in the reactant state).
        acceptor: Acceptor atom index (bonded to reactive in the product state).
        reactive: Transferring atom index.
        D_e: Morse well depth in kJ/mol.
        alpha: Morse width parameter in nm^-1.
        r0: Morse equilibrium distance in nm.
        nonbonded_cutoff: PME real-space cutoff in nm.

    Returns:
        An OpenMM System whose group-1 energy is V1 and group-2 energy is V2.

    Raises:
        ValueError: If the two topologies differ in particle count.
    """
    from molecular_simulations.simulate.free_energy import EVBCalculation

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

    # Replace each state's reactive harmonic bond with a Morse bond so the
    # diabatic energy stays bounded when the transferring atom is far from its
    # partner (otherwise the energy gap explodes and the windows never overlap).
    if donor is not None and acceptor is not None and reactive is not None:
        EVBCalculation.remove_harmonic_bond(react, donor, reactive)
        react.addForce(EVBCalculation.morse_bond_force(donor, reactive, D_e, alpha, r0))
        EVBCalculation.remove_harmonic_bond(prod, acceptor, reactive)
        prod.addForce(
            EVBCalculation.morse_bond_force(acceptor, reactive, D_e, alpha, r0)
        )

    # OpenMM requires all NonbondedForces in a System to share one exclusion-pair
    # set. The two diabatic states differ only in the reactive atom's bonds, so a
    # few pairs are excluded in one state but not the other; reconcile them.
    nb_react = _nonbonded_force(react)
    nb_prod = _nonbonded_force(prod)
    if nb_react is not None and nb_prod is not None:
        _reconcile_exceptions(nb_react, nb_prod)

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


def _nonbonded_force(system: mm.System) -> mm.NonbondedForce | None:
    """Return the system's NonbondedForce, or None."""
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    return None


def _reconcile_exceptions(nb_a: mm.NonbondedForce, nb_b: mm.NonbondedForce) -> None:
    """Give two NonbondedForces the same exception-pair set (params may differ).

    A pair excluded in one diabatic state but not the other is added to the state
    that lacks it as an exception carrying *that state's* real direct nonbonded
    interaction (Lorentz-Berthelot combined), so the pair still interacts
    normally there while the two forces share one exclusion-pair set (an OpenMM
    requirement for multiple PME NonbondedForces). Only reactive-atom pairs
    differ, so this touches a handful of short-range pairs; the reciprocal-space
    approximation this introduces is negligible and, for a symmetric reaction,
    cancels between the two states.

    Args:
        nb_a: First NonbondedForce (modified in place).
        nb_b: Second NonbondedForce (modified in place).
    """

    def pairset(nb: mm.NonbondedForce) -> set[tuple[int, int]]:
        pairs = set()
        for k in range(nb.getNumExceptions()):
            i, j, *_ = nb.getExceptionParameters(k)
            pairs.add((min(i, j), max(i, j)))
        return pairs

    pa, pb = pairset(nb_a), pairset(nb_b)
    for nb, missing in ((nb_a, pb - pa), (nb_b, pa - pb)):
        for i, j in missing:
            qi, si, ei = nb.getParticleParameters(i)
            qj, sj, ej = nb.getParticleParameters(j)
            charge_prod = qi.value_in_unit(unit.elementary_charge) * qj.value_in_unit(
                unit.elementary_charge
            )
            sigma = 0.5 * (
                si.value_in_unit(unit.nanometer) + sj.value_in_unit(unit.nanometer)
            )
            epsilon = np.sqrt(
                ei.value_in_unit(unit.kilojoule_per_mole)
                * ej.value_in_unit(unit.kilojoule_per_mole)
            )
            nb.addException(
                i,
                j,
                charge_prod * unit.elementary_charge**2,
                sigma * unit.nanometer,
                epsilon * unit.kilojoule_per_mole,
            )


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


# ---------------------------------------------------------------------------
# Multi-GPU orchestration (parsl)
# ---------------------------------------------------------------------------


@python_app
def run_mapping_window(
    reactant_prmtop: str,
    product_prmtop: str,
    coord_file: str,
    lam: float,
    out_npz: str,
    temperature: float,
    friction: float,
    dt: float,
    n_equil: int,
    n_prod: int,
    sample_interval: int,
    platform: str,
    nonbonded_cutoff: float,
    donor: int | None,
    acceptor: int | None,
    reactive: int | None,
    D_e: float,
    alpha: float,
    r0: float,
) -> str:
    """Parsl app: sample one mapping window and save (v1, v2) to ``out_npz``.

    Separate module-level app for serialization; rebuilds the dual-force-group
    system on the worker so only file paths cross the wire.
    """
    import numpy as np
    from openmm import app

    from molecular_simulations.simulate.evb_mapping import (
        build_mapping_system,
        run_lambda_window,
    )

    inp = app.AmberInpcrdFile(coord_file)
    system = build_mapping_system(
        reactant_prmtop,
        product_prmtop,
        donor=donor,
        acceptor=acceptor,
        reactive=reactive,
        D_e=D_e,
        alpha=alpha,
        r0=r0,
        nonbonded_cutoff=nonbonded_cutoff,
    )
    v1, v2 = run_lambda_window(
        system,
        inp.positions,
        inp.boxVectors,
        lam,
        temperature=temperature,
        friction=friction,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
        sample_interval=sample_interval,
        platform=platform,
    )
    np.savez(out_npz, v1=v1, v2=v2, lam=lam)
    return out_npz


class EVBMapping:
    """Distribute EVB energy-gap lambda windows across local GPUs with parsl.

    Consumes the two diabatic topologies from
    :class:`~molecular_simulations.build.EVBBuilder` (shared coordinates) and
    runs one mapping window per lambda, one worker per GPU. ``run`` writes a
    per-window ``window{i}.npz`` of (v1, v2); ``analyze`` reduces them to the
    energy-gap free-energy profile via :func:`analyze_gap`.

    Args:
        reactant_prmtop: Reactant diabatic topology.
        product_prmtop: Product diabatic topology (shared atoms/coords).
        coordinates: Shared coordinate file (AMBER inpcrd).
        parsl_config: Parsl Config (e.g. LocalSettings(mpi_launcher=False)).
        out_path: Output directory for the per-window npz files.
        lambdas: Explicit lambda schedule; if None, ``n_windows`` evenly spaced.
        n_windows: Number of windows when ``lambdas`` is None. Defaults to 11.
        temperature: Temperature in K.
        friction: Langevin collision rate in 1/ps.
        dt: Timestep in ps (1 fs; the reactive bond is unconstrained).
        n_equil: Equilibration steps per window.
        n_prod: Production steps per window.
        sample_interval: Steps between V1/V2 samples.
        platform: OpenMM platform name.
        nonbonded_cutoff: PME real-space cutoff in nm.
    """

    def __init__(
        self,
        reactant_prmtop: PathLike,
        product_prmtop: PathLike,
        coordinates: PathLike,
        parsl_config,
        out_path: PathLike,
        lambdas: np.ndarray | None = None,
        n_windows: int = 11,
        temperature: float = 300.0,
        friction: float = 1.0,
        dt: float = 0.001,
        n_equil: int = 20000,
        n_prod: int = 200000,
        sample_interval: int = 100,
        platform: str = 'CUDA',
        nonbonded_cutoff: float = 1.0,
        donor: int | None = None,
        acceptor: int | None = None,
        reactive: int | None = None,
        D_e: float = 460.0,
        alpha: float = 22.0,
        r0: float = 0.097,
    ):
        """Initialize the EVBMapping orchestrator.

        ``donor``/``acceptor``/``reactive`` (0-based indices, from EVBBuilder's
        evb_meta.json) trigger Morse-ification of each state's reactive bond,
        which is required for the diabatic energy gap to stay bounded.
        """
        self.reactant_prmtop = str(Path(reactant_prmtop).resolve())
        self.product_prmtop = str(Path(product_prmtop).resolve())
        self.coordinates = str(Path(coordinates).resolve())
        self.parsl_config = parsl_config
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.donor = donor
        self.acceptor = acceptor
        self.reactive = reactive
        self.D_e = D_e
        self.alpha = alpha
        self.r0 = r0

        self.lambdas = (
            default_lambda_schedule(n_windows)
            if lambdas is None
            else np.asarray(lambdas, dtype=float)
        )
        self.temperature = temperature
        self.friction = friction
        self.dt = dt
        self.n_equil = n_equil
        self.n_prod = n_prod
        self.sample_interval = sample_interval
        self.platform = platform
        self.nonbonded_cutoff = nonbonded_cutoff

        self.dfk = None
        self._owns_parsl = False

    def initialize(self) -> None:
        """Load parsl (reusing a parent DFK if one is already running)."""
        import parsl

        if self.dfk is None:
            try:
                self.dfk = parsl.dfk()
                self._owns_parsl = False
            except Exception:
                self.dfk = parsl.load(self.parsl_config)
                self._owns_parsl = True

    def shutdown(self) -> None:
        """Clean up parsl if this instance loaded it."""
        import parsl

        if self._owns_parsl and self.dfk:
            self.dfk.cleanup()
            parsl.clear()
        self.dfk = None
        self._owns_parsl = False

    def run(self) -> list[str]:
        """Distribute the windows and wait for all to finish.

        Returns:
            The per-window npz paths (window order = lambda order).
        """
        self.initialize()
        futures, out_files = [], []
        for i, lam in enumerate(self.lambdas):
            out_npz = str(self.out_path / f'window{i}.npz')
            out_files.append(out_npz)
            futures.append(
                run_mapping_window(
                    self.reactant_prmtop,
                    self.product_prmtop,
                    self.coordinates,
                    float(lam),
                    out_npz,
                    self.temperature,
                    self.friction,
                    self.dt,
                    self.n_equil,
                    self.n_prod,
                    self.sample_interval,
                    self.platform,
                    self.nonbonded_cutoff,
                    self.donor,
                    self.acceptor,
                    self.reactive,
                    self.D_e,
                    self.alpha,
                    self.r0,
                )
            )
        for fut in futures:
            fut.result()
        if self._owns_parsl:
            self.shutdown()
        return out_files

    def analyze(self, h12: float = 0.0, n_bins: int = 50) -> EVBGapResult:
        """Reduce the per-window npz files to the energy-gap free energy.

        Args:
            h12: Off-diagonal coupling (0 = diabatic upper-bound barrier).
            n_bins: Number of energy-gap bins.

        Returns:
            The :class:`EVBGapResult`.
        """
        v1, v2 = [], []
        for i in range(len(self.lambdas)):
            data = np.load(self.out_path / f'window{i}.npz')
            v1.append(data['v1'])
            v2.append(data['v2'])
        return analyze_gap(
            self.lambdas, v1, v2, temperature=self.temperature, h12=h12, n_bins=n_bins
        )
