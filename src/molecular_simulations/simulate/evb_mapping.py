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
import warnings
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
    soft_core: bool = False,
    sc_alpha: float = 0.5,
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
        soft_core: If True (and transfer atoms supplied), replace the hard LJ of
            the reactive atom's diabat-flipping nonbonded pairs with a bounded
            soft-core term. Keeps the diabatic energy finite at the endpoints,
            where the transferring atom clashes with a partner it is not bonded to
            in that diabat (the Morse bond only bounds the *bonded* term). Opt-in
            prototype; off by default preserves the exact prior behaviour.
        sc_alpha: Soft-core offset (fraction of sigma^6); larger = softer plateau.

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
        added_react, added_prod = _reconcile_exceptions(nb_react, nb_prod)
        # Optionally soft-core the reactive atom's *nonbonded* clash with the
        # partner it is not bonded to in each diabat -- the flip pairs reconcile
        # just added. Keeps the diabatic energy finite at the endpoints (the
        # Morse bond only bounds the bonded reactive term). Prototype; opt-in.
        if soft_core and reactive is not None:
            _soft_core_reactive_pairs(react, nb_react, reactive, added_react, sc_alpha)
            _soft_core_reactive_pairs(prod, nb_prod, reactive, added_prod, sc_alpha)

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


def _reconcile_exceptions(
    nb_a: mm.NonbondedForce, nb_b: mm.NonbondedForce
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Give two NonbondedForces the same exception-pair set (params may differ).

    A pair excluded in one diabatic state but not the other is added to the state
    that lacks it as an exception carrying *that state's* real direct nonbonded
    interaction (Lorentz-Berthelot combined), so the pair still interacts
    normally there while the two forces share one exclusion-pair set (an OpenMM
    requirement for multiple PME NonbondedForces). Only reactive-atom pairs
    differ, so this touches a handful of short-range pairs; the reciprocal-space
    approximation this introduces is negligible and, for a symmetric reaction,
    cancels between the two states.

    These added pairs are exactly the ones whose LJ diverges when the diabat is
    evaluated at the *other* state's geometry (the transferring atom near a
    partner it is not bonded to here), so the returned lists drive the optional
    reactive-atom soft-core (:func:`_soft_core_reactive_pairs`).

    Args:
        nb_a: First NonbondedForce (modified in place).
        nb_b: Second NonbondedForce (modified in place).

    Returns:
        ``(added_to_a, added_to_b)`` -- the (i, j) pairs added to each force.
    """

    def pairset(nb: mm.NonbondedForce) -> set[tuple[int, int]]:
        pairs = set()
        for k in range(nb.getNumExceptions()):
            i, j, *_ = nb.getExceptionParameters(k)
            pairs.add((min(i, j), max(i, j)))
        return pairs

    pa, pb = pairset(nb_a), pairset(nb_b)
    added: dict[int, list[tuple[int, int]]] = {id(nb_a): [], id(nb_b): []}
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
            added[id(nb)].append((i, j))
    return added[id(nb_a)], added[id(nb_b)]


def _soft_core_reactive_pairs(
    system: mm.System,
    nb: mm.NonbondedForce,
    reactive: int,
    pairs: list[tuple[int, int]],
    sc_alpha: float,
) -> None:
    """Soft-core the LJ of the given reactive-atom nonbonded pairs (prototype).

    The two-topology diabatic energy explodes at the endpoints because the
    transferring atom sits ~1 Angstrom from a partner it is *not* bonded to in
    this diabat, so a bare r^-12 LJ term blows up (the Morse fix only bounds the
    *bonded* reactive term, not this nonbonded clash). Here each such pair's hard
    LJ is moved out of the NonbondedForce into a bounded soft-core CustomBondForce

        V_sc(r) = 4 eps (u^2 - u),   u = sigma^6 / (sc_alpha*sigma^6 + r^6),

    which -> the true LJ at large r but plateaus at ``4 eps (1/sc_alpha^2 -
    1/sc_alpha)`` as r -> 0. Coulomb on the pair is left intact (small; ~r^-1).
    Only the handful of flip pairs from :func:`_reconcile_exceptions` are touched,
    so the physical basins (where these pairs are at normal range, u ~ (sigma/r)^6)
    are essentially unchanged.

    Args:
        system: System to add the soft-core CustomBondForce to (its own group).
        nb: The diabat's NonbondedForce (its exception LJ is zeroed in place).
        reactive: Transferring atom index (pairs not involving it are ignored).
        pairs: Candidate (i, j) pairs (the reconcile-added flip pairs).
        sc_alpha: Soft-core offset (nm^6 fraction of sigma^6); larger = softer.
    """
    pairs = [(i, j) for (i, j) in pairs if reactive in (i, j)]
    if not pairs:
        return

    by_pair: dict[frozenset, int] = {}
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        by_pair[frozenset((i, j))] = k

    cbf = mm.CustomBondForce(
        '4*epsilon*(u*u - u); u = s6/(sc_alpha*s6 + r6); s6 = sigma^6; r6 = r^6'
    )
    cbf.addGlobalParameter('sc_alpha', sc_alpha)
    cbf.addPerBondParameter('sigma')
    cbf.addPerBondParameter('epsilon')

    n_added = 0
    for pair in pairs:
        k = by_pair.get(frozenset(pair))
        if k is None:
            continue
        i, j, q, s, e = nb.getExceptionParameters(k)
        eps = e.value_in_unit(unit.kilojoule_per_mole)
        if eps <= 0.0:
            continue  # pure exclusion / no LJ to soften
        # Strip the hard LJ from the exception (keep its Coulomb chargeProd)...
        nb.setExceptionParameters(k, i, j, q, s, 0.0 * unit.kilojoule_per_mole)
        # ...and re-add it soft-cored.
        cbf.addBond(i, j, [s.value_in_unit(unit.nanometer), eps])
        n_added += 1

    if n_added:
        system.addForce(cbf)


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


def endpoint_dense_schedule(
    n_windows: int, lam_min: float = 0.0, lam_max: float = 1.0
) -> np.ndarray:
    """Cosine-spaced lambda windows -- dense near both ends of [lam_min, lam_max].

    The diabatic energy gap is largest in the reactant basin (near lam_min) and
    steep through the crossing (near the barrier), so a uniform schedule leaves
    adjacent windows there with too little overlap for BAR. Chebyshev/cosine
    (Gauss-Lobatto) spacing clusters windows at the two endpoints, keeping the
    per-step work within BAR's overlap range where the gap changes fastest.

    For a reactant->TS barrier one typically passes ``lam_max`` ~0.55 (just past
    the crossing) so the product endpoint -- which needs its own dense tail -- is
    not sampled.

    Args:
        n_windows: Number of windows (>= 2).
        lam_min: Reactant-side endpoint (default 0.0).
        lam_max: Product-side endpoint (default 1.0).

    Returns:
        Increasing lambda values in ``[lam_min, lam_max]`` (inclusive).
    """
    if n_windows < 2:
        return np.array([lam_min], dtype=float)
    i = np.arange(n_windows)
    x = 0.5 * (1.0 - np.cos(np.pi * i / (n_windows - 1)))  # 0..1, dense at ends
    return lam_min + (lam_max - lam_min) * x


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
    seed: int | None = None,
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
        seed: Random seed for the thermostat + initial velocities. Pass distinct
            seeds to generate independent replicas (see :func:`aggregate_replicas`);
            None draws a fresh nondeterministic seed each call.

    Returns:
        ``(v1, v2)`` arrays of per-sample diabatic energies (kJ/mol).
    """
    integ = mapping_integrator(temperature, friction, dt, lam)
    if seed is not None:
        integ.setRandomNumberSeed(int(seed))
    context = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    if minimize:
        mm.LocalEnergyMinimizer.minimize(context, maxIterations=1000)
    if seed is not None:
        context.setVelocitiesToTemperature(temperature, int(seed))
    else:
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


class BARConvergenceWarning(UserWarning):
    """Warns that a BAR/ladder step could not be estimated (no window overlap).

    Emitted instead of raising when adjacent lambda windows are spaced too far
    apart for their work distributions to overlap; the affected free energies are
    ``nan`` and the caller gets a partial result. The remedy is a denser lambda
    schedule in the flagged range.
    """


class GapSamplingWarning(UserWarning):
    """Warns that the gap PMF does not span both basins.

    Emitted when one side of the crossing (reactant gap > 0 or product gap < 0)
    is unsampled, so ``dG_barrier``/``dG_rxn`` come back ``nan``. The remedy is to
    extend the lambda range across the crossing or sample more.
    """


def bar(work_f: np.ndarray, work_r: np.ndarray, kT: float) -> float:
    """Bennett acceptance ratio free-energy difference dG = G_B - G_A.

    The BAR self-consistency root is bracketed on ``|dG| <= 500 kT`` -- a
    generous trust region for a single lambda step. If the root lies outside it,
    the forward/reverse work distributions do not overlap (the two windows are
    spaced too far apart) and BAR is undefined; this returns ``nan`` and warns a
    :class:`BARConvergenceWarning` rather than raising, so a poorly-spaced ladder
    degrades to a partial result instead of crashing.

    Args:
        work_f: Forward works U_B - U_A sampled in state A (kJ/mol).
        work_r: Reverse works U_A - U_B sampled in state B (kJ/mol).
        kT: Thermal energy kB*T (kJ/mol).

    Returns:
        The free-energy difference G_B - G_A in kJ/mol, or ``nan`` if the two
        states' work distributions do not overlap.
    """
    from scipy.optimize import brentq  # ty: ignore[unresolved-import]
    from scipy.special import expit  # ty: ignore[unresolved-import]

    uf = np.asarray(work_f) / kT
    ur = np.asarray(work_r) / kT
    nf, nr = len(uf), len(ur)
    m = np.log(nf / nr)

    def objective(d: float) -> float:
        # Shirts & Chodera (2008) two-state BAR self-consistency:
        #   sum_A f(ln(nf/nr) + u_f - d) = sum_B f(ln(nr/nf) + u_r + d),
        # with u_f = beta(U_B - U_A) from A and u_r = beta(U_A - U_B) from B.
        # expit(-x) = 1/(1+exp(x)) is the overflow-safe Fermi function, so a
        # huge (non-overlapping) work does not raise/emit numpy overflow noise.
        lhs = np.sum(expit(-(m + uf - d)))
        rhs = np.sum(expit(-(-m + ur + d)))
        return lhs - rhs

    lo, hi = -500.0, 500.0
    # Strict sign change only: when a non-overlapping (huge) work underflows the
    # objective to exactly 0 at a bound, that is not a real bracket -- treat it
    # as non-overlap (nan) rather than letting brentq return a spurious root.
    if objective(lo) * objective(hi) < 0:
        return brentq(objective, lo, hi, xtol=1e-8) * kT

    warnings.warn(
        f'BAR did not converge: the forward/reverse work distributions do not '
        f'overlap (mean work_f={np.mean(work_f):.4g}, '
        f'mean work_r={np.mean(work_r):.4g} kJ/mol, |dG| > 500 kT). The two '
        f'lambda windows are spaced too far apart; returning nan.',
        BARConvergenceWarning,
        stacklevel=2,
    )
    return float('nan')


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
        step = bar(w_f, w_r, kT)
        if not np.isfinite(step):
            warnings.warn(
                f'lambda ladder broken between window {m} (lambda='
                f'{lambdas[m]:.4g}) and window {m + 1} (lambda={lambdas[m + 1]:.4g}):'
                f' the windows do not overlap. Free energies past window {m} are '
                f'undefined (nan); use a denser lambda schedule in this range.',
                BARConvergenceWarning,
                stacklevel=2,
            )
        dG[m + 1] = dG[m] + step  # nan propagates: downstream windows unresolved
    return dG


@dataclass
class EVBGapResult:
    """Recovered energy-gap free-energy profile and observables."""

    gap_centers: np.ndarray  # reaction coordinate dE bin centers (kJ/mol)
    pmf: np.ndarray  # free energy along dE (kJ/mol), min-referenced
    dG_rxn: float  # product basin - reactant basin (kJ/mol)
    dG_barrier: float  # diabatic (or adiabatic) barrier (kJ/mol)
    ladder: np.ndarray  # dG_m(lambda) mapping free energies (kJ/mol)
    dG_rxn_err: float = float('nan')  # bootstrap SE of dG_rxn (nan if n_boot=0)
    dG_barrier_err: float = float('nan')  # bootstrap SE of dG_barrier


def analyze_gap(
    lambdas: np.ndarray,
    v1: list[np.ndarray],
    v2: list[np.ndarray],
    temperature: float = 300.0,
    h12: float = 0.0,
    n_bins: int = 50,
    n_boot: int = 0,
    alpha: float = 0.0,
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
        n_boot: If > 0, resample each window's frames with replacement this many
            times and re-estimate to set ``dG_rxn_err`` / ``dG_barrier_err``.
            NOTE: a plain (non-block) bootstrap over correlated MD frames is a
            *lower bound* on the true statistical error -- for the honest error
            run independent replicas and combine with :func:`aggregate_replicas`.
        alpha: EVB diabat-2 gas-phase shift in kJ/mol (H22 = V2 + alpha). A pure
            post-processing constant -- it does not change the sampled
            trajectories -- calibrated with :func:`calibrate_evb` so the diabatic
            reaction free energy matches a reference.

    Returns:
        An :class:`EVBGapResult`.
    """
    kT = KB * temperature
    lambdas = np.asarray(lambdas)
    # H22 = V2 + alpha (EVB gas-phase shift). In the mapping potential alpha is a
    # per-window constant (lambda*alpha), so it does not change the sampled
    # trajectories; it is applied here, in post-processing, only.
    v2 = [np.asarray(b) + alpha for b in v2]
    gaps = [np.asarray(b) - np.asarray(a) for a, b in zip(v1, v2, strict=True)]

    dG_m = ladder_free_energies(lambdas, gaps, kT)

    all_gap = np.concatenate(gaps)
    # Equal-population (quantile) bin edges: the densely sampled crossing and
    # basin-floor regions get fine resolution while a sparse extreme tail (e.g.
    # the deep reactant-basin window near lambda=0, whose gap is huge) gets a few
    # wide bins -- so a wide gap range does not wash out the crossing under uniform
    # binning. Collapsed edges are deduped; fall back to uniform if degenerate.
    edges = np.unique(np.quantile(all_gap, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        edges = np.linspace(all_gap.min(), all_gap.max() + 1e-9, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)  # effective bin count after dedup

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
    # Windows past a broken ladder step carry a nan free-energy reference; drop
    # them here so the bins covered by the resolvable windows still get a PMF.
    pmf = np.full(n_bins, np.nan)
    for bin_idx in range(n_bins):
        w = counts[:, bin_idx]
        vals = g_est[:, bin_idx]
        ok = (w > 0) & np.isfinite(vals)
        if ok.any():
            pmf[bin_idx] = np.average(vals[ok], weights=w[ok])

    finite = np.isfinite(pmf)
    pmf = pmf - np.nanmin(pmf)

    dG_rxn, dG_barrier = _gap_observables(centers, pmf, finite)
    if not np.isfinite(dG_barrier):
        warnings.warn(
            'gap PMF does not span both basins (one side of the crossing is '
            'unsampled); dG_barrier/dG_rxn are nan. Extend the lambda range across '
            'the crossing (raise lam_max) or sample more.',
            GapSamplingWarning,
            stacklevel=2,
        )

    dG_rxn_err = dG_barrier_err = float('nan')
    if n_boot > 0:
        rng = np.random.default_rng(0)
        rxn_b, bar_b = [], []
        for _ in range(n_boot):
            rv1, rv2 = [], []
            for a, b in zip(v1, v2, strict=True):
                a = np.asarray(a)
                idx = rng.integers(0, len(a), len(a))
                rv1.append(a[idx])
                rv2.append(np.asarray(b)[idx])
            r = analyze_gap(lambdas, rv1, rv2, temperature, h12, n_bins, n_boot=0)
            rxn_b.append(r.dG_rxn)
            bar_b.append(r.dG_barrier)
        dG_rxn_err = float(np.nanstd(rxn_b))
        dG_barrier_err = float(np.nanstd(bar_b))

    return EVBGapResult(
        centers, pmf, dG_rxn, dG_barrier, dG_m, dG_rxn_err, dG_barrier_err
    )


def aggregate_replicas(results: list[EVBGapResult]) -> dict[str, float]:
    """Combine independent-replica results into mean +/- standard error.

    Independent replicas (different random seeds / initial velocities) are the
    honest error estimate for an EVB barrier -- they capture the slow
    conformational sampling that a within-run bootstrap misses. Feed one
    :class:`EVBGapResult` per replica (from :func:`analyze_gap` on that replica's
    windows).

    Args:
        results: One result per replica (nan observables are ignored).

    Returns:
        Dict with ``dG_barrier``/``dG_rxn`` means, ``_sem`` standard errors, and
        ``n`` (number of finite replicas used).
    """
    bar = np.array([r.dG_barrier for r in results], dtype=float)
    rxn = np.array([r.dG_rxn for r in results], dtype=float)

    def mean_sem(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float('nan'), float('nan')
        sem = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float('nan')
        return float(np.mean(x)), sem

    bar_m, bar_s = mean_sem(bar)
    rxn_m, rxn_s = mean_sem(rxn)
    return {
        'dG_barrier': bar_m,
        'dG_barrier_sem': bar_s,
        'dG_rxn': rxn_m,
        'dG_rxn_sem': rxn_s,
        'n': int(np.isfinite(bar).sum()),
    }


# ---------------------------------------------------------------------------
# EVB calibration: fit (alpha, H12) to a reference dG_rxn + dG_barrier
# ---------------------------------------------------------------------------


@dataclass
class EVBCalibration:
    """Calibrated EVB parameters and the observables they reproduce.

    ``alpha`` (diabat-2 gas-phase shift) and ``h12`` (off-diagonal coupling) are
    calibrated once on a reference (e.g. WT enzyme, or the reaction in water) and
    then held FIXED across mutants of the same reaction -- the EVB transferability
    premise. Apply them via ``analyze_gap(..., h12=cal.h12, alpha=cal.alpha)``.
    """

    alpha: float  # diabat-2 gas-phase shift (kJ/mol)
    h12: float  # off-diagonal coupling (kJ/mol)
    dG_rxn: float  # achieved reaction free energy (kJ/mol)
    dG_barrier: float  # achieved adiabatic barrier (kJ/mol)
    converged: bool  # both targets hit within tol


def calibrate_evb(
    lambdas: np.ndarray,
    v1: list[np.ndarray],
    v2: list[np.ndarray],
    dG_rxn_ref: float,
    dG_barrier_ref: float,
    temperature: float = 300.0,
    n_bins: int = 50,
    max_iter: int = 20,
    tol: float = 0.5,
) -> EVBCalibration:
    """Fit the EVB (alpha, H12) so the profile matches a reference.

    Both parameters are pure post-processing constants (see :func:`analyze_gap`),
    so this re-analyzes the *already-sampled* windows -- no new simulation. It
    alternates two 1-D solves until both targets are met:

      * ``alpha`` sets the reaction free energy (raising V2 by alpha raises the
        product basin, so dG_rxn increases ~linearly with alpha) and barely
        affects the barrier;
      * ``h12`` lowers the barrier (E_g = V - H12 at the crossing) and barely
        affects the well-separated basins,

    so the two solves are nearly decoupled and converge in a few iterations.
    Calibrate on a reference system (WT, or the reaction in water) and reuse the
    result across mutants.

    Args:
        lambdas: Window lambda values.
        v1: Per-window reactant diabatic energies.
        v2: Per-window product diabatic energies.
        dG_rxn_ref: Target reaction free energy (kJ/mol).
        dG_barrier_ref: Target activation free energy (kJ/mol). Must be below the
            diabatic (h12=0) barrier, which H12 can only lower.
        temperature: Temperature in K.
        n_bins: Energy-gap bins.
        max_iter: Maximum alternating iterations.
        tol: Convergence tolerance on both targets (kJ/mol).

    Returns:
        An :class:`EVBCalibration` (``converged`` False if a target was not met --
        e.g. the reference barrier exceeds the diabatic upper bound).
    """

    def rxn(a: float, h: float) -> float:
        return analyze_gap(
            lambdas, v1, v2, temperature=temperature, h12=h, n_bins=n_bins, alpha=a
        ).dG_rxn

    def barrier(a: float, h: float) -> float:
        return analyze_gap(
            lambdas, v1, v2, temperature=temperature, h12=h, n_bins=n_bins, alpha=a
        ).dG_barrier

    from scipy.optimize import brentq  # ty: ignore[unresolved-import]

    # alpha is only valid within (-max_gap, -min_gap): outside it the shifted gap
    # V2+alpha-V1 no longer straddles the crossing, a basin empties and dG_rxn is
    # nan. Bracket brentq inside that window (inset off the degenerate edges).
    all_gap = np.concatenate(
        [np.asarray(b) - np.asarray(a) for a, b in zip(v1, v2, strict=True)]
    )
    inset = 0.1 * float(np.max(all_gap) - np.min(all_gap))
    a_lo, a_hi = -float(np.max(all_gap)) + inset, -float(np.min(all_gap)) - inset

    alpha, h12 = 0.0, 0.0
    d_rxn = d_bar = float('nan')
    for _ in range(max_iter):
        if a_hi <= a_lo:
            break
        # alpha matches dG_rxn (monotone increasing over the valid window).
        try:
            alpha = brentq(
                lambda a, h=h12: rxn(a, h) - dG_rxn_ref, a_lo, a_hi, xtol=1e-4
            )
        except ValueError:
            break  # reference dG_rxn unachievable with this sampling

        # h12 >= 0 matches dG_barrier (monotone decreasing; E_g = V - H12 at the
        # crossing, so the barrier drops ~H12). Expand the bound until it brackets.
        b0 = barrier(alpha, 0.0)
        if not np.isfinite(b0) or b0 <= dG_barrier_ref:
            h12 = 0.0  # target already at/below the diabatic upper bound
        else:
            hi = b0 - dG_barrier_ref + 1.0
            for _ in range(40):
                if barrier(alpha, hi) - dG_barrier_ref <= 0:
                    break
                hi *= 2.0
            try:
                h12 = brentq(
                    lambda h, a=alpha: barrier(a, h) - dG_barrier_ref,
                    0.0,
                    hi,
                    xtol=1e-4,
                )
            except ValueError:
                h12 = 0.0

        d_rxn, d_bar = rxn(alpha, h12), barrier(alpha, h12)
        if abs(d_rxn - dG_rxn_ref) < tol and abs(d_bar - dG_barrier_ref) < tol:
            return EVBCalibration(alpha, h12, d_rxn, d_bar, True)

    return EVBCalibration(alpha, h12, d_rxn, d_bar, False)


def ddg_barrier(mutant: EVBGapResult, reference: EVBGapResult) -> tuple[float, float]:
    """ddG_dagger = dG_barrier(mutant) - dG_barrier(reference), with error.

    Both results must be analyzed with the SAME calibrated (alpha, h12). The
    relative barrier is the screening observable: systematic errors largely cancel
    in the difference, and it maps to experiment via ddG_dagger ~ -RT ln(kcat_mut
    / kcat_wt). Errors are combined in quadrature (use per-result bootstrap or
    replica SEs).

    Returns:
        ``(ddG_dagger, error)`` in kJ/mol.
    """
    d = mutant.dG_barrier - reference.dG_barrier
    err = float(np.hypot(mutant.dG_barrier_err, reference.dG_barrier_err))
    return d, err


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
    soft_core: bool = False,
    sc_alpha: float = 0.5,
    seed: int | None = None,
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
        soft_core=soft_core,
        sc_alpha=sc_alpha,
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
        seed=seed,
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

    This class does **not** own the parsl DataFlowKernel: the caller loads and
    cleans up parsl, so its lifetime is managed at the scope where it is created.
    ``run`` submits to whatever DFK is currently loaded. A typical driver::

        import parsl
        from molecular_simulations.utils.parsl_settings import LocalSettings

        with parsl.load(LocalSettings(mpi_launcher=False).config_factory(rundir)):
            evb = EVBMapping(reactant, product, coords, out, ...)
            evb.run()
        result = evb.analyze()  # analysis needs no parsl

    Args:
        reactant_prmtop: Reactant diabatic topology.
        product_prmtop: Product diabatic topology (shared atoms/coords).
        coordinates: Shared coordinate file (AMBER inpcrd).
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
        soft_core: bool = False,
        sc_alpha: float = 0.5,
    ):
        """Initialize the EVBMapping orchestrator.

        ``donor``/``acceptor``/``reactive`` (0-based indices, from EVBBuilder's
        evb_meta.json) trigger Morse-ification of each state's reactive bond,
        which is required for the diabatic energy gap to stay bounded. Set
        ``soft_core=True`` to also bound the reactive atom's endpoint nonbonded
        clash (needed to sample the reactant basin near lambda=0 cleanly).
        """
        self.reactant_prmtop = str(Path(reactant_prmtop).resolve())
        self.product_prmtop = str(Path(product_prmtop).resolve())
        self.coordinates = str(Path(coordinates).resolve())
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.donor = donor
        self.acceptor = acceptor
        self.reactive = reactive
        self.D_e = D_e
        self.alpha = alpha
        self.r0 = r0
        self.soft_core = soft_core
        self.sc_alpha = sc_alpha

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

    def run(self, seed: int | None = None) -> list[str]:
        """Distribute the windows and wait for all to finish.

        Requires parsl to already be loaded by the caller (see the class
        docstring); this method neither loads nor cleans up the DataFlowKernel.

        Args:
            seed: Base random seed. Each window gets ``seed*1000 + i`` so windows
                are independent yet reproducible; pass distinct base seeds (with
                distinct ``out_path``\\ s) to generate replicas for
                :func:`aggregate_replicas`. None = nondeterministic.

        Returns:
            The per-window npz paths (window order = lambda order).
        """
        futures, out_files = [], []
        for i, lam in enumerate(self.lambdas):
            out_npz = str(self.out_path / f'window{i}.npz')
            out_files.append(out_npz)
            win_seed = None if seed is None else int(seed) * 1000 + i
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
                    self.soft_core,
                    self.sc_alpha,
                    win_seed,
                )
            )
        for fut in futures:
            fut.result()
        return out_files

    def analyze(
        self,
        h12: float = 0.0,
        n_bins: int = 50,
        n_boot: int = 0,
        alpha: float = 0.0,
    ) -> EVBGapResult:
        """Reduce the per-window npz files to the energy-gap free energy.

        Args:
            h12: Off-diagonal coupling (0 = diabatic upper-bound barrier).
            n_bins: Number of energy-gap bins.
            n_boot: Bootstrap resamples for the error bars (0 = none).
            alpha: EVB diabat-2 gas-phase shift (kJ/mol); calibrate with
                :func:`calibrate_evb`. Note this is unrelated to the Morse ``alpha``
                width passed at construction.

        Returns:
            The :class:`EVBGapResult`.
        """
        v1, v2 = [], []
        for i in range(len(self.lambdas)):
            data = np.load(self.out_path / f'window{i}.npz')
            v1.append(data['v1'])
            v2.append(data['v2'])
        return analyze_gap(
            self.lambdas,
            v1,
            v2,
            temperature=self.temperature,
            h12=h12,
            n_bins=n_bins,
            n_boot=n_boot,
            alpha=alpha,
        )
