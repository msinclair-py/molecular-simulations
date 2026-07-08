# ruff: noqa: B006, B008
"""Constant pH molecular dynamics simulations using OpenMM.

This module provides classes and functions for running constant pH simulations
with replica exchange across different pH values. It uses Parsl for distributed
execution of simulation replicas.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import parsl
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, Topology
from parsl import Config, python_app

from .constantph.constantph import ConstantPH
from .constantph.logging import setup_task_logger


@python_app
def run_cph_sim(
    params: dict[str, Any],
    temperature: float,
    n_cycles: int,
    n_steps: int,
    log_params: dict[str, str],
    path: str,
) -> None:
    """Run a constant pH simulation as a Parsl python app.

    This function executes a single constant pH simulation replica,
    performing Monte Carlo titration steps between MD propagation cycles.

    Args:
        params: Dictionary of ConstantPH parameters including topology,
            coordinates, residue variants, and reference energies.
        temperature: Simulation temperature as float.
        n_cycles: Number of MD/MC cycles to perform.
        n_steps: Number of MD steps per cycle.
        log_params: Dictionary containing logging configuration with
            run_id, task_id, and log_dir.
        path: Path to the simulation directory for logging purposes.
    """
    from openmm import LangevinIntegrator
    from openmm.app import PME, CutoffNonPeriodic, HBonds
    from openmm.unit import (  # ty: ignore[unresolved-import]
        amu,
        kelvin,
        kilojoules_per_mole,
        nanometers,
        picosecond,
    )

    variants = params['residueVariants']

    logger = setup_task_logger(**log_params)

    # build OpenMM stuff here to avoid serializing SWIG objects (bad idea)
    temp = temperature * kelvin  # ty: ignore[unsupported-operator]
    expl_params = dict(
        nonbondedMethod=PME,
        nonbondedCutoff=params['nonbonded_cutoff'] * nanometers,
        constraints=HBonds,
        hydrogenMass=params['hmr'] * amu,
    )
    impl_params = dict(
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=params['implicit_cutoff'] * nanometers,
        constraints=HBonds,
    )

    reference_energies = params['referenceEnergies']
    for key, vals in reference_energies.items():
        reference_energies[key] = [val * kilojoules_per_mole for val in vals]

    cph_params = {
        'prmtop_file': params['prmtop_file'],
        'inpcrd_file': params['inpcrd_file'],
        'pH': params['pH'],
        'relaxationSteps': params['relaxationSteps'],
        'explicitArgs': expl_params,
        'implicitArgs': impl_params,
        'integrator': LangevinIntegrator(temp, 1.0 / picosecond, 0.004 * picosecond),
        'relaxationIntegrator': LangevinIntegrator(
            temp, 10.0 / picosecond, 0.002 * picosecond
        ),
        'residueVariants': variants,
        'referenceEnergies': reference_energies,
    }

    cph = ConstantPH(**cph_params)
    cph.simulation.minimizeEnergy()
    cph.simulation.context.setVelocitiesToTemperature(temperature)

    resids = list(variants.keys())

    logger.info(
        f'Running {n_cycles} cycles of constant pH simulation!',
        extra={'path': path, 'pH': None, 'step': None, 'resids': resids},
    )
    for i in range(n_cycles):
        cph.simulation.step(n_steps)
        cph.attemptMCStep(temperature)
        pH = cph.pH[cph.currentPHIndex]
        current_variants = [
            variants[index][cph.titrations[index].currentIndex] for index in variants
        ]
        logger.info(
            f'Step complete {i}!',
            extra={'path': path, 'pH': pH, 'step': i, 'resnames': current_variants},
        )


class ConstantPHEnsemble:
    """Orchestrator for running constant pH simulation ensembles.

    Manages distributed execution of multiple constant pH simulation replicas
    using Parsl. Each replica can sample across a range of pH values with
    Monte Carlo titration moves.
    """

    def __init__(
        self,
        paths: list[Path],
        reference_energies: dict[str, list[float]],
        log_dir: Path | list[Path],
        pHs: list[float] = [x + 0.5 for x in range(14)],
        temperature: float = 300.0,
        platform: str = 'CUDA',
        properties: dict[str, str] = {'Precision': 'mixed'},
        parsl_config: Config | None = None,
        variant_sel: str | None = None,
    ):
        """Initialize the constant pH ensemble.

        Args:
            paths: List of paths to simulation directories, each containing
                system.prmtop and system.inpcrd files.
            reference_energies: Dictionary mapping residue names to lists of
                reference free energies for each protonation state.
            parsl_config: Parsl configuration for distributed execution.
            log_dir: Directory path for writing simulation logs.
            pHs: List of pH values to sample. Defaults to [0.5, 1.5, ..., 13.5].
            temperature: Simulation temperature in Kelvin. Defaults to 300.
            platform: OpenMM platform name to run on. Defaults to 'CUDA'.
            properties: Platform-specific properties passed to OpenMM. Defaults
                to {'Precision': 'mixed'}.
            variant_sel: Optional MDAnalysis selection string to limit which
                titratable residues are included. Defaults to None.
        """
        self.paths = paths
        self.ref_energies = reference_energies

        self.parsl_config = parsl_config
        self.dfk = None

        self.log_dir = log_dir
        self.pHs = pHs
        self.variant_sel = variant_sel

        self.temperature = temperature
        self.platform = platform
        self.properties = properties

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def initialize(self) -> None:
        """Initialize Parsl for distributed simulation execution."""
        self.dfk = parsl.load(self.parsl_config)

    def shutdown(self) -> None:
        """Clean up Parsl resources after simulation completion."""
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    def load_files(self, path: Path) -> tuple[Topology, np.ndarray]:
        """Load topology and coordinates from AMBER files.

        Args:
            path: Directory containing system.prmtop and system.inpcrd files.

        Returns:
            Tuple of (OpenMM Topology, atomic positions array).
        """
        prmtop = AmberPrmtopFile(str(path / 'system.prmtop'))
        inpcrd = AmberInpcrdFile(str(path / 'system.inpcrd'))

        return prmtop.topology, inpcrd.positions

    def build_dicts(
        self, path: Path, top: Topology
    ) -> tuple[dict[str, list[str]], dict[int, list[float]]]:
        """Build residue variant and reference energy dictionaries.

        Identifies titratable residues in the topology and maps them to
        their possible protonation states and corresponding reference energies.

        Args:
            path: Path to the structure file for MDAnalysis loading.
            top: OpenMM Topology object containing residue information.

        Returns:
            Tuple of (variants dict, reference_energies dict) where variants
            maps residue indices to lists of residue name variants and
            reference_energies maps residue indices to lists of reference
            free energies in kJ/mol.
        """
        _variants = {
            'CYS': ['CYX', 'CYS'],
            'ASP': ['ASP', 'ASH'],
            'GLU': ['GLU', 'GLH'],
            'LYS': ['LYN', 'LYS'],
            'HIS': ['HID', 'HIP'],
        }
        # A titratable residue may appear under its default name or, when the
        # system was built for constant pH, under its protonated variant. Map
        # both to the canonical key used by _variants and the reference-energy
        # table (ConstantPHSolvent builds ASP->ASH, GLU->GLH, HIS->HIP).
        _canonical = {name: name for name in _variants}
        _canonical.update({'ASH': 'ASP', 'GLH': 'GLU', 'HIP': 'HIS', 'LYN': 'LYS'})
        variants = {}
        reference_energies = {}
        for residue in top.residues():
            canonical = _canonical.get(residue.name)
            if canonical is not None:
                variants[residue.index] = _variants[canonical]
                reference_energies[residue.index] = [
                    x for x in self.ref_energies[canonical]
                ]

        u = mda.Universe(str(path / 'system.prmtop'), str(path / 'system.inpcrd'))
        sel = u.select_atoms('protein')
        bad_keys = [sel[0].resindex, sel[-1].resindex]  # termini

        if self.variant_sel is not None:
            var = u.select_atoms(self.variant_sel)
            bad_keys = [
                resid
                for resid in sel.residues.resindices
                if resid not in var.residues.resindices
            ]

        for bad_key in bad_keys:
            variants.pop(bad_key, None)
            reference_energies.pop(bad_key, None)

        return variants, reference_energies

    def run(
        self,
        n_cycles: int = 500,
        n_steps: int = 500,
        parsl_func: Callable = run_cph_sim,
    ) -> dict:
        """Run the constant pH simulation ensemble.

        Distributes simulation replicas across available resources using
        Parsl and waits for all replicas to complete.

        Args:
            n_cycles: Number of MD/MC cycles per replica. Defaults to 500.
            n_steps: Number of MD steps per cycle. Defaults to 500.
            parsl_func: Parsl python app used to run each replica. Defaults to
                run_cph_sim.

        Returns:
            Dictionary mapping each replica index to None on success or to the
            raised Exception on failure. Contains one entry per replica (it is
            not empty when all replicas succeed).
        """
        futures = []
        for i, path in enumerate(self.paths):
            top, _ = self.load_files(path)
            variants, reference_energies = self.build_dicts(path, top)

            cph_params = self.get_params(path)

            cph_params.update(
                {
                    'residueVariants': variants,
                    'referenceEnergies': reference_energies,
                }
            )

            if isinstance(self.log_dir, list):
                log_dir = self.log_dir[i]
            else:
                log_dir = self.log_dir

            log_params = {
                'run_id': self.run_id,
                'task_id': f'{i:05d}',
                'log_dir': log_dir,
            }

            futures.append(
                parsl_func(
                    cph_params,
                    self.temperature,
                    n_cycles,
                    n_steps,
                    log_params,
                    str(path),
                )
            )

        results = {}
        for i, future in enumerate(futures):
            try:
                future.result()
                results[i] = None  # Success
            except Exception as e:
                results[i] = e  # Failure

        return results

    def get_params(self, path: Path) -> dict[str, Any]:
        """Build the parameter dictionary for ConstantPH initialization.

        Args:
            path: Directory containing system.prmtop and system.inpcrd files.

        Returns:
            Dictionary containing the parameters needed to initialize a
            ConstantPH simulation: file paths (prmtop_file, inpcrd_file), pH
            values, relaxationSteps, the nonbonded and implicit cutoffs,
            hmr, platform_name, and properties.
        """
        params = {
            'prmtop_file': path / 'system.prmtop',
            'inpcrd_file': path / 'system.inpcrd',
            'pH': self.pHs,
            'relaxationSteps': 1000,
            'nonbonded_cutoff': 0.9,
            'hmr': 1.5,
            'implicit_cutoff': 2.0,
            'platform_name': self.platform,
            'properties': self.properties,
        }

        return params
