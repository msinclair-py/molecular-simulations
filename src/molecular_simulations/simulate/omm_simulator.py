# ruff: noqa: B006, SIM115
"""OpenMM molecular dynamics simulation module.

This module provides classes for running molecular dynamics simulations using
OpenMM with the AMBER force field. It supports both explicit and implicit
solvent simulations, with built-in equilibration protocols and production MD.

Classes:
    Simulator: Main class for explicit solvent OpenMM simulations.
    ImplicitSimulator: Simulator for implicit solvent (GB) simulations.
    CustomForcesSimulator: Simulator with user-defined custom forces.
    Minimizer: Simple energy minimization utility.
"""

import os
from copy import deepcopy
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    Platform,
    System,
)
from openmm.app import (
    PME,
    AmberInpcrdFile,
    AmberPrmtopFile,
    CheckpointReporter,
    DCDReporter,
    ForceField,
    GBn2,
    HBonds,
    NoCutoff,
    PDBFile,
    Simulation,
    StateDataReporter,
)
from openmm.unit import (
    amu,
    angstroms,
    bar,
    kelvin,
    kilocalories_per_mole,
    nanometer,  # ty: ignore[unresolved-import]
    nanometers,  # ty: ignore[unresolved-import]
    picosecond,  # ty: ignore[unresolved-import]
    picoseconds,  # ty: ignore[unresolved-import]
)

PathLike = Path | str
OptPath = Path | str | None

class Simulator:
    """Class for performing OpenMM simulations on AMBER FF inputs.

    Inputs must conform to naming conventions found below in the init.
    Supports explicit solvent simulations with PME electrostatics and
    hydrogen mass repartitioning for longer timesteps.

    Args:
        path: Path to simulation inputs, same as output path.
        top_name: Optional topology file name. If not provided, assumes
            'system.prmtop'.
        coor_name: Optional coordinate file name. If not provided, assumes
            'system.inpcrd'.
        out_path: Optional output path for simulation outputs. If not
            provided, uses the same path as inputs.
        ff: Force field to use. Only 'amber' is supported.
        heat_steps: Number of heating timesteps. Defaults to 100,000 (200 ps
            at 2 fs timestep).
        equil_steps: Number of equilibration timesteps. Defaults to 1,250,000
            (2.5 ns at 2 fs timestep).
        prod_steps: Number of production timesteps. Defaults to 250,000,000
            (1 µs at 4 fs timestep).
        prod_dt_in_ps: Production integration timestep in picoseconds. Defaults
            to 0.004.
        hydrogen_mass_repartitioning: Hydrogen mass in amu used for hydrogen
            mass repartitioning. Defaults to 1.5.
        n_equil_cycles: Number of unrestrained equilibration cycles after
            restraint relaxation. Defaults to 3.
        temperature: Simulation temperature in Kelvin. Defaults to 300.0.
        eq_reporter_frequency: Reporter frequency during equilibration in
            timesteps. Defaults to 1,000.
        prod_reporter_frequency: Reporter frequency during production in
            timesteps. Defaults to 10,000.
        platform: OpenMM platform to use. Options: 'CUDA', 'CPU', 'OpenCL'.
            Defaults to 'CUDA'.
        device_ids: List of GPU device IDs to use. Defaults to [0].
        force_constant: Harmonic restraint force constant in kcal/mol*Å².
            Defaults to 10.0.
        membrane: Whether this is a membrane system requiring anisotropic
            pressure coupling. Defaults to False.

    Attributes:
        path: Path object for simulation directory.
        top_file: Path to topology file.
        coor_file: Path to coordinate file.
        temperature: Simulation temperature.
        simulation: OpenMM Simulation object (available after production).

    Example:
        >>> sim = Simulator(
        ...     path='./simulation',
        ...     equil_steps=500_000,
        ...     prod_steps=50_000_000,
        ...     device_ids=[0, 1],
        ... )
        >>> sim.run()
    """

    def __init__(
        self,
        path: PathLike,
        top_name: str | None = None,
        coor_name: str | None = None,
        out_path: Path | None = None,
        ff: str = 'amber',
        heat_steps: int = 100_000,
        equil_steps: int = 1_250_000,
        prod_steps: int = 250_000_000,
        prod_dt_in_ps: float = 0.004,
        hydrogen_mass_repartitioning: float = 1.5,
        n_equil_cycles: int = 3,
        temperature: float = 300.0,
        eq_reporter_frequency: int = 1_000,
        prod_reporter_frequency: int = 10_000,
        platform: str = 'CUDA',
        device_ids: list[int] = [0],
        force_constant: float = 10.0,
        membrane: bool = False,
    ):
        self.path = Path(path)  # enforce path object
        self.top_file = (
            self.path / top_name
            if top_name is not None
            else self.path / 'system.prmtop'
        )
        self.coor_file = (
            self.path / coor_name
            if coor_name is not None
            else self.path / 'system.inpcrd'
        )
        self.temperature = temperature

        self.ff = ff.lower()
        self.setup_barostat(membrane)

        p = Path(out_path) if out_path is not None else self.path

        # make sure path exists
        p.mkdir(exist_ok=True, parents=True)

        # logging/checkpointing stuff
        self.eq_state = p / 'eq.state'
        self.eq_chkpt = p / 'eq.chk'
        self.eq_log = p / 'eq.log'
        self.eq_dcd = p / 'eq.dcd'
        self.eq_freq = eq_reporter_frequency

        self.dcd = p / 'prod.dcd'
        self.restart = p / 'prod.rst.chk'
        self.state = p / 'prod.state'
        self.chkpt = p / 'prod.chk'
        self.prod_log = p / 'prod.log'
        self.prod_freq = prod_reporter_frequency
        self.dt = prod_dt_in_ps
        self.h_mass = hydrogen_mass_repartitioning * amu  # ty: ignore[unsupported-operator]

        # simulation details
        self.heat_steps = heat_steps
        self.equil_cycles = n_equil_cycles
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.k = force_constant
        self.platform = Platform.getPlatformByName(platform)

        if platform == 'CPU':
            self.properties = {}

        elif platform == 'CUDA':
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # CUDA_VISIBLE_DEVICES (set by Parsl/the scheduler) already masks
                # GPUs and re-indexes them from 0, so the in-process device index
                # is always 0 regardless of the physical GPU assigned.
                device_index = '0'
            else:
                device_index = ','.join([str(x) for x in device_ids])

            self.properties = {'DeviceIndex': device_index, 'Precision': 'mixed'}

        elif platform == 'OpenCL':
            if device_id := os.environ.get('ZE_AFFINITY_MASK', False):
                if isinstance(device_id, list):
                    device_index = ','.join([str(x) for x in device_id])
                else:
                    device_index = str(device_id)
            else:
                device_index = ','.join([str(x) for x in device_ids])

            self.properties = {'Precision': 'mixed'}
            # self.properties = {'DeviceIndex': device_index,
            #                   'Precision': 'mixed',
            #                   'OpenCLPlatformIndex': '1'}
        else:
            raise AttributeError(f'Platform: {platform} not available!')

    def setup_barostat(self, is_membrane_system: bool) -> None:
        """Configure the barostat based on system type.

        Chooses the correct barostat for the system based on whether or not
        there is a membrane present. Membrane systems use MonteCarloMembraneBarostat
        with anisotropic pressure coupling.

        Args:
            is_membrane_system: True if this is a membrane-containing system.
        """
        self.barostat_args = {
            'defaultPressure': 1 * bar,  # ty: ignore[unsupported-operator]
        }

        if is_membrane_system:
            self.barostat = MonteCarloMembraneBarostat
            self.barostat_args.update(
                {
                    'defaultSurfaceTension': 0 * bar * nanometer,  # ty: ignore[unsupported-operator]
                    'defaultTemperature': self.temperature * kelvin,  # ty: ignore[unsupported-operator]
                    'xymode': MonteCarloMembraneBarostat.XYIsotropic,
                    'zmode': MonteCarloMembraneBarostat.ZFree,
                }
            )
        else:
            self.barostat = MonteCarloBarostat
            self.barostat_args.update({'temperature': self.temperature * kelvin})  # ty: ignore[unsupported-operator]

    def load_system(self) -> System:
        """Load the molecular system from AMBER inputs.

        Returns:
            OpenMM System object configured with the AMBER force field.

        Raises:
            AttributeError: If an unsupported force field type is specified.
        """
        if self.ff != 'amber':
            raise AttributeError("self.ff must be 'amber'!")

        system = self.load_amber_files()

        if not hasattr(self, 'indices'):
            self.indices = self.get_restraint_indices()

        return system

    def load_amber_files(self) -> System:
        """Build an OpenMM system from AMBER prmtop/inpcrd files.

        Uses PME for electrostatics with a 1 nm non-bonded cutoff and
        1.5 amu hydrogen mass repartitioning for longer timesteps.

        Returns:
            OpenMM System configured for AMBER force field.
        """
        if not hasattr(self, 'coordinate'):
            self.coordinate = AmberInpcrdFile(str(self.coor_file))
            self.topology = AmberPrmtopFile(
                str(self.top_file), periodicBoxVectors=self.coordinate.boxVectors
            )

        system = self.topology.createSystem(  # ty: ignore[missing-argument]
            nonbondedMethod=PME,
            removeCMMotion=False,
            nonbondedCutoff=1.0 * nanometer,
            constraints=HBonds,
            hydrogenMass=1.5 * amu,  # ty: ignore[unsupported-operator]
        )

        return system

    def setup_sim(self, system: System, dt: float) -> tuple[Simulation, LangevinMiddleIntegrator]:
        """Build OpenMM Simulation and Integrator objects.

        Creates a LangevinMiddleIntegrator with the specified timestep and
        builds the Simulation object with the configured platform.

        Args:
            system: OpenMM System object to simulate.
            dt: Integration timestep in picoseconds.

        Returns:
            Tuple containing (Simulation, LangevinMiddleIntegrator) objects.
        """
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin, 1 / picosecond, dt * picoseconds  # ty: ignore[unsupported-operator]
        )
        simulation = Simulation(
            self.topology.topology,
            system,
            integrator,
            platform=self.platform,
            platformProperties=self.properties,
        )

        return simulation, integrator

    def run(self) -> None:
        """Execute the full simulation workflow.

        Determines whether to restart based on existing equilibration outputs.
        Checks production log to resume from checkpoints if available. Runs
        equilibration if needed, then production MD.
        """
        # Store original total for progress reporting
        self.total_prod_steps = self.prod_steps

        skip_eq = all([f.exists() for f in [self.eq_state, self.eq_chkpt, self.eq_log]])
        if not skip_eq:
            self.equilibrate()

        if self.restart.exists():
            self.check_num_steps_left()
            self.production(chkpt=str(self.restart), restart=True)
        else:
            self.production(chkpt=str(self.eq_chkpt), restart=False)

    def equilibrate(self) -> Simulation:
        """Run the equilibration protocol.

        Performs energy minimization, slow heating, and gradual restraint
        relaxation followed by unrestrained NPT equilibration.

        Returns:
            Equilibrated OpenMM Simulation object.
        """
        system = self.load_system()
        system = self.add_backbone_posres(
            system,
            self.coordinate.positions,
            self.topology.topology.atoms(),
            self.indices,
            self.k,
        )

        simulation, integrator = self.setup_sim(system, dt=0.002)

        simulation.context.setPositions(self.coordinate.positions)
        simulation.minimizeEnergy()

        simulation.reporters.append(
            StateDataReporter(
                str(self.eq_log),
                self.eq_freq,
                step=True,
                potentialEnergy=True,
                speed=True,
                temperature=True,
            )
        )
        simulation.reporters.append(DCDReporter(str(self.eq_dcd), self.eq_freq))

        simulation, integrator = self._heating(simulation, integrator)
        simulation = self._equilibrate(simulation)

        return simulation

    def production(self, chkpt: PathLike, restart: bool = False) -> None:
        """Run production molecular dynamics.

        Loads a new system with barostat, loads checkpoint, attaches reporters,
        and runs production MD for the specified number of steps.

        Args:
            chkpt: Path to checkpoint file (equilibration or previous production).
            restart: If True, append to existing log/DCD files instead of
                overwriting. Defaults to False.
        """
        system = self.load_system()
        simulation, _ = self.setup_sim(system, dt=self.dt)

        system.addForce(self.barostat(*self.barostat_args.values()))
        simulation.context.reinitialize(True)

        log_file = open(self.prod_log, 'a') if restart else str(self.prod_log)

        simulation = self.load_checkpoint(simulation, chkpt)
        simulation = self.attach_reporters(
            simulation, self.dcd, log_file, str(self.restart), restart=restart
        )

        self.simulation = self._production(simulation)  # save simulation object

    def load_checkpoint(
        self, simulation: Simulation, checkpoint: PathLike
    ) -> Simulation:
        """Load a checkpoint into the simulation.

        Args:
            simulation: OpenMM Simulation object to load checkpoint into.
            checkpoint: Path to OpenMM checkpoint file.

        Returns:
            Simulation with restored positions, velocities, and step count.
        """
        simulation.loadCheckpoint(checkpoint)
        return simulation

    def attach_reporters(
        self,
        simulation: Simulation,
        dcd_file: PathLike,
        log_file: PathLike | TextIOWrapper,
        rst_file: PathLike,
        restart: bool = False,
    ) -> Simulation:
        """Attach trajectory, logging, and checkpoint reporters.

        Args:
            simulation: OpenMM Simulation object.
            dcd_file: Path for DCD trajectory output.
            log_file: Path for state data log output.
            rst_file: Path for checkpoint file output.
            restart: If True, append to existing DCD file. Defaults to False.

        Returns:
            Simulation with reporters attached.
        """
        # Use total_prod_steps for progress reporting if available (set by run()),
        # otherwise fall back to prod_steps for direct production() calls
        total_steps = getattr(self, 'total_prod_steps', self.prod_steps)

        simulation.reporters.extend(
            [
                DCDReporter(dcd_file, self.prod_freq, append=restart),
                StateDataReporter(
                    log_file,
                    self.prod_freq,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    progress=True,
                    remainingTime=True,
                    speed=True,
                    volume=True,
                    totalSteps=total_steps,
                    separator='\t',
                ),
                CheckpointReporter(rst_file, self.prod_freq * 10),
            ]
        )

        return simulation

    def _heating(
        self, simulation: Simulation, integrator: LangevinMiddleIntegrator
    ) -> tuple[Simulation, LangevinMiddleIntegrator]:
        """Perform slow heating protocol.

        Gradually heats the system from 5K to the target temperature over
        `self.heat_steps` timesteps in 1,000 discrete temperature increments.

        Args:
            simulation: OpenMM Simulation object.
            integrator: OpenMM LangevinMiddleIntegrator object.

        Returns:
            Tuple of (Simulation, LangevinMiddleIntegrator) after heating.
        """
        simulation.context.setVelocitiesToTemperature(5 * kelvin)  # ty: ignore[unsupported-operator]
        T = 5

        integrator.setTemperature(T * kelvin)  # ty: ignore[unsupported-operator]
        n_steps = 1000
        length = self.heat_steps // n_steps
        tstep = (self.temperature - T) / length
        for i in range(length):
            simulation.step(n_steps)
            temp = T + tstep * (1 + i)

            if temp > self.temperature:
                temp = self.temperature

            integrator.setTemperature(temp * kelvin)  # ty: ignore[unsupported-operator]

        return simulation, integrator

    def _equilibrate(self, simulation: Simulation) -> Simulation:
        """Run equilibration with restraint relaxation.

        Protocol:
        1. 5-step restraint relaxation in NVT (each step removes 1/5 of restraint)
        2. One step of unrestrained NVT
        3. Turn on barostat for NPT
        4. Run equil_cycles of NPT equilibration

        Args:
            simulation: OpenMM Simulation object.

        Returns:
            Equilibrated Simulation with saved state and checkpoint.
        """
        simulation.context.reinitialize(True)
        n_levels = 5
        d_k = self.k / n_levels
        eq_steps = self.equil_steps // (n_levels + self.equil_cycles)

        for i in range(n_levels):
            simulation.step(eq_steps)
            k = float(self.k - (i * d_k))
            simulation.context.setParameter(
                'k', (k * kilocalories_per_mole / angstroms**2)
            )

        simulation.context.setParameter('k', 0)
        simulation.step(eq_steps)

        simulation.system.addForce(self.barostat(*self.barostat_args.values()))
        simulation.step(self.equil_cycles * eq_steps)

        simulation.saveState(str(self.eq_state))
        simulation.saveCheckpoint(str(self.eq_chkpt))

        return simulation

    def _production(self, simulation: Simulation) -> Simulation:
        """Run production MD and save final state.

        Args:
            simulation: OpenMM Simulation object.

        Returns:
            Simulation after production run with saved state and checkpoint.
        """
        simulation.step(self.prod_steps)
        simulation.saveState(str(self.state))
        simulation.saveCheckpoint(str(self.chkpt))

        return simulation

    def get_restraint_indices(self, addtl_selection: str = '') -> list[int]:
        """Get atom indices for harmonic restraints.

        Uses MDAnalysis to select protein/nucleic acid backbone atoms.
        Additional selections can be included for ligand restraints.

        Args:
            addtl_selection: Additional MDAnalysis selection string to include
                in restraints. Defaults to empty string.

        Returns:
            List of atom indices to restrain.
        """
        u = mda.Universe(str(self.top_file), str(self.coor_file))
        if addtl_selection:
            sel = u.select_atoms(f'backbone or nucleicbackbone or {addtl_selection}')
        else:
            sel = u.select_atoms('backbone or nucleicbackbone')

        return sel.atoms.ix

    def check_num_steps_left(self) -> None:
        """Check production log to determine remaining simulation steps.

        Reads the log file to find the last completed step, then calculates
        the remaining steps needed. Also handles duplicate frames that may
        occur when restarting from checkpoints (since checkpoint frequency
        may not align with reporter frequency).
        """
        with open(str(self.prod_log)) as f:
            prod_log = f.readlines()

        try:
            last_line = prod_log[-1]
            last_step = int(last_line.split()[1].strip())
        except (IndexError, ValueError):
            try:
                last_line = prod_log[-2]
                last_step = int(last_line.split()[1].strip())
            except (
                IndexError,
                ValueError,
            ):  # something weird happened just run full time
                return

        # Calculate steps remaining from the checkpoint position
        # Checkpoint is written every prod_freq * 10 steps, so checkpoint_step
        # is the most recent multiple of checkpoint_freq before last_step
        checkpoint_freq = self.prod_freq * 10
        n_repeat_timesteps = last_step % checkpoint_freq
        time_left_from_log = self.prod_steps - last_step

        if time_left_from_log > 0:
            # Add back steps that will be re-run from checkpoint
            self.prod_steps = time_left_from_log + n_repeat_timesteps

            # Log duplicate frames that will be created
            if n_repeat_timesteps:
                n_repeat_frames = n_repeat_timesteps // self.prod_freq
                n_total_frames = last_step // self.prod_freq

                lines = [f'{n_total_frames - n_repeat_frames},{n_total_frames}']
                duplicate_log = self.path / 'duplicate_frames.log'
                if duplicate_log.exists():
                    mode = 'a'
                else:
                    mode = 'w'
                    lines = ['first_frame,last_frame', *lines]

                with open(str(duplicate_log), mode) as fout:
                    fout.write('\n'.join(lines))

    @staticmethod
    def add_backbone_posres(
        system: System,
        positions: Any,
        atoms: Any,
        indices: list[int],
        restraint_force: float = 10.0,
    ) -> System:
        """Add harmonic position restraints to selected atoms.

        Args:
            system: OpenMM System object.
            positions: Position array for all atoms (with units).
            atoms: List of OpenMM Atom objects from topology.
            indices: List of atom indices to restrain.
            restraint_force: Force constant in kcal/mol*Å². Defaults to 10.0.

        Returns:
            Copy of System with harmonic restraints added.
        """
        force = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')

        force_amount = restraint_force * kilocalories_per_mole / angstroms**2
        force.addGlobalParameter('k', force_amount)
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')

        for i, (atom_crd, atom) in enumerate(zip(positions, atoms, strict=True)):
            if atom.index in indices:
                force.addParticle(i, atom_crd.value_in_unit(nanometers))

        posres_sys = deepcopy(system)
        posres_sys.addForce(force)

        return posres_sys


class ImplicitSimulator(Simulator):
    """Simulator for implicit solvent (Generalized Born) simulations.

    Inherits from Simulator and overloads methods for implicit solvent
    compatibility. Uses GBn2 model by default with ionic screening.

    Args:
        path: Path to simulation inputs, same as output path.
        top_name: Optional topology file name. If not provided, assumes
            'system.prmtop'.
        coor_name: Optional coordinate file name. If not provided, assumes
            'system.inpcrd'.
        out_path: Optional output path for simulation outputs.
        ff: Force field to use. Only 'amber' is supported.
        equil_steps: Number of equilibration timesteps. Defaults to 1,250,000.
        prod_steps: Number of production timesteps. Defaults to 250,000,000.
        n_equil_cycles: Number of unrestrained equilibration cycles.
        temperature: Simulation temperature in Kelvin. Defaults to 300.0.
        eq_reporter_frequency: Reporter frequency during equilibration.
        prod_reporter_frequency: Reporter frequency during production.
        platform: OpenMM platform to use. Defaults to 'CUDA'.
        device_ids: List of GPU device IDs to use. Defaults to [0].
        force_constant: Harmonic restraint force constant in kcal/mol*Å².
        implicit_solvent: GB model to use. Defaults to GBn2.
        solute_dielectric: Solute dielectric constant. Defaults to 1.0.
        solvent_dielectric: Solvent dielectric constant. Defaults to 78.5.

    Example:
        >>> sim = ImplicitSimulator(
        ...     path='./simulation', implicit_solvent=GBn2, prod_steps=100_000_000
        ... )
        >>> sim.run()
    """

    def __init__(
        self,
        path: PathLike,
        top_name: str | None = None,
        coor_name: str | None = None,
        out_path: Path | None = None,
        ff: str = 'amber',
        equil_steps: int = 1_250_000,
        prod_steps: int = 250_000_000,
        n_equil_cycles: int = 3,
        temperature: float = 300.0,
        eq_reporter_frequency: int = 1_000,
        prod_reporter_frequency: int = 10_000,
        platform: str = 'CUDA',
        device_ids: list[int] = [0],
        force_constant: float = 10.0,
        implicit_solvent: Any = GBn2,
        solute_dielectric: float = 1.0,
        solvent_dielectric: float = 78.5,
        **kwargs: Any,
    ):
        super().__init__(
            path=path,
            top_name=top_name,
            coor_name=coor_name,
            out_path=out_path,
            ff=ff,
            equil_steps=equil_steps,
            prod_steps=prod_steps,
            n_equil_cycles=n_equil_cycles,
            temperature=temperature,
            eq_reporter_frequency=eq_reporter_frequency,
            prod_reporter_frequency=prod_reporter_frequency,
            platform=platform,
            device_ids=device_ids,
            force_constant=force_constant,
        )
        self.solvent = implicit_solvent
        self.solute_dielectric = solute_dielectric
        self.solvent_dielectric = solvent_dielectric
        # solvent screening parameter for 150mM ions
        # k = 367.434915 * sqrt(conc. [M] / (solvent_dielectric * T))
        self.kappa = 367.434915 * np.sqrt(0.15 / (solvent_dielectric * 300))

    def load_amber_files(self) -> System:
        """Build an OpenMM system with implicit solvent.

        Uses the specified GB model with ionic screening for 150mM salt.

        Returns:
            OpenMM System configured for implicit solvent simulation.
        """
        if not hasattr(self, 'coordinate'):
            self.coordinate = AmberInpcrdFile(str(self.coor_file))
            self.topology = AmberPrmtopFile(str(self.top_file))

        system = self.topology.createSystem(  # ty: ignore[missing-argument]
            nonbondedMethod=NoCutoff,
            removeCMMotion=False,
            constraints=HBonds,
            hydrogenMass=self.h_mass,
            implicitSolvent=self.solvent,
            soluteDielectric=self.solute_dielectric,
            solventDielectric=self.solvent_dielectric,
            implicitSolventKappa=self.kappa / nanometer,
        )

        return system

    def equilibrate(self) -> Simulation:
        """Run reduced equilibration protocol for implicit solvent.

        Due to faster convergence with implicit solvent, uses a simplified
        equilibration protocol compared to explicit solvent.

        Returns:
            Equilibrated OpenMM Simulation object.
        """
        system = self.load_system()
        system = self.add_backbone_posres(
            system,
            self.coordinate.positions,
            self.topology.topology.atoms(),
            self.indices,
            self.k,
        )

        simulation, integrator = self.setup_sim(system, dt=0.002)

        simulation.context.setPositions(self.coordinate.positions)
        state = simulation.context.getState(getEnergy=True)
        print(f'Energy before minimization: {state.getPotentialEnergy()}')
        simulation.minimizeEnergy()
        state = simulation.context.getState(getEnergy=True)
        print(f'Energy after minimization: {state.getPotentialEnergy()}')

        simulation.reporters.append(
            StateDataReporter(
                str(self.eq_log),
                self.eq_freq,
                step=True,
                potentialEnergy=True,
                speed=True,
                temperature=True,
            )
        )
        simulation.reporters.append(DCDReporter(str(self.eq_dcd), self.eq_freq))

        simulation, integrator = self._heating(simulation, integrator)
        simulation = self._equilibrate(simulation)

        return simulation

    def production(self, chkpt: PathLike, restart: bool = False) -> None:
        """Run production MD for implicit solvent (no barostat).

        Args:
            chkpt: Path to checkpoint file.
            restart: If True, append to existing files. Defaults to False.
        """
        system = self.load_system()
        simulation, _ = self.setup_sim(system, dt=0.004)

        simulation.context.reinitialize(True)

        log_file = open(self.prod_log, 'a') if restart else str(self.prod_log)

        simulation = self.load_checkpoint(simulation, chkpt)
        simulation = self.attach_reporters(
            simulation, self.dcd, log_file, str(self.restart), restart=restart
        )

        self.simulation = self._production(simulation)  # save simulation object


class CustomForcesSimulator(Simulator):
    """Simulator with user-defined custom forces.

    Inherits from Simulator and provides injection of custom OpenMM forces
    for enhanced sampling or specialized simulations.

    Args:
        path: Path to simulation inputs.
        custom_force_objects: List of OpenMM Force objects to add to system.
        equil_steps: Number of equilibration timesteps. Defaults to 1,250,000.
        prod_steps: Number of production timesteps. Defaults to 250,000,000.
        n_equil_cycles: Number of unrestrained equilibration cycles.
        reporter_frequency: Reporter frequency in timesteps. Defaults to 1,000.
        platform: OpenMM platform to use. Defaults to 'CUDA'.
        device_ids: List of GPU device IDs. Defaults to [0].
        equilibration_force_constant: Restraint force constant in kcal/mol*Å².

    Example:
        >>> from openmm import CustomBondForce
        >>> force = CustomBondForce('0.5*k*(r-r0)^2')
        >>> force.addGlobalParameter('k', 1000)
        >>> force.addGlobalParameter('r0', 0.3)
        >>> sim = CustomForcesSimulator('./sim', [force])
        >>> sim.run()
    """

    def __init__(
        self,
        path: PathLike,
        custom_force_objects: list,
        equil_steps: int = 1_250_000,
        prod_steps: int = 250_000_000,
        n_equil_cycles: int = 3,
        reporter_frequency: int = 1_000,
        platform: str = 'CUDA',
        device_ids: list[int] = [0],
        equilibration_force_constant: float = 10.0,
    ):
        super().__init__(
            path=path,
            equil_steps=equil_steps,
            prod_steps=prod_steps,
            n_equil_cycles=n_equil_cycles,
            eq_reporter_frequency=reporter_frequency,
            prod_reporter_frequency=reporter_frequency,
            platform=platform,
            device_ids=device_ids,
            force_constant=equilibration_force_constant,
        )
        self.custom_forces = custom_force_objects

    def load_amber_files(self) -> System:
        """Load OpenMM system and add custom forces.

        Returns:
            OpenMM System with custom forces added.
        """
        if not hasattr(self, 'coordinate'):
            self.coor_file = self.path / 'system.inpcrd'
            self.top_file = self.path / 'system.prmtop'
            self.coordinate = AmberInpcrdFile(str(self.coor_file))
            self.topology = AmberPrmtopFile(
                str(self.top_file), periodicBoxVectors=self.coordinate.boxVectors
            )

        system = self.topology.createSystem(  # ty: ignore[missing-argument]
            nonbondedMethod=PME,
            removeCMMotion=False,
            nonbondedCutoff=1.0 * nanometer,
            constraints=HBonds,
            hydrogenMass=1.5 * amu,  # ty: ignore[unsupported-operator]
        )

        system = self.add_forces(system)

        return system

    def add_forces(self, system: System) -> System:
        """Add all custom forces to the system.

        Args:
            system: OpenMM System object.

        Returns:
            System with all custom forces added.
        """
        for custom_force in self.custom_forces:
            system.addForce(custom_force)

        return system


class Minimizer:
    """Simple energy minimization utility for molecular structures.

    Supports AMBER and PDB input formats. Performs energy
    minimization and writes out the minimized structure as a PDB file.

    Args:
        topology: Path to topology file (prmtop, top, or pdb).
        coordinates: Path to coordinate file (inpcrd, gro, or same pdb).
        out: Output filename for minimized structure. Defaults to 'min.pdb'.
        platform: OpenMM platform to use. Defaults to 'OpenCL'.
        device_ids: List of GPU device IDs, or None for CPU. Defaults to [0].

    Example:
        >>> minimizer = Minimizer(
        ...     topology='system.prmtop',
        ...     coordinates='system.inpcrd',
        ...     out='minimized.pdb',
        ... )
        >>> minimizer.minimize()
    """

    def __init__(
        self,
        topology: PathLike,
        coordinates: PathLike,
        out: PathLike = 'min.pdb',
        platform: str = 'OpenCL',
        device_ids: list[int] | None = [0],
    ):
        self.topology: Any = Path(topology)
        self.coordinates: Any = Path(coordinates)

        self.path = Path(topology).parent
        self.out = self.path / out
        self.platform = Platform.getPlatformByName(platform)
        self.properties = {'Precision': 'mixed'}

        if device_ids is not None:
            self.properties.update(
                {'DeviceIndex': ','.join([str(x) for x in device_ids])}
            )

    def minimize(self) -> None:
        """Perform energy minimization and save structure.

        Loads the system, runs OpenMM energy minimization, and writes
        the final coordinates to a PDB file.
        """
        system = self.load_files()
        integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1 / picosecond, 0.001 * picoseconds  # ty: ignore[unsupported-operator]
        )
        simulation = Simulation(
            self.topology, system, integrator, self.platform, self.properties
        )

        simulation.context.setPositions(self.coordinates.positions)

        simulation.minimizeEnergy()

        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()

        PDBFile.writeFile(
            self.topology.topology, positions, file=str(self.out), keepIds=True
        )

    def load_files(self) -> System:
        """Load system based on topology file extension.

        Returns:
            OpenMM System object.

        Raises:
            FileNotFoundError: If no valid input files are found.
        """
        match self.topology.suffix:
            case '.prmtop':
                system = self.load_amber()
            case '.pdb':
                system = self.load_pdb()
            case _:
                raise FileNotFoundError(
                    f'No viable simulation input files foundat path: {self.path}!'
                )

        return system

    def load_amber(self) -> System:
        """Load AMBER input files into OpenMM System.

        Returns:
            OpenMM System object from AMBER files.
        """
        self.coordinates = AmberInpcrdFile(str(self.coordinates))
        self.topology = AmberPrmtopFile(
            str(self.topology), periodicBoxVectors=self.coordinates.boxVectors
        )

        system = self.topology.createSystem(
            nonbondedMethod=NoCutoff, constraints=HBonds
        )

        return system

    def load_pdb(self) -> System:
        """Load PDB file into OpenMM System.

        Uses AMBER14 force field with automatic topology generation.

        Returns:
            OpenMM System object from PDB file.
        """
        self.coordinates = PDBFile(str(self.topology))
        self.topology = self.coordinates.topology
        forcefield = ForceField('amber14-all.xml')

        system = forcefield.createSystem(
            self.topology, nonbondedMethod=NoCutoff, constraints=HBonds
        )

        return system
