"""
Unit tests for simulate/omm_simulator.py module

This module contains both unit tests (with mocks) and integration tests that use
real OpenMM when available. Integration tests are marked with @pytest.mark.requires_openmm
and will be skipped if OpenMM is not installed or if no suitable platform is available.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Fixtures for conditional real OpenMM usage
# ============================================================================


def _amber_kwargs(real_amber_system_files, **extra):
    """Constructor kwargs pointing a Simulator at the real Ace-Ala-Nme system.

    Uses the committed tleap prmtop/inpcrd (copied into tmp_path by the
    real_amber_system_files fixture) on the real CPU platform, so simulators can
    be built and run for real instead of mocking Platform/AmberPrmtopFile.
    """
    return {
        "path": real_amber_system_files["path"],
        "top_name": "ala_dipeptide.prmtop",
        "coor_name": "ala_dipeptide.inpcrd",
        "platform": "CPU",
        **extra,
    }


def _build_real_implicit_simulation(
    real_amber_system_files, dt=0.002, minimize=True, **extra
):
    """Build a real CPU ImplicitSimulator + Simulation on the Ace-Ala-Nme system.

    The committed Ace-Ala-Nme inputs are an implicit-solvent/vacuum system with
    no periodic box, so the implicit (NoCutoff, GBn2) simulator is the one that
    can actually run on the CPU platform. Returns ``(sim, simulation, integrator)``
    with positions set (and optionally lightly minimized for numerical stability).
    """
    from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

    sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files, **extra))
    system = sim.load_system()
    simulation, integrator = sim.setup_sim(system, dt=dt)
    simulation.context.setPositions(sim.coordinate.positions)
    if minimize:
        simulation.minimizeEnergy(maxIterations=50)

    return sim, simulation, integrator


def _build_real_explicit_simulation(real_amber_explicit_files, **extra):
    """Build a real CPU Simulator + restrained Simulation on the solvated box.

    The solvated Ace-Ala-Nme carries a periodic box, so PME and the NPT
    MonteCarloBarostat added by _equilibrate run for real. Adds the backbone
    position restraint (the global 'k' parameter _equilibrate relaxes) and lightly
    minimizes. Returns ``(sim, simulation)``.
    """
    from molecular_simulations.simulate.omm_simulator import Simulator

    sim = Simulator(
        path=real_amber_explicit_files["path"],
        top_name="ala_dipeptide_solv.prmtop",
        coor_name="ala_dipeptide_solv.inpcrd",
        platform="CPU",
        **extra,
    )
    system = sim.load_system()
    system = sim.add_backbone_posres(
        system,
        sim.coordinate.positions,
        sim.topology.topology.atoms(),
        sim.indices,
        sim.k,
    )
    simulation, _ = sim.setup_sim(system, dt=0.002)
    simulation.context.setPositions(sim.coordinate.positions)
    simulation.minimizeEnergy(maxIterations=20)

    return sim, simulation


def _check_openmm_available():
    """Check if OpenMM is available with a working platform."""
    try:
        from openmm import Platform

        # Try to get CPU platform which should always work
        Platform.getPlatformByName("CPU")
        return True
    except Exception:
        return False


def _has_opencl():
    """Check whether an OpenCL OpenMM platform is registered.

    CI's pip-installed OpenMM ships only Reference and CPU, so real-OpenCL tests
    must skip there rather than fail.
    """
    try:
        from openmm import Platform

        Platform.getPlatformByName("OpenCL")
        return True
    except Exception:
        return False


# Custom marker for tests requiring OpenMM
requires_openmm = pytest.mark.skipif(
    not _check_openmm_available(), reason="OpenMM not available or no working platform"
)


@pytest.fixture
def openmm_platform():
    """Return a real OpenMM CPU platform if available, else skip."""
    try:
        from openmm import Platform

        return Platform.getPlatformByName("CPU")
    except Exception:
        pytest.skip("OpenMM CPU platform not available")


@pytest.fixture
def simple_pdb_system(tmp_path):
    """Create a simple PDB file for testing with real OpenMM.

    Returns a path to a valid ACE-ALA-NME (alanine dipeptide) PDB file
    that can be loaded by OpenMM with AMBER force field.
    """
    # Alanine dipeptide with ACE and NME caps - standard test system
    # PDB format: columns must be properly aligned for OpenMM
    pdb_content = """\
HEADER    ALANINE DIPEPTIDE
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
ATOM      1 HH31 ACE A   1      -2.060   0.018   0.016  1.00  0.00           H
ATOM      2  CH3 ACE A   1      -1.506  -0.034   0.964  1.00  0.00           C
ATOM      3 HH32 ACE A   1      -1.834   0.778   1.609  1.00  0.00           H
ATOM      4 HH33 ACE A   1      -1.774  -0.979   1.458  1.00  0.00           H
ATOM      5  C   ACE A   1       0.000   0.000   0.692  1.00  0.00           C
ATOM      6  O   ACE A   1       0.627  -0.959   0.253  1.00  0.00           O
ATOM      7  N   ALA A   2       0.527   1.120   1.199  1.00  0.00           N
ATOM      8  H   ALA A   2      -0.016   1.915   1.515  1.00  0.00           H
ATOM      9  CA  ALA A   2       1.955   1.244   1.354  1.00  0.00           C
ATOM     10  HA  ALA A   2       2.412   0.485   0.715  1.00  0.00           H
ATOM     11  CB  ALA A   2       2.270   0.955   2.824  1.00  0.00           C
ATOM     12  HB1 ALA A   2       1.825   1.714   3.469  1.00  0.00           H
ATOM     13  HB2 ALA A   2       3.350   0.948   2.965  1.00  0.00           H
ATOM     14  HB3 ALA A   2       1.869  -0.022   3.102  1.00  0.00           H
ATOM     15  C   ALA A   2       2.468   2.632   0.949  1.00  0.00           C
ATOM     16  O   ALA A   2       1.694   3.586   0.826  1.00  0.00           O
ATOM     17  N   NME A   3       3.771   2.755   0.730  1.00  0.00           N
ATOM     18  H   NME A   3       4.366   1.944   0.817  1.00  0.00           H
ATOM     19  CH3 NME A   3       4.343   4.057   0.369  1.00  0.00           C
ATOM     20 HH31 NME A   3       5.352   3.897   0.003  1.00  0.00           H
ATOM     21 HH32 NME A   3       3.741   4.532  -0.410  1.00  0.00           H
ATOM     22 HH33 NME A   3       4.379   4.719   1.238  1.00  0.00           H
TER
END
"""
    pdb_file = tmp_path / "test_system.pdb"
    pdb_file.write_text(pdb_content)
    return pdb_file


# ============================================================================
# Integration tests using real OpenMM (when available)
# ============================================================================


class TestOpenMMIntegration:
    """Integration tests using real OpenMM objects.

    These tests verify actual OpenMM behavior rather than mocked interactions.
    They are skipped if OpenMM is not available.
    """

    @requires_openmm
    def test_real_platform_initialization(self, openmm_platform):
        """Test that we can get a real OpenMM Platform."""
        from openmm import Platform

        assert openmm_platform is not None
        assert openmm_platform.getName() == "CPU"

        # Verify we can access platform properties
        num_platforms = Platform.getNumPlatforms()
        assert num_platforms >= 1

    @requires_openmm
    def test_real_simulation_setup_from_pdb(self, simple_pdb_system, tmp_path):
        """Test creating a real Simulation object from a PDB file."""
        from openmm import LangevinMiddleIntegrator, Platform
        from openmm.app import ForceField, NoCutoff, PDBFile, Simulation
        from openmm.unit import kelvin, picoseconds

        # Load PDB
        pdb = PDBFile(str(simple_pdb_system))

        # Create force field and system
        ff = ForceField("amber14-all.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff)

        # Create integrator
        integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1.0 / picoseconds, 0.002 * picoseconds
        )

        # Create simulation with CPU platform
        platform = Platform.getPlatformByName("CPU")
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Verify simulation was created
        assert simulation is not None
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        assert energy is not None

    @requires_openmm
    def test_real_minimization(self, simple_pdb_system, tmp_path):
        """Test real energy minimization."""
        from openmm import LangevinMiddleIntegrator, Platform
        from openmm.app import ForceField, NoCutoff, PDBFile, Simulation
        from openmm.unit import kelvin, picoseconds

        pdb = PDBFile(str(simple_pdb_system))
        ff = ForceField("amber14-all.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
        integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1.0 / picoseconds, 0.002 * picoseconds
        )

        platform = Platform.getPlatformByName("CPU")
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Get initial energy
        state_before = simulation.context.getState(getEnergy=True)
        energy_before = state_before.getPotentialEnergy()

        # Minimize
        simulation.minimizeEnergy(maxIterations=100)

        # Get final energy
        state_after = simulation.context.getState(getEnergy=True)
        energy_after = state_after.getPotentialEnergy()

        # Energy should decrease or stay the same after minimization
        assert energy_after <= energy_before

    @requires_openmm
    def test_real_integrator_creation(self):
        """Test creating real OpenMM integrators with proper units."""
        from openmm import LangevinMiddleIntegrator
        from openmm.unit import kelvin, picoseconds

        temperature = 300.0 * kelvin
        friction = 1.0 / picoseconds
        timestep = 0.002 * picoseconds

        integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

        assert integrator is not None
        assert integrator.getTemperature() == temperature
        assert integrator.getStepSize() == timestep

    @requires_openmm
    def test_real_barostat_creation(self):
        """Test creating real OpenMM MonteCarloBarostat."""
        from openmm import MonteCarloBarostat
        from openmm.unit import bar, kelvin

        pressure = 1.0 * bar
        temperature = 300.0 * kelvin
        frequency = 25

        barostat = MonteCarloBarostat(pressure, temperature, frequency)

        assert barostat is not None
        assert barostat.getDefaultPressure() == pressure
        assert barostat.getDefaultTemperature() == temperature

    @requires_openmm
    def test_real_custom_external_force(self):
        """Test creating real CustomExternalForce for position restraints."""
        from openmm import CustomExternalForce

        # Create the restraint force expression
        force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")

        # Add parameters
        force.addGlobalParameter("k", 1000.0)  # kJ/mol/nm^2
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        # Add a particle
        force.addParticle(0, [0.0, 0.0, 0.0])

        assert force is not None
        assert force.getNumParticles() == 1
        assert force.getNumGlobalParameters() == 1
        assert force.getNumPerParticleParameters() == 3


class TestSimulatorInit:
    """Real Simulator construction tests (CPU platform; __init__ stores paths).

    Simulator.__init__ resolves the platform and stores paths without parsing
    the topology, so these need only the real CPU platform and a directory.
    """

    def test_simulator_init_defaults(self, tmp_path):
        """Test Simulator initialization with defaults"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(path=tmp_path, platform="CPU")

        assert sim.path == tmp_path
        assert sim.top_file == tmp_path / "system.prmtop"
        assert sim.coor_file == tmp_path / "system.inpcrd"
        assert sim.temperature == 300.0
        assert sim.equil_steps == 1_250_000
        assert sim.prod_steps == 250_000_000

    def test_simulator_init_custom_files(self, tmp_path):
        """Test Simulator initialization with custom file names"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(
            path=tmp_path,
            top_name="custom.prmtop",
            coor_name="custom.inpcrd",
            platform="CPU",
        )

        assert sim.top_file == tmp_path / "custom.prmtop"
        assert sim.coor_file == tmp_path / "custom.inpcrd"

    def test_simulator_init_custom_output_path(self, tmp_path):
        """Test Simulator initialization with custom output path"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        out_path = tmp_path / "output"
        out_path.mkdir()

        sim = Simulator(path=tmp_path, out_path=out_path, platform="CPU")

        assert sim.dcd == out_path / "prod.dcd"
        assert sim.prod_log == out_path / "prod.log"

    def test_simulator_init_cpu_platform(self, tmp_path):
        """Test Simulator initialization with the real CPU platform"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(path=tmp_path, platform="CPU")

        # The real CPU platform takes no precision properties
        assert sim.properties == {}


class TestSimulatorBarostat:
    """Test suite for Simulator barostat setup"""

    def test_setup_barostat_non_membrane(self, tmp_path):
        """Test barostat setup for non-membrane system"""
        from openmm import MonteCarloBarostat

        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(path=tmp_path, platform="CPU", membrane=False)

        assert sim.barostat is MonteCarloBarostat
        assert "temperature" in sim.barostat_args

    @pytest.mark.skip(reason="Source code has bug - nm is not imported")
    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    @patch("molecular_simulations.simulate.omm_simulator.MonteCarloMembraneBarostat")
    def test_setup_barostat_membrane(self, mock_membrane_barostat, mock_platform):
        """Test barostat setup for membrane system"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            sim = Simulator(path=path, membrane=True)

            assert "defaultSurfaceTension" in sim.barostat_args
            assert "xymode" in sim.barostat_args
            assert "zmode" in sim.barostat_args


class TestSimulatorLoadSystem:
    """Test suite for Simulator load_system method"""

    def test_load_system_invalid_ff(self, tmp_path):
        """Test load_system with invalid force field"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # A non-amber ff hits the dispatch error branch before any file is parsed.
        sim = Simulator(path=tmp_path, platform="CPU", ff="invalid")

        with pytest.raises(AttributeError, match="amber"):
            sim.load_system()

    def test_load_amber_files(self, real_amber_explicit_files):
        """load_amber_files builds a real PME system from the solvated box."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(
            path=real_amber_explicit_files["path"],
            top_name="ala_dipeptide_solv.prmtop",
            coor_name="ala_dipeptide_solv.inpcrd",
            platform="CPU",
        )
        system = sim.load_amber_files()

        # Real solvated system: 913 atoms with a periodic box (PME)
        assert system.getNumParticles() == 913
        assert system.usesPeriodicBoundaryConditions()


class TestSimulatorSetupSim:
    """Test suite for Simulator setup_sim method"""

    def test_setup_sim(self, real_amber_system_files):
        """Test setup_sim builds a real Simulation and integrator on CPU."""
        from openmm import LangevinMiddleIntegrator
        from openmm.app import Simulation
        from openmm.unit import kelvin, picoseconds

        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files))
        system = sim.load_system()

        simulation, integrator = sim.setup_sim(system, dt=0.002)

        assert isinstance(simulation, Simulation)
        assert isinstance(integrator, LangevinMiddleIntegrator)
        assert integrator.getStepSize().value_in_unit(picoseconds) == 0.002
        assert integrator.getTemperature().value_in_unit(kelvin) == 300.0
        assert simulation.system.getNumParticles() == 22


class TestSimulatorCheckpoint:
    """Test suite for Simulator checkpoint methods"""

    def test_load_checkpoint(self, real_amber_system_files):
        """Test load_checkpoint restores a real OpenMM checkpoint.

        loadCheckpoint() restores positions, velocities, and step count, so a
        real round-trip (save then load) verifies the method end to end.
        """
        sim, simulation, _ = _build_real_implicit_simulation(real_amber_system_files)

        chkpt = real_amber_system_files["path"] / "test.chk"
        simulation.step(5)
        simulation.saveCheckpoint(str(chkpt))
        saved_step = simulation.currentStep

        # Advance further, then load the checkpoint to roll back the step count.
        simulation.step(5)
        assert simulation.currentStep != saved_step

        result = sim.load_checkpoint(simulation, str(chkpt))

        assert result is simulation
        assert simulation.currentStep == saved_step


class TestSimulatorReporters:
    """Test suite for Simulator reporter methods"""

    def test_attach_reporters(self, real_amber_system_files):
        """Test attach_reporters attaches three real OpenMM reporters."""
        from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter

        path = real_amber_system_files["path"]
        sim, simulation, _ = _build_real_implicit_simulation(real_amber_system_files)
        simulation.reporters = []

        dcd_file = path / "test.dcd"
        log_file = path / "test.log"
        rst_file = path / "test.rst"

        result = sim.attach_reporters(
            simulation, str(dcd_file), str(log_file), str(rst_file)
        )

        assert len(result.reporters) == 3
        assert isinstance(result.reporters[0], DCDReporter)
        assert isinstance(result.reporters[1], StateDataReporter)
        assert isinstance(result.reporters[2], CheckpointReporter)
        # The DCD reporter creates its output file on construction.
        assert dcd_file.exists()


class TestSimulatorRestraintIndices:
    """Test suite for Simulator restraint methods"""

    def test_get_restraint_indices(self, real_amber_system_files):
        """Test get_restraint_indices selects the real Ace-Ala-Nme backbone."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(**_amber_kwargs(real_amber_system_files))
        indices = sim.get_restraint_indices()

        # MDAnalysis 'backbone or nucleicbackbone' on Ace-Ala-Nme.
        assert list(indices) == [4, 5, 6, 8, 14, 15, 16, 18]

    def test_get_restraint_indices_with_additional_selection(
        self, real_amber_system_files
    ):
        """Test get_restraint_indices honors an additional selection string."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(**_amber_kwargs(real_amber_system_files))

        backbone = list(sim.get_restraint_indices())
        # 'resname ALA' adds the alanine side-chain/cap atoms beyond the backbone.
        combined = list(sim.get_restraint_indices(addtl_selection="resname ALA"))

        assert set(backbone).issubset(set(combined))
        assert len(combined) > len(backbone)


class TestSimulatorAddBackbonePosres:
    """Test suite for add_backbone_posres static method"""

    def test_add_backbone_posres(self, real_amber_system_files):
        """Test add_backbone_posres adds a real CustomExternalForce."""
        from openmm import CustomExternalForce

        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files))
        system = sim.load_system()
        n_forces_before = system.getNumForces()

        indices = list(sim.indices)
        posres_system = sim.add_backbone_posres(
            system,
            sim.coordinate.positions,
            sim.topology.topology.atoms(),
            indices,
            restraint_force=10.0,
        )

        # A new restraint force is added on a deepcopy; the original is untouched.
        assert system.getNumForces() == n_forces_before
        assert posres_system.getNumForces() == n_forces_before + 1

        new_force = posres_system.getForce(posres_system.getNumForces() - 1)
        assert isinstance(new_force, CustomExternalForce)
        # One restrained particle per backbone index.
        assert new_force.getNumParticles() == len(indices)


class TestSimulatorCheckNumStepsLeft:
    """Test suite for check_num_steps_left method"""

    def test_check_num_steps_left_normal(self, tmp_path):
        """Test check_num_steps_left with normal log file"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create production log
        log_content = "header\tstep\tenergy\n0\t100000\t-1000.0\n1\t200000\t-1001.0\n"
        (tmp_path / "prod.log").write_text(log_content)

        sim = Simulator(path=tmp_path, platform="CPU", prod_steps=500000)
        sim.prod_log = tmp_path / "prod.log"

        sim.check_num_steps_left()

        # prod_steps should be decremented
        assert sim.prod_steps < 500000

    def test_check_num_steps_left_empty_log(self, tmp_path):
        """Test check_num_steps_left with empty log file"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create empty production log
        (tmp_path / "prod.log").write_text("")

        sim = Simulator(path=tmp_path, platform="CPU", prod_steps=500000)
        sim.prod_log = tmp_path / "prod.log"

        # Should not raise, just return
        sim.check_num_steps_left()


@pytest.mark.skip(reason="Source code has bug - passes args to super() in wrong order")
class TestCustomForcesSimulator:
    """Test suite for CustomForcesSimulator class"""

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    def test_custom_forces_simulator_init(self, mock_platform):
        """Test CustomForcesSimulator initialization"""
        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_force = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            sim = CustomForcesSimulator(path=path, custom_force_objects=[mock_force])

            assert sim.custom_forces == [mock_force]

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    def test_add_forces(self, mock_platform):
        """Test add_forces method"""
        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_force1 = MagicMock()
        mock_force2 = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            sim = CustomForcesSimulator(
                path=path, custom_force_objects=[mock_force1, mock_force2]
            )

            mock_system = MagicMock()
            sim.add_forces(mock_system)

            assert mock_system.addForce.call_count == 2


class TestMinimizer:
    """Test suite for Minimizer class"""

    def test_minimizer_init(self, tmp_path):
        """Test Minimizer initialization"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        top_file = tmp_path / "system.prmtop"
        top_file.write_text("mock topology")
        coor_file = tmp_path / "system.inpcrd"
        coor_file.write_text("mock coordinates")

        minimizer = Minimizer(
            topology=str(top_file),
            coordinates=str(coor_file),
            out="minimized.pdb",
            platform="CPU",
        )

        assert minimizer.topology == top_file
        assert minimizer.coordinates == coor_file
        assert minimizer.out == tmp_path / "minimized.pdb"

    def test_minimizer_init_cpu(self, tmp_path):
        """Test Minimizer initialization without GPU"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        top_file = tmp_path / "system.prmtop"
        top_file.write_text("mock topology")
        coor_file = tmp_path / "system.inpcrd"
        coor_file.write_text("mock coordinates")

        minimizer = Minimizer(
            topology=str(top_file),
            coordinates=str(coor_file),
            platform="CPU",
            device_ids=None,
        )

        assert "DeviceIndex" not in minimizer.properties

    def test_load_files_invalid(self, tmp_path):
        """Test load_files with invalid file type"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        top_file = tmp_path / "system.xyz"  # Invalid extension
        top_file.write_text("mock topology")
        coor_file = tmp_path / "system.xyz"
        coor_file.write_text("mock coordinates")

        minimizer = Minimizer(
            topology=str(top_file), coordinates=str(coor_file), platform="CPU"
        )

        with pytest.raises(FileNotFoundError, match="No viable simulation"):
            minimizer.load_files()

    def test_load_amber(self, real_amber_system_files):
        """Test load_amber builds a real System from prmtop/inpcrd."""
        from openmm import System

        from molecular_simulations.simulate.omm_simulator import Minimizer

        minimizer = Minimizer(
            topology=str(real_amber_system_files["prmtop"]),
            coordinates=str(real_amber_system_files["inpcrd"]),
            platform="CPU",
        )

        system = minimizer.load_amber()
        assert isinstance(system, System)
        assert system.getNumParticles() == 22

    def test_load_pdb(self, real_amber_system_files):
        """Test load_pdb builds a real System from a PDB via amber14 FF."""
        from openmm import System

        from molecular_simulations.simulate.omm_simulator import Minimizer

        pdb_file = real_amber_system_files["pdb"]
        minimizer = Minimizer(
            topology=str(pdb_file), coordinates=str(pdb_file), platform="CPU"
        )

        system = minimizer.load_pdb()
        assert isinstance(system, System)
        assert system.getNumParticles() == 22


class TestSimulatorRun:
    """Test suite for Simulator run method"""

    def test_run_skip_equilibration(self, tmp_path):
        """Test run method when equilibration files exist."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create eq files to skip equilibration
        (tmp_path / "eq.state").write_text("mock state")
        (tmp_path / "eq.chk").write_bytes(b"mock checkpoint")
        (tmp_path / "eq.log").write_text("mock log")

        sim = Simulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "equilibrate") as mock_eq,
            patch.object(sim, "production") as mock_prod,
            patch.object(sim, "check_num_steps_left"),
        ):
            sim.run()
            # equilibrate should not be called
            mock_eq.assert_not_called()
            mock_prod.assert_called_once()

    def test_run_with_restart(self, tmp_path):
        """Test run method with restart checkpoint."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create all files including restart
        (tmp_path / "eq.state").write_text("mock state")
        (tmp_path / "eq.chk").write_bytes(b"mock checkpoint")
        (tmp_path / "eq.log").write_text("mock log")
        (tmp_path / "prod.rst.chk").write_bytes(b"mock restart")
        (tmp_path / "prod.log").write_text("header\tstep\tenergy\n0\t100000\t-1000.0\n")

        sim = Simulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "production") as mock_prod,
            patch.object(sim, "check_num_steps_left"),
        ):
            sim.run()
            # Should call production with restart=True
            mock_prod.assert_called_once()
            call_kwargs = mock_prod.call_args[1]
            assert call_kwargs.get("restart") is True


class TestSimulatorEquilibration:
    """Test suite for equilibration methods."""

    def test_equilibrate(self, tmp_path):
        """Test equilibrate orchestration (real reporters, mocked MD steps)."""
        from openmm.app import DCDReporter, StateDataReporter

        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "load_system") as mock_load,
            patch.object(sim, "add_backbone_posres") as mock_posres,
            patch.object(sim, "setup_sim") as mock_setup,
            patch.object(sim, "_heating") as mock_heat,
            patch.object(sim, "_equilibrate") as mock_eq,
        ):
            mock_system = MagicMock()
            mock_load.return_value = mock_system
            mock_posres.return_value = mock_system

            mock_simulation = MagicMock()
            mock_simulation.reporters = []
            mock_integrator = MagicMock()
            mock_setup.return_value = (mock_simulation, mock_integrator)
            mock_heat.return_value = (mock_simulation, mock_integrator)
            mock_eq.return_value = mock_simulation

            sim.coordinate = MagicMock()
            sim.coordinate.positions = [[0, 0, 0]]
            sim.topology = MagicMock()
            sim.topology.topology.atoms.return_value = []
            sim.indices = [0]

            sim.equilibrate()

            mock_load.assert_called_once()
            mock_posres.assert_called_once()
            mock_setup.assert_called_once()
            mock_heat.assert_called_once()
            mock_eq.assert_called_once()
            # Real equilibration reporters were attached before heating.
            assert any(
                isinstance(r, StateDataReporter) for r in mock_simulation.reporters
            )
            assert any(isinstance(r, DCDReporter) for r in mock_simulation.reporters)
            assert sim.eq_dcd.exists()


class TestSimulatorProduction:
    """Test suite for production methods."""

    def test_production_no_restart(self, tmp_path):
        """Test production method without restart."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        chkpt = tmp_path / "test.chk"
        chkpt.write_bytes(b"mock checkpoint")

        sim = Simulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "load_system") as mock_load,
            patch.object(sim, "setup_sim") as mock_setup,
            patch.object(sim, "load_checkpoint") as mock_load_chk,
            patch.object(sim, "attach_reporters") as mock_attach,
            patch.object(sim, "_production") as mock_prod,
        ):
            mock_system = MagicMock()
            mock_load.return_value = mock_system

            mock_simulation = MagicMock()
            mock_integrator = MagicMock()
            mock_setup.return_value = (mock_simulation, mock_integrator)
            mock_load_chk.return_value = mock_simulation
            mock_attach.return_value = mock_simulation
            mock_prod.return_value = mock_simulation

            sim.production(str(chkpt), restart=False)

            mock_load.assert_called_once()
            mock_setup.assert_called_once()
            mock_load_chk.assert_called_once()
            mock_attach.assert_called_once()
            mock_prod.assert_called_once()

    def test_production_with_restart(self, tmp_path):
        """Test production method with restart."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        chkpt = tmp_path / "test.chk"
        chkpt.write_bytes(b"mock checkpoint")

        sim = Simulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "load_system") as mock_load,
            patch.object(sim, "setup_sim") as mock_setup,
            patch.object(sim, "load_checkpoint") as mock_load_chk,
            patch.object(sim, "attach_reporters") as mock_attach,
            patch.object(sim, "_production") as mock_prod,
        ):
            mock_system = MagicMock()
            mock_load.return_value = mock_system

            mock_simulation = MagicMock()
            mock_integrator = MagicMock()
            mock_setup.return_value = (mock_simulation, mock_integrator)
            mock_load_chk.return_value = mock_simulation
            mock_attach.return_value = mock_simulation
            mock_prod.return_value = mock_simulation

            sim.production(str(chkpt), restart=True)

            # With restart=True, log file should be opened in append mode
            mock_attach.assert_called_once()
            call_args = mock_attach.call_args[0]
            # log_file arg should be a file object (from open), not string
            assert not isinstance(call_args[2], str)


class TestMinimizerMethods:
    """Additional tests for Minimizer class."""

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    @patch("molecular_simulations.simulate.omm_simulator.AmberInpcrdFile")
    @patch("molecular_simulations.simulate.omm_simulator.AmberPrmtopFile")
    @patch("molecular_simulations.simulate.omm_simulator.LangevinMiddleIntegrator")
    @patch("molecular_simulations.simulate.omm_simulator.Simulation")
    @patch("molecular_simulations.simulate.omm_simulator.PDBFile")
    def test_minimizer_minimize(
        self,
        mock_pdb,
        mock_sim_class,
        mock_integrator,
        mock_prmtop,
        mock_inpcrd,
        mock_platform,
    ):
        """Test Minimizer minimize method."""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_inpcrd_instance = MagicMock(boxVectors=None)
        mock_inpcrd_instance.positions = [[0, 0, 0]]
        mock_inpcrd.return_value = mock_inpcrd_instance
        mock_topology = MagicMock()
        mock_topology.createSystem.return_value = MagicMock()
        mock_topology.topology = MagicMock()
        mock_prmtop.return_value = mock_topology

        mock_simulation = MagicMock()
        mock_state = MagicMock()
        mock_state.getPositions.return_value = [[0, 0, 0]]
        mock_simulation.context.getState.return_value = mock_state
        mock_sim_class.return_value = mock_simulation
        mock_integrator.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / "system.prmtop"
            top_file.write_text("mock topology")
            coor_file = path / "system.inpcrd"
            coor_file.write_text("mock coordinates")

            minimizer = Minimizer(
                topology=str(top_file), coordinates=str(coor_file), out="minimized.pdb"
            )

            minimizer.minimize()

            mock_simulation.minimizeEnergy.assert_called_once()
            mock_pdb.writeFile.assert_called_once()


class TestImplicitSimulator:
    """Real ImplicitSimulator tests on the Ace-Ala-Nme system (CPU)."""

    def test_implicit_simulator_init(self, real_amber_system_files):
        """Test ImplicitSimulator initialization and implicit-solvent setup."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files))

        assert sim.path == real_amber_system_files["path"]
        assert sim.temperature == 300.0
        assert sim.solute_dielectric == 1.0
        assert sim.solvent_dielectric == 78.5
        # Debye-Huckel screening parameter is computed from the salt conc.
        assert sim.kappa > 0

    def test_implicit_simulator_init_with_ff(self, real_amber_system_files):
        """Test ImplicitSimulator initialization with force field parameter."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files, ff="amber"))

        assert sim.ff == "amber"

    def test_implicit_load_amber_files(self, real_amber_system_files):
        """load_amber_files builds a real implicit-solvent system (GB force)."""
        from openmm import CustomGBForce, GBSAOBCForce

        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files))
        system = sim.load_amber_files()

        assert system.getNumParticles() == 22
        # An implicit-solvent (Generalized Born) force is present
        assert any(
            isinstance(f, (CustomGBForce, GBSAOBCForce)) for f in system.getForces()
        )


class TestSimulatorHeating:
    """Test suite for _heating method."""

    def test_heating_gradual_temperature_increase(self, real_amber_system_files):
        """Test _heating ramps a real integrator up to the target temperature."""
        from openmm.unit import kelvin

        sim, simulation, integrator = _build_real_implicit_simulation(
            real_amber_system_files, dt=0.002, temperature=300.0
        )
        # ImplicitSimulator does not expose heat_steps via its constructor; set a
        # tiny ramp so the real heating runs quickly on CPU.
        sim.heat_steps = 2000

        step_before = simulation.currentStep
        result_sim, result_int = sim._heating(simulation, integrator)

        # The simulation actually advanced (1000-step blocks * heat_steps//1000).
        assert simulation.currentStep > step_before
        # Heating ends at the target temperature.
        assert result_int.getTemperature().value_in_unit(kelvin) == 300.0
        assert result_sim is simulation

    def test_heating_temperature_caps_at_target(self, real_amber_system_files):
        """Test _heating caps temperature at the target temperature."""
        from openmm.unit import kelvin

        sim, simulation, integrator = _build_real_implicit_simulation(
            real_amber_system_files, dt=0.002, temperature=300.0
        )
        sim.heat_steps = 2000

        sim._heating(simulation, integrator)

        # Final integrator temperature must not exceed the target.
        assert integrator.getTemperature().value_in_unit(kelvin) <= 300.0


class TestSimulatorEquilibrateMethod:
    """Test suite for _equilibrate method."""

    def test_equilibrate_restraint_relaxation(self, real_amber_explicit_files):
        """_equilibrate relaxes the restraint to zero and saves state/checkpoint."""
        sim, simulation = _build_real_explicit_simulation(
            real_amber_explicit_files, equil_steps=8, n_equil_cycles=1
        )

        sim._equilibrate(simulation)

        # The backbone restraint is fully relaxed to k=0 by the end
        assert simulation.context.getParameter("k") == 0.0
        # State and checkpoint are written
        assert sim.eq_state.exists()
        assert sim.eq_chkpt.exists()

    def test_equilibrate_adds_barostat(self, real_amber_explicit_files):
        """_equilibrate adds a MonteCarloBarostat for the NPT phase."""
        from openmm import MonteCarloBarostat

        sim, simulation = _build_real_explicit_simulation(
            real_amber_explicit_files, equil_steps=8, n_equil_cycles=1
        )

        before = sum(
            isinstance(f, MonteCarloBarostat) for f in simulation.system.getForces()
        )
        sim._equilibrate(simulation)
        after = sum(
            isinstance(f, MonteCarloBarostat) for f in simulation.system.getForces()
        )

        assert before == 0
        assert after == 1


class TestSimulatorProductionMethod:
    """Test suite for _production method."""

    def test_production_runs_steps(self, real_amber_system_files):
        """Test _production runs the requested steps and saves outputs."""
        sim, simulation, _ = _build_real_implicit_simulation(
            real_amber_system_files, dt=0.004, prod_steps=10
        )

        step_before = simulation.currentStep
        sim._production(simulation)

        # Verify simulation advanced by prod_steps.
        assert simulation.currentStep == step_before + 10
        # Verify state and checkpoint were saved to disk.
        assert sim.state.exists()
        assert sim.chkpt.exists()

    def test_production_returns_simulation(self, real_amber_system_files):
        """Test _production returns the simulation object."""
        sim, simulation, _ = _build_real_implicit_simulation(
            real_amber_system_files, dt=0.004, prod_steps=10
        )

        result = sim._production(simulation)

        assert result is simulation


class TestSimulatorPlatformInit:
    """Test suite for platform initialization."""

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    def test_cuda_platform_with_env_variable(self, mock_platform):
        """Test CUDA platform uses CUDA_VISIBLE_DEVICES env variable."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            # Set environment variable
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                sim = Simulator(path=path, platform="CUDA")

            assert sim.properties["DeviceIndex"] == "0"
            assert sim.properties["Precision"] == "mixed"

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    def test_cuda_platform_without_env_variable(self, mock_platform):
        """Test CUDA platform uses device_ids when no env variable."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            # Ensure no env variable
            with patch.dict(os.environ, {}, clear=True):
                # Remove CUDA_VISIBLE_DEVICES if present
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                sim = Simulator(path=path, platform="CUDA", device_ids=[0, 1])

            assert sim.properties["DeviceIndex"] == "0,1"

    @pytest.mark.skipif(
        not _has_opencl(), reason="OpenCL platform not available in this OpenMM build"
    )
    def test_opencl_platform_with_env_variable(self, tmp_path):
        """Test OpenCL platform uses ZE_AFFINITY_MASK env variable."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        with patch.dict(os.environ, {"ZE_AFFINITY_MASK": "0"}):
            sim = Simulator(path=tmp_path, platform="OpenCL")

        # OpenCL currently only sets Precision, not DeviceIndex
        assert sim.properties["Precision"] == "mixed"

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    def test_invalid_platform_raises(self, mock_platform):
        """Test invalid platform raises AttributeError."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            with pytest.raises(AttributeError, match="not available"):
                Simulator(path=path, platform="InvalidPlatform")


class TestSimulatorCheckNumStepsLeftAdvanced:
    """Advanced tests for check_num_steps_left method."""

    def test_check_num_steps_left_with_duplicate_frames(self, tmp_path):
        """Test check_num_steps_left creates duplicate_frames.log."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create log with steps that don't align with checkpoint frequency
        # prod_freq=10000 means checkpoint_freq=100000
        # Last step 150000 means 50000 steps after last checkpoint
        log_content = "#header\tstep\tenergy\n0\t10000\t-1000.0\n1\t150000\t-1001.0\n"
        (tmp_path / "prod.log").write_text(log_content)

        sim = Simulator(
            path=tmp_path,
            platform="CPU",
            prod_steps=500000,
            prod_reporter_frequency=10000,
        )
        sim.prod_log = tmp_path / "prod.log"

        sim.check_num_steps_left()

        # Should create duplicate frames log
        dup_log = tmp_path / "duplicate_frames.log"
        assert dup_log.exists()

    def test_check_num_steps_left_appends_to_existing_log(self, tmp_path):
        """Test check_num_steps_left appends to existing duplicate_frames.log."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create existing duplicate frames log
        dup_log = tmp_path / "duplicate_frames.log"
        dup_log.write_text("first_frame,last_frame\n0,5\n")

        log_content = "#header\tstep\tenergy\n0\t10000\t-1000.0\n1\t150000\t-1001.0\n"
        (tmp_path / "prod.log").write_text(log_content)

        sim = Simulator(
            path=tmp_path,
            platform="CPU",
            prod_steps=500000,
            prod_reporter_frequency=10000,
        )
        sim.prod_log = tmp_path / "prod.log"

        sim.check_num_steps_left()

        # Should have appended new entry
        content = dup_log.read_text()
        assert content.count("\n") >= 2

    def test_check_num_steps_left_handles_malformed_log(self, tmp_path):
        """Test check_num_steps_left handles malformed log gracefully."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Create malformed log (no valid step data)
        log_content = "malformed line with no tab separated values\n"
        (tmp_path / "prod.log").write_text(log_content)

        sim = Simulator(path=tmp_path, platform="CPU", prod_steps=500000)
        sim.prod_log = tmp_path / "prod.log"

        # Should not raise, just return
        sim.check_num_steps_left()

    def test_check_num_steps_left_simulation_complete(self, tmp_path):
        """Test check_num_steps_left when simulation is already complete."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # Last step equals prod_steps
        log_content = "#header\tstep\tenergy\n0\t100000\t-1000.0\n1\t500000\t-1001.0\n"
        (tmp_path / "prod.log").write_text(log_content)

        sim = Simulator(
            path=tmp_path,
            platform="CPU",
            prod_steps=500000,
            prod_reporter_frequency=10000,
        )
        sim.prod_log = tmp_path / "prod.log"

        sim.check_num_steps_left()

        # When time_left_from_log is 0, prod_steps should not be modified
        # (the condition is time_left_from_log > 0)


class TestSimulatorRunEquilibration:
    """Test suite for run method calling equilibration."""

    def test_run_calls_equilibrate_when_no_eq_files(self, tmp_path):
        """Test run method calls equilibrate when eq files don't exist."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        # No eq files exist in tmp_path.
        sim = Simulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "equilibrate") as mock_eq,
            patch.object(sim, "production") as mock_prod,
        ):
            sim.run()

            # equilibrate should be called
            mock_eq.assert_called_once()
            mock_prod.assert_called_once()


class TestSimulatorMembraneBarostat:
    """Test suite for membrane barostat setup."""

    def test_membrane_barostat_setup(self, tmp_path):
        """Test setup_barostat configures membrane barostat correctly."""
        from openmm import MonteCarloMembraneBarostat

        from molecular_simulations.simulate.omm_simulator import Simulator

        sim = Simulator(path=tmp_path, platform="CPU", membrane=True)

        assert sim.barostat == MonteCarloMembraneBarostat
        assert "defaultSurfaceTension" in sim.barostat_args
        assert "xymode" in sim.barostat_args
        assert "zmode" in sim.barostat_args
        assert "defaultTemperature" in sim.barostat_args


class TestImplicitSimulatorProduction:
    """Test suite for ImplicitSimulator production method."""

    def test_implicit_production_no_barostat(self, tmp_path):
        """Test ImplicitSimulator production doesn't add barostat."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        chkpt = tmp_path / "test.chk"
        chkpt.write_bytes(b"mock checkpoint")

        sim = ImplicitSimulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "load_system") as mock_load,
            patch.object(sim, "setup_sim") as mock_setup,
            patch.object(sim, "load_checkpoint") as mock_load_chk,
            patch.object(sim, "attach_reporters") as mock_attach,
            patch.object(sim, "_production") as mock_prod,
        ):
            mock_system = MagicMock()
            mock_load.return_value = mock_system

            mock_simulation = MagicMock()
            mock_integrator = MagicMock()
            mock_setup.return_value = (mock_simulation, mock_integrator)
            mock_load_chk.return_value = mock_simulation
            mock_attach.return_value = mock_simulation
            mock_prod.return_value = mock_simulation

            sim.production(str(chkpt), restart=False)

            # Should reinitialize context
            mock_simulation.context.reinitialize.assert_called_once_with(True)
            # Implicit production loads the system but never adds a barostat.
            mock_system.addForce.assert_not_called()


class TestImplicitSimulatorEquilibration:
    """Test suite for ImplicitSimulator equilibration method."""

    def test_implicit_equilibrate_prints_energy(self, tmp_path):
        """Test ImplicitSimulator equilibrate prints energy before/after minimization."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(path=tmp_path, platform="CPU")

        with (
            patch.object(sim, "load_system") as mock_load,
            patch.object(sim, "add_backbone_posres") as mock_posres,
            patch.object(sim, "setup_sim") as mock_setup,
            patch.object(sim, "_heating") as mock_heat,
            patch.object(sim, "_equilibrate") as mock_eq,
            patch("builtins.print") as mock_print,
        ):
            mock_system = MagicMock()
            mock_load.return_value = mock_system
            mock_posres.return_value = mock_system

            mock_simulation = MagicMock()
            mock_simulation.reporters = []
            mock_state_obj = MagicMock()
            mock_state_obj.getPotentialEnergy.return_value = -1000.0
            mock_simulation.context.getState.return_value = mock_state_obj
            mock_integrator = MagicMock()
            mock_setup.return_value = (mock_simulation, mock_integrator)
            mock_heat.return_value = (mock_simulation, mock_integrator)
            mock_eq.return_value = mock_simulation

            sim.coordinate = MagicMock()
            sim.coordinate.positions = [[0, 0, 0]]
            sim.topology = MagicMock()
            sim.topology.topology.atoms.return_value = []
            sim.indices = [0]

            sim.equilibrate()

            # Should print energy messages
            assert mock_print.call_count == 2


class TestSimulatorAttachReportersWithRestart:
    """Test suite for attach_reporters with restart option."""

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    @patch("molecular_simulations.simulate.omm_simulator.DCDReporter")
    @patch("molecular_simulations.simulate.omm_simulator.StateDataReporter")
    @patch("molecular_simulations.simulate.omm_simulator.CheckpointReporter")
    def test_attach_reporters_with_restart(
        self, mock_chkpt, mock_state, mock_dcd, mock_platform
    ):
        """Test attach_reporters with restart=True sets append mode."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            sim = Simulator(path=path)

            mock_simulation = MagicMock()
            mock_simulation.reporters = []

            dcd_file = path / "test.dcd"
            log_file = path / "test.log"
            rst_file = path / "test.rst"

            sim.attach_reporters(
                mock_simulation,
                str(dcd_file),
                str(log_file),
                str(rst_file),
                restart=True,
            )

            # Verify DCDReporter was called with append=True
            mock_dcd.assert_called_once()
            call_kwargs = mock_dcd.call_args[1]
            assert call_kwargs.get("append") is True


class TestSimulatorTotalProdSteps:
    """Test suite for total_prod_steps handling in attach_reporters."""

    @patch("molecular_simulations.simulate.omm_simulator.Platform")
    @patch("molecular_simulations.simulate.omm_simulator.DCDReporter")
    @patch("molecular_simulations.simulate.omm_simulator.StateDataReporter")
    @patch("molecular_simulations.simulate.omm_simulator.CheckpointReporter")
    def test_attach_reporters_uses_total_prod_steps(
        self, mock_chkpt, mock_state, mock_dcd, mock_platform
    ):
        """Test attach_reporters uses total_prod_steps for progress reporting."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock topology")
            (path / "system.inpcrd").write_text("mock coordinates")

            sim = Simulator(path=path, prod_steps=1000000)
            # Set total_prod_steps as run() method does
            sim.total_prod_steps = 1000000

            mock_simulation = MagicMock()
            mock_simulation.reporters = []

            dcd_file = path / "test.dcd"
            log_file = path / "test.log"
            rst_file = path / "test.rst"

            sim.attach_reporters(
                mock_simulation, str(dcd_file), str(log_file), str(rst_file)
            )

            # StateDataReporter should be called with totalSteps=total_prod_steps
            mock_state.assert_called_once()
            call_kwargs = mock_state.call_args[1]
            assert call_kwargs.get("totalSteps") == 1000000


class TestImplicitSimulatorInit:
    """Test suite for ImplicitSimulator initialization."""

    def test_implicit_simulator_init(self, tmp_path):
        """Test ImplicitSimulator stores implicit solvent parameters."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(path=tmp_path, platform="CPU", prod_steps=1000)

        assert sim.solute_dielectric == 1.0
        assert sim.solvent_dielectric == 78.5
        assert sim.kappa > 0  # Should be positive for 150mM ions

    def test_implicit_simulator_custom_dielectrics(self, tmp_path):
        """Test ImplicitSimulator with custom dielectric constants."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(
            path=tmp_path,
            platform="CPU",
            prod_steps=1000,
            solute_dielectric=2.0,
            solvent_dielectric=80.0,
        )

        assert sim.solute_dielectric == 2.0
        assert sim.solvent_dielectric == 80.0


class TestImplicitSimulatorLoadAmberFiles:
    """Test suite for ImplicitSimulator.load_amber_files method."""

    @requires_openmm
    def test_load_amber_files_creates_system(self, real_amber_system_files):
        """Test load_amber_files returns a real implicit-solvent OpenMM System."""
        from openmm import CustomGBForce, GBSAOBCForce, System

        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        sim = ImplicitSimulator(**_amber_kwargs(real_amber_system_files))
        system = sim.load_amber_files()

        assert isinstance(system, System)
        assert system.getNumParticles() == 22
        # A Generalized Born (implicit solvent) force is present.
        assert any(
            isinstance(f, (CustomGBForce, GBSAOBCForce)) for f in system.getForces()
        )


class TestCustomForcesSimulatorInit:
    """Test suite for CustomForcesSimulator initialization."""

    def test_custom_forces_simulator_stores_forces(self, tmp_path):
        """Test CustomForcesSimulator stores custom force objects."""
        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        mock_force1 = MagicMock()
        mock_force2 = MagicMock()

        sim = CustomForcesSimulator(
            path=str(tmp_path),
            custom_force_objects=[mock_force1, mock_force2],
            platform="CPU",
            prod_steps=1000,
        )

        assert sim.custom_forces == [mock_force1, mock_force2]


class TestCustomForcesSimulatorAddForces:
    """Test suite for CustomForcesSimulator.add_forces method."""

    def test_add_forces_adds_all_custom_forces(self, tmp_path):
        """Test add_forces adds each custom force to a real System."""
        from openmm import CustomBondForce, System

        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        force1 = CustomBondForce("0")
        force2 = CustomBondForce("0")

        sim = CustomForcesSimulator(
            path=str(tmp_path),
            custom_force_objects=[force1, force2],
            platform="CPU",
            prod_steps=1000,
        )

        system = System()
        result = sim.add_forces(system)

        assert result is system
        assert system.getNumForces() == 2
        assert {type(system.getForce(i)) for i in range(system.getNumForces())} == {
            CustomBondForce
        }

    def test_add_forces_empty_list(self, tmp_path):
        """Test add_forces with no custom forces leaves the System untouched."""
        from openmm import System

        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        sim = CustomForcesSimulator(
            path=str(tmp_path),
            custom_force_objects=[],
            platform="CPU",
            prod_steps=1000,
        )

        system = System()
        result = sim.add_forces(system)

        assert result is system
        assert system.getNumForces() == 0


class TestCustomForcesSimulatorLoadAmberFiles:
    """Test suite for CustomForcesSimulator.load_amber_files."""

    def test_load_amber_files_adds_custom_forces(self, tmp_path):
        """Test load_amber_files creates system and adds custom forces.

        The explicit-solvent PME path requires a periodic box, which the
        committed Ace-Ala-Nme inputs (implicit/vacuum) lack, so the AMBER file
        loaders are mocked here; only the unnecessary Platform mock is removed.
        """
        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        mock_force = MagicMock()

        sim = CustomForcesSimulator(
            path=str(tmp_path),
            custom_force_objects=[mock_force],
            platform="CPU",
            prod_steps=1000,
        )

        mock_prmtop = MagicMock()
        mock_system = MagicMock()
        mock_prmtop.createSystem.return_value = mock_system

        mock_inpcrd = MagicMock()
        mock_inpcrd.boxVectors = MagicMock()

        with (
            patch(
                "molecular_simulations.simulate.omm_simulator.AmberPrmtopFile",
                return_value=mock_prmtop,
            ),
            patch(
                "molecular_simulations.simulate.omm_simulator.AmberInpcrdFile",
                return_value=mock_inpcrd,
            ),
        ):
            system = sim.load_amber_files()

        mock_system.addForce.assert_called_once_with(mock_force)
        assert system is mock_system


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
