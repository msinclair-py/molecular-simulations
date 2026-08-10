"""
Unit tests for simulate/free_energy.py module

This module tests the EVB (Empirical Valence Bond) calculation classes
used for free energy simulations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


def _has_opencl() -> bool:
    """Return True if an OpenCL OpenMM platform is registered.

    CI's pip-installed OpenMM ships only Reference and CPU platforms, so the
    real-OpenCL test must skip there rather than fail.
    """
    try:
        from openmm import Platform

        Platform.getPlatformByName('OpenCL')
        return True
    except Exception:
        return False


def _has_cuda() -> bool:
    """Return True if a CUDA OpenMM platform is registered.

    CI has no GPU, so the real-CUDA precision test must skip there rather
    than fall back to mocking the Simulator engine.
    """
    try:
        from openmm import Platform

        Platform.getPlatformByName('CUDA')
        return True
    except Exception:
        return False


class TestEVBInit:
    """Test suite for EVB class initialization."""

    def test_evb_init_with_valid_inputs(self, alanine_dipeptide_pdb) -> None:
        """Test EVB initialization with valid input files and parameters.

        Verifies that the EVB class correctly initializes all attributes
        including paths, atom indices, and simulation parameters.
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.3, 0.3, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            assert evb.topology == alanine_dipeptide_pdb
            assert evb.coordinates == alanine_dipeptide_pdb
            assert evb.parsl_config is None
            assert evb.log_path == log_path

    def test_evb_init_custom_parameters(self, alanine_dipeptide_pdb) -> None:
        """Test EVB initialization with custom simulation parameters.

        Verifies that custom values for force constants, timestep, and
        other simulation parameters are correctly set.
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.3, 0.3, 0.1],
                parsl_config=None,
                log_path=log_path,
                steps=1000000,
                dt=0.001,
                k=200000.0,
                k_path=150.0,
                D_e=400.0,
                alpha=15.0,
                r0=0.11,
                platform='CPU',
            )

            assert evb.steps == 1000000
            assert evb.dt == 0.001
            assert evb.k == 200000.0
            assert evb.k_path == 150.0
            assert evb.D_e == 400.0
            assert evb.alpha == 15.0
            assert evb.r0 == 0.11
            assert evb.platform == 'CPU'

    def test_evb_init_default_parameters(self, alanine_dipeptide_pdb) -> None:
        """Test EVB initialization with default parameter values."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.3, 0.3, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            # Check default values
            assert evb.log_prefix == 'reactant'
            assert evb.rc_freq == 5
            assert evb.steps == 500000
            assert evb.dt == 0.002
            assert evb.k == 160000.0
            assert evb.k_path == 100.0
            assert evb.D_e == 392.46
            assert evb.alpha == 13.275
            assert evb.r0 == 0.109
            assert evb.platform == 'CUDA'
            assert evb.restraint_sel is None


class TestEVBConstructRC:
    """Test suite for EVB reaction coordinate construction."""

    def test_construct_rc_basic(self, alanine_dipeptide_pdb) -> None:
        """Test construction of linearly spaced reaction coordinate.

        The reaction coordinate is specified as [start, end, increment]
        and should produce an array from start to end (inclusive).
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            expected = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
            np.testing.assert_array_almost_equal(
                evb.reaction_coordinate, expected, decimal=5
            )

    def test_construct_rc_single_step(self, alanine_dipeptide_pdb) -> None:
        """Test reaction coordinate with large increment resulting in few windows."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[0.0, 0.5, 0.5],
                parsl_config=None,
                log_path=log_path,
            )

            expected = np.array([0.0, 0.5])
            np.testing.assert_array_almost_equal(
                evb.reaction_coordinate, expected, decimal=5
            )

    def test_construct_rc_negative_range(self, alanine_dipeptide_pdb) -> None:
        """Test reaction coordinate spanning negative to positive values."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.3, 0.3, 0.05],
                parsl_config=None,
                log_path=log_path,
            )

            # Should have 13 windows
            assert evb.reaction_coordinate.shape[0] == 13

    def test_construct_rc_direct_method(self, alanine_dipeptide_pdb) -> None:
        """Test construct_rc method directly."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            # Test the method directly
            rc = evb.construct_rc([0.0, 1.0, 0.2])
            expected = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            np.testing.assert_array_almost_equal(rc, expected, decimal=5)


class TestEVBProperties:
    """Test suite for EVB property methods."""

    def test_umbrella_property(self, alanine_dipeptide_pdb) -> None:
        """Test umbrella property returns correct dictionary structure.

        The umbrella property should return a dictionary containing:
        - atom_i, atom_j, atom_k: atom indices
        - k: umbrella force constant
        - k_path: path restraint force constant
        - rc0: None (set at runtime per window)
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 10',
                acceptor_atom='index 15',
                reactive_atom='index 20',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
                k=180000.0,
                k_path=120.0,
            )

            umbrella = evb.umbrella

            assert umbrella['atom_i'] == 10
            assert umbrella['atom_j'] == 15
            assert umbrella['atom_k'] == 20
            assert umbrella['k'] == 180000.0
            assert umbrella['k_path'] == 120.0
            assert umbrella['rc0'] is None

    def test_morse_bond_property(self, alanine_dipeptide_pdb) -> None:
        """Test morse_bond property returns correct dictionary structure.

        The morse_bond property should return a dictionary containing:
        - atom_i, atom_j: atom indices for the bond
        - D_e: well depth (bond dissociation energy)
        - alpha: width parameter
        - r0: equilibrium distance
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 10',
                acceptor_atom='index 15',
                reactive_atom='index 20',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
                D_e=400.0,
                alpha=14.0,
                r0=0.11,
            )

            morse = evb.morse_bond

            assert morse['atom_i'] == 10
            assert morse['atom_j'] == 20
            assert morse['D_e'] == 400.0
            assert morse['alpha'] == 14.0
            assert morse['r0'] == 0.11

    def test_morse_bond2_none_by_default(self, alanine_dipeptide_pdb) -> None:
        """morse_bond2 is None unless a symmetric second Morse is requested."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 10',
                acceptor_atom='index 15',
                reactive_atom='index 20',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=Path(tmpdir) / 'logs',
            )

            assert evb.morse_bond2 is None

    def test_morse_bond2_property(self, alanine_dipeptide_pdb) -> None:
        """With second_morse=True, morse_bond2 bonds acceptor-reactive.

        It mirrors morse_bond but between the acceptor (atom_j of the umbrella)
        and the shared reactive atom, reusing the same well parameters so the
        two ends of a symmetric transfer are equivalent.
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 10',
                acceptor_atom='index 15',
                reactive_atom='index 20',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=Path(tmpdir) / 'logs',
                D_e=400.0,
                alpha=14.0,
                r0=0.11,
                second_morse=True,
            )

            morse2 = evb.morse_bond2

            assert morse2 is not None
            assert morse2['atom_i'] == 15  # acceptor
            assert morse2['atom_j'] == 20  # reactive
            assert morse2['D_e'] == 400.0
            assert morse2['alpha'] == 14.0
            assert morse2['r0'] == 0.11


class TestEVBParslManagement:
    """Test suite for EVB Parsl initialization and shutdown."""

    def test_initialize_loads_parsl(
        self, alanine_dipeptide_pdb, local_parsl_config
    ) -> None:
        """initialize() loads a real Parsl DataFlowKernel from the config."""
        import parsl

        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=local_parsl_config,
                log_path=Path(tmpdir) / 'logs',
            )

            assert evb.dfk is None

            evb.initialize()
            try:
                # A real DataFlowKernel is now loaded and is the process-wide one.
                assert evb.dfk is parsl.dfk()
                assert evb._owns_parsl is True
            finally:
                evb.shutdown()

    def test_shutdown_cleans_up_parsl(
        self, alanine_dipeptide_pdb, local_parsl_config
    ) -> None:
        """shutdown() tears down the real DataFlowKernel it loaded."""
        import parsl

        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=local_parsl_config,
                log_path=Path(tmpdir) / 'logs',
            )

            evb.initialize()
            evb.shutdown()

            assert evb.dfk is None
            assert evb._owns_parsl is False
            # The global kernel is gone, so a fresh load() would be required.
            from parsl.errors import NoDataFlowKernelError

            with pytest.raises(NoDataFlowKernelError):
                parsl.dfk()
                assert evb.dfk is None

    def test_shutdown_when_not_initialized(
        self, alanine_dipeptide_pdb, local_parsl_config
    ) -> None:
        """shutdown() is a no-op (no raise) when nothing was ever loaded."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=local_parsl_config,
                log_path=Path(tmpdir) / 'logs',
            )

            # Never initialized: shutdown must not raise and must leave dfk None.
            evb.shutdown()
            assert evb.dfk is None
            assert evb._owns_parsl is False


class TestEVBCalculationInit:
    """Test suite for EVBCalculation class initialization."""

    def test_evb_calculation_init(self, real_amber_system_files) -> None:
        """Test EVBCalculation initialization builds a real Simulator engine."""
        from molecular_simulations.simulate.free_energy import (
            EVBCalculation,
            Simulator,
        )

        topology = real_amber_system_files['prmtop']
        coord_file = real_amber_system_files['inpcrd']
        out_path = real_amber_system_files['path'] / 'output'
        rc_file = real_amber_system_files['path'] / 'rc.log'

        umbrella = {
            'atom_i': 0,
            'atom_j': 1,
            'atom_k': 2,
            'k': 160000.0,
            'k_path': 100.0,
            'rc0': 0.1,
        }
        morse_bond = {
            'atom_i': 0,
            'atom_j': 2,
            'D_e': 392.46,
            'alpha': 13.275,
            'r0': 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform='CPU',
        )

        assert isinstance(evb_calc.sim_engine, Simulator)
        assert evb_calc.rc_file == rc_file
        assert evb_calc.umbrella == umbrella
        assert evb_calc.morse_bond == morse_bond

    @pytest.mark.skipif(
        not _has_cuda(), reason='CUDA platform not available in this OpenMM build'
    )
    def test_evb_calculation_cuda_precision(self, real_amber_system_files) -> None:
        """EVBCalculation sets mixed precision for the real CUDA platform."""
        from molecular_simulations.simulate.free_energy import EVBCalculation

        topology = real_amber_system_files['prmtop']
        coord_file = real_amber_system_files['inpcrd']
        out_path = real_amber_system_files['path'] / 'output'
        rc_file = real_amber_system_files['path'] / 'rc.log'

        umbrella = {
            'atom_i': 0,
            'atom_j': 1,
            'atom_k': 2,
            'k': 160000.0,
            'k_path': 100.0,
            'rc0': 0.1,
        }
        morse_bond = {
            'atom_i': 0,
            'atom_j': 2,
            'D_e': 392.46,
            'alpha': 13.275,
            'r0': 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform='CUDA',
        )

        # CUDA (like OpenCL) uses mixed precision.
        assert evb_calc.sim_engine.properties == {'Precision': 'mixed'}

    def test_evb_calculation_cpu_no_precision(self, real_amber_system_files) -> None:
        """Test EVBCalculation does not set precision for CPU platform."""
        from molecular_simulations.simulate.free_energy import EVBCalculation

        topology = real_amber_system_files['prmtop']
        coord_file = real_amber_system_files['inpcrd']
        out_path = real_amber_system_files['path'] / 'output'
        rc_file = real_amber_system_files['path'] / 'rc.log'

        umbrella = {
            'atom_i': 0,
            'atom_j': 1,
            'atom_k': 2,
            'k': 160000.0,
            'k_path': 100.0,
            'rc0': 0.1,
        }
        morse_bond = {
            'atom_i': 0,
            'atom_j': 2,
            'D_e': 392.46,
            'alpha': 13.275,
            'r0': 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform='CPU',
        )

        # CPU platform sets no precision properties
        assert evb_calc.sim_engine.properties == {}

    @pytest.mark.skipif(
        not _has_opencl(), reason='OpenCL platform not available in this OpenMM build'
    )
    def test_evb_calculation_opencl_precision(self, real_amber_system_files) -> None:
        """Test EVBCalculation sets mixed precision for OpenCL platform."""
        from molecular_simulations.simulate.free_energy import EVBCalculation

        topology = real_amber_system_files['prmtop']
        coord_file = real_amber_system_files['inpcrd']
        out_path = real_amber_system_files['path'] / 'output'
        rc_file = real_amber_system_files['path'] / 'rc.log'

        umbrella = {
            'atom_i': 0,
            'atom_j': 1,
            'atom_k': 2,
            'k': 160000.0,
            'k_path': 100.0,
            'rc0': 0.1,
        }
        morse_bond = {
            'atom_i': 0,
            'atom_j': 2,
            'D_e': 392.46,
            'alpha': 13.275,
            'r0': 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform='OpenCL',
        )

        # OpenCL (like CUDA) uses mixed precision
        assert evb_calc.sim_engine.properties == {'Precision': 'mixed'}


class TestEVBCalculationStaticMethods:
    """Test suite for EVBCalculation static force-generation methods."""

    def test_umbrella_force_parameters(self) -> None:
        """Test umbrella_force static method creates correct force.

        The umbrella force uses the difference of distances formula:
        V = 0.5 * k * ((r13 - r23) - rc0)^2
        """
        from molecular_simulations.simulate.free_energy import EVBCalculation

        force = EVBCalculation.umbrella_force(
            atom_i=0,
            atom_j=1,
            atom_k=2,
            k=160000.0,
            rc0=0.1,
        )

        # Verify force type
        from openmm import CustomCompoundBondForce

        assert isinstance(force, CustomCompoundBondForce)

        # Force should have 1 bond added
        assert force.getNumBonds() == 1

    def test_umbrella_force_ignores_extra_kwargs(self) -> None:
        """Test umbrella_force ignores extra keyword arguments.

        This is important because the umbrella dict may contain k_path
        which is used by path_restraint, not umbrella_force.
        """
        from molecular_simulations.simulate.free_energy import EVBCalculation

        # Should not raise despite extra kwargs
        force = EVBCalculation.umbrella_force(
            atom_i=0,
            atom_j=1,
            atom_k=2,
            k=160000.0,
            rc0=0.1,
            k_path=100.0,  # Extra kwarg that should be ignored
            extra_param='ignored',
        )

        from openmm import CustomCompoundBondForce

        assert isinstance(force, CustomCompoundBondForce)

    def test_path_restraint_parameters(self) -> None:
        """Test path_restraint static method creates correct force.

        The path restraint enforces collinearity using cosine angle:
        V = k_path * (1 - cos(theta))^2
        """
        from molecular_simulations.simulate.free_energy import EVBCalculation

        force = EVBCalculation.path_restraint(
            atom_i=0,
            atom_j=1,
            atom_k=2,
            k_path=100.0,
        )

        from openmm import CustomCompoundBondForce

        assert isinstance(force, CustomCompoundBondForce)
        assert force.getNumBonds() == 1

    def test_morse_bond_force_parameters(self) -> None:
        """Test morse_bond_force static method creates correct force.

        The Morse potential has the form:
        V(r) = D_e * (1 - exp(-alpha * (r - r0)))^2
        """
        from molecular_simulations.simulate.free_energy import EVBCalculation

        force = EVBCalculation.morse_bond_force(
            atom_i=0,
            atom_j=1,
            D_e=392.46,
            alpha=13.275,
            r0=0.1,
        )

        from openmm import CustomBondForce

        assert isinstance(force, CustomBondForce)
        assert force.getNumBonds() == 1

    def test_evb_coupled_force_type_and_cvs(self) -> None:
        """evb_coupled_force returns a CustomCVForce with two Morse CVs."""
        from openmm import CustomCVForce

        from molecular_simulations.simulate.free_energy import EVBCalculation

        force = EVBCalculation.evb_coupled_force(
            atom_donor=0,
            atom_acceptor=1,
            atom_reactive=2,
            D_e=460.0,
            alpha=22.0,
            r0=0.097,
            h12=50.0,
        )

        assert isinstance(force, CustomCVForce)
        assert force.getNumCollectiveVariables() == 2

    @pytest.mark.parametrize('h12', [0.0, 50.0, 120.0])
    def test_evb_coupled_force_energy_matches_eigenvalue(self, h12: float) -> None:
        """The force energy equals the 2x2 EVB ground-state eigenvalue.

        Evaluate the CustomCVForce on the Reference platform for a fixed
        geometry and compare against 0.5(V1+V2) - sqrt(0.25(V1-V2)^2 + H12^2)
        computed by hand from the two Morse potentials. Also checks the coupling
        never raises the ground state above min(V1, V2).
        """
        import numpy as np
        from openmm import Context, Platform, System, VerletIntegrator
        from openmm.unit import kilojoules_per_mole, nanometer

        from molecular_simulations.simulate.free_energy import EVBCalculation

        D_e, alpha, r0 = 460.0, 22.0, 0.097

        def morse(r: float) -> float:
            return D_e * (1.0 - np.exp(-alpha * (r - r0))) ** 2

        # donor-reactive = 0.10 nm, acceptor-reactive = 0.16 nm (collinear).
        d_dr, d_ar = 0.10, 0.16
        v1, v2 = morse(d_dr), morse(d_ar)
        expected = 0.5 * (v1 + v2) - np.sqrt(0.25 * (v1 - v2) ** 2 + h12**2)

        system = System()
        for _ in range(3):
            system.addParticle(1.0)
        system.addForce(
            EVBCalculation.evb_coupled_force(
                atom_donor=0,
                atom_acceptor=1,
                atom_reactive=2,
                D_e=D_e,
                alpha=alpha,
                r0=r0,
                h12=h12,
            )
        )

        context = Context(
            system,
            VerletIntegrator(0.001),
            Platform.getPlatformByName('Reference'),
        )
        # donor at origin, reactive on +x at d_dr, acceptor beyond it so that
        # acceptor-reactive = d_ar.
        positions = np.array([[0.0, 0, 0], [d_dr + d_ar, 0, 0], [d_dr, 0, 0]])
        context.setPositions(positions * nanometer)
        energy = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(kilojoules_per_mole)
        )

        assert energy == pytest.approx(expected, abs=1e-3)
        assert energy <= min(v1, v2) + 1e-6  # coupling only lowers the state


class TestEVBCalculationRemoveHarmonicBond:
    """Test suite for remove_harmonic_bond static method."""

    def test_remove_harmonic_bond_zeros_force_constant(self) -> None:
        """Test that remove_harmonic_bond zeros out the bond force constant.

        When replacing a harmonic bond with a Morse potential, we need to
        zero out the original harmonic bond to avoid double-counting.
        """
        from openmm import HarmonicBondForce, System
        from openmm.unit import kilojoules_per_mole, nanometers

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)

        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.1, 1000.0)  # length=0.1nm, k=1000 kJ/mol/nm^2
        system.addForce(bond_force)

        removed = EVBCalculation.remove_harmonic_bond(system, 0, 1)

        # Check force constant is now zero (OpenMM returns Quantity with units)
        _p1, _p2, length, k = bond_force.getBondParameters(0)
        assert k.value_in_unit(kilojoules_per_mole / nanometers**2) == 0.0
        assert length.value_in_unit(nanometers) == pytest.approx(0.1)
        # ...and the removed bond's (r0, k) are returned for Morse derivation.
        assert removed is not None
        r0, k_bond = removed
        assert r0 == pytest.approx(0.1)
        assert k_bond == pytest.approx(1000.0)

    def test_remove_harmonic_bond_returns_none_for_constraint(self) -> None:
        """A removed SHAKE constraint carries no force constant -> returns None."""
        from openmm import System

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addConstraint(0, 1, 0.1)
        assert EVBCalculation.remove_harmonic_bond(system, 0, 1) is None

    def test_remove_harmonic_bond_removes_constraint(self) -> None:
        """Test that remove_harmonic_bond removes SHAKE constraints."""
        from openmm import System

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addConstraint(0, 1, 0.1)

        assert system.getNumConstraints() == 1

        EVBCalculation.remove_harmonic_bond(system, 0, 1)

        assert system.getNumConstraints() == 0

    def test_remove_harmonic_bond_handles_missing_bond(self) -> None:
        """Test remove_harmonic_bond handles case where bond does not exist."""
        from openmm import HarmonicBondForce, System
        from openmm.unit import kilojoules_per_mole, nanometers

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)

        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.1, 1000.0)
        system.addForce(bond_force)

        # Try to remove bond between atoms 1 and 2 (doesn't exist)
        # Should not raise, just print warning
        EVBCalculation.remove_harmonic_bond(system, 1, 2)

        # Original bond should be unchanged
        _p1, _p2, _length, k = bond_force.getBondParameters(0)
        assert k.value_in_unit(kilojoules_per_mole / nanometers**2) == 1000.0

    def test_remove_harmonic_bond_reversed_indices(self) -> None:
        """Test remove_harmonic_bond works with reversed atom indices."""
        from openmm import HarmonicBondForce, System
        from openmm.unit import kilojoules_per_mole, nanometers

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)

        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.1, 1000.0)
        system.addForce(bond_force)

        # Remove with reversed indices (1, 0 instead of 0, 1)
        EVBCalculation.remove_harmonic_bond(system, 1, 0)

        # Check force constant is now zero
        _p1, _p2, _length, k = bond_force.getBondParameters(0)
        assert k.value_in_unit(kilojoules_per_mole / nanometers**2) == 0.0


class TestEVBCalculationExcludeNonbonded:
    """Test suite for exclude_nonbonded static method (symmetric 2nd Morse)."""

    def test_exclude_nonbonded_zeros_interaction(self) -> None:
        """exclude_nonbonded adds a charge=0, epsilon=0 exception for the pair.

        The acceptor-reactive pair of a symmetric second Morse bond is not
        bonded in the original topology, so it keeps full Coulomb/LJ unless we
        add this exclusion to mirror the (already-excluded) donor-reactive pair.
        """
        from openmm import NonbondedForce, System
        from openmm.unit import elementary_charge, kilojoules_per_mole

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        for _ in range(3):
            system.addParticle(1.0)

        nb = NonbondedForce()
        for q in (0.5, -0.5, 0.3):
            nb.addParticle(q, 0.3, 0.5)
        system.addForce(nb)

        assert nb.getNumExceptions() == 0

        EVBCalculation.exclude_nonbonded(system, 0, 2)

        assert nb.getNumExceptions() == 1
        p1, p2, chargeprod, _sigma, epsilon = nb.getExceptionParameters(0)
        assert {p1, p2} == {0, 2}
        assert chargeprod.value_in_unit(elementary_charge**2) == 0.0
        assert epsilon.value_in_unit(kilojoules_per_mole) == 0.0

    def test_exclude_nonbonded_replaces_existing(self) -> None:
        """Calling exclude_nonbonded twice on a pair does not raise (replace)."""
        from openmm import NonbondedForce, System

        from molecular_simulations.simulate.free_energy import EVBCalculation

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.5, 0.3, 0.5)
        nb.addParticle(-0.5, 0.3, 0.5)
        system.addForce(nb)

        EVBCalculation.exclude_nonbonded(system, 0, 1)
        EVBCalculation.exclude_nonbonded(system, 0, 1)  # must not raise

        assert nb.getNumExceptions() == 1


@pytest.mark.parametrize(
    'rc_input,expected_length',
    [
        ([-0.3, 0.3, 0.1], 7),
        ([-0.2, 0.2, 0.05], 9),
        ([0.0, 1.0, 0.25], 5),
        ([-0.5, 0.5, 0.5], 3),
    ],
)
class TestEVBConstructRCParametrized:
    """Parametrized tests for reaction coordinate construction."""

    def test_construct_rc_lengths(
        self,
        alanine_dipeptide_pdb,
        rc_input: list[float],
        expected_length: int,
    ) -> None:
        """Test that reaction coordinate has expected number of windows."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=rc_input,
                parsl_config=None,
                log_path=log_path,
            )

            assert len(evb.reaction_coordinate) == expected_length


class TestEVBPath:
    """Test suite for EVB path handling."""

    def test_evb_creates_correct_path(self, alanine_dipeptide_pdb) -> None:
        """Test EVB creates correct path for output directory."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'logs'

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom='index 0',
                acceptor_atom='index 1',
                reactive_atom='index 2',
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            # EVB path should be parent of topology / 'evb'
            expected_path = alanine_dipeptide_pdb.parent / 'evb'
            assert evb.path == expected_path


class TestEVBAnalyzerSaveMetadata:
    """EVBAnalyzer.save_metadata writes a real, re-readable TOML file."""

    def test_save_metadata_default_path(self, tmp_path) -> None:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        analyzer = EVBAnalyzer(
            log_path=tmp_path,
            log_prefix='reactant',
            k_umbrella=160000.0,
            rc0_values=[0.0, 0.5, 1.0],
        )

        out = analyzer.save_metadata()

        assert out == tmp_path / 'evb_metadata.toml'
        assert out.exists()

        # Round-trips through the standard TOML reader with real values.
        with open(out, 'rb') as f:
            meta = tomllib.load(f)

        evb = meta['evb']
        assert evb['log_path'] == str(tmp_path)
        assert evb['log_prefix'] == 'reactant'
        assert evb['k_umbrella'] == 160000.0
        assert evb['rc0_values'] == [0.0, 0.5, 1.0]
        # output_path defaults to log_path, so it is intentionally omitted.
        assert 'output_path' not in evb

    def test_save_metadata_custom_output_path(self, tmp_path) -> None:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        out_dir = tmp_path / 'results'
        out_dir.mkdir()
        analyzer = EVBAnalyzer(
            log_path=tmp_path,
            log_prefix='product',
            k_umbrella=1600.0,
            rc0_values=[-0.2, 0.2],
            output_path=out_dir,
        )

        # Write to an explicit, caller-supplied destination.
        dest = tmp_path / 'meta.toml'
        out = analyzer.save_metadata(output_path=dest)

        assert out == dest
        with open(dest, 'rb') as f:
            evb = tomllib.load(f)['evb']

        assert evb['log_prefix'] == 'product'
        assert evb['rc0_values'] == [-0.2, 0.2]
        # output_path differs from log_path, so it is recorded.
        assert evb['output_path'] == str(out_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
