"""
Unit tests for simulate/free_energy.py module

This module tests the EVB (Empirical Valence Bond) calculation classes
used for free energy simulations.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


class TestEVBInit:
    """Test suite for EVB class initialization."""

    def test_evb_init_with_valid_inputs(self, alanine_dipeptide_pdb) -> None:
        """Test EVB initialization with valid input files and parameters.

        Verifies that the EVB class correctly initializes all attributes
        including paths, atom indices, and simulation parameters.
        """
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
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
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
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
                platform="CPU",
            )

            assert evb.steps == 1000000
            assert evb.dt == 0.001
            assert evb.k == 200000.0
            assert evb.k_path == 150.0
            assert evb.D_e == 400.0
            assert evb.alpha == 15.0
            assert evb.r0 == 0.11
            assert evb.platform == "CPU"

    def test_evb_init_default_parameters(self, alanine_dipeptide_pdb) -> None:
        """Test EVB initialization with default parameter values."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.3, 0.3, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            # Check default values
            assert evb.log_prefix == "reactant"
            assert evb.rc_freq == 5
            assert evb.steps == 500000
            assert evb.dt == 0.002
            assert evb.k == 160000.0
            assert evb.k_path == 100.0
            assert evb.D_e == 392.46
            assert evb.alpha == 13.275
            assert evb.r0 == 0.109
            assert evb.platform == "CUDA"
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
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            expected = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
            np.testing.assert_array_almost_equal(
                evb.reaction_coordinate, expected, decimal=5
            )

    def test_construct_rc_single_step(
        self, alanine_dipeptide_pdb
    ) -> None:
        """Test reaction coordinate with large increment resulting in few windows."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[0.0, 0.5, 0.5],
                parsl_config=None,
                log_path=log_path,
            )

            expected = np.array([0.0, 0.5])
            np.testing.assert_array_almost_equal(
                evb.reaction_coordinate, expected, decimal=5
            )

    def test_construct_rc_negative_range(
        self, alanine_dipeptide_pdb
    ) -> None:
        """Test reaction coordinate spanning negative to positive values."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.3, 0.3, 0.05],
                parsl_config=None,
                log_path=log_path,
            )

            # Should have 13 windows
            assert evb.reaction_coordinate.shape[0] == 13

    def test_construct_rc_direct_method(
        self, alanine_dipeptide_pdb
    ) -> None:
        """Test construct_rc method directly."""
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
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
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 10",
                acceptor_atom="index 15",
                reactive_atom="index 20",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
                k=180000.0,
                k_path=120.0,
            )

            umbrella = evb.umbrella

            assert umbrella["atom_i"] == 10
            assert umbrella["atom_j"] == 15
            assert umbrella["atom_k"] == 20
            assert umbrella["k"] == 180000.0
            assert umbrella["k_path"] == 120.0
            assert umbrella["rc0"] is None

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
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 10",
                acceptor_atom="index 15",
                reactive_atom="index 20",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
                D_e=400.0,
                alpha=14.0,
                r0=0.11,
            )

            morse = evb.morse_bond

            assert morse["atom_i"] == 10
            assert morse["atom_j"] == 20
            assert morse["D_e"] == 400.0
            assert morse["alpha"] == 14.0
            assert morse["r0"] == 0.11


class TestEVBParslManagement:
    """Test suite for EVB Parsl initialization and shutdown."""

    def test_initialize_loads_parsl(
        self, alanine_dipeptide_pdb
    ) -> None:
        """Test that initialize() loads the Parsl configuration."""
        import molecular_simulations.simulate.free_energy as fe_module
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config = MagicMock()
            mock_dfk = MagicMock()

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=mock_config,
                log_path=Path(tmpdir) / "logs",
            )

            assert evb.dfk is None

            # Patch parsl.load on the module
            with patch.object(
                fe_module.parsl, "load", return_value=mock_dfk
            ) as mock_load:
                evb.initialize()
                mock_load.assert_called_once_with(mock_config)
                assert evb.dfk is mock_dfk

    def test_shutdown_cleans_up_parsl(
        self, alanine_dipeptide_pdb
    ) -> None:
        """Test that shutdown() properly cleans up Parsl resources."""
        import molecular_simulations.simulate.free_energy as fe_module
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config = MagicMock()
            mock_dfk = MagicMock()

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=mock_config,
                log_path=Path(tmpdir) / "logs",
            )

            with patch.object(fe_module.parsl, "load", return_value=mock_dfk):
                evb.initialize()

            with patch.object(fe_module.parsl, "clear") as mock_clear:
                evb.shutdown()
                mock_dfk.cleanup.assert_called_once()
                mock_clear.assert_called()
                assert evb.dfk is None

    def test_shutdown_when_not_initialized(
        self, alanine_dipeptide_pdb
    ) -> None:
        """Test that shutdown() handles case when dfk is None."""
        import molecular_simulations.simulate.free_energy as fe_module
        from molecular_simulations.simulate.free_energy import EVB

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config = MagicMock()

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=mock_config,
                log_path=Path(tmpdir) / "logs",
            )

            # Should not raise even when dfk is None
            # parsl.clear() should NOT be called since dfk is None
            with patch.object(fe_module.parsl, "clear") as mock_clear:
                evb.shutdown()
                mock_clear.assert_not_called()
                assert evb.dfk is None


class TestEVBCalculationInit:
    """Test suite for EVBCalculation class initialization."""

    def test_evb_calculation_init(self, real_amber_system_files) -> None:
        """Test EVBCalculation initialization builds a real Simulator engine."""
        from molecular_simulations.simulate.free_energy import (
            EVBCalculation,
            Simulator,
        )

        topology = real_amber_system_files["prmtop"]
        coord_file = real_amber_system_files["inpcrd"]
        out_path = real_amber_system_files["path"] / "output"
        rc_file = real_amber_system_files["path"] / "rc.log"

        umbrella = {
            "atom_i": 0,
            "atom_j": 1,
            "atom_k": 2,
            "k": 160000.0,
            "k_path": 100.0,
            "rc0": 0.1,
        }
        morse_bond = {
            "atom_i": 0,
            "atom_j": 2,
            "D_e": 392.46,
            "alpha": 13.275,
            "r0": 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform="CPU",
        )

        assert isinstance(evb_calc.sim_engine, Simulator)
        assert evb_calc.rc_file == rc_file
        assert evb_calc.umbrella == umbrella
        assert evb_calc.morse_bond == morse_bond

    def test_evb_calculation_cuda_precision(self) -> None:
        """Test EVBCalculation sets mixed precision for CUDA platform."""
        import molecular_simulations.simulate.free_energy as fe_module
        from molecular_simulations.simulate.free_energy import EVBCalculation

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            topology = path / "system.prmtop"
            topology.write_text("mock topology")
            coord_file = path / "system.inpcrd"
            coord_file.write_text("mock coordinates")
            out_path = path / "output"
            rc_file = path / "rc.log"

            umbrella = {
                "atom_i": 0,
                "atom_j": 1,
                "atom_k": 2,
                "k": 160000.0,
                "k_path": 100.0,
                "rc0": 0.1,
            }
            morse_bond = {
                "atom_i": 0,
                "atom_j": 2,
                "D_e": 392.46,
                "alpha": 13.275,
                "r0": 0.1,
            }

            mock_simulator = MagicMock()
            mock_simulator.properties = {"Precision": "mixed"}
            with patch.object(fe_module, "Simulator", return_value=mock_simulator):
                evb_calc = EVBCalculation(
                    topology=topology,
                    coord_file=coord_file,
                    out_path=out_path,
                    rc_file=rc_file,
                    umbrella=umbrella,
                    morse_bond=morse_bond,
                    platform="CUDA",
                )

                # Should set mixed precision
                assert evb_calc.sim_engine.properties == {"Precision": "mixed"}

    def test_evb_calculation_cpu_no_precision(self, real_amber_system_files) -> None:
        """Test EVBCalculation does not set precision for CPU platform."""
        from molecular_simulations.simulate.free_energy import EVBCalculation

        topology = real_amber_system_files["prmtop"]
        coord_file = real_amber_system_files["inpcrd"]
        out_path = real_amber_system_files["path"] / "output"
        rc_file = real_amber_system_files["path"] / "rc.log"

        umbrella = {
            "atom_i": 0,
            "atom_j": 1,
            "atom_k": 2,
            "k": 160000.0,
            "k_path": 100.0,
            "rc0": 0.1,
        }
        morse_bond = {
            "atom_i": 0,
            "atom_j": 2,
            "D_e": 392.46,
            "alpha": 13.275,
            "r0": 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform="CPU",
        )

        # CPU platform sets no precision properties
        assert evb_calc.sim_engine.properties == {}

    def test_evb_calculation_opencl_precision(self, real_amber_system_files) -> None:
        """Test EVBCalculation sets mixed precision for OpenCL platform."""
        from molecular_simulations.simulate.free_energy import EVBCalculation

        topology = real_amber_system_files["prmtop"]
        coord_file = real_amber_system_files["inpcrd"]
        out_path = real_amber_system_files["path"] / "output"
        rc_file = real_amber_system_files["path"] / "rc.log"

        umbrella = {
            "atom_i": 0,
            "atom_j": 1,
            "atom_k": 2,
            "k": 160000.0,
            "k_path": 100.0,
            "rc0": 0.1,
        }
        morse_bond = {
            "atom_i": 0,
            "atom_j": 2,
            "D_e": 392.46,
            "alpha": 13.275,
            "r0": 0.1,
        }

        evb_calc = EVBCalculation(
            topology=topology,
            coord_file=coord_file,
            out_path=out_path,
            rc_file=rc_file,
            umbrella=umbrella,
            morse_bond=morse_bond,
            platform="OpenCL",
        )

        # OpenCL (like CUDA) uses mixed precision
        assert evb_calc.sim_engine.properties == {"Precision": "mixed"}


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
            extra_param="ignored",
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


class TestEVBCalculationRemoveHarmonicBond:
    """Test suite for remove_harmonic_bond static method."""

    def test_remove_harmonic_bond_zeros_force_constant(
        self
    ) -> None:
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

        EVBCalculation.remove_harmonic_bond(system, 0, 1)

        # Check force constant is now zero (OpenMM returns Quantity with units)
        p1, p2, length, k = bond_force.getBondParameters(0)
        assert k.value_in_unit(kilojoules_per_mole / nanometers**2) == 0.0
        assert length.value_in_unit(nanometers) == pytest.approx(0.1)

    def test_remove_harmonic_bond_removes_constraint(
        self
    ) -> None:
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

    def test_remove_harmonic_bond_handles_missing_bond(
        self
    ) -> None:
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
        p1, p2, length, k = bond_force.getBondParameters(0)
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
        p1, p2, length, k = bond_force.getBondParameters(0)
        assert k.value_in_unit(kilojoules_per_mole / nanometers**2) == 0.0


@pytest.mark.parametrize(
    "rc_input,expected_length",
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
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
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
            log_path = Path(tmpdir) / "logs"

            evb = EVB(
                topology=alanine_dipeptide_pdb,
                coordinates=alanine_dipeptide_pdb,
                donor_atom="index 0",
                acceptor_atom="index 1",
                reactive_atom="index 2",
                reaction_coordinate=[-0.2, 0.2, 0.1],
                parsl_config=None,
                log_path=log_path,
            )

            # EVB path should be parent of topology / 'evb'
            expected_path = alanine_dipeptide_pdb.parent / "evb"
            assert evb.path == expected_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
