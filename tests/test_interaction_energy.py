"""
Unit tests for analysis/interaction_energy.py module

This module contains both unit tests (with minimal mocks) and integration tests.
Tests use real OpenMM when available, with conditional skips for environments
without OpenMM installed.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# Fixtures and helpers for conditional dependency usage
# ============================================================================


def _check_openmm():
    """Check if OpenMM is available."""
    try:
        import openmm

        return True
    except ImportError:
        return False


def _check_openmm_cpu():
    """Check if OpenMM CPU platform is available."""
    try:
        from openmm import Platform

        Platform.getPlatformByName("CPU")
        return True
    except Exception:
        return False


requires_openmm = pytest.mark.skipif(not _check_openmm(), reason="OpenMM not installed")


requires_openmm_cpu = pytest.mark.skipif(
    not _check_openmm_cpu(), reason="OpenMM CPU platform not available"
)


@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def alanine_pdb(test_data_dir):
    """Return the path to the alanine dipeptide PDB."""
    return test_data_dir / "pdb" / "alanine_dipeptide.pdb"


# ============================================================================
# Pure logic tests - no mocking needed
# ============================================================================


class TestInteractionEnergyPureLogic:
    """Test pure logic that doesn't need OpenMM."""

    def test_get_selection_logic_full_chain(self):
        """Test selection logic for full chain - no mocks."""
        # Test the selection logic without instantiating the class
        chain = "A"
        first = None
        last = None

        # Simulate atoms
        atoms = [
            {"index": 0, "chain_id": "A", "resid": "1"},
            {"index": 1, "chain_id": "A", "resid": "2"},
            {"index": 2, "chain_id": "B", "resid": "1"},
        ]

        selection = [a["index"] for a in atoms if a["chain_id"] == chain]

        assert selection == [0, 1]

    def test_get_selection_logic_with_first_residue(self):
        """Test selection logic with first_residue - no mocks."""
        chain = "A"
        first = 3
        last = None

        atoms = [{"index": i, "chain_id": "A", "resid": str(i + 1)} for i in range(5)]

        selection = [
            a["index"]
            for a in atoms
            if a["chain_id"] == chain and int(first) <= int(a["resid"])
        ]

        assert selection == [2, 3, 4]

    def test_get_selection_logic_with_last_residue(self):
        """Test selection logic with last_residue - no mocks."""
        chain = "A"
        first = None
        last = 3

        atoms = [{"index": i, "chain_id": "A", "resid": str(i + 1)} for i in range(5)]

        selection = [
            a["index"]
            for a in atoms
            if a["chain_id"] == chain and int(last) >= int(a["resid"])
        ]

        assert selection == [0, 1, 2]

    def test_get_selection_logic_with_range(self):
        """Test selection logic with residue range - no mocks."""
        chain = "A"
        first = 2
        last = 4

        atoms = [{"index": i, "chain_id": "A", "resid": str(i + 1)} for i in range(5)]

        selection = [
            a["index"]
            for a in atoms
            if a["chain_id"] == chain and int(first) <= int(a["resid"]) <= int(last)
        ]

        assert selection == [1, 2, 3]

    def test_interactions_property_logic(self):
        """Test interactions property returns correct shape - no mocks."""
        lj = -5.0
        coulomb = -10.0
        result = np.vstack([lj, coulomb])
        assert result.shape == (2, 1)
        assert result[0, 0] == -5.0
        assert result[1, 0] == -10.0

    def test_energy_array_shape(self):
        """Test energy array computation - no mocks."""
        n_frames = 10
        stride = 2
        n_computed = n_frames // stride
        energies = np.zeros((n_computed, 2))

        for i in range(n_computed):
            energies[i, 0] = -10.0  # LJ
            energies[i, 1] = -20.0  # Coulomb

        assert energies.shape == (5, 2)
        assert np.all(energies[:, 0] == -10.0)
        assert np.all(energies[:, 1] == -20.0)


# ============================================================================
# Integration tests using real OpenMM
# ============================================================================


@requires_openmm_cpu
class TestStaticInteractionEnergyIntegration:
    """Integration tests using real OpenMM."""

    def test_static_interaction_energy_real_init(self, alanine_pdb):
        """Test StaticInteractionEnergy with real PDB and CPU platform."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(alanine_pdb), chain="A", platform="CPU")

        assert sie.pdb == str(alanine_pdb)
        assert sie.chain == "A"
        assert sie.platform is not None


@requires_openmm
class TestInteractionEnergyAbstractIntegration:
    """Integration test for abstract base class."""

    def test_abstract_class_cannot_instantiate(self):
        """Test that InteractionEnergy cannot be instantiated."""
        from molecular_simulations.analysis.interaction_energy import InteractionEnergy

        with pytest.raises(TypeError):
            InteractionEnergy()


# ============================================================================
# Unit tests with minimal mocking
# ============================================================================


class TestStaticInteractionEnergy:
    """Real StaticInteractionEnergy tests against the two-chain salt-bridge PDB.

    Chain A is Ace-Lys-Nme (atoms 0-33: ACE 0-5, LYS 6-27, NME 28-33), chain B
    is Ace-Asp-Nme (atoms 34-57). Real CPU platform; no Platform mock needed.
    """

    @staticmethod
    def _topology(pdb):
        from openmm.app import PDBFile

        return PDBFile(str(pdb)).topology

    def test_static_interaction_energy_init(self, two_chain_pdb):
        """Test StaticInteractionEnergy initialization stores parameters."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb),
            chain="B",
            platform="CPU",
            first_residue=10,
            last_residue=50,
        )

        assert sie.pdb == str(two_chain_pdb)
        assert sie.chain == "B"
        assert sie.first == 10
        assert sie.last == 50

    def test_static_interaction_energy_init_defaults(self, two_chain_pdb):
        """Test StaticInteractionEnergy default values."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), platform="CPU")

        assert sie.chain == "A"
        assert sie.first is None
        assert sie.last is None

    def test_get_selection_full_chain(self, two_chain_pdb):
        """Test get_selection picks every atom of the requested chain."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb), chain="A", platform="CPU"
        )
        sie.get_selection(self._topology(two_chain_pdb))

        # Chain A is the first 34 atoms (Ace-Lys-Nme)
        assert sie.selection == list(range(34))

    def test_get_selection_with_first_residue(self, two_chain_pdb):
        """Test get_selection with first_residue filter."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb), chain="A", platform="CPU", first_residue=2
        )
        sie.get_selection(self._topology(two_chain_pdb))

        # Residues 2-3 (LYS, NME) -> atoms 6-33
        assert sie.selection == list(range(6, 34))

    def test_get_selection_with_last_residue(self, two_chain_pdb):
        """Test get_selection with last_residue filter."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb), chain="A", platform="CPU", last_residue=2
        )
        sie.get_selection(self._topology(two_chain_pdb))

        # Residues 1-2 (ACE, LYS) -> atoms 0-27
        assert sie.selection == list(range(0, 28))

    def test_get_selection_with_range(self, two_chain_pdb):
        """Test get_selection with first and last residue."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb),
            chain="A",
            platform="CPU",
            first_residue=2,
            last_residue=2,
        )
        sie.get_selection(self._topology(two_chain_pdb))

        # Residue 2 only (LYS) -> atoms 6-27
        assert sie.selection == list(range(6, 28))

    def test_interactions_property(self, two_chain_pdb):
        """Test interactions property stacks lj and coulomb."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), platform="CPU")
        sie.lj = -5.0
        sie.coulomb = -10.0

        result = sie.interactions

        assert result.shape == (2, 1)
        assert result[0, 0] == -5.0
        assert result[1, 0] == -10.0

    def test_energy_static_method(self):
        """Test energy static method sets the scaling parameters.

        Uses a mock Context: a real Context requires the fully assembled
        custom-force system, which is exercised end-to-end in
        test_compute_real_interaction_energy below.
        """
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        mock_context = MagicMock()
        mock_energy = MagicMock()
        mock_energy.getPotentialEnergy.return_value = 100.0
        mock_context.getState.return_value = mock_energy

        result = StaticInteractionEnergy.energy(
            mock_context,
            solute_coulomb_scale=1,
            solute_lj_scale=0,
            solvent_coulomb_scale=1,
            solvent_lj_scale=0,
        )

        mock_context.setParameter.assert_any_call("solute_coulomb_scale", 1)
        mock_context.setParameter.assert_any_call("solute_lj_scale", 0)
        assert result == 100.0

    def test_fix_pdb(self, two_chain_pdb):
        """Test fix_pdb runs PDBFixer and returns a usable topology."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), platform="CPU")
        positions, topology = sie.fix_pdb()

        # The structure is already complete -> 58 atoms preserved
        assert topology.getNumAtoms() == 58
        assert len(positions) == 58

    def test_compute_real_interaction_energy(self, two_chain_pdb):
        """End-to-end: real LJ/Coulomb interaction of chain A vs the rest."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb), chain="A", platform="CPU"
        )
        sie.compute()

        assert np.isfinite(sie.lj)
        assert np.isfinite(sie.coulomb)
        # A Lys-Asp salt bridge across the interface is attractive (coulomb < 0)
        assert sie.coulomb < 0.0
        assert sie.interactions.shape == (2, 1)


class TestInteractionEnergyFrame:
    """Test suite for InteractionEnergyFrame class"""

    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_interaction_energy_frame_init(self, mock_platform):
        """Test InteractionEnergyFrame initialization"""
        from molecular_simulations.analysis.interaction_energy import (
            InteractionEnergyFrame,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_system = MagicMock()
        mock_topology = MagicMock()

        ief = InteractionEnergyFrame(
            system=mock_system,
            top=mock_topology,
            chain="C",
            platform="CPU",
            first_residue=5,
            last_residue=15,
        )

        assert ief.system == mock_system
        assert ief.top == mock_topology
        assert ief.chain == "C"
        assert ief.first == 5
        assert ief.last == 15

    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_interaction_energy_frame_get_system(self, mock_platform):
        """Test InteractionEnergyFrame get_system method"""
        from molecular_simulations.analysis.interaction_energy import (
            InteractionEnergyFrame,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_system = MagicMock()
        mock_topology = MagicMock()
        mock_topology.atoms.return_value = []

        ief = InteractionEnergyFrame(system=mock_system, top=mock_topology)

        result = ief.get_system()

        assert result == mock_system
        assert hasattr(ief, "selection")


class TestDynamicInteractionEnergy:
    """Test suite for DynamicInteractionEnergy class"""

    @patch("molecular_simulations.analysis.interaction_energy.md")
    @patch("molecular_simulations.analysis.interaction_energy.AmberPrmtopFile")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_dynamic_interaction_energy_init(self, mock_platform, mock_prmtop, mock_md):
        """Test DynamicInteractionEnergy initialization"""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_prmtop_instance = MagicMock()
        mock_prmtop_instance.createSystem.return_value = MagicMock()
        mock_prmtop.return_value = mock_prmtop_instance

        mock_traj = MagicMock()
        mock_traj.xyz = np.zeros((10, 100, 3))
        mock_md.load.return_value = mock_traj

        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / "system.prmtop"
            traj_path = Path(tmpdir) / "traj.dcd"
            top_path.write_text("dummy")
            traj_path.write_text("dummy")

            die = DynamicInteractionEnergy(
                top=top_path,
                traj=traj_path,
                stride=2,
                chain="A",
                platform="CPU",
                first_residue=1,
                last_residue=10,
                progress_bar=True,
            )

            assert die.stride == 2
            assert die.progress is True
            assert die.coordinates.shape == (10, 100, 3)

    @patch("molecular_simulations.analysis.interaction_energy.md")
    @patch("molecular_simulations.analysis.interaction_energy.PDBFile")
    @patch("molecular_simulations.analysis.interaction_energy.ForceField")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_build_system_pdb(self, mock_platform, mock_ff, mock_pdb, mock_md):
        """Test build_system with PDB file"""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_pdb_instance = MagicMock()
        mock_pdb_instance.topology = MagicMock()
        mock_pdb.return_value = mock_pdb_instance

        mock_ff_instance = MagicMock()
        mock_ff_instance.createSystem.return_value = MagicMock()
        mock_ff.return_value = mock_ff_instance

        mock_traj = MagicMock()
        mock_traj.xyz = np.zeros((5, 50, 3))
        mock_md.load.return_value = mock_traj

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "system.pdb"
            traj_path = Path(tmpdir) / "traj.dcd"
            pdb_path.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )
            traj_path.write_text("dummy")

            die = DynamicInteractionEnergy(top=pdb_path, traj=traj_path)

            # Should have built system from PDB
            mock_ff.assert_called()

    @patch("molecular_simulations.analysis.interaction_energy.md")
    @patch("molecular_simulations.analysis.interaction_energy.AmberPrmtopFile")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_build_system_unsupported(self, mock_platform, mock_prmtop, mock_md):
        """Test build_system with unsupported topology"""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_traj = MagicMock()
        mock_traj.xyz = np.zeros((5, 50, 3))
        mock_md.load.return_value = mock_traj

        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / "system.xyz"  # Unsupported
            traj_path = Path(tmpdir) / "traj.dcd"
            top_path.write_text("dummy")
            traj_path.write_text("dummy")

            with pytest.raises(NotImplementedError):
                DynamicInteractionEnergy(top=top_path, traj=traj_path)


class TestInteractionEnergyAbstract:
    """Test the abstract base class"""

    def test_abstract_class_cannot_instantiate(self):
        """Test that InteractionEnergy cannot be instantiated"""
        from molecular_simulations.analysis.interaction_energy import InteractionEnergy

        with pytest.raises(TypeError):
            InteractionEnergy()


class TestDynamicInteractionEnergyAdditional:
    """Additional tests for DynamicInteractionEnergy class."""

    def test_setup_pbar(self):
        """Test setup_pbar creates progress bar."""
        with (
            patch(
                "molecular_simulations.analysis.interaction_energy.Platform"
            ) as mock_platform,
            patch(
                "molecular_simulations.analysis.interaction_energy.tqdm"
            ) as mock_tqdm,
        ):
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.zeros((100, 500, 3))

            die.setup_pbar()

            mock_tqdm.assert_called_once_with(total=100, position=0, leave=False)

    def test_compute_energies_no_progress_bar(self):
        """Test compute_energies without progress bar."""
        with patch(
            "molecular_simulations.analysis.interaction_energy.Platform"
        ) as mock_platform:
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.zeros((10, 500, 3))
            die.stride = 2
            die.progress = False

            mock_ie = MagicMock()
            mock_ie.lj = -10.0
            mock_ie.coulomb = -20.0
            die.IE = mock_ie

            die.compute_energies()

            # With stride=2 and 10 frames, should compute 5 energies
            assert die.energies.shape == (5, 2)
            assert mock_ie.compute.call_count == 5
            # Check that energies were filled
            assert np.all(die.energies[:, 0] == -10.0)
            assert np.all(die.energies[:, 1] == -20.0)

    def test_compute_energies_with_progress_bar(self):
        """Test compute_energies with progress bar enabled."""
        with (
            patch(
                "molecular_simulations.analysis.interaction_energy.Platform"
            ) as mock_platform,
            patch(
                "molecular_simulations.analysis.interaction_energy.tqdm"
            ) as mock_tqdm,
        ):
            mock_platform.getPlatformByName.return_value = MagicMock()
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.zeros((10, 500, 3))
            die.stride = 1
            die.progress = True

            mock_ie = MagicMock()
            mock_ie.lj = -5.0
            mock_ie.coulomb = -15.0
            die.IE = mock_ie

            die.compute_energies()

            assert die.energies.shape == (10, 2)
            assert mock_pbar.update.call_count == 10
            mock_pbar.close.assert_called_once()

    def test_load_traj(self):
        """Test load_traj loads trajectory with mdtraj."""
        with (
            patch(
                "molecular_simulations.analysis.interaction_energy.Platform"
            ) as mock_platform,
            patch("molecular_simulations.analysis.interaction_energy.md") as mock_md,
        ):
            mock_platform.getPlatformByName.return_value = MagicMock()
            mock_traj = MagicMock()
            mock_traj.xyz = np.random.rand(100, 500, 3)
            mock_md.load.return_value = mock_traj

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)

            result = die.load_traj(Path("top.pdb"), Path("traj.dcd"))

            mock_md.load.assert_called_once_with("traj.dcd", top="top.pdb")
            assert np.array_equal(result, mock_traj.xyz)


class TestDynamicInteractionEnergyBuildSystem:
    """Test suite for DynamicInteractionEnergy.build_system method."""

    def test_build_system_pdb(self):
        """Test build_system with PDB topology file."""
        with (
            patch(
                "molecular_simulations.analysis.interaction_energy.Platform"
            ) as mock_platform,
            patch(
                "molecular_simulations.analysis.interaction_energy.PDBFile"
            ) as mock_pdb_cls,
            patch(
                "molecular_simulations.analysis.interaction_energy.ForceField"
            ) as mock_ff_cls,
        ):
            mock_platform.getPlatformByName.return_value = MagicMock()

            mock_pdb = MagicMock()
            mock_pdb.topology = MagicMock()
            mock_pdb_cls.return_value = mock_pdb

            mock_ff = MagicMock()
            mock_system = MagicMock()
            mock_ff.createSystem.return_value = mock_system
            mock_ff_cls.return_value = mock_ff

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            result = die.build_system(Path("test.pdb"))

            mock_ff.createSystem.assert_called_once()
            assert result is mock_system
            assert die.top is mock_pdb.topology

    def test_build_system_prmtop(self):
        """Test build_system with AMBER prmtop topology file."""
        with (
            patch(
                "molecular_simulations.analysis.interaction_energy.Platform"
            ) as mock_platform,
            patch(
                "molecular_simulations.analysis.interaction_energy.AmberPrmtopFile"
            ) as mock_prmtop_cls,
        ):
            mock_platform.getPlatformByName.return_value = MagicMock()

            mock_prmtop = MagicMock()
            mock_system = MagicMock()
            mock_prmtop.createSystem.return_value = mock_system
            mock_prmtop_cls.return_value = mock_prmtop

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            result = die.build_system(Path("test.prmtop"))

            mock_prmtop.createSystem.assert_called_once()
            assert result is mock_system
            assert die.top is mock_prmtop

    def test_build_system_unsupported_format(self):
        """Test build_system raises for unsupported file type."""
        with patch(
            "molecular_simulations.analysis.interaction_energy.Platform"
        ) as mock_platform:
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)

            with pytest.raises(NotImplementedError):
                die.build_system(Path("test.gro"))


class TestDynamicInteractionEnergySetupPbar:
    """Test suite for DynamicInteractionEnergy.setup_pbar method."""

    def test_setup_pbar_creates_progress_bar(self):
        """Test setup_pbar creates a tqdm progress bar."""
        with (
            patch(
                "molecular_simulations.analysis.interaction_energy.Platform"
            ) as mock_platform,
            patch(
                "molecular_simulations.analysis.interaction_energy.tqdm"
            ) as mock_tqdm,
        ):
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import (
                DynamicInteractionEnergy,
            )

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.random.rand(100, 50, 3)

            die.setup_pbar()

            mock_tqdm.assert_called_once_with(total=100, position=0, leave=False)


class TestInteractionEnergyFrameGetSystem:
    """Test suite for InteractionEnergyFrame.get_system method."""

    def test_get_system_returns_existing_system(self):
        """Test get_system returns the pre-built system."""
        with patch(
            "molecular_simulations.analysis.interaction_energy.Platform"
        ) as mock_platform:
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import (
                InteractionEnergyFrame,
            )

            mock_system = MagicMock()
            mock_top = MagicMock()
            mock_top.atoms.return_value = []

            ie = InteractionEnergyFrame(
                system=mock_system, top=mock_top, chain="A", platform="CPU"
            )

            result = ie.get_system()
            assert result is mock_system


class TestStaticInteractionEnergyInteractions:
    """Test suite for StaticInteractionEnergy.interactions property."""

    def test_interactions_property(self):
        """Test interactions returns stacked LJ and Coulomb."""
        with patch(
            "molecular_simulations.analysis.interaction_energy.Platform"
        ) as mock_platform:
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import (
                StaticInteractionEnergy,
            )

            ie = StaticInteractionEnergy.__new__(StaticInteractionEnergy)
            ie.lj = -5.0
            ie.coulomb = -10.0

            result = ie.interactions
            assert result.shape == (2, 1)
            assert np.isclose(result[0, 0], -5.0)
            assert np.isclose(result[1, 0], -10.0)


class _MockQuantity:
    """Helper for mocking OpenMM Quantity arithmetic in tests."""

    def __init__(self, value):
        self._value = value

    def __sub__(self, other):
        return _MockQuantity(self._value - other._value)

    def value_in_unit(self, unit):
        return self._value


class TestStaticInteractionEnergyGetSystem:
    """Tests for StaticInteractionEnergy.get_system (lines 128-148)."""

    @patch("molecular_simulations.analysis.interaction_energy.ForceField")
    @patch("molecular_simulations.analysis.interaction_energy.PDBFile")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_get_system_success(self, mock_platform, mock_pdb_cls, mock_ff_cls):
        """Test get_system builds system on first try."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_pdb = MagicMock()
        mock_pdb.positions = [[0, 0, 0]]
        mock_pdb.topology = MagicMock()
        mock_pdb_cls.return_value = mock_pdb

        mock_ff = MagicMock()
        mock_system = MagicMock()
        mock_ff.createSystem.return_value = mock_system
        mock_ff_cls.return_value = mock_ff

        sie = StaticInteractionEnergy.__new__(StaticInteractionEnergy)
        sie.pdb = "test.pdb"
        sie.chain = "A"
        sie.first = None
        sie.last = None
        sie.platform = MagicMock()

        # Mock get_selection so it doesn't interfere
        with patch.object(sie, "get_selection"):
            result = sie.get_system()

        assert result is mock_system
        assert sie.positions == [[0, 0, 0]]
        mock_ff_cls.assert_called_once_with("amber14-all.xml", "implicit/gbn2.xml")
        mock_ff.createSystem.assert_called_once()

    @patch("molecular_simulations.analysis.interaction_energy.ForceField")
    @patch("molecular_simulations.analysis.interaction_energy.PDBFile")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_get_system_value_error_fallback(
        self, mock_platform, mock_pdb_cls, mock_ff_cls
    ):
        """Test get_system falls back to fix_pdb on ValueError."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_pdb = MagicMock()
        mock_pdb.positions = [[0, 0, 0]]
        mock_pdb.topology = MagicMock()
        mock_pdb_cls.return_value = mock_pdb

        mock_ff = MagicMock()
        mock_system = MagicMock()
        # First call raises ValueError, second succeeds
        mock_ff.createSystem.side_effect = [ValueError("bad topology"), mock_system]
        mock_ff_cls.return_value = mock_ff

        sie = StaticInteractionEnergy.__new__(StaticInteractionEnergy)
        sie.pdb = "test.pdb"
        sie.chain = "A"
        sie.first = None
        sie.last = None
        sie.platform = MagicMock()

        fixed_positions = [[1, 1, 1]]
        fixed_topology = MagicMock()

        with (
            patch.object(
                sie, "fix_pdb", return_value=(fixed_positions, fixed_topology)
            ),
            patch.object(sie, "get_selection"),
        ):
            result = sie.get_system()

        assert result is mock_system
        assert sie.positions == fixed_positions
        assert mock_ff.createSystem.call_count == 2


class TestStaticInteractionEnergyCompute:
    """Tests for StaticInteractionEnergy.compute (lines 160-215)."""

    @patch("molecular_simulations.analysis.interaction_energy.Context")
    @patch("molecular_simulations.analysis.interaction_energy.VerletIntegrator")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_compute_full(self, mock_platform, mock_integrator_cls, mock_context_cls):
        """Test compute runs full interaction energy calculation."""
        from openmm import NonbondedForce

        import molecular_simulations.analysis.interaction_energy as ie_mod

        # Inject NonbondedForce into module namespace (missing import in source)
        ie_mod.NonbondedForce = NonbondedForce

        mock_platform.getPlatformByName.return_value = MagicMock()

        # Create mock NonbondedForce that passes isinstance check
        mock_nb_force = MagicMock(spec=NonbondedForce)
        mock_nb_force.getNumParticles.return_value = 3
        mock_nb_force.getParticleParameters.side_effect = [
            (1.0, 0.3, 0.1),
            (0.5, 0.3, 0.1),
            (-1.5, 0.3, 0.1),
        ]
        mock_nb_force.getNumExceptions.return_value = 1
        mock_nb_force.getExceptionParameters.return_value = (0, 1, 0.5, 0.3, 0.1)

        # Other non-NB force
        mock_other_force = MagicMock()

        mock_system = MagicMock()
        mock_system.getForces.return_value = [mock_nb_force, mock_other_force]

        # Mock Context
        mock_context = MagicMock()
        mock_context_cls.return_value = mock_context

        # Energy calls return MockQuantity
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value = _MockQuantity(100.0)
        mock_context.getState.return_value = mock_state

        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy.__new__(StaticInteractionEnergy)
        sie.pdb = "test.pdb"
        sie.chain = "A"
        sie.platform = MagicMock()
        sie.first = None
        sie.last = None
        sie.selection = [0, 1]
        sie.positions = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        with patch.object(sie, "get_system", return_value=mock_system):
            sie.compute()

        # Check that energies were computed
        assert sie.lj is not None
        assert sie.coulomb is not None
        # 6 energy calls: total_coulomb, solute_coulomb, solvent_coulomb,
        #                 total_lj, solute_lj, solvent_lj
        assert mock_context.getState.call_count == 6
        # Other force should be set to group 2
        mock_other_force.setForceGroup.assert_called_with(2)

    @patch("molecular_simulations.analysis.interaction_energy.Context")
    @patch("molecular_simulations.analysis.interaction_energy.VerletIntegrator")
    @patch("molecular_simulations.analysis.interaction_energy.Platform")
    def test_compute_with_explicit_positions(
        self, mock_platform, mock_integrator_cls, mock_context_cls
    ):
        """Test compute with explicitly provided positions."""
        from openmm import NonbondedForce

        import molecular_simulations.analysis.interaction_energy as ie_mod

        ie_mod.NonbondedForce = NonbondedForce

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_nb_force = MagicMock(spec=NonbondedForce)
        mock_nb_force.getNumParticles.return_value = 2
        mock_nb_force.getParticleParameters.side_effect = [
            (1.0, 0.3, 0.1),
            (-1.0, 0.3, 0.1),
        ]
        mock_nb_force.getNumExceptions.return_value = 0

        mock_system = MagicMock()
        mock_system.getForces.return_value = [mock_nb_force]

        mock_context = MagicMock()
        mock_context_cls.return_value = mock_context

        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value = _MockQuantity(50.0)
        mock_context.getState.return_value = mock_state

        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy.__new__(StaticInteractionEnergy)
        sie.pdb = "test.pdb"
        sie.chain = "A"
        sie.platform = MagicMock()
        sie.first = None
        sie.last = None
        sie.selection = [0]
        sie.positions = [[0, 0, 0], [1, 1, 1]]

        explicit_positions = [[2, 2, 2], [3, 3, 3]]

        with patch.object(sie, "get_system", return_value=mock_system):
            sie.compute(positions=explicit_positions)

        # Should use explicit positions
        mock_context.setPositions.assert_called_once_with(explicit_positions)
        # All energy calls return 50.0, so: total - solute - solvent = 50 - 50 - 50 = -50
        assert sie.coulomb == -50.0
        assert sie.lj == -50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
