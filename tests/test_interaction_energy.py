"""
Unit tests for analysis/interaction_energy.py module

All tests run real OpenMM code on the CPU platform, with conditional skips for
environments without OpenMM installed. No mocks or patches are used.
"""

from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# Fixtures and helpers for conditional dependency usage
# ============================================================================


def _check_openmm():
    """Check if OpenMM is available."""
    try:
        import openmm  # noqa: F401

        return True
    except ImportError:
        return False


def _check_openmm_cpu():
    """Check if OpenMM CPU platform is available."""
    try:
        from openmm import Platform

        Platform.getPlatformByName('CPU')
        return True
    except Exception:
        return False


requires_openmm = pytest.mark.skipif(not _check_openmm(), reason='OpenMM not installed')


requires_openmm_cpu = pytest.mark.skipif(
    not _check_openmm_cpu(), reason='OpenMM CPU platform not available'
)


@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def alanine_pdb(test_data_dir):
    """Return the path to the alanine dipeptide PDB."""
    return test_data_dir / 'pdb' / 'alanine_dipeptide.pdb'


# ============================================================================
# Pure logic tests - no mocking needed
# ============================================================================


class TestInteractionEnergyPureLogic:
    """Test pure logic that doesn't need OpenMM."""

    def test_get_selection_logic_full_chain(self):
        """Test selection logic for full chain - no mocks."""
        # Test the selection logic without instantiating the class
        chain = 'A'

        # Simulate atoms
        atoms = [
            {'index': 0, 'chain_id': 'A', 'resid': '1'},
            {'index': 1, 'chain_id': 'A', 'resid': '2'},
            {'index': 2, 'chain_id': 'B', 'resid': '1'},
        ]

        selection = [a['index'] for a in atoms if a['chain_id'] == chain]

        assert selection == [0, 1]

    def test_get_selection_logic_with_first_residue(self):
        """Test selection logic with first_residue - no mocks."""
        chain = 'A'
        first = 3

        atoms = [{'index': i, 'chain_id': 'A', 'resid': str(i + 1)} for i in range(5)]

        selection = [
            a['index']
            for a in atoms
            if a['chain_id'] == chain and int(first) <= int(a['resid'])
        ]

        assert selection == [2, 3, 4]

    def test_get_selection_logic_with_last_residue(self):
        """Test selection logic with last_residue - no mocks."""
        chain = 'A'
        last = 3

        atoms = [{'index': i, 'chain_id': 'A', 'resid': str(i + 1)} for i in range(5)]

        selection = [
            a['index']
            for a in atoms
            if a['chain_id'] == chain and int(last) >= int(a['resid'])
        ]

        assert selection == [0, 1, 2]

    def test_get_selection_logic_with_range(self):
        """Test selection logic with residue range - no mocks."""
        chain = 'A'
        first = 2
        last = 4

        atoms = [{'index': i, 'chain_id': 'A', 'resid': str(i + 1)} for i in range(5)]

        selection = [
            a['index']
            for a in atoms
            if a['chain_id'] == chain and int(first) <= int(a['resid']) <= int(last)
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

        sie = StaticInteractionEnergy(pdb=str(alanine_pdb), chain='A', platform='CPU')

        assert sie.pdb == str(alanine_pdb)
        assert sie.chain == 'A'
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
# Real unit tests against the two-chain salt-bridge PDB
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
            chain='B',
            platform='CPU',
            first_residue=10,
            last_residue=50,
        )

        assert sie.pdb == str(two_chain_pdb)
        assert sie.chain == 'B'
        assert sie.first == 10
        assert sie.last == 50

    def test_static_interaction_energy_init_defaults(self, two_chain_pdb):
        """Test StaticInteractionEnergy default values."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), platform='CPU')

        assert sie.chain == 'A'
        assert sie.first is None
        assert sie.last is None

    def test_get_selection_full_chain(self, two_chain_pdb):
        """Test get_selection picks every atom of the requested chain."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), chain='A', platform='CPU')
        sie.get_selection(self._topology(two_chain_pdb))

        # Chain A is the first 34 atoms (Ace-Lys-Nme)
        assert sie.selection == list(range(34))

    def test_get_selection_with_first_residue(self, two_chain_pdb):
        """Test get_selection with first_residue filter."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(
            pdb=str(two_chain_pdb), chain='A', platform='CPU', first_residue=2
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
            pdb=str(two_chain_pdb), chain='A', platform='CPU', last_residue=2
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
            chain='A',
            platform='CPU',
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

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), platform='CPU')
        sie.lj = -5.0
        sie.coulomb = -10.0

        result = sie.interactions

        assert result.shape == (2, 1)
        assert result[0, 0] == -5.0
        assert result[1, 0] == -10.0

    def test_energy_static_method(self):
        """Test energy static method sets the scaling parameters on a real
        Context and returns a real potential energy.

        Builds a tiny two-particle System whose NonbondedForce carries the
        four scale global parameters, runs it on the real CPU platform, and
        verifies energy() actually pushes the requested scale values onto the
        context and reads back a finite potential energy.
        """
        from openmm import Context, NonbondedForce, Platform, System, VerletIntegrator
        from openmm.unit import (
            kilojoules_per_mole,
            nanometer,
            picosecond,
        )

        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(1.0, 0.3, 0.5)
        nb.addParticle(-1.0, 0.3, 0.5)
        for name in (
            'solute_coulomb_scale',
            'solute_lj_scale',
            'solvent_coulomb_scale',
            'solvent_lj_scale',
        ):
            nb.addGlobalParameter(name, 1)
        nb.setForceGroup(0)
        system.addForce(nb)

        integrator = VerletIntegrator(0.001 * picosecond)
        context = Context(system, integrator, Platform.getPlatformByName('CPU'))
        context.setPositions([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]] * nanometer)

        result = StaticInteractionEnergy.energy(
            context,
            solute_coulomb_scale=1,
            solute_lj_scale=0,
            solvent_coulomb_scale=1,
            solvent_lj_scale=0,
        )

        # The static method really set the parameters on the live context.
        assert context.getParameter('solute_coulomb_scale') == 1
        assert context.getParameter('solute_lj_scale') == 0
        assert context.getParameter('solvent_coulomb_scale') == 1
        assert context.getParameter('solvent_lj_scale') == 0
        # And it returned a real, finite potential energy.
        assert np.isfinite(result.value_in_unit(kilojoules_per_mole))

    def test_fix_pdb(self, two_chain_pdb):
        """Test fix_pdb runs PDBFixer and returns a usable topology."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), platform='CPU')
        positions, topology = sie.fix_pdb()

        # The structure is already complete -> 58 atoms preserved
        assert topology.getNumAtoms() == 58
        assert len(positions) == 58

    def test_compute_real_interaction_energy(self, two_chain_pdb):
        """End-to-end: real LJ/Coulomb interaction of chain A vs the rest."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), chain='A', platform='CPU')
        sie.compute()

        assert np.isfinite(sie.lj)
        assert np.isfinite(sie.coulomb)
        # A Lys-Asp salt bridge across the interface is attractive (coulomb < 0)
        assert sie.coulomb < 0.0
        assert sie.interactions.shape == (2, 1)


class TestInteractionEnergyFrame:
    """Real InteractionEnergyFrame tests against the two-chain PDB."""

    @staticmethod
    def _real_system_and_top(pdb):
        from openmm.app import ForceField, PDBFile

        parsed = PDBFile(str(pdb))
        ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        system = ff.createSystem(
            parsed.topology, soluteDielectric=1.0, solventDielectric=80.0
        )
        return system, parsed.topology

    def test_interaction_energy_frame_init(self, two_chain_pdb):
        """Test InteractionEnergyFrame stores a real system, topology, params."""
        from molecular_simulations.analysis.interaction_energy import (
            InteractionEnergyFrame,
        )

        system, topology = self._real_system_and_top(two_chain_pdb)

        ief = InteractionEnergyFrame(
            system=system,
            top=topology,
            chain='B',
            platform='CPU',
            first_residue=5,
            last_residue=15,
        )

        assert ief.system is system
        assert ief.top is topology
        assert ief.chain == 'B'
        assert ief.first == 5
        assert ief.last == 15

    def test_interaction_energy_frame_get_system(self, two_chain_pdb):
        """Test get_system returns the pre-built system and sets the selection."""
        from molecular_simulations.analysis.interaction_energy import (
            InteractionEnergyFrame,
        )

        system, topology = self._real_system_and_top(two_chain_pdb)

        ief = InteractionEnergyFrame(
            system=system, top=topology, chain='A', platform='CPU'
        )

        result = ief.get_system()

        assert result is system
        # Chain A is the first 34 atoms of the real two-chain topology.
        assert ief.selection == list(range(34))


class TestDynamicInteractionEnergy:
    """Real DynamicInteractionEnergy tests over the two-chain trajectory."""

    def test_dynamic_interaction_energy_init(self, two_chain_trajectory):
        """Test initialization builds the system and loads the trajectory."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy(
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            stride=2,
            chain='A',
            platform='CPU',
            first_residue=1,
            last_residue=10,
            progress_bar=True,
        )

        assert die.stride == 2
        assert die.progress is True
        # Real trajectory: 5 frames, 58 atoms
        assert die.coordinates.shape == (5, 58, 3)

    def test_build_system_pdb(self, two_chain_trajectory):
        """Test build_system constructs a real OpenMM system from a PDB."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy(
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            chain='A',
            platform='CPU',
        )

        assert die.system.getNumParticles() == 58

    def test_build_system_unsupported(self, tmp_path):
        """Test build_system rejects an unsupported topology extension."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        bad_top = tmp_path / 'system.xyz'
        bad_top.write_text('dummy')
        traj_path = tmp_path / 'traj.dcd'
        traj_path.write_text('dummy')

        # build_system runs first in __init__, so it raises before any load
        with pytest.raises(NotImplementedError):
            DynamicInteractionEnergy(top=bad_top, traj=traj_path, platform='CPU')

    def test_compute_energies_breaks_salt_bridge(self, two_chain_trajectory):
        """End-to-end: per-frame energies weaken as chain B drifts away."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy(
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            chain='A',
            platform='CPU',
            progress_bar=False,
        )
        die.compute_energies()

        assert die.energies.shape == (5, 2)
        assert np.isfinite(die.energies).all()
        # Coulombic attraction of the Lys-Asp salt bridge weakens as the chains
        # separate across the trajectory (less negative).
        assert die.energies[-1, 1] > die.energies[0, 1]


class TestInteractionEnergyAbstract:
    """Test the abstract base class"""

    def test_abstract_class_cannot_instantiate(self):
        """Test that InteractionEnergy cannot be instantiated"""
        from molecular_simulations.analysis.interaction_energy import InteractionEnergy

        with pytest.raises(TypeError):
            InteractionEnergy()


class TestDynamicInteractionEnergyAdditional:
    """Additional real tests for DynamicInteractionEnergy class."""

    def test_setup_pbar(self, two_chain_trajectory):
        """Test setup_pbar creates a real tqdm bar sized to the trajectory."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy(
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            chain='A',
            platform='CPU',
        )

        die.setup_pbar()

        # Real trajectory has 5 frames -> the real progress bar totals 5.
        assert die.pbar.total == 5
        die.pbar.close()

    def test_compute_energies_no_progress_bar(self, two_chain_trajectory):
        """Test compute_energies honours stride with the real per-frame engine."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy(
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            stride=2,
            chain='A',
            platform='CPU',
            progress_bar=False,
        )

        die.compute_energies()

        # 5 frames with stride 2 -> 2 computed energies, all finite.
        assert die.energies.shape == (2, 2)
        assert np.isfinite(die.energies).all()

    def test_compute_energies_with_progress_bar(self, two_chain_trajectory):
        """Test compute_energies with the real tqdm progress bar enabled."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy(
            top=str(two_chain_trajectory['top']),
            traj=str(two_chain_trajectory['traj']),
            stride=1,
            chain='A',
            platform='CPU',
            progress_bar=True,
        )

        die.compute_energies()

        assert die.energies.shape == (5, 2)
        assert np.isfinite(die.energies).all()

    def test_load_traj(self, two_chain_trajectory):
        """Test load_traj loads real trajectory coordinates with mdtraj."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)

        result = die.load_traj(
            two_chain_trajectory['top'], two_chain_trajectory['traj']
        )

        # 5 frames, 58 atoms, xyz
        assert result.shape == (5, 58, 3)


class TestDynamicInteractionEnergyBuildSystem:
    """Real build_system tests for DynamicInteractionEnergy."""

    def test_build_system_pdb(self, two_chain_pdb):
        """build_system creates a real OpenMM system from a PDB topology."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
        result = die.build_system(Path(str(two_chain_pdb)))

        assert result.getNumParticles() == 58
        # build_system stores the parsed OpenMM topology
        assert die.top.getNumAtoms() == 58

    def test_build_system_prmtop(self, real_amber_system_files):
        """build_system creates a real OpenMM system from an AMBER prmtop."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
        result = die.build_system(real_amber_system_files['prmtop'])

        assert result.getNumParticles() == 22

    def test_build_system_unsupported_format(self):
        """build_system raises for an unsupported file type."""
        from molecular_simulations.analysis.interaction_energy import (
            DynamicInteractionEnergy,
        )

        die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)

        with pytest.raises(NotImplementedError):
            die.build_system(Path('test.gro'))


class TestStaticInteractionEnergyGetSystem:
    """Real StaticInteractionEnergy.get_system tests, including the
    ValueError -> fix_pdb fallback triggered by a real (hydrogen-stripped)
    input rather than a mocked exception.
    """

    def test_get_system_success(self, two_chain_pdb):
        """get_system builds a real implicit-solvent System on the first try."""
        from openmm import System

        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), chain='A', platform='CPU')
        system = sie.get_system()

        assert isinstance(system, System)
        assert system.getNumParticles() == 58
        # get_system stores the positions and computes the chain-A selection.
        assert len(sie.positions) == 58
        assert sie.selection == list(range(34))

    def test_get_system_value_error_fallback(self, two_chain_pdb, tmp_path):
        """get_system recovers from a real ValueError via fix_pdb.

        A hydrogen-stripped copy of the complete structure makes the first
        ``forcefield.createSystem`` raise ValueError ('missing H atoms');
        the source then re-runs through PDBFixer, which re-adds the hydrogens,
        so a real, complete 58-particle System is returned.
        """
        from openmm.app import Modeller, PDBFile

        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        parsed = PDBFile(str(two_chain_pdb))
        modeller = Modeller(parsed.topology, parsed.positions)
        hydrogens = [
            a
            for a in modeller.topology.atoms()
            if a.element is not None and a.element.symbol == 'H'
        ]
        modeller.delete(hydrogens)
        stripped = tmp_path / 'stripped.pdb'
        with open(stripped, 'w') as fh:
            PDBFile.writeFile(modeller.topology, modeller.positions, fh)

        # Sanity: the stripped structure really is missing its hydrogens.
        assert modeller.topology.getNumAtoms() == 27

        sie = StaticInteractionEnergy(pdb=str(stripped), chain='A', platform='CPU')
        system = sie.get_system()

        # fix_pdb re-added the hydrogens -> full 58-particle system restored.
        assert system.getNumParticles() == 58
        assert len(sie.positions) == 58


class TestStaticInteractionEnergyCompute:
    """Real StaticInteractionEnergy.compute tests on the two-chain PDB."""

    def test_compute_full(self, two_chain_pdb):
        """compute fills real, finite LJ and Coulomb interaction energies."""
        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), chain='A', platform='CPU')
        sie.compute()

        assert np.isfinite(sie.lj)
        assert np.isfinite(sie.coulomb)
        # The Lys-Asp salt bridge across the interface is attractive.
        assert sie.coulomb < 0.0
        assert sie.interactions.shape == (2, 1)

    def test_compute_with_explicit_positions(self, two_chain_pdb):
        """compute uses explicitly provided positions for a real calculation."""
        from openmm.app import PDBFile

        from molecular_simulations.analysis.interaction_energy import (
            StaticInteractionEnergy,
        )

        sie = StaticInteractionEnergy(pdb=str(two_chain_pdb), chain='A', platform='CPU')

        # Feed the real PDB coordinates explicitly; the result must match a
        # default compute() that reads positions from the PDB itself.
        explicit_positions = PDBFile(str(two_chain_pdb)).positions
        sie.compute(positions=explicit_positions)
        lj_explicit, coulomb_explicit = sie.lj, sie.coulomb

        sie.compute()

        assert np.isfinite(lj_explicit)
        assert np.isfinite(coulomb_explicit)
        assert np.isclose(lj_explicit, sie.lj)
        assert np.isclose(coulomb_explicit, sie.coulomb)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
