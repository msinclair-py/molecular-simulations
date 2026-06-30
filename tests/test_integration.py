"""
Integration tests for molecular_simulations library

These tests verify that different components can work together:
- Builders create valid outputs for simulators
- Simulators produce valid trajectories for analysis
- Analysis tools can process simulator outputs

Run with: pytest tests/test_integration.py --run-integration
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration


# Helper function to create minimal PDB file
def create_sample_pdb(path):
    """Create a minimal valid PDB file for testing"""
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.989  -0.744   1.232  1.00  0.00           C
ATOM      6  N   GLY A   2       3.331   1.539   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       4.021   2.826   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.528   2.661   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.089   1.563   0.000  1.00  0.00           O
END
"""
    Path(path).write_text(pdb_content)
    return Path(path)


class TestBuildToSimulate:
    """Test that builders create valid inputs for simulators"""

    @patch.dict('os.environ', {'AMBERHOME': '/mock/amber/path'})
    @pytest.mark.requires_amber
    def test_implicit_solvent_builder_creates_simulator_inputs(self, tmp_path):
        """Test ImplicitSolvent builder creates files that Minimizer can use"""
        from molecular_simulations.build import ImplicitSolvent
        from molecular_simulations.simulate import Minimizer

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'input.pdb')

        # Build system
        builder = ImplicitSolvent(
            path=tmp_path, pdb=sample_pdb, protein=True, out='system.pdb'
        )

        # Mock tleap execution
        with patch.object(builder, 'temp_tleap'):
            # Create dummy output files
            (tmp_path / 'system.pdb').write_text(sample_pdb.read_text())
            (tmp_path / 'system.prmtop').write_text('%FLAG POINTERS\n')
            (tmp_path / 'system.inpcrd').write_text('coords\n')

            builder.build()

        # Verify files exist
        assert (tmp_path / 'system.pdb').exists()
        assert (tmp_path / 'system.prmtop').exists()
        assert (tmp_path / 'system.inpcrd').exists()

        # These files should be usable by Minimizer
        # (We can't actually run minimization without OpenMM, but check init)
        minimizer = Minimizer(
            topology=tmp_path / 'system.prmtop',
            coordinates=tmp_path / 'system.inpcrd',
            out='min.pdb',
            platform='CPU',
        )
        assert minimizer.topology.exists()
        assert minimizer.coordinates.exists()

    def test_builder_output_structure_for_analysis(self, tmp_path):
        """Test that builder output PDB can be loaded for analysis"""
        import MDAnalysis as mda

        from molecular_simulations.analysis import SASA

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'input.pdb')

        # Use the sample PDB as if it were builder output
        output_pdb = tmp_path / 'built_system.pdb'
        shutil.copy(sample_pdb, output_pdb)

        # Should be loadable by MDAnalysis
        u = mda.Universe(str(output_pdb))
        assert u.atoms.n_atoms > 0

        # Should be usable for analysis (though we mock the analysis)
        with patch('molecular_simulations.analysis.sasa.KDTree'):
            sasa = SASA(u.atoms[:5])  # Just use first 5 atoms
            assert sasa.ag.n_atoms == 5


class TestSimulateToAnalyze:
    """Test that simulators produce valid outputs for analysis"""

    def test_trajectory_output_readable_by_analysis(self, tmp_path):
        """Test that trajectory format is compatible with analysis tools"""
        import MDAnalysis as mda

        # Create a minimal PDB for topology
        topology = create_sample_pdb(tmp_path / 'topology.pdb')

        # Create mock universe (in real test, would load actual DCD)
        u = mda.Universe(str(topology))

        # Analysis tools should be able to work with this
        from molecular_simulations.analysis import SASA

        with patch('molecular_simulations.analysis.sasa.KDTree'):
            sasa = SASA(u.atoms)
            sasa._prepare()
            assert hasattr(sasa.results, 'sasa')

    def test_energy_log_parseable(self, tmp_path):
        """Test that simulation logs can be parsed for analysis"""
        # Simulate what a log file might look like
        log_content = 'Step\tPotential Energy (kJ/mole)\tTemperature (K)\n'
        for i in range(10):
            log_content += f'{i * 1000}\t{-5000 + np.random.rand() * 100}\t{300 + np.random.rand() * 5}\n'

        log_file = tmp_path / 'simulation.log'
        log_file.write_text(log_content)

        # Should be parseable
        import pandas as pd

        df = pd.read_csv(log_file, sep='\t')

        assert 'Step' in df.columns
        assert 'Potential Energy (kJ/mole)' in df.columns
        assert len(df) == 10


class TestFullPipeline:
    """Test complete workflows from start to finish"""

    def test_simple_analysis_pipeline(self, tmp_path):
        """Test: Load structure -> Calculate property -> Save result"""
        import MDAnalysis as mda
        import numpy as np

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'structure.pdb')

        # Load structure (as if from builder)
        u = mda.Universe(str(sample_pdb))

        # Calculate simple property (center of mass)
        com = u.atoms.center_of_mass()
        assert com.shape == (3,)

        # Save result
        result_file = tmp_path / 'analysis_result.npy'
        np.save(result_file, com)

        # Verify saved result
        assert result_file.exists()
        loaded_com = np.load(result_file)
        assert np.allclose(com, loaded_com)

    def test_clustering_workflow(self, tmp_path):
        """Test: Generate data -> Cluster -> Save clusters"""
        from molecular_simulations.analysis.autocluster import (
            AutoKMeans,
        )

        # Generate mock trajectory data (as if from simulation)
        n_frames = 50
        n_features = 10

        for i in range(3):
            data = np.random.rand(n_frames, n_features)
            np.save(tmp_path / f'traj_{i}.npy', data)

        # Run clustering analysis
        with patch(
            'molecular_simulations.analysis.autocluster.silhouette_score'
        ) as mock_score:
            mock_score.return_value = 0.5

            auto_km = AutoKMeans(tmp_path, pattern='traj_', max_clusters=5, stride=1)

            # Run clustering
            auto_km.reduce_dimensionality()
            auto_km.sweep_n_clusters([2, 3])
            auto_km.map_centers_to_frames()
            auto_km.save_centers()
            auto_km.save_labels()

        # Verify outputs exist
        assert (tmp_path / 'cluster_centers.json').exists()
        assert (tmp_path / 'cluster_assignments.parquet').exists()

    @pytest.mark.slow
    def test_fingerprint_calculation_workflow(self, tmp_path):
        """Test: Load structure -> Calculate fingerprint -> Save"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'structure.pdb')

        # This test would normally require OpenMM and proper topology
        # Here we just verify the workflow structure

        # Mock the OpenMM parts
        with (
            patch('molecular_simulations.analysis.fingerprinter.AmberPrmtopFile'),
            patch(
                'molecular_simulations.analysis.fingerprinter.mda.Universe'
            ) as mock_u,
        ):
            # Setup mocks
            mock_universe = MagicMock()
            mock_ag = MagicMock()
            mock_ag.residues = [MagicMock() for _ in range(5)]
            for i, res in enumerate(mock_ag.residues):
                res.atoms.ix = np.array([i])
            mock_universe.select_atoms.return_value = mock_ag
            mock_universe.atoms.positions = np.random.rand(10, 3)
            mock_universe.trajectory = [MagicMock()]
            mock_u.return_value = mock_universe

            fp = Fingerprinter(
                topology=sample_pdb, target_selection='segid A', out_path=tmp_path
            )

            # Verify workflow can be set up
            assert fp.topology.exists()
            assert fp.out.parent == tmp_path


class TestDataFlow:
    """Test that data flows correctly between components"""

    def test_coordinate_preservation(self, tmp_path):
        """Test that coordinates are preserved through analysis chain"""
        import MDAnalysis as mda

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'structure.pdb')

        # Load coordinates
        u = mda.Universe(str(sample_pdb))
        original_coords = u.atoms.positions.copy()

        # Perform some analysis (that shouldn't modify coords)
        u.atoms.center_of_mass()

        # Verify coordinates unchanged
        assert np.allclose(u.atoms.positions, original_coords)

    def test_residue_indexing_consistency(self, tmp_path):
        """Test that residue indexing is consistent across tools"""
        import MDAnalysis as mda

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'structure.pdb')

        u = mda.Universe(str(sample_pdb))

        # Get residue indices different ways
        resids_from_atoms = np.unique([atom.resid for atom in u.atoms])
        resids_from_residues = u.residues.resids

        # Should match
        assert np.array_equal(resids_from_atoms, resids_from_residues)

    def test_selection_consistency(self, tmp_path):
        """Test that selections work consistently"""
        import MDAnalysis as mda

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'structure.pdb')

        u = mda.Universe(str(sample_pdb))

        # Different selection methods should give same result
        sel1 = u.select_atoms('protein')
        sel2 = u.select_atoms('all') & u.select_atoms('protein')

        assert sel1.n_atoms == sel2.n_atoms
        assert np.array_equal(sel1.indices, sel2.indices)


class TestErrorHandling:
    """Test that errors are handled gracefully in pipelines"""

    def test_missing_file_handling(self, tmp_path):
        """Test graceful handling of missing input files"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        # Try to create with non-existent file
        with pytest.raises((FileNotFoundError, Exception)):
            fp = Fingerprinter(
                topology=tmp_path / 'nonexistent.prmtop',
                trajectory=tmp_path / 'nonexistent.dcd',
            )
            # Some initialization might not fail until we try to use it
            fp.load_pdb()

    def test_incompatible_topology_trajectory(self, tmp_path):
        """Test detection of incompatible topology/trajectory"""
        # This would test that mismatched atom counts are caught
        # Implementation depends on specific error handling in your code
        pass

    def test_empty_selection_handling(self, tmp_path):
        """Test handling of empty atom selections"""
        import MDAnalysis as mda

        # Create sample PDB
        sample_pdb = create_sample_pdb(tmp_path / 'structure.pdb')

        u = mda.Universe(str(sample_pdb))

        # Empty selection
        empty_sel = u.select_atoms('resname NOTEXIST')
        assert empty_sel.n_atoms == 0

        # Analysis on empty selection should handle gracefully

        if empty_sel.n_atoms == 0:
            # Should either skip or raise informative error
            # Depends on your implementation
            pass


@pytest.fixture
def integration_test_system(tmp_path):
    """
    Fixture that creates a complete test system with all necessary files
    for integration testing
    """
    system_dir = tmp_path / 'test_system'
    system_dir.mkdir()

    # Create sample structure
    create_sample_pdb(system_dir / 'input.pdb')

    # Create dummy topology files
    (system_dir / 'system.prmtop').write_text('%FLAG POINTERS\n')
    (system_dir / 'system.inpcrd').write_text('coordinates\n')

    # Create dummy trajectory data
    traj_data = np.random.rand(10, 10, 3)
    np.save(system_dir / 'trajectory.npy', traj_data)

    return system_dir
