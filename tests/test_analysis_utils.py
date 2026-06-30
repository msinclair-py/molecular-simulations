"""
Unit tests for analysis/utils.py module
"""

import shutil
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest


@pytest.fixture
def pdb_copy(tmp_path: Path, alanine_dipeptide_pdb: Path) -> Path:
    """A writable copy of the real Ace-Ala-Nme PDB (3 residues) in tmp_path.

    EmbedData writes back to its input path and backs the original up, so tests
    must operate on a copy rather than the committed fixture file.
    """
    dest = tmp_path / 'structure.pdb'
    shutil.copyfile(alanine_dipeptide_pdb, dest)
    return dest


class TestEmbedData:
    """Test suite for EmbedData class"""

    def test_embed_data_init(self, pdb_copy):
        """Test EmbedData initialization loads a real Universe."""
        from molecular_simulations.analysis.utils import EmbedData

        embedding_dict = {'protein': np.array([1.0, 2.0, 3.0])}
        embedder = EmbedData(pdb_copy, embedding_dict)

        assert embedder.pdb == pdb_copy
        assert embedder.out == pdb_copy  # No output specified
        assert embedder.embeddings == embedding_dict
        # Real Universe parsed from the PDB
        assert embedder.u.residues.n_residues == 3

    def test_embed_data_with_output(self, pdb_copy, tmp_path):
        """Test EmbedData with custom output path"""
        from molecular_simulations.analysis.utils import EmbedData

        out_path = tmp_path / 'output.pdb'
        embedding_dict = {'protein': np.array([1.0, 2.0, 3.0])}
        embedder = EmbedData(pdb_copy, embedding_dict, out=out_path)

        assert embedder.out == out_path

    def test_embed_selection(self, pdb_copy):
        """Test embed_selection writes data into each residue's beta column."""
        from molecular_simulations.analysis.utils import EmbedData

        embedder = EmbedData(pdb_copy, {'protein': np.array([1.0, 2.0, 3.0])})

        embedder.embed_selection('protein', np.array([1.0, 2.0, 3.0]))

        # Each residue's atoms get their beta factor set to the matching datum
        residues = embedder.u.select_atoms('protein').residues
        assert np.allclose(residues[0].atoms.tempfactors, 1.0)
        assert np.allclose(residues[1].atoms.tempfactors, 2.0)
        assert np.allclose(residues[2].atoms.tempfactors, 3.0)

    def test_write_new_pdb_with_backup(self, pdb_copy):
        """Test write_new_pdb backs up the original and writes a valid PDB."""
        from molecular_simulations.analysis.utils import EmbedData

        embedder = EmbedData(pdb_copy, {'protein': np.array([1.0, 2.0, 3.0])})
        embedder.write_new_pdb()

        # Writing back over the input creates a one-time backup
        backup = pdb_copy.with_suffix('.orig.pdb')
        assert backup.exists()
        # The output is a real, re-parseable PDB with the same atom count
        rewritten = mda.Universe(str(pdb_copy))
        assert rewritten.atoms.n_atoms == 23

    def test_embed_method(self, pdb_copy, tmp_path):
        """Test the full embed workflow writes embedded beta factors."""
        from molecular_simulations.analysis.utils import EmbedData

        out_path = tmp_path / 'embedded.pdb'
        embedder = EmbedData(
            pdb_copy, {'protein': np.array([1.0, 2.0, 3.0])}, out=out_path
        )

        embedder.embed()

        assert out_path.exists()
        # The embedded beta factors survive the round-trip
        result = mda.Universe(str(out_path))
        assert np.allclose(result.residues[1].atoms.tempfactors, 2.0)


class TestEmbedEnergyData:
    """Test suite for EmbedEnergyData class"""

    def test_embed_energy_data_init(self, pdb_copy):
        """Test EmbedEnergyData initialization and preprocessing"""
        from molecular_simulations.analysis.utils import EmbedEnergyData

        # Energy data with LJ and coulombic terms: 2 frames, 3 residues, 2 terms
        energy_data = np.array(
            [
                [[1.0, -0.5], [2.0, -1.0], [3.0, -1.5]],
                [[1.5, -0.7], [2.5, -1.2], [3.5, -1.7]],
            ]
        )

        embedder = EmbedEnergyData(pdb_copy, {'protein': energy_data})

        # Preprocessing reduces to one value per residue
        assert 'protein' in embedder.embeddings
        assert embedder.embeddings['protein'].shape == (3,)

    def test_sanitize_data_3d(self):
        """Test sanitize_data with 3D input (n_frames, n_residues, n_terms)"""
        from molecular_simulations.analysis.utils import EmbedEnergyData

        # 3D data: (2 frames, 3 residues, 2 terms)
        data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]],
            ]
        )

        result = EmbedEnergyData.sanitize_data(data)

        # Should average over frames, then sum terms
        # Average: [[1.25, 2.25], [3.25, 4.25], [5.25, 6.25]]
        # Sum: [3.5, 7.5, 11.5]
        expected = np.array([3.5, 7.5, 11.5])

        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sanitize_data_2d(self):
        """Test sanitize_data with 2D input (n_residues, n_terms)"""
        from molecular_simulations.analysis.utils import EmbedEnergyData

        # 2D data: (3 residues, 2 terms)
        data = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )

        result = EmbedEnergyData.sanitize_data(data)

        # Should sum terms
        expected = np.array([3.0, 7.0, 11.0])

        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, expected)

    def test_preprocess_rescaling(self, pdb_copy):
        """Test that preprocessing rescales data appropriately"""
        from molecular_simulations.analysis.utils import EmbedEnergyData

        # Simple energy data: 2 frames, 3 residues, 2 terms
        energy_data = np.array(
            [
                [[-10.0, -5.0], [-8.0, -4.0], [-6.0, -3.0]],
                [[-20.0, -10.0], [-16.0, -8.0], [-12.0, -6.0]],
            ]
        )

        embedder = EmbedEnergyData(pdb_copy, {'sel1': energy_data})

        # All rescaled values should be <= 1
        for data in embedder.embeddings.values():
            assert np.all(data <= 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
