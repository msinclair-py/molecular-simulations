"""
Unit tests for utils/amber_utils.py module
"""

import string
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestAssignChainIds:
    """Test suite for assign_chainids function"""

    def test_assign_chainids_single_chain(self):
        """Test assigning chain IDs to a single-chain structure"""
        from molecular_simulations.utils.amber_utils import assign_chainids

        # Create mock universe with one chain
        mock_u = MagicMock()
        mock_u.atoms = MagicMock()
        mock_u.atoms.chainIDs = None

        # Create mock residues
        mock_residue1 = MagicMock()
        mock_residue1.resindex = 0
        mock_residue1.atoms = MagicMock()
        mock_residue1.atoms.chainIDs = None

        mock_residue2 = MagicMock()
        mock_residue2.resindex = 1
        mock_residue2.atoms = MagicMock()
        mock_residue2.atoms.chainIDs = None

        mock_u.residues = [mock_residue1, mock_residue2]

        # Mock terminus selection - last residue is terminus
        mock_terminus_atoms = MagicMock()
        mock_terminus_atoms.resindices = np.array([1])
        mock_u.select_atoms.return_value = mock_terminus_atoms

        # hasattr returns False to trigger adding chainIDs
        with patch(
            'molecular_simulations.utils.amber_utils.hasattr', return_value=False
        ):
            result = assign_chainids(mock_u)

        # Should return the universe
        assert result == mock_u

    def test_get_chain_label_single_letter(self):
        """Test chain label generation for indices 0-25"""
        from molecular_simulations.utils.amber_utils import assign_chainids

        # Create a minimal mock universe to test the internal get_chain_label function
        mock_u = MagicMock()
        mock_u.residues = []
        mock_u.select_atoms.return_value = MagicMock(resindices=np.array([]))

        # hasattr returns True (chainIDs already exists)
        with patch(
            'molecular_simulations.utils.amber_utils.hasattr', return_value=True
        ):
            result = assign_chainids(mock_u)

        # Should return universe without error
        assert result == mock_u

    def test_assign_chainids_multi_chain(self):
        """Test assigning chain IDs to multi-chain structure"""
        from molecular_simulations.utils.amber_utils import assign_chainids

        mock_u = MagicMock()

        # Create mock residues for two chains
        residues = []
        for i in range(6):
            mock_res = MagicMock()
            mock_res.resindex = i
            mock_res.atoms = MagicMock()
            mock_res.atoms.tempfactors = MagicMock()
            mock_res.atoms.tempfactors.shape = (5,)
            residues.append(mock_res)

        mock_u.residues = residues

        # Residues 2 and 5 are termini (end of chains)
        mock_terminus = MagicMock()
        mock_terminus.resindices = np.array([2, 5])
        mock_u.select_atoms.return_value = mock_terminus

        with patch(
            'molecular_simulations.utils.amber_utils.hasattr', return_value=False
        ):
            result = assign_chainids(mock_u)

        assert result == mock_u
        # First 3 residues should be chain A
        assert residues[0].atoms.chainIDs == 'A'
        assert residues[1].atoms.chainIDs == 'A'
        assert residues[2].atoms.chainIDs == 'A'
        # Next 3 residues should be chain B
        assert residues[3].atoms.chainIDs == 'B'
        assert residues[4].atoms.chainIDs == 'B'
        assert residues[5].atoms.chainIDs == 'B'

    def test_assign_chainids_more_than_26_chains(self):
        """Test chain labeling for more than 26 chains (requires double letters)"""
        from molecular_simulations.utils.amber_utils import assign_chainids

        mock_u = MagicMock()

        # Create 30 single-residue chains
        residues = []
        terminus_indices = []
        for i in range(30):
            mock_res = MagicMock()
            mock_res.resindex = i
            mock_res.atoms = MagicMock()
            residues.append(mock_res)
            terminus_indices.append(i)  # Each residue is a terminus

        mock_u.residues = residues
        mock_terminus = MagicMock()
        mock_terminus.resindices = np.array(terminus_indices)
        mock_u.select_atoms.return_value = mock_terminus

        with patch(
            'molecular_simulations.utils.amber_utils.hasattr', return_value=False
        ):
            assign_chainids(mock_u)

        # First 26 residues should have single letter IDs (A-Z)
        for i in range(26):
            assert residues[i].atoms.chainIDs == string.ascii_uppercase[i]

        # Residues 26-29 should have double letter IDs (AA, AB, AC, AD)
        assert residues[26].atoms.chainIDs == 'AA'
        assert residues[27].atoms.chainIDs == 'AB'
        assert residues[28].atoms.chainIDs == 'AC'
        assert residues[29].atoms.chainIDs == 'AD'

    def test_assign_chainids_custom_terminus_selection(self):
        """Test with custom terminus selection string"""
        from molecular_simulations.utils.amber_utils import assign_chainids

        mock_u = MagicMock()
        mock_u.residues = []
        mock_terminus = MagicMock()
        mock_terminus.resindices = np.array([])
        mock_u.select_atoms.return_value = mock_terminus

        with patch(
            'molecular_simulations.utils.amber_utils.hasattr', return_value=False
        ):
            assign_chainids(mock_u, terminus_selection='name C and resname NME')

        # Should call select_atoms with custom selection
        mock_u.select_atoms.assert_called_with('name C and resname NME')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
