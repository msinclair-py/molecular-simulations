"""
Unit tests for utils/amber_utils.py module
"""

import string

import MDAnalysis as mda
import numpy as np
import pytest

from molecular_simulations.utils.amber_utils import assign_chainids


class TestAssignChainIds:
    """Test suite for assign_chainids function using real MDAnalysis Universes."""

    def test_assign_chainids_single_chain(self, alanine_dipeptide_pdb):
        """A single (uncapped-terminus) chain gets one label for every residue."""
        u = mda.Universe(str(alanine_dipeptide_pdb))

        # alanine dipeptide (Ace-Ala-Nme) carries no OXT terminus atom, so the
        # default selection finds nothing and the chain index never advances.
        result = assign_chainids(u)

        assert result is u
        assert hasattr(u.atoms, 'chainIDs')
        labels = [str(res.atoms.chainIDs[0]) for res in u.residues]
        assert labels == ['A', 'A', 'A']

    def test_assign_chainids_multi_chain(self, two_chain_pdb):
        """Two NME-capped chains split into A and B at their termini."""
        u = mda.Universe(str(two_chain_pdb))

        # Chains here are capped with NME rather than OXT, so the terminus is the
        # backbone C of each NME residue (resindices 2 and 5).
        result = assign_chainids(u, terminus_selection='name C and resname NME')

        assert result is u
        labels = [str(res.atoms.chainIDs[0]) for res in u.residues]
        # Residues 0-2 (ACE/LYS/NME) -> A, residues 3-5 (ACE/ASP/NME) -> B
        assert labels == ['A', 'A', 'A', 'B', 'B', 'B']

    def test_assign_chainids_custom_terminus_selection(self, two_chain_pdb):
        """The terminus selection string controls where chains are split."""
        # Default selection ('name OXT') finds no termini in this NME-capped
        # system, so every residue collapses into a single chain.
        u_default = mda.Universe(str(two_chain_pdb))
        assign_chainids(u_default)
        default_labels = [str(res.atoms.chainIDs[0]) for res in u_default.residues]
        assert set(default_labels) == {'A'}

        # A custom selection targeting the NME backbone C splits into two chains.
        u_custom = mda.Universe(str(two_chain_pdb))
        assign_chainids(u_custom, terminus_selection='name C and resname NME')
        custom_labels = [str(res.atoms.chainIDs[0]) for res in u_custom.residues]
        assert custom_labels[:3] == ['A', 'A', 'A']
        assert custom_labels[3:] == ['B', 'B', 'B']

    def test_assign_chainids_more_than_26_chains(self):
        """Chain labels wrap to AA, AB, ... once 26 single letters are exhausted."""
        n = 30
        # Build a real Universe of n single-atom, single-residue "chains". Every
        # atom is named OXT so the default terminus selection treats each residue
        # as its own chain terminus.
        u = mda.Universe.empty(
            n_atoms=n,
            n_residues=n,
            atom_resindex=np.arange(n),
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['OXT'] * n)
        u.add_TopologyAttr('resid', list(range(1, n + 1)))
        u.add_TopologyAttr('resname', ['ALA'] * n)

        assign_chainids(u)

        labels = [str(res.atoms.chainIDs[0]) for res in u.residues]

        # First 26 residues get single-letter IDs A-Z.
        assert labels[:26] == list(string.ascii_uppercase)
        # Residues 26-29 wrap to double letters AA, AB, AC, AD.
        assert labels[26] == 'AA'
        assert labels[27] == 'AB'
        assert labels[28] == 'AC'
        assert labels[29] == 'AD'

    def test_assign_chainids_adds_attribute_when_missing(self):
        """assign_chainids adds the chainIDs topology attribute if absent."""
        n = 3
        u = mda.Universe.empty(
            n_atoms=n,
            n_residues=n,
            atom_resindex=np.arange(n),
            trajectory=True,
        )
        u.add_TopologyAttr('name', ['OXT'] * n)
        u.add_TopologyAttr('resid', list(range(1, n + 1)))
        u.add_TopologyAttr('resname', ['ALA'] * n)

        assert not hasattr(u.atoms, 'chainIDs')
        assign_chainids(u)
        assert hasattr(u.atoms, 'chainIDs')
        labels = [str(res.atoms.chainIDs[0]) for res in u.residues]
        assert labels == ['A', 'B', 'C']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
