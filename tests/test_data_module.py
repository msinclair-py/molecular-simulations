"""
Unit tests for data/ module - reference energy data.
"""

import pytest


class TestGetRefEnergies:
    """Test suite for get_ref_energies function."""

    def test_amber19_default(self):
        """Test default amber19 reference energies."""
        from molecular_simulations.data import get_ref_energies

        ref = get_ref_energies()

        assert 'CYS' in ref
        assert 'ASP' in ref
        assert 'GLU' in ref
        assert 'LYS' in ref
        assert 'HIS' in ref

        # Each entry should have 2 states [deprotonated, protonated]
        for _name, energies in ref.items():
            assert len(energies) == 2
            assert energies[0] == 0.0  # deprotonated is always 0

    def test_amber19_explicit(self):
        """Test explicit amber19 force field parameter."""
        from molecular_simulations.data import get_ref_energies

        ref = get_ref_energies(ff='amber19')
        assert ref['ASP'][1] < 0  # protonated state has negative reference energy

    def test_unknown_forcefield_raises(self):
        """Test that unknown force field raises ValueError."""
        from molecular_simulations.data import get_ref_energies

        with pytest.raises(ValueError, match='not yet computed'):
            get_ref_energies(ff='charmm36')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
