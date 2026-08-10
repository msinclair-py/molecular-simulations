"""Unit tests for the BDE -> Morse D_e helper (ALFABET wrapper).

The ALFABET model call itself needs the `alfabet` package + TensorFlow, so the
prediction wrapper is exercised only when it is installed. The dependency-light
logic -- unit conversion and bond selection -- is tested against real pandas
DataFrames (not mocks).
"""

import pytest

from molecular_simulations.build.morse_bde import (
    KCAL_TO_KJ,
    bde_to_de,
    select_bde,
)

pd = pytest.importorskip('pandas')


class TestBdeToDe:
    def test_kcal_to_kj(self):
        # ALFABET C-H example: 93.8 kcal/mol -> 392.46 kJ/mol (the documented D_e).
        assert bde_to_de(93.8) == pytest.approx(93.8 * KCAL_TO_KJ)
        assert bde_to_de(93.8) == pytest.approx(392.46, abs=0.02)

    def test_kj_passthrough(self):
        assert bde_to_de(392.46, units='kJ/mol') == pytest.approx(392.46)

    def test_rejects_bad_units(self):
        with pytest.raises(ValueError, match='units'):
            bde_to_de(93.8, units='eV')

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match='positive'):
            bde_to_de(0.0)


def _preds():
    return pd.DataFrame(
        {
            'bond_index': [0, 3, 5],
            'bond_type': ['C-H', 'C-O', 'O-H'],
            'bde': [96.1, 84.0, 104.2],
        }
    )


class TestSelectBde:
    def test_by_index(self):
        assert select_bde(_preds(), bond_index=3) == pytest.approx(84.0)

    def test_by_unique_type(self):
        assert select_bde(_preds(), bond_type='C-H') == pytest.approx(96.1)

    def test_ambiguous_type_raises(self):
        two = pd.DataFrame(
            {'bond_index': [0, 1], 'bond_type': ['C-H', 'C-H'], 'bde': [96.1, 98.0]}
        )
        with pytest.raises(ValueError, match='ambiguous'):
            select_bde(two, bond_type='C-H')

    def test_missing_raises(self):
        with pytest.raises(ValueError, match='no'):
            select_bde(_preds(), bond_index=99)
        with pytest.raises(ValueError, match='available types'):
            select_bde(_preds(), bond_type='N-H')

    def test_requires_exactly_one_selector(self):
        with pytest.raises(ValueError, match='exactly one'):
            select_bde(_preds())
        with pytest.raises(ValueError, match='exactly one'):
            select_bde(_preds(), bond_index=0, bond_type='C-H')
