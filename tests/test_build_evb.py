"""Unit tests for the dependency-light logic in build_evb.

The tleap/antechamber assembly is exercised separately against a real
AmberTools install; here we test the pure pieces: mol2 parsing/merging, the
reactant/product consistency check, box reading, and constructor validation.
"""

from pathlib import Path

import pytest

from molecular_simulations.build.build_evb import (
    EVBBuilder,
    EVBBuildError,
    _as_list,
    _merge_mol2,
    _Mol2,
    _parse_mol2,
    _write_mol2,
)

pytestmark = pytest.mark.unit


_METHANOL_MOL2 = """@<TRIPOS>MOLECULE
LIG
    6     5     1     0     0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1   -0.3680  -0.0390  -0.0230 c3  1 LIG  0.116700
      2 O1    0.9840  -0.3980  -0.2410 oh  1 LIG -0.598800
      3 H1   -0.5650   0.0090   1.0510 h1  1 LIG  0.028700
      4 H2   -1.0140  -0.7950  -0.4760 h1  1 LIG  0.028700
      5 H3   -0.5680   0.9310  -0.4840 h1  1 LIG  0.028700
      6 H4    1.5320   0.2910   0.1720 ho  1 LIG  0.396000
@<TRIPOS>BOND
     1     1     2 1
     2     1     3 1
     3     1     4 1
     4     1     5 1
     5     2     6 1
"""

_METHANE_MOL2 = """@<TRIPOS>MOLECULE
LIG
    5     4     1     0     0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1    0.0000   0.0000   0.0000 c3  1 LIG -0.106000
      2 H1    0.6300   0.6300   0.6300 hc  1 LIG  0.026500
      3 H2   -0.6300  -0.6300   0.6300 hc  1 LIG  0.026500
      4 H3   -0.6300   0.6300  -0.6300 hc  1 LIG  0.026500
      5 H4    0.6300  -0.6300  -0.6300 hc  1 LIG  0.026500
@<TRIPOS>BOND
     1     1     2 1
     2     1     3 1
     3     1     4 1
     4     1     5 1
"""


@pytest.fixture
def methanol(tmp_path: Path) -> Path:
    p = tmp_path / 'methanol.mol2'
    p.write_text(_METHANOL_MOL2)
    return p


@pytest.fixture
def methane(tmp_path: Path) -> Path:
    p = tmp_path / 'methane.mol2'
    p.write_text(_METHANE_MOL2)
    return p


class TestParseMol2:
    def test_parses_atoms_and_bonds(self, methanol: Path):
        m = _parse_mol2(methanol)
        assert m.n_atoms == 6
        assert m.names == ['C1', 'O1', 'H1', 'H2', 'H3', 'H4']
        assert m.types[1] == 'oh'
        assert m.charges[1] == pytest.approx(-0.5988)
        assert m.charges[5] == pytest.approx(0.396)
        assert len(m.bonds) == 5
        assert (2, 6, '1') in m.bonds

    def test_empty_raises(self, tmp_path: Path):
        p = tmp_path / 'empty.mol2'
        p.write_text('@<TRIPOS>MOLECULE\nX\n0 0 0 0 0\n')
        with pytest.raises(EVBBuildError, match='no ATOM records'):
            _parse_mol2(p)


class TestWriteRoundTrip:
    def test_roundtrip_preserves_atoms_and_charges(self, methanol: Path, tmp_path):
        m = _parse_mol2(methanol)
        out = tmp_path / 'rt.mol2'
        _write_mol2(m, out, 'LIG')
        m2 = _parse_mol2(out)
        assert m2.names == m.names
        assert m2.types == m.types
        assert m2.charges == pytest.approx(m.charges)
        assert m2.bonds == m.bonds


class TestMergeMol2:
    def test_atom_and_bond_counts(self, methanol: Path, methane: Path, tmp_path):
        out = tmp_path / 'merged.mol2'
        merged = _merge_mol2([methanol, methane], out, 'LIG')
        assert merged.n_atoms == 11  # 6 + 5
        assert len(merged.bonds) == 9  # 5 + 4

    def test_bond_indices_offset(self, methanol: Path, methane: Path, tmp_path):
        out = tmp_path / 'merged.mol2'
        merged = _merge_mol2([methanol, methane], out, 'LIG')
        # methane's first bond (1-2) must be offset by 6 -> (7, 8).
        assert (7, 8, '1') in merged.bonds
        # methanol bonds are unchanged.
        assert (2, 6, '1') in merged.bonds

    def test_charges_concatenated_in_order(self, methanol, methane, tmp_path):
        merged = _merge_mol2([methanol, methane], tmp_path / 'm.mol2', 'LIG')
        assert merged.charges[1] == pytest.approx(-0.5988)  # methanol O
        assert merged.charges[6] == pytest.approx(-0.106)  # methane C

    def test_single_species_passthrough(self, methanol: Path, tmp_path):
        merged = _merge_mol2([methanol], tmp_path / 's.mol2', 'LIG')
        assert merged.n_atoms == 6


class TestStateConsistency:
    def _builder(self, tmp_path, monkeypatch, reactant, product):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        return EVBBuilder(
            out_path=tmp_path / 'out',
            reactant=reactant,
            product=product,
        )

    def test_atom_count_mismatch_raises(self, methanol, methane, tmp_path, monkeypatch):
        b = self._builder(tmp_path, monkeypatch, methanol, methane)
        r = _parse_mol2(methanol)
        p = _parse_mol2(methane)
        with pytest.raises(EVBBuildError, match='atom count mismatch'):
            b._check_states_consistent(r, p)

    def test_name_order_mismatch_raises(self, methanol, tmp_path, monkeypatch):
        b = self._builder(tmp_path, monkeypatch, methanol, methanol)
        r = _parse_mol2(methanol)
        p = _Mol2(
            names=list(reversed(r.names)),
            types=r.types,
            coords=r.coords,
            charges=r.charges,
            bonds=r.bonds,
        )
        with pytest.raises(EVBBuildError, match='NAMES/order differ'):
            b._check_states_consistent(r, p)

    def test_consistent_states_pass(self, methanol, tmp_path, monkeypatch):
        b = self._builder(tmp_path, monkeypatch, methanol, methanol)
        r = _parse_mol2(methanol)
        b._check_states_consistent(r, r)  # no raise


class TestReadBox:
    def test_reads_box_lengths(self, tmp_path: Path):
        inpcrd = tmp_path / 'system.inpcrd'
        inpcrd.write_text(
            'title\n     6\n'
            '  1.0  1.0  1.0  2.0  2.0  2.0\n'
            '  28.3291785  27.6494425  27.8865755  90.0  90.0  90.0\n'
        )
        box = EVBBuilder._read_box(inpcrd)
        assert box == pytest.approx([28.3291785, 27.6494425, 27.8865755])

    def test_missing_box_raises(self, tmp_path: Path):
        inpcrd = tmp_path / 'nobox.inpcrd'
        inpcrd.write_text('title\n  3\n  1.0  2.0\n')
        with pytest.raises(EVBBuildError, match='no box'):
            EVBBuilder._read_box(inpcrd)


class TestConstructor:
    def test_unknown_water_raises(self, methanol, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        with pytest.raises(EVBBuildError, match='unknown water'):
            EVBBuilder(
                out_path=tmp_path / 'o',
                reactant=methanol,
                product=methanol,
                water='spce',
            )

    def test_missing_amberhome_raises(self, methanol, tmp_path, monkeypatch):
        monkeypatch.delenv('AMBERHOME', raising=False)
        with pytest.raises(EVBBuildError, match='AMBERHOME'):
            EVBBuilder(out_path=tmp_path / 'o', reactant=methanol, product=methanol)

    def test_single_path_becomes_list(self, methanol, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        b = EVBBuilder(out_path=tmp_path / 'o', reactant=methanol, product=methanol)
        assert b.reactant_inputs == [methanol.resolve()]
        assert b.product_inputs == [methanol.resolve()]


class TestAsList:
    def test_wraps_scalar(self):
        assert _as_list('a.mol2') == ['a.mol2']
        assert _as_list(Path('a.mol2')) == [Path('a.mol2')]

    def test_passes_list(self):
        assert _as_list(['a', 'b']) == ['a', 'b']
