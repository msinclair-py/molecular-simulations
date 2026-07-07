"""Unit tests for the pure structure-prep logic in scripts/benchmark_hewl.py."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_hewl.py'


def _load():
    spec = importlib.util.spec_from_file_location('benchmark_hewl', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A tiny PDB: two protein residues (chain A), a water HETATM, and a B-chain atom.
SAMPLE_PDB = """\
ATOM      1  N   ASP A  18       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ASP A  18       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  N  AGLU A  35       2.000   0.000   0.000  1.00  0.00           N
ATOM      4  N  BGLU A  35       2.100   0.000   0.000  0.50  0.00           N
ATOM      5  CA  GLU A  35       3.000   0.000   0.000  1.00  0.00           C
TER
ATOM      6  N   LYS B   1       4.000   0.000   0.000  1.00  0.00           N
HETATM    7  O   HOH A 200       9.000   9.000   9.000  1.00  0.00           O
END
"""


def test_clean_pdb_strips_water_and_keeps_protein(tmp_path):
    mod = _load()
    src = tmp_path / 'in.pdb'
    src.write_text(SAMPLE_PDB)
    dest = tmp_path / 'out.pdb'

    residues = mod.clean_pdb(src, dest, chain='A')

    # Two chain-A residues retained, in order; water and chain B dropped.
    assert residues == [('18', 'ASP'), ('35', 'GLU')]
    text = dest.read_text()
    assert 'HOH' not in text
    assert ' B   1' not in text


def test_clean_pdb_keeps_only_first_altloc(tmp_path):
    mod = _load()
    src = tmp_path / 'in.pdb'
    src.write_text(SAMPLE_PDB)
    dest = tmp_path / 'out.pdb'

    mod.clean_pdb(src, dest, chain='A')

    # The B altloc of Glu35 N is dropped; the A altloc is kept with a blank altloc.
    lines = [ln for ln in dest.read_text().splitlines() if ln.startswith('ATOM')]
    n_glu35_n = [ln for ln in lines if ln[12:16].strip() == 'N' and '35' in ln[22:27]]
    assert len(n_glu35_n) == 1
    assert n_glu35_n[0][16] == ' '  # altloc normalized to blank


def test_clean_pdb_without_chain_filter_keeps_all_chains(tmp_path):
    mod = _load()
    src = tmp_path / 'in.pdb'
    src.write_text(SAMPLE_PDB)
    dest = tmp_path / 'out.pdb'

    residues = mod.clean_pdb(src, dest, chain=None)

    # Now the chain-B Lys is retained too (still no water).
    assert ('1', 'LYS') in residues
    assert 'HOH' not in dest.read_text()
