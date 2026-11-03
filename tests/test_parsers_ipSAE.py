import numpy as np
import pytest

from molecular_simulations.analysis.ipSAE import ModelParser

pytestmark = pytest.mark.fast  # quick to run


def test_ipSAE_pdb_atom_line_parsing(tmp_path):
    """
    Tests PDB ATOM/HETATM line parsing in ModelParser.read_atoms.
    """
    pdb = tmp_path / 'model.pdb'
    pdb.write_text(
        'ATOM      1  N   MET A   1      38.428  13.947   8.678  1.00 54.69           N  \n'
    )

    atoms = ModelParser(pdb).read_atoms()

    assert len(atoms) == 1
    assert atoms.record[0] == 'ATOM'
    assert atoms.atom_name[0] == 'N'
    assert atoms.resname[0] == 'MET'
    assert atoms.chain[0] == 'A'
    assert atoms.resnum[0] == '1'
    assert atoms.element[0] == 'N'
    assert np.allclose(atoms.xyz[0], [38.428, 13.947, 8.678])


def test_ipSAE_cif_atom_line_parsing(tmp_path):
    """
    Tests mmCIF atom_site parsing in ModelParser.read_atoms.
    """
    cif = tmp_path / 'model.cif'
    cif.write_text(
        '\n'.join(
            [
                'loop_',
                '_atom_site.group_PDB',
                '_atom_site.id',
                '_atom_site.label_atom_id',
                '_atom_site.label_alt_id',
                '_atom_site.label_comp_id',
                '_atom_site.label_asym_id',
                '_atom_site.label_seq_id',
                '_atom_site.Cartn_x',
                '_atom_site.Cartn_y',
                '_atom_site.Cartn_z',
                '_atom_site.occupancy',
                '_atom_site.B_iso_or_equiv',
                '_atom_site.type_symbol',
                'ATOM 1 N . MET A 1 38.428 13.947 8.678 1.00 54.69 N',
            ]
        )
        + '\n'
    )

    atoms = ModelParser(cif).read_atoms()

    assert len(atoms) == 1
    assert atoms.record[0] == 'ATOM'
    assert atoms.atom_name[0] == 'N'
    assert atoms.resname[0] == 'MET'
    assert atoms.chain[0] == 'A'
    assert atoms.resnum[0] == '1'
    assert atoms.element[0] == 'N'
    assert np.allclose(atoms.xyz[0], [38.428, 13.947, 8.678])
