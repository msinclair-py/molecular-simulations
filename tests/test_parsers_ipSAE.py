import numpy as np
import pytest

pytestmark = pytest.mark.fast  # quick to run


def test_ipSAE_pdb_atom_line_parsing():
    """
    Tests PDB ATOM/HETATM line parser from ModelParser.
    """
    try:
        from molecular_simulations.analysis.ipSAE import ModelParser
    except Exception as e:
        pytest.skip(f'ModelParser not importable from ipSAE: {e}')

    pdb_line = 'ATOM      1  N   MET A   1      38.428  13.947   8.678  1.00 54.69           N  '

    # ModelParser has parse_pdb_line as a static method
    if not hasattr(ModelParser, 'parse_pdb_line'):
        pytest.skip('ModelParser has no parse_pdb_line method')

    # Call the static method
    rec = ModelParser.parse_pdb_line(pdb_line)

    # Generic assertions
    assert rec is not None
    assert isinstance(rec, dict)

    # Check that parsed data contains expected fields
    assert 'atom_num' in rec
    assert 'atom_name' in rec
    assert 'res' in rec
    assert 'chain_id' in rec
    assert 'resid' in rec
    assert 'coor' in rec

    # Verify specific values
    assert rec['atom_num'] == 1
    assert rec['atom_name'] == 'N'
    assert rec['res'] == 'MET'
    assert rec['chain_id'] == 'A'
    assert rec['resid'] == 1
    assert np.allclose(rec['coor'], [38.428, 13.947, 8.678])


def test_ipSAE_cif_atom_line_parsing():
    """
    Tests CIF-style atom line parser from ModelParser.
    """
    try:
        from molecular_simulations.analysis.ipSAE import ModelParser
    except Exception as e:
        pytest.skip(f'ModelParser not importable from ipSAE: {e}')

    # ModelParser has parse_cif_line as a static method
    if not hasattr(ModelParser, 'parse_cif_line'):
        pytest.skip('ModelParser has no parse_cif_line method')

    # mmCIF format line
    cif_line = 'ATOM 1 N . MET A 1 38.428 13.947 8.678 1.00 54.69 N'

    # Define field indices for mmCIF format
    fields = {
        'id': 1,  # Atom serial number
        'label_atom_id': 2,  # Atom name
        'label_comp_id': 4,  # Residue name
        'label_asym_id': 5,  # Chain ID
        'label_seq_id': 6,  # Residue ID
        'Cartn_x': 7,  # X coordinate
        'Cartn_y': 8,  # Y coordinate
        'Cartn_z': 9,  # Z coordinate
    }

    # Call the static method
    rec = ModelParser.parse_cif_line(cif_line, fields)

    # Generic assertions
    assert rec is not None
    assert isinstance(rec, dict)

    # Check that parsed data contains expected fields
    assert 'atom_num' in rec
    assert 'atom_name' in rec
    assert 'res' in rec
    assert 'chain_id' in rec
    assert 'resid' in rec
    assert 'coor' in rec

    # Verify specific values
    assert rec['atom_num'] == 1
    assert rec['atom_name'] == 'N'
    assert rec['res'] == 'MET'
    assert rec['chain_id'] == 'A'
    assert rec['resid'] == 1
    assert np.allclose(rec['coor'], [38.428, 13.947, 8.678])
