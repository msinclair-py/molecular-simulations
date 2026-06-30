from pathlib import Path
from unittest.mock import patch

from molecular_simulations.build import ImplicitSolvent


@patch.dict('os.environ', {'AMBERHOME': '/mock/amber/path'})
def test_tleap_it_writes_file(tmp_path: Path):
    pdb_path = tmp_path / 'test.pdb'
    pdb_path.write_text(
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
    )
    imp = ImplicitSolvent(path=tmp_path, pdb=str(pdb_path))
    imp.debug = True
    with patch('subprocess.run'):
        imp.tleap_it()
    leap_path = tmp_path / 'tleap.in'
    assert leap_path.exists()
    content = leap_path.read_text()
    assert 'leaprc.protein.ff19SB' in content


@patch.dict('os.environ', {'AMBERHOME': '/mock/amber/path'})
def test_tleap_it_creates_leap_in_path(tmp_path: Path):
    pdb_path = tmp_path / 'test.pdb'
    pdb_path.write_text(
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
    )
    imp = ImplicitSolvent(path=tmp_path, pdb=str(pdb_path))
    imp.debug = True
    with patch('subprocess.run'):
        imp.tleap_it()
    leap_path = tmp_path / 'tleap.in'
    assert leap_path.parent == tmp_path
    assert leap_path.suffix == '.in'
