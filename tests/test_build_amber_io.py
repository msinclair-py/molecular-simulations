"""I/O-focused tests for ImplicitSolvent.tleap_it.

These run the REAL ``tleap_it`` (no mocked subprocess). The ``fake_amberhome``
fixture provides a stub ``tleap`` executable so the build code's real
``subprocess.run`` succeeds and its real ``tleap.in`` is written to disk, which
the tests then read back.
"""

from pathlib import Path

from molecular_simulations.build import ImplicitSolvent


def test_tleap_it_writes_file(tmp_path: Path, fake_amberhome):
    pdb_path = tmp_path / 'test.pdb'
    pdb_path.write_text(
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
    )
    imp = ImplicitSolvent(path=tmp_path, pdb=str(pdb_path), debug=True)
    imp.tleap_it()

    leap_path = tmp_path / 'tleap.in'
    assert leap_path.exists()
    content = leap_path.read_text()
    assert 'leaprc.protein.ff19SB' in content


def test_tleap_it_creates_leap_in_path(tmp_path: Path, fake_amberhome):
    pdb_path = tmp_path / 'test.pdb'
    pdb_path.write_text(
        'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
    )
    imp = ImplicitSolvent(path=tmp_path, pdb=str(pdb_path), debug=True)
    imp.tleap_it()

    leap_path = tmp_path / 'tleap.in'
    assert leap_path.parent == tmp_path
    assert leap_path.suffix == '.in'
