"""Tests for build/build_amber.py.

These tests run against the REAL build code with NO mocking of ``subprocess``.

They split into two groups:

* No-binary tests run in any environment (including CI without AmberTools).
  They cover the pure logic (force-field selection, box sizing, ion counts,
  directory cleanup) and the REAL input-file/command generation. For the latter
  the ``fake_amberhome`` fixture supplies stub ``tleap``/``cpptraj`` executables
  so the build code's real ``subprocess.run`` calls succeed and write their real
  input files, which the tests read back and assert on.

* Binary tests are gated behind ``skip_without_amber`` and actually run tleap to
  produce a topology/coordinate pair. They run where AmberTools is installed and
  skip cleanly otherwise.
"""

import os
import shutil
from pathlib import Path

import pytest

from molecular_simulations.build.build_amber import (
    ConstantPHSolvent,
    ExplicitSolvent,
    ImplicitSolvent,
)

PDB_TEXT = 'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'

# A few titratable residues with distinct resids for renaming tests. Columns are
# real PDB columns so both the byte-slice rewrite and MDAnalysis can read them.
TITRATABLE_PDB_TEXT = (
    'ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00  0.00           N  \n'
    'ATOM      2  CA  ASP A   1       1.000   0.000   0.000  1.00  0.00           C  \n'
    'ATOM      3  N   GLU A   2       2.000   0.000   0.000  1.00  0.00           N  \n'
    'ATOM      4  CA  GLU A   2       3.000   0.000   0.000  1.00  0.00           C  \n'
    'ATOM      5  N   HIS A   3       4.000   0.000   0.000  1.00  0.00           N  \n'
    'ATOM      6  CA  HIS A   3       5.000   0.000   0.000  1.00  0.00           C  \n'
    'ATOM      7  N   LYS A   4       6.000   0.000   0.000  1.00  0.00           N  \n'
    'ATOM      8  CA  LYS A   4       7.000   0.000   0.000  1.00  0.00           C  \n'
)


def _resnames(pdb_path: Path) -> list[str]:
    """Return the residue name of every ATOM line, in file order."""
    return [
        line[17:20].strip()
        for line in Path(pdb_path).read_text().splitlines()
        if line.startswith(('ATOM', 'HETATM'))
    ]


def _write_pdb(directory: Path, text: str = PDB_TEXT) -> Path:
    pdb = directory / 'test.pdb'
    pdb.write_text(text)
    return pdb


def _real_amberhome() -> str | None:
    """Resolve a real AMBERHOME for skip-gated binary tests."""
    home = os.environ.get('AMBERHOME')
    if home and (Path(home) / 'bin' / 'tleap').exists():
        return home
    tleap = shutil.which('tleap')
    if tleap is not None:
        return str(Path(tleap).resolve().parent.parent)
    return None


# ---------------------------------------------------------------------------
# ImplicitSolvent: construction + logic (no binary)
# ---------------------------------------------------------------------------


class TestImplicitSolvent:
    """Construction and force-field logic for ImplicitSolvent."""

    def test_init_with_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(path=tmp_path, pdb=str(pdb), protein=True)

        assert builder.path == tmp_path.resolve()
        assert 'leaprc.protein.ff19SB' in builder.ffs
        assert builder.tleap == str(tmp_path / 'bin' / 'tleap')

    def test_init_none_path_uses_pdb_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(path=None, pdb=str(pdb), protein=True)

        assert builder.path == tmp_path.resolve()

    def test_no_amberhome_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv('AMBERHOME', raising=False)
        pdb = _write_pdb(tmp_path)
        with pytest.raises(ValueError, match='AMBERHOME is not set'):
            ImplicitSolvent(path=tmp_path, pdb=str(pdb), amberhome=None)

    def test_custom_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(path=tmp_path, pdb=str(pdb), out='custom_output.pdb')

        assert builder.out.name == 'custom_output.pdb'

    def test_forcefield_selection(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(
            path=tmp_path, pdb=str(pdb), protein=True, rna=True, dna=True
        )

        assert 'leaprc.protein.ff19SB' in builder.ffs
        assert 'leaprc.RNA.Shaw' in builder.ffs
        assert 'leaprc.DNA.OL21' in builder.ffs

    def test_kwargs_set_as_attributes(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(
            path=tmp_path, pdb=str(pdb), custom_param='custom_value', another_param=42
        )

        assert builder.custom_param == 'custom_value'
        assert builder.another_param == 42


# ---------------------------------------------------------------------------
# ImplicitSolvent: REAL tleap input generation (stub binary, no AmberTools)
# ---------------------------------------------------------------------------


class TestImplicitTleapInput:
    """The real tleap input file generated by tleap_it()."""

    def test_tleap_it_writes_real_input(self, tmp_path, fake_amberhome):
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(path=tmp_path, pdb=str(pdb), debug=True)

        # Runs the real tleap_it -> debug_tleap -> subprocess against the stub.
        builder.tleap_it()

        leap_file = tmp_path / 'tleap.in'
        assert leap_file.exists()
        content = leap_file.read_text()
        assert 'source leaprc.protein.ff19SB' in content
        assert f'loadpdb {builder.pdb}' in content
        assert 'set default pbradii mbondi3' in content
        assert f'savepdb prot {builder.out}' in content
        assert 'saveamberparm prot' in content
        assert 'quit' in content

        # The stub tleap was really invoked on that file (it copied it back).
        captured = (fake_amberhome / 'tleap_input.txt').read_text()
        assert captured == content

    def test_temp_tleap_runs_real_subprocess(self, tmp_path, fake_amberhome):
        """temp_tleap writes a temp input file and really invokes tleap on it."""
        pdb = _write_pdb(tmp_path)
        builder = ImplicitSolvent(path=tmp_path, pdb=str(pdb))

        builder.temp_tleap('source leaprc.protein.ff19SB\nquit\n')

        # The stub captured the exact input temp_tleap handed to tleap -f.
        captured = (fake_amberhome / 'tleap_input.txt').read_text()
        assert 'source leaprc.protein.ff19SB' in captured
        assert 'quit' in captured


# ---------------------------------------------------------------------------
# ExplicitSolvent: construction + logic (no binary)
# ---------------------------------------------------------------------------


class TestExplicitSolvent:
    """Construction and pure logic for ExplicitSolvent."""

    def test_init_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), padding=15.0)

        assert builder.pad == 15.0
        assert 'leaprc.water.opc' in builder.ffs
        assert builder.water_box == 'OPCBOX'

    def test_polarizable(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), polarizable=True)

        assert 'leaprc.protein.ff15ipq' in builder.ffs
        assert 'leaprc.water.spceb' in builder.ffs
        assert builder.water_box == 'SPCBOX'

    def test_disulfides(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(
            path=tmp_path, pdb=str(pdb), disulfide_residues=[10, 20]
        )

        assert 'protein.10 = CYX' in builder.disulfides
        assert 'protein.20 = CYX' in builder.disulfides

    def test_get_ion_numbers(self):
        num_ions = ExplicitSolvent.get_ion_numbers(1_000_000)
        assert isinstance(num_ions, int)
        assert num_ions > 0

    def test_get_ion_numbers_scales_with_volume(self):
        small = ExplicitSolvent.get_ion_numbers(125_000)
        large = ExplicitSolvent.get_ion_numbers(1_000_000)
        assert large > small

    def test_get_pdb_extent(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb_text = (
            'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            'ATOM      2  CA  ALA A   1      10.000   5.000   3.000  1.00  0.00\n'
            'ATOM      3  C   ALA A   1       5.000  15.000   8.000  1.00  0.00\n'
        )
        pdb = _write_pdb(tmp_path, pdb_text)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), padding=10.0)
        builder.pdb = str(pdb)

        # max extent is Y (15) + 2*10 padding = 35
        assert builder.get_pdb_extent() == 35

    def test_clean_up_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        (tmp_path / 'leap.log').write_text('log content')
        (tmp_path / 'protein.pdb').write_text('pdb content')
        (tmp_path / 'system.prmtop').write_text('topology')
        (tmp_path / 'system.inpcrd').write_text('coordinates')

        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb))
        builder.clean_up_directory()

        assert (tmp_path / 'build').exists()
        assert (tmp_path / 'system.prmtop').exists()
        assert (tmp_path / 'system.inpcrd').exists()
        # Intermediate files were moved into build/.
        assert (tmp_path / 'build' / 'leap.log').exists()
        assert (tmp_path / 'build' / 'protein.pdb').exists()


# ---------------------------------------------------------------------------
# ExplicitSolvent: REAL input generation (stub binary, no AmberTools)
# ---------------------------------------------------------------------------


class TestExplicitInputGeneration:
    """Real tleap/cpptraj inputs generated by ExplicitSolvent."""

    def test_assemble_system_writes_real_input(self, tmp_path, fake_amberhome):
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), protein=True, debug=True)

        builder.assemble_system(dim=60.0, num_ions=15)

        content = (tmp_path / 'tleap.in').read_text()
        assert 'source leaprc.protein.ff19SB' in content
        assert 'source leaprc.water.opc' in content
        assert f'PROT = loadpdb {builder.pdb}' in content
        assert 'set PROT box {60.0 60.0 60.0}' in content
        assert 'solvatebox PROT OPCBOX {0 0 0}' in content
        assert 'addIonsRand PROT Na+ 15 Cl- 15' in content
        assert 'saveamberparm PROT' in content

    def test_prep_pdb_builds_real_cpptraj_input(self, tmp_path, fake_amberhome):
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), protein=True)

        builder.prep_pdb()

        # The stub cpptraj captured the exact stdin the build code piped to it.
        cpptraj_input = (fake_amberhome / 'cpptraj_input.txt').read_text()
        assert f'parm {pdb}' in cpptraj_input
        assert f'loadcrd {pdb} name IN' in cpptraj_input
        assert 'prepareforleap' in cpptraj_input
        assert 'existingdisulfides' in cpptraj_input
        assert 'quit' in cpptraj_input

        # prep_pdb advances the working PDB and records the ss-bond leap file.
        assert builder.pdb.endswith('protein.pdb')
        assert builder.ss_bonds_leap == tmp_path / 'ss_bonds.leap'

    def test_search_disulfides_switches_cpptraj_keyword(self, tmp_path, fake_amberhome):
        """search_disulfides=True asks cpptraj to detect disulfides by distance."""
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(
            path=tmp_path, pdb=str(pdb), protein=True, search_disulfides=True
        )

        builder.prep_pdb()

        cpptraj_input = (fake_amberhome / 'cpptraj_input.txt').read_text()
        assert 'searchdisulfides' in cpptraj_input
        assert 'existingdisulfides' not in cpptraj_input

    def test_read_disulfide_bonds_keeps_only_bond_lines(self, tmp_path, monkeypatch):
        """The loadpdb line cpptraj writes is dropped; only bond commands survive."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), protein=True)
        builder.ss_bonds_leap = tmp_path / 'ss_bonds.leap'
        builder.ss_bonds_leap.write_text(
            'PROT = loadpdb protein.pdb\n'
            'bond PROT.6.SG PROT.127.SG\n'
            'bond PROT.30.SG PROT.115.SG\n'
        )

        bonds = builder._read_disulfide_bonds()

        assert bonds == 'bond PROT.6.SG PROT.127.SG\nbond PROT.30.SG PROT.115.SG'
        assert 'loadpdb' not in bonds

    def test_read_disulfide_bonds_absent_file(self, tmp_path, monkeypatch):
        """No ss_bonds.leap -> empty string, not an error."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), protein=True)
        builder.ss_bonds_leap = tmp_path / 'does_not_exist.leap'

        assert builder._read_disulfide_bonds() == ''

    def test_detected_disulfide_bonds_injected_into_tleap(
        self, tmp_path, fake_amberhome
    ):
        """Bonds captured by prep_pdb reach the tleap input (the wiring that was missing)."""
        pdb = _write_pdb(tmp_path)
        builder = ExplicitSolvent(path=tmp_path, pdb=str(pdb), protein=True, debug=True)
        builder.disulfide_bonds = 'bond PROT.2.SG PROT.4.SG'

        builder.assemble_system(dim=60.0, num_ions=15)

        content = (tmp_path / 'tleap.in').read_text()
        assert 'bond PROT.2.SG PROT.4.SG' in content


# ---------------------------------------------------------------------------
# Skip-gated: real AmberTools execution
# ---------------------------------------------------------------------------


class TestRealAmberBuild:
    """End-to-end builds that actually run tleap (require AmberTools)."""

    def test_implicit_build_produces_topology(
        self, tmp_path, alanine_dipeptide_pdb, skip_without_amber
    ):
        home = _real_amberhome()
        builder = ImplicitSolvent(
            path=tmp_path,
            pdb=str(alanine_dipeptide_pdb),
            protein=True,
            glycans=False,
            amberhome=home,
        )
        builder.build()

        assert builder.out.with_suffix('.prmtop').exists()
        assert builder.out.with_suffix('.inpcrd').exists()

    def test_explicit_build_produces_topology(
        self, tmp_path, alanine_dipeptide_pdb, skip_without_amber
    ):
        home = _real_amberhome()
        builder = ExplicitSolvent(
            path=tmp_path,
            pdb=str(alanine_dipeptide_pdb),
            protein=True,
            padding=10.0,
            amberhome=home,
        )
        builder.build()

        assert builder.out.with_suffix('.prmtop').exists()
        assert builder.out.with_suffix('.inpcrd').exists()

    @staticmethod
    def _sg_sg_bonds(prmtop) -> list[tuple[int, int]]:
        """Residue-number pairs (1-indexed) of every SG-SG bond in a prmtop."""
        import parmed as pmd

        p = pmd.load_file(str(prmtop))
        return [
            (b.atom1.residue.idx + 1, b.atom2.residue.idx + 1)
            for b in p.bonds
            if b.atom1.name == 'SG' and b.atom2.name == 'SG'
        ]

    def test_search_disulfides_forms_ss_bond(
        self, tmp_path, disulfide_peptide_pdb, skip_without_amber
    ):
        """A geometric disulfide in the input becomes a real SG-SG bond in the prmtop.

        Regression guard for the two bugs that left HEWL fully reduced: cpptraj
        must be run with ``searchdisulfides`` (distance detection), and the bond
        commands it emits must actually reach tleap. The Cys are plain ``CYS``
        with no SSBOND records, so nothing forms unless both halves work.
        """
        import parmed as pmd

        home = _real_amberhome()
        builder = ExplicitSolvent(
            path=tmp_path,
            pdb=str(disulfide_peptide_pdb),
            padding=10.0,
            search_disulfides=True,
            amberhome=home,
        )
        builder.build()

        prmtop = builder.out.with_suffix('.prmtop')
        assert self._sg_sg_bonds(prmtop) == [(2, 4)]

        # The bonded cysteines are oxidized (CYX) and shed their thiol H.
        p = pmd.load_file(str(prmtop))
        cyx = {r.idx + 1 for r in p.residues if r.name == 'CYX'}
        assert cyx == {2, 4}
        assert not any(
            a.name == 'HG' for r in p.residues if r.name == 'CYX' for a in r.atoms
        )

    def test_disulfides_not_formed_without_opt_in(
        self, tmp_path, disulfide_peptide_pdb, skip_without_amber
    ):
        """The binder-safe default never forms distance-based disulfides silently."""
        home = _real_amberhome()
        builder = ExplicitSolvent(
            path=tmp_path,
            pdb=str(disulfide_peptide_pdb),
            padding=10.0,
            amberhome=home,
        )
        builder.build()

        assert self._sg_sg_bonds(builder.out.with_suffix('.prmtop')) == []


class TestConstantPHSolvent:
    """Titratable-residue protonation logic for ConstantPHSolvent (no binary)."""

    def _builder(self, tmp_path, monkeypatch, text, **kwargs):
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        pdb = tmp_path / 'prepared.pdb'
        pdb.write_text(text)
        builder = ConstantPHSolvent(path=tmp_path, pdb=str(pdb), **kwargs)
        # protonate_titratable operates on the prepared (H-free) structure.
        builder.pdb = str(pdb)
        return builder

    def test_sets_mbondi3_radii(self, tmp_path, monkeypatch):
        builder = self._builder(tmp_path, monkeypatch, PDB_TEXT)
        assert builder.pbradii == 'mbondi3'

    def test_defaults_to_search_disulfides(self, tmp_path, monkeypatch):
        """Native-state titration should keep the fold's disulfides by default."""
        builder = self._builder(tmp_path, monkeypatch, PDB_TEXT)
        assert builder.search_disulfides is True

    def test_search_disulfides_can_be_overridden(self, tmp_path, monkeypatch):
        """The caller can still opt out (e.g. an intentionally reduced construct)."""
        builder = self._builder(
            tmp_path, monkeypatch, PDB_TEXT, search_disulfides=False
        )
        assert builder.search_disulfides is False

    def test_protonate_renames_all_titratable_by_default(self, tmp_path, monkeypatch):
        builder = self._builder(tmp_path, monkeypatch, TITRATABLE_PDB_TEXT)

        builder.protonate_titratable()

        # ASP->ASH, GLU->GLH, HIS->HIP; LYS already protonated, left as-is.
        assert _resnames(builder.pdb) == [
            'ASH',
            'ASH',
            'GLH',
            'GLH',
            'HIP',
            'HIP',
            'LYS',
            'LYS',
        ]
        assert builder.pdb.endswith('protein_protonated.pdb')
        assert set(builder.protonated_residues) == {
            'ASP1->ASH',
            'GLU2->GLH',
            'HIS3->HIP',
        }

    def test_protonate_preserves_other_columns(self, tmp_path, monkeypatch):
        builder = self._builder(tmp_path, monkeypatch, TITRATABLE_PDB_TEXT)

        builder.protonate_titratable()

        # Only the 3-char residue-name field changes; coordinates/serials intact.
        for original, rewritten in zip(
            TITRATABLE_PDB_TEXT.splitlines(),
            Path(builder.pdb).read_text().splitlines(),
            strict=True,
        ):
            assert original[:17] == rewritten[:17]  # record, serial, atom name
            assert original[20:] == rewritten[20:]  # chain, resid, coords, element

    def test_protonate_respects_selection(self, tmp_path, monkeypatch):
        builder = self._builder(
            tmp_path, monkeypatch, TITRATABLE_PDB_TEXT, titratable_sel='resid 2'
        )

        builder.protonate_titratable()

        # Only the selected GLU (resid 2) is protonated; ASP/HIS keep their names.
        assert _resnames(builder.pdb) == [
            'ASP',
            'ASP',
            'GLH',
            'GLH',
            'HIS',
            'HIS',
            'LYS',
            'LYS',
        ]
        assert builder.protonated_residues == ['GLU2->GLH']

    def test_assemble_system_writes_mbondi3(self, tmp_path, fake_amberhome):
        pdb = _write_pdb(tmp_path)
        builder = ConstantPHSolvent(path=tmp_path, pdb=str(pdb), debug=True)

        builder.assemble_system(dim=60.0, num_ions=15)

        content = (tmp_path / 'tleap.in').read_text()
        assert 'set default pbradii mbondi3' in content


class TestConstantPHSolventRealBuild:
    """Skip-gated: ConstantPHSolvent builds a real protonated titratable system."""

    def test_build_produces_protonated_asp(
        self, tmp_path, two_chain_pdb, skip_without_amber
    ):
        home = _real_amberhome()
        builder = ConstantPHSolvent(
            path=tmp_path, pdb=str(two_chain_pdb), padding=10.0, amberhome=home
        )
        builder.build()

        prmtop = builder.out.with_suffix('.prmtop')
        assert prmtop.exists()
        assert builder.out.with_suffix('.inpcrd').exists()
        assert any(
            tag.startswith('ASP') and tag.endswith('ASH')
            for tag in builder.protonated_residues
        )

        # The built Asp (now ASH) must carry its labile HD2 as a real particle.
        import parmed as pmd

        parm = pmd.load_file(str(prmtop))
        ash = [r for r in parm.residues if r.name == 'ASH']
        assert ash, 'expected at least one ASH residue in the built system'
        assert all(any(a.name == 'HD2' for a in r.atoms) for r in ash)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
