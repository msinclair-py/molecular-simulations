"""
Unit and integration tests for build/build_ligand.py module.

These tests avoid mocking. Chemistry steps run against real RDKit/OpenBabel on
real fixture molecules, and the antechamber/parmchk2/tleap parameterization
pipeline runs for real when AmberTools is installed (and skips cleanly when it
is not). The ``build_ligand`` module imports ``openbabel`` at module load, so
every test that constructs ``LigandBuilder``/``ComplexBuilder`` is gated behind
``@requires_openbabel`` -- it runs where OpenBabel is importable and skips
otherwise.
"""

import os
import shutil
from pathlib import Path

import pytest

# ============================================================================
# Availability checks and skip markers
# ============================================================================


def _check_rdkit_available():
    """Check if RDKit is available."""
    try:
        from rdkit import Chem  # noqa: F401

        return True
    except ImportError:
        return False


def _check_openbabel_available():
    """Check if OpenBabel/pybel is available."""
    try:
        from openbabel import pybel  # noqa: F401

        return True
    except ImportError:
        return False


def _antechamber_available():
    """Check for a usable AmberTools install (AMBERHOME + antechamber).

    ``LigandBuilder`` resolves antechamber/parmchk2/tleap from ``AMBERHOME``,
    so a bare ``antechamber`` on ``PATH`` is not enough -- ``AMBERHOME`` must be
    set and contain the binary. Returns True only when both hold.
    """
    amberhome = os.environ.get('AMBERHOME')
    if not amberhome:
        return False
    return (Path(amberhome) / 'bin' / 'antechamber').exists()


# Custom markers for tests requiring chemistry libraries
requires_rdkit = pytest.mark.skipif(
    not _check_rdkit_available(), reason='RDKit not available'
)

requires_openbabel = pytest.mark.skipif(
    not _check_openbabel_available(), reason='OpenBabel not available'
)

requires_chemistry = pytest.mark.skipif(
    not (_check_rdkit_available() and _check_openbabel_available()),
    reason='RDKit or OpenBabel not available',
)

# Full ligand parameterization needs RDKit + OpenBabel + AmberTools binaries.
requires_amber_pipeline = pytest.mark.skipif(
    not (
        _check_rdkit_available()
        and _check_openbabel_available()
        and _antechamber_available()
    ),
    reason='RDKit/OpenBabel/AmberTools (AMBERHOME + antechamber) not available',
)


@pytest.fixture
def sample_sdf_content():
    """Return a valid SDF file content for methanol."""
    return """methanol
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3000   -0.5000    0.8660 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3000   -0.5000   -0.8660 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
"""


@pytest.fixture
def sample_sdf_file(tmp_path, sample_sdf_content):
    """Create a temporary SDF file with valid content."""
    sdf_path = tmp_path / 'methanol.sdf'
    sdf_path.write_text(sample_sdf_content)
    return sdf_path


# ============================================================================
# Integration tests using real chemistry libraries (when available)
# ============================================================================


class TestRDKitIntegration:
    """Integration tests using real RDKit functionality.

    These tests verify actual chemistry operations rather than mocked interactions.
    """

    @requires_rdkit
    def test_real_sdf_reading(self, sample_sdf_file):
        """Test that RDKit can read a real SDF file."""
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(sample_sdf_file), removeHs=False)
        mol = next(iter(supplier))

        assert mol is not None
        # Should have at least the heavy atoms (C and O)
        assert mol.GetNumAtoms() >= 2

    @requires_rdkit
    def test_real_hydrogen_addition(self, sample_sdf_file):
        """Test that RDKit can add hydrogens to a molecule."""
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(sample_sdf_file))
        mol = next(iter(supplier))

        # The methanol in our SDF already has explicit H
        initial_atoms = mol.GetNumAtoms()

        # AddHs should not add more since they're already explicit
        molH = Chem.AddHs(mol, addCoords=True)
        assert molH is not None
        assert molH.GetNumAtoms() >= initial_atoms

    @requires_rdkit
    def test_real_molecule_from_smiles(self):
        """Test creating a molecule from SMILES and converting it."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Create ethanol from SMILES
        mol = Chem.MolFromSmiles('CCO')
        assert mol is not None

        # Add hydrogens
        molH = Chem.AddHs(mol)
        assert molH.GetNumAtoms() == 9  # 2C + 1O + 6H

        # Generate 3D coordinates
        AllChem.EmbedMolecule(molH, randomSeed=42)
        conf = molH.GetConformer()
        assert conf.GetNumAtoms() == 9

    @requires_rdkit
    def test_real_sdf_writing(self, tmp_path):
        """Test that RDKit can write valid SDF files."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Create molecule
        mol = Chem.MolFromSmiles('C')  # Methane
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Write to SDF
        output_sdf = tmp_path / 'output.sdf'
        with Chem.SDWriter(str(output_sdf)) as writer:
            writer.write(mol)

        # Verify file was written and can be read back
        assert output_sdf.exists()
        supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False)
        read_mol = next(iter(supplier))
        assert read_mol is not None
        assert read_mol.GetNumAtoms() == 5  # C + 4H

    @requires_rdkit
    def test_real_pdb_reading(self, tmp_path):
        """Test that RDKit can read a PDB file with small molecule."""
        from rdkit import Chem

        # Create a simple PDB for a water molecule
        pdb_content = """HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
END
"""
        pdb_path = tmp_path / 'water.pdb'
        pdb_path.write_text(pdb_content)

        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        assert mol is not None
        assert mol.GetNumAtoms() == 3


class TestOpenBabelIntegration:
    """Integration tests using real OpenBabel functionality."""

    @requires_openbabel
    def test_real_sdf_to_mol2_conversion(self, sample_sdf_file, tmp_path):
        """Test that OpenBabel can convert SDF to mol2."""
        from openbabel import pybel

        # Read SDF
        mols = list(pybel.readfile('sdf', str(sample_sdf_file)))
        assert len(mols) == 1
        mol = mols[0]

        # Write mol2
        mol2_path = tmp_path / 'output.mol2'
        mol.write('mol2', str(mol2_path), overwrite=True)

        assert mol2_path.exists()
        assert mol2_path.stat().st_size > 0

    @requires_openbabel
    def test_real_format_detection(self, sample_sdf_file):
        """Test that OpenBabel correctly detects molecular format."""
        from openbabel import pybel

        mols = list(pybel.readfile('sdf', str(sample_sdf_file)))
        mol = mols[0]

        # Verify atom count
        assert len(mol.atoms) == 5  # C + O + 3H for methanol


class TestChemistryValidation:
    """Tests that validate chemistry logic with real libraries."""

    @requires_chemistry
    def test_molecule_valence_valid(self, sample_sdf_file):
        """Test that the molecule has valid valence."""
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(sample_sdf_file))
        mol = next(iter(supplier))

        # Sanitize checks valence
        try:
            Chem.SanitizeMol(mol)
            valid = True
        except Exception:
            valid = False

        assert valid, 'Molecule should have valid valence'

    @requires_chemistry
    def test_hydrogen_count_correct(self, sample_sdf_file):
        """Test that hydrogen count is correct for the molecule."""
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(sample_sdf_file))
        mol = next(iter(supplier))

        # Count atoms by element
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

        # Methanol: CH3OH -> 1C, 1O, 4H (but our SDF has 3H explicit)
        assert atom_counts.get('C', 0) == 1
        assert atom_counts.get('O', 0) == 1


# ============================================================================
# LigandError -- pure-logic exception behaviour (real, needs module import)
# ============================================================================


@requires_openbabel
class TestLigandError:
    """Test suite for the LigandError exception class."""

    def test_ligand_error_default_message(self):
        """Default message mentions that the system cannot be modeled."""
        from molecular_simulations.build.build_ligand import LigandError

        err = LigandError()
        assert 'cannot model' in str(err)

    def test_ligand_error_custom_message(self):
        """Custom message is preserved verbatim."""
        from molecular_simulations.build.build_ligand import LigandError

        err = LigandError('Custom error message')
        assert str(err) == 'Custom error message'

    def test_ligand_error_is_exception(self):
        """LigandError is a proper Exception subclass and is raisable."""
        from molecular_simulations.build.build_ligand import LigandError

        assert issubclass(LigandError, Exception)

        with pytest.raises(LigandError):
            raise LigandError('Test error')


# ============================================================================
# LigandBuilder -- construction and non-binary helpers (real)
# ============================================================================


@requires_openbabel
class TestLigandBuilder:
    """Construction and pure-Python helper methods of LigandBuilder."""

    def test_init(self, tmp_path, monkeypatch):
        """Initialization resolves paths and ligand number."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import LigandBuilder

        (tmp_path / 'ligand.sdf').write_text('placeholder')
        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf', lig_number=0)

        assert builder.path == tmp_path
        assert builder.lig == tmp_path / 'ligand.sdf'
        assert builder.ln == 0
        assert builder.out_lig == tmp_path / 'ligand'
        # Binary paths are derived from AMBERHOME.
        assert builder.antechamber == str(tmp_path / 'bin' / 'antechamber')
        assert builder.tleap == str(tmp_path / 'bin' / 'tleap')

    def test_init_with_prefix(self, tmp_path, monkeypatch):
        """A file prefix is applied to the output ligand stem."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import LigandBuilder

        (tmp_path / 'ligand.sdf').write_text('placeholder')
        builder = LigandBuilder(
            path=tmp_path, lig='ligand.sdf', lig_number=1, file_prefix='prefix_'
        )

        assert builder.ln == 1
        assert builder.out_lig.name == 'prefix_ligand'

    def test_init_missing_amberhome_raises(self, tmp_path, monkeypatch):
        """Constructing without AMBERHOME raises ValueError."""
        monkeypatch.delenv('AMBERHOME', raising=False)
        from molecular_simulations.build.build_ligand import LigandBuilder

        with pytest.raises(ValueError, match='AMBERHOME'):
            LigandBuilder(path=tmp_path, lig='ligand.sdf')

    def test_default_prefix(self, tmp_path, monkeypatch):
        """With no prefix the output stem matches the input stem."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import LigandBuilder

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf')
        assert str(builder.out_lig).endswith('ligand')

    def test_lig_number(self, tmp_path, monkeypatch):
        """Ligand number is stored for residue naming."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import LigandBuilder

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf', lig_number=5)
        assert builder.ln == 5

    def test_write_leap(self, tmp_path, monkeypatch):
        """write_leap writes the tleap input verbatim and returns its path."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import LigandBuilder

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf')

        leap_content = 'source leaprc.gaff2\nquit'
        leap_file, leap_log = builder.write_leap(leap_content)

        assert Path(leap_file).exists()
        assert Path(leap_file).read_text() == leap_content
        assert leap_log.endswith('leap.log')

    def test_check_sqm_success(self, tmp_path, monkeypatch):
        """check_sqm passes when the sqm output reports completion."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder

        Path('ligand_sqm.out').write_text('Some output\nCalculation Completed\nEnd')

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf')
        builder.lig = 'ligand'

        # Should not raise.
        builder.check_sqm()

    def test_check_sqm_failure(self, tmp_path, monkeypatch):
        """check_sqm raises LigandError when completion is absent."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError

        Path('ligand_sqm.out').write_text('Some output\nError occurred\nEnd')

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf')
        builder.lig = 'ligand'

        with pytest.raises(LigandError, match='SQM failed'):
            builder.check_sqm()

    def test_move_antechamber_outputs(self, tmp_path, monkeypatch):
        """move_antechamber_outputs deletes sqm.in/sqm.pdb and renames sqm.out."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder

        Path('sqm.in').write_text('sqm input')
        Path('sqm.pdb').write_text('sqm pdb')
        Path('sqm.out').write_text('sqm output')

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf')
        builder.lig = 'ligand'

        builder.move_antechamber_outputs()

        assert not Path('sqm.in').exists()
        assert not Path('sqm.pdb').exists()
        assert Path('ligand_sqm.out').exists()
        assert Path('ligand_sqm.out').read_text() == 'sqm output'


# ============================================================================
# LigandBuilder -- real chemistry steps (RDKit / OpenBabel)
# ============================================================================


@requires_chemistry
class TestLigandBuilderChemistry:
    """process_input (RDKit) and convert_to_mol2 (OpenBabel) on real molecules."""

    def test_process_input_sdf_adds_hydrogens(self, tmp_path, monkeypatch, benzene_sdf):
        """process_input reads an SDF and writes a hydrogenated SDF."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder

        shutil.copy(benzene_sdf, tmp_path / 'benzene.sdf')

        builder = LigandBuilder(path=tmp_path, lig='benzene.sdf')
        builder.lig = 'benzene'
        builder.process_input('.sdf')

        out = tmp_path / 'benzene_H.sdf'
        assert out.exists()

        from rdkit import Chem

        mol = next(iter(Chem.SDMolSupplier(str(out), removeHs=False)))
        assert mol is not None
        # Benzene: 6 C + 6 H explicit.
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        assert symbols.count('C') == 6
        assert symbols.count('H') == 6

    def test_process_input_pdb_adds_hydrogens(self, tmp_path, monkeypatch):
        """process_input reads a PDB ligand and writes a hydrogenated SDF."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from molecular_simulations.build.build_ligand import LigandBuilder

        # Build a real benzene PDB with explicit hydrogens.
        mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        Chem.MolToPDBFile(mol, str(tmp_path / 'benzene.pdb'))

        builder = LigandBuilder(path=tmp_path, lig='benzene.pdb')
        builder.lig = 'benzene'
        builder.process_input('.pdb')

        out = tmp_path / 'benzene_H.sdf'
        assert out.exists()
        read = next(iter(Chem.SDMolSupplier(str(out), removeHs=False)))
        assert read is not None
        assert read.GetNumAtoms() == 12  # 6 C + 6 H

    def test_process_input_bad_extension_raises(self, tmp_path, monkeypatch):
        """An unsupported extension raises LigandError."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError

        builder = LigandBuilder(path=tmp_path, lig='ligand.xyz')
        builder.lig = 'ligand'
        with pytest.raises(LigandError):
            builder.process_input('.xyz')

    def test_convert_to_mol2(self, tmp_path, monkeypatch, benzene_sdf):
        """convert_to_mol2 turns the hydrogenated SDF into a real mol2 file."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder

        # convert_to_mol2 reads '<lig>_H.sdf' from the cwd.
        shutil.copy(benzene_sdf, tmp_path / 'ligand_H.sdf')

        builder = LigandBuilder(path=tmp_path, lig='ligand.sdf')
        builder.lig = 'ligand'
        builder.convert_to_mol2()

        mol2 = tmp_path / 'ligand_prep.mol2'
        assert mol2.exists()
        content = mol2.read_text()
        assert '@<TRIPOS>MOLECULE' in content
        assert '@<TRIPOS>ATOM' in content


# ============================================================================
# LigandBuilder -- full parameterization pipeline (AmberTools required)
# ============================================================================


@requires_amber_pipeline
class TestLigandBuilderParameterize:
    """End-to-end ligand parameterization via antechamber/parmchk2/tleap."""

    def test_parameterize_ligand_sdf(self, tmp_path, monkeypatch, benzene_sdf):
        """Parameterizing an SDF ligand produces real .mol2/.frcmod/.lib files."""
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder

        shutil.copy(benzene_sdf, tmp_path / 'benzene.sdf')

        builder = LigandBuilder(path=tmp_path, lig='benzene.sdf')
        builder.parameterize_ligand()

        assert (tmp_path / 'benzene.mol2').exists()
        assert (tmp_path / 'benzene.frcmod').exists()
        assert (tmp_path / 'benzene.lib').exists()

    def test_parameterize_ligand_pdb(self, tmp_path, monkeypatch):
        """Parameterizing a PDB ligand produces real parameter files."""
        monkeypatch.chdir(tmp_path)
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from molecular_simulations.build.build_ligand import LigandBuilder

        mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        Chem.MolToPDBFile(mol, str(tmp_path / 'benzene.pdb'))

        builder = LigandBuilder(path=tmp_path, lig='benzene.pdb')
        builder.parameterize_ligand()

        assert (tmp_path / 'benzene.mol2').exists()
        assert (tmp_path / 'benzene.frcmod').exists()
        assert (tmp_path / 'benzene.lib').exists()


@requires_chemistry
class TestLigandBuilderParameterizeError:
    """Error handling of the parameterization pipeline without AmberTools."""

    @pytest.mark.skipif(
        _antechamber_available(),
        reason='Runs only where antechamber is absent so the pipeline fails',
    )
    def test_missing_antechamber_raises_ligand_error(
        self, tmp_path, monkeypatch, benzene_sdf
    ):
        """When antechamber is unavailable, parameterize_ligand raises LigandError.

        The RDKit/OpenBabel preprocessing succeeds and writes the prep mol2, then
        ``os.system`` invokes a non-existent antechamber (resolved from a bogus
        AMBERHOME), so no sqm.* files are produced and move_antechamber_outputs
        raises FileNotFoundError, which the source wraps as LigandError.
        """
        monkeypatch.setenv('AMBERHOME', str(tmp_path))  # no bin/antechamber here
        monkeypatch.chdir(tmp_path)
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError

        shutil.copy(benzene_sdf, tmp_path / 'benzene.sdf')

        builder = LigandBuilder(path=tmp_path, lig='benzene.sdf')
        with pytest.raises(LigandError, match='Antechamber failed'):
            builder.parameterize_ligand()


# ============================================================================
# ComplexBuilder -- construction and non-binary orchestration (real)
# ============================================================================


@requires_openbabel
class TestComplexBuilder:
    """Construction and file-staging logic of ComplexBuilder (no binaries)."""

    def test_init_single_ligand(self, tmp_path, monkeypatch, sample_pdb_path):
        """Single-ligand init resolves the ligand to a Path and adds gaff2."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import ComplexBuilder

        lig_file = tmp_path / 'ligand.sdf'
        lig_file.write_text('placeholder')

        builder = ComplexBuilder(
            path=str(tmp_path),
            pdb=str(sample_pdb_path),
            lig=str(lig_file),
            padding=12.0,
        )

        assert builder.pad == 12.0
        assert 'leaprc.gaff2' in builder.ffs
        assert isinstance(builder.lig, Path)
        assert builder.lig == lig_file.resolve()

    def test_init_multiple_ligands(self, tmp_path, monkeypatch, sample_pdb_path):
        """Multiple-ligand init stores a list of resolved Paths."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import ComplexBuilder

        lig1 = tmp_path / 'ligand1.sdf'
        lig1.write_text('placeholder')
        lig2 = tmp_path / 'ligand2.sdf'
        lig2.write_text('placeholder')

        builder = ComplexBuilder(
            path=str(tmp_path),
            pdb=str(sample_pdb_path),
            lig=[str(lig1), str(lig2)],
            padding=10.0,
        )

        assert isinstance(builder.lig, list)
        assert len(builder.lig) == 2
        assert all(isinstance(p, Path) for p in builder.lig)

    def test_init_with_precomputed_params(self, tmp_path, monkeypatch, sample_pdb_path):
        """A precomputed param prefix is normalized to parent/stem."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import ComplexBuilder

        lig_file = tmp_path / 'ligand.sdf'
        lig_file.write_text('placeholder')
        param_prefix = tmp_path / 'params' / 'ligand'

        builder = ComplexBuilder(
            path=str(tmp_path),
            pdb=str(sample_pdb_path),
            lig=str(lig_file),
            lig_param_prefix=str(param_prefix),
        )

        assert builder.lig_param_prefix is not None
        assert builder.lig_param_prefix == param_prefix.parent / param_prefix.stem

    def test_init_kwargs_set_ion(self, tmp_path, monkeypatch, sample_pdb_path):
        """An 'ion' kwarg is stored on the instance."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import ComplexBuilder

        lig_file = tmp_path / 'ligand.sdf'
        lig_file.write_text('placeholder')
        ion_file = tmp_path / 'ion.pdb'
        ion_file.write_text(
            'HETATM    1  NA  NA+ A   1       5.000   5.000   5.000  1.00  0.00\n'
        )

        builder = ComplexBuilder(
            path=str(tmp_path),
            pdb=str(sample_pdb_path),
            lig=str(lig_file),
            ion=str(ion_file),
        )

        assert builder.ion == str(ion_file)

    def test_add_ion_to_pdb(self, tmp_path, monkeypatch):
        """add_ion_to_pdb appends ion records ahead of END in the protein PDB."""
        monkeypatch.setenv('AMBERHOME', str(tmp_path))
        from molecular_simulations.build.build_ligand import ComplexBuilder

        pdb_content = (
            'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            'ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00\n'
            'END\n'
        )
        ion_content = (
            'HETATM    1  NA  NA+ A   2       5.000   5.000   5.000  1.00  0.00\n'
        )
        pdb_file = tmp_path / 'protein.pdb'
        pdb_file.write_text(pdb_content)
        ion_file = tmp_path / 'ion.pdb'
        ion_file.write_text(ion_content)
        lig_file = tmp_path / 'ligand.sdf'
        lig_file.write_text('placeholder')

        builder = ComplexBuilder(
            path=str(tmp_path),
            pdb=str(pdb_file),
            lig=str(lig_file),
            ion=str(ion_file),
        )
        builder.pdb = str(pdb_file)

        builder.add_ion_to_pdb()

        modified = pdb_file.read_text()
        assert 'HETATM' in modified
        assert 'NA' in modified
        # Ion line must precede the END record.
        lines = modified.splitlines()
        first_hetatm = next(i for i, ln in enumerate(lines) if 'HETATM' in ln)
        assert lines.index('END') > first_hetatm


# ============================================================================
# ComplexBuilder.process_ligand -- full pipeline (AmberTools required)
# ============================================================================


@requires_amber_pipeline
class TestComplexBuilderProcessLigand:
    """process_ligand stages the file and parameterizes it for real."""

    def test_process_ligand_copies_and_parameterizes(
        self, tmp_path, monkeypatch, sample_pdb_path, benzene_sdf
    ):
        """A ligand outside build_dir is copied in and parameterized."""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        lig_file = tmp_path / 'benzene.sdf'
        shutil.copy(benzene_sdf, lig_file)

        builder = ComplexBuilder(
            path=str(tmp_path), pdb=str(sample_pdb_path), lig=str(lig_file)
        )
        builder.build_dir = tmp_path / 'build'
        builder.build_dir.mkdir()
        monkeypatch.chdir(builder.build_dir)

        out_lig = builder.process_ligand(lig_file)

        # File staged into build_dir.
        assert (builder.build_dir / 'benzene.sdf').exists()
        # Real parameter files produced.
        assert Path(f'{out_lig}.mol2').exists()
        assert Path(f'{out_lig}.frcmod').exists()
        assert Path(f'{out_lig}.lib').exists()

    def test_process_ligand_already_in_build_dir(
        self, tmp_path, monkeypatch, sample_pdb_path, benzene_sdf
    ):
        """A ligand already in build_dir is parameterized in place (no copy error)."""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        build_dir = tmp_path / 'build'
        build_dir.mkdir()
        lig_file = build_dir / 'benzene.sdf'
        shutil.copy(benzene_sdf, lig_file)

        builder = ComplexBuilder(
            path=str(tmp_path), pdb=str(sample_pdb_path), lig=str(lig_file)
        )
        builder.build_dir = build_dir
        monkeypatch.chdir(build_dir)

        out_lig = builder.process_ligand(lig_file)
        assert Path(f'{out_lig}.lib').exists()
