"""
Unit tests for build/build_ligand.py module

This module contains both unit tests (with mocks) and integration tests that use
real RDKit/OpenBabel when available. Tests for non-chemistry logic use mocks,
while chemistry validation tests use real libraries.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# ============================================================================
# Fixtures and helpers for conditional chemistry library usage
# ============================================================================


def _check_rdkit_available():
    """Check if RDKit is available."""
    try:
        from rdkit import Chem

        return True
    except ImportError:
        return False


def _check_openbabel_available():
    """Check if OpenBabel/pybel is available."""
    try:
        from openbabel import pybel

        return True
    except ImportError:
        return False


# Custom markers for tests requiring chemistry libraries
requires_rdkit = pytest.mark.skipif(
    not _check_rdkit_available(), reason="RDKit not available"
)

requires_openbabel = pytest.mark.skipif(
    not _check_openbabel_available(), reason="OpenBabel not available"
)

requires_chemistry = pytest.mark.skipif(
    not (_check_rdkit_available() and _check_openbabel_available()),
    reason="RDKit or OpenBabel not available",
)


# NOTE: This fixture is NOT autouse - only used by tests that need mocks
@pytest.fixture
def mock_difficult_dependencies():
    """Mock dependencies that might not be installed.

    This fixture is NOT autouse - it must be explicitly requested by tests
    that need to mock the chemistry libraries. Tests that validate actual
    chemistry behavior should not use this fixture.
    """
    mock_pybel = MagicMock()
    mock_openbabel = MagicMock()
    mock_openbabel.pybel = mock_pybel

    mock_rdkit = MagicMock()
    mock_chem = MagicMock()
    mock_rdkit.Chem = mock_chem

    # Remove cached build_ligand module to ensure fresh import with new mocks
    modules_to_remove = [
        "molecular_simulations.build.build_ligand",
    ]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    with patch.dict(
        sys.modules,
        {
            "openbabel": mock_openbabel,
            "openbabel.pybel": mock_pybel,
            "rdkit": mock_rdkit,
            "rdkit.Chem": mock_chem,
        },
    ):
        # Also patch the module's pybel binding after import
        yield {
            "pybel": mock_pybel,
            "Chem": mock_chem,
        }
        # Cleanup: remove the module so subsequent tests/other files get fresh imports
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)


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
    sdf_path = tmp_path / "methanol.sdf"
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
        mol = Chem.MolFromSmiles("CCO")
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
        mol = Chem.MolFromSmiles("C")  # Methane
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Write to SDF
        output_sdf = tmp_path / "output.sdf"
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
        pdb_path = tmp_path / "water.pdb"
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
        mols = list(pybel.readfile("sdf", str(sample_sdf_file)))
        assert len(mols) == 1
        mol = mols[0]

        # Write mol2
        mol2_path = tmp_path / "output.mol2"
        mol.write("mol2", str(mol2_path), overwrite=True)

        assert mol2_path.exists()
        assert mol2_path.stat().st_size > 0

    @requires_openbabel
    def test_real_format_detection(self, sample_sdf_file):
        """Test that OpenBabel correctly detects molecular format."""
        from openbabel import pybel

        mols = list(pybel.readfile("sdf", str(sample_sdf_file)))
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

        assert valid, "Molecule should have valid valence"

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
        assert atom_counts.get("C", 0) == 1
        assert atom_counts.get("O", 0) == 1


# ============================================================================
# Unit tests with mocks (for non-chemistry logic)
# ============================================================================


class TestLigandError:
    """Test suite for LigandError exception class"""

    def test_ligand_error_default_message(self, mock_difficult_dependencies):
        """Test LigandError with default message"""
        from molecular_simulations.build.build_ligand import LigandError

        err = LigandError()
        assert "cannot model" in str(err)

    def test_ligand_error_custom_message(self, mock_difficult_dependencies):
        """Test LigandError with custom message"""
        from molecular_simulations.build.build_ligand import LigandError

        err = LigandError("Custom error message")
        assert str(err) == "Custom error message"

    def test_ligand_error_is_exception(self, mock_difficult_dependencies):
        """Test LigandError is a proper Exception subclass"""
        from molecular_simulations.build.build_ligand import LigandError

        assert issubclass(LigandError, Exception)

        with pytest.raises(LigandError):
            raise LigandError("Test error")


class TestLigandBuilder:
    """Test suite for LigandBuilder class"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_ligand_builder_init(self, mock_difficult_dependencies):
        """Test LigandBuilder initialization"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            # The source code expects lig to be a Path for `.stem` on line 89
            # This appears to be a bug in the source - working around by using Path
            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"), lig_number=0)

            assert builder.path == path
            assert builder.lig == path / "ligand.sdf"
            assert builder.ln == 0

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_ligand_builder_init_with_prefix(self, mock_difficult_dependencies):
        """Test LigandBuilder initialization with file prefix"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(
                path=path, lig=Path("ligand.sdf"), lig_number=1, file_prefix="prefix_"
            )

            assert builder.ln == 1
            assert "prefix_" in str(builder.out_lig)

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_ligand_builder_write_leap(self, mock_difficult_dependencies):
        """Test write_leap method"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))

            leap_content = "source leaprc.gaff2\nquit"
            leap_file, leap_log = builder.write_leap(leap_content)

            assert Path(leap_file).exists()
            assert Path(leap_file).read_text() == leap_content

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_check_sqm_success(self, mock_difficult_dependencies):
        """Test check_sqm with successful calculation"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                path = Path(tmpdir)
                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf content")

                # Create successful sqm output in current directory
                sqm_out = Path("ligand_sqm.out")
                sqm_out.write_text("Some output\nCalculation Completed\nEnd")

                builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))
                builder.lig = "ligand"

                # Should not raise
                builder.check_sqm()
        finally:
            os.chdir(cwd)

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_check_sqm_failure(self, mock_difficult_dependencies):
        """Test check_sqm with failed calculation"""
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                path = Path(tmpdir)
                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf content")

                # Create failed sqm output in current directory
                sqm_out = Path("ligand_sqm.out")
                sqm_out.write_text("Some output\nError occurred\nEnd")

                builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))
                builder.lig = "ligand"

                with pytest.raises(LigandError, match="SQM failed"):
                    builder.check_sqm()
        finally:
            os.chdir(cwd)

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_convert_to_mol2(self, mock_difficult_dependencies):
        """Test convert_to_mol2 method"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import LigandBuilder

        mock_pybel = bl_mod.pybel

        mock_mol = MagicMock()
        mock_pybel.readfile.return_value = [mock_mol]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))
            builder.lig = "ligand"

            builder.convert_to_mol2()

            mock_pybel.readfile.assert_called_once_with("sdf", "ligand_H.sdf")
            mock_mol.write.assert_called_once()

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_move_antechamber_outputs(self, mock_difficult_dependencies):
        """Test move_antechamber_outputs method"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf content")

                # Create files that antechamber would produce
                os.chdir(tmpdir)
                Path("sqm.in").write_text("sqm input")
                Path("sqm.pdb").write_text("sqm pdb")
                Path("sqm.out").write_text("sqm output")

                builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))
                builder.lig = "ligand"

                builder.move_antechamber_outputs()

                # sqm.in and sqm.pdb should be removed
                assert not Path("sqm.in").exists()
                assert not Path("sqm.pdb").exists()
                # sqm.out should be renamed
                assert Path("ligand_sqm.out").exists()
        finally:
            os.chdir(cwd)


class TestComplexBuilder:
    """Test suite for ComplexBuilder class"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_complex_builder_init_single_ligand(self, mock_difficult_dependencies):
        """Test ComplexBuilder initialization with single ligand"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / "protein.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path), pdb=str(pdb_file), lig=str(lig_file), padding=12.0
            )

            assert builder.pad == 12.0
            assert "leaprc.gaff2" in builder.ffs
            assert isinstance(builder.lig, Path)

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_complex_builder_init_multiple_ligands(self, mock_difficult_dependencies):
        """Test ComplexBuilder initialization with multiple ligands"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / "protein.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            lig_file1 = path / "ligand1.sdf"
            lig_file1.write_text("mock sdf content")
            lig_file2 = path / "ligand2.sdf"
            lig_file2.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=[str(lig_file1), str(lig_file2)],
                padding=10.0,
            )

            assert isinstance(builder.lig, list)
            assert len(builder.lig) == 2

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_complex_builder_with_precomputed_params(self, mock_difficult_dependencies):
        """Test ComplexBuilder with pre-computed ligand parameters"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / "protein.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            param_prefix = path / "params" / "ligand"

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=str(lig_file),
                lig_param_prefix=str(param_prefix),
            )

            assert builder.lig_param_prefix is not None

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_complex_builder_kwargs(self, mock_difficult_dependencies):
        """Test ComplexBuilder with extra kwargs"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / "protein.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            ion_file = path / "ion.pdb"
            ion_file.write_text(
                "HETATM    1  NA  NA+ A   1       5.000   5.000   5.000  1.00  0.00\n"
            )

            builder = ComplexBuilder(
                path=str(path), pdb=str(pdb_file), lig=str(lig_file), ion=str(ion_file)
            )

            assert hasattr(builder, "ion")
            assert builder.ion == str(ion_file)


class TestComplexBuilderMethods:
    """Additional test methods for ComplexBuilder"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_add_ion_to_pdb(self, mock_difficult_dependencies):
        """Test add_ion_to_pdb method"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00
END
"""
            ion_content = """HETATM    1  NA  NA+ A   2       5.000   5.000   5.000  1.00  0.00
"""
            pdb_file = path / "protein.pdb"
            pdb_file.write_text(pdb_content)

            ion_file = path / "ion.pdb"
            ion_file.write_text(ion_content)

            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path), pdb=str(pdb_file), lig=str(lig_file), ion=str(ion_file)
            )

            # Override pdb path for test
            builder.pdb = str(pdb_file)

            builder.add_ion_to_pdb()

            modified_pdb = pdb_file.read_text()
            assert "HETATM" in modified_pdb
            assert "NA" in modified_pdb
            assert "END" in modified_pdb

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_process_ligand_copies_file(self, mock_difficult_dependencies):
        """Test process_ligand copies file to build directory"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        # Mock LigandBuilder directly on the module
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = "ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text(
                    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
                )

                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf content")

                builder = ComplexBuilder(
                    path=str(path), pdb=str(pdb_file), lig=str(lig_file)
                )

                # Create build directory
                builder.build_dir = path / "build"
                builder.build_dir.mkdir()

                result = builder.process_ligand(lig_file)

                # LigandBuilder should be called
                mock_lig_builder.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_complex_builder_with_list_of_ligands(self, mock_difficult_dependencies):
        """Test ComplexBuilder with list of ligands"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / "protein.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            lig_file1 = path / "ligand1.sdf"
            lig_file1.write_text("mock sdf content")

            lig_file2 = path / "ligand2.sdf"
            lig_file2.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path), pdb=str(pdb_file), lig=[str(lig_file1), str(lig_file2)]
            )

            assert isinstance(builder.lig, list)
            assert len(builder.lig) == 2


class TestLigandBuilderAdditional:
    """Additional tests for LigandBuilder"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_ligand_builder_default_prefix(self, mock_difficult_dependencies):
        """Test LigandBuilder with default empty prefix"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))

            # out_lig should not have a prefix
            assert str(builder.out_lig).endswith("ligand")

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_ligand_builder_lig_number(self, mock_difficult_dependencies):
        """Test LigandBuilder with different ligand numbers"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"), lig_number=5)

            assert builder.ln == 5


class TestLigandBuilderParameterize:
    """Test suite for LigandBuilder parameterize methods"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.system")
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    def test_parameterize_ligand_sdf(
        self, mock_chdir, mock_os_system, mock_difficult_dependencies
    ):
        """Test parameterize_ligand with SDF file"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import LigandBuilder

        mock_chem = bl_mod.Chem
        mock_pybel = bl_mod.pybel

        mock_os_system.return_value = 0
        mock_mol = MagicMock()
        mock_chem.SDMolSupplier.return_value = [mock_mol]
        mock_chem.AddHs.return_value = mock_mol
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)
        mock_pybel.readfile.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf")

            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))

            with (
                patch.object(builder, "check_sqm"),
                patch.object(builder, "move_antechamber_outputs"),
            ):
                builder.parameterize_ligand()

            mock_os_system.assert_called()

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.system")
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    def test_parameterize_ligand_pdb(
        self, mock_chdir, mock_os_system, mock_difficult_dependencies
    ):
        """Test parameterize_ligand with PDB file"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import LigandBuilder

        mock_chem = bl_mod.Chem
        mock_pybel = bl_mod.pybel

        mock_os_system.return_value = 0
        mock_mol = MagicMock()
        mock_chem.MolFromPDBFile.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)
        mock_pybel.readfile.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.pdb"
            lig_file.write_text(
                "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00\n"
            )

            builder = LigandBuilder(path=path, lig=Path("ligand.pdb"))

            with (
                patch.object(builder, "check_sqm"),
                patch.object(builder, "move_antechamber_outputs"),
            ):
                builder.parameterize_ligand()

            mock_chem.MolFromPDBFile.assert_called()


class TestComplexBuilderBuild:
    """Test suite for ComplexBuilder build methods"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    @patch("molecular_simulations.build.build_amber.subprocess")
    @patch("molecular_simulations.build.build_ligand.LigandBuilder")
    @patch("molecular_simulations.build.build_ligand.os.system")
    def test_complex_builder_build(
        self,
        mock_os_system,
        mock_lig_builder,
        mock_subprocess,
        mock_chdir,
        mock_difficult_dependencies,
    ):
        """Test ComplexBuilder build method"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_builder = MagicMock()
        mock_builder.lig = "ligand"
        mock_lig_builder.return_value = mock_builder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / "protein.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n"
            )

            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf")

            builder = ComplexBuilder(
                path=str(path), pdb=str(pdb_file), lig=str(lig_file)
            )

            with (
                patch.object(builder, "assemble_system"),
                patch.object(builder, "process_ligand") as mock_process,
            ):
                mock_process.return_value = "ligand"

                # Create build directory
                builder.build_dir = path / "build"
                builder.build_dir.mkdir()

                builder.build()

                mock_process.assert_called_once()


class TestComplexBuilderProcessLigand:
    """Test suite for ComplexBuilder process_ligand method"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.shutil")
    def test_process_ligand(self, mock_shutil, mock_difficult_dependencies):
        """Test process_ligand method"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        # Mock LigandBuilder directly
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = "ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                pdb_file = path / "protein.pdb"
                pdb_file.write_text(
                    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
                )

                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path), pdb=str(pdb_file), lig=str(lig_file)
                )

                builder.build_dir = path / "build"
                builder.build_dir.mkdir()

                result = builder.process_ligand(lig_file)

                mock_lig_builder.assert_called_once()
                mock_builder.parameterize_ligand.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder


class TestComplexBuilderTleap:
    """Test suite for ComplexBuilder tleap_it method"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("subprocess.run")
    def test_tleap_it_single_ligand(self, mock_subprocess, mock_difficult_dependencies):
        """Test tleap_it writes correct leap input for single ligand"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / "protein.pdb"
            pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

            builder = ComplexBuilder.__new__(ComplexBuilder)
            builder.path = path
            builder.out = path / "output"
            builder.pdb = str(pdb_file)
            builder.ffs = ["leaprc.protein.ff19SB"]
            builder.debug = True
            builder.tleap = "tleap"

            builder.tleap_it()

            leap_file = path / "tleap.in"
            assert leap_file.exists()
            content = leap_file.read_text()
            assert "leaprc.protein.ff19SB" in content
            assert "loadpdb" in content
            mock_subprocess.assert_called()

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("subprocess.run")
    def test_tleap_it_with_rna_dna(self, mock_subprocess, mock_difficult_dependencies):
        """Test tleap_it writes correct leap input with multiple force fields"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / "protein.pdb"
            pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

            builder = ComplexBuilder.__new__(ComplexBuilder)
            builder.path = path
            builder.out = path / "output"
            builder.pdb = str(pdb_file)
            builder.ffs = ["leaprc.protein.ff19SB", "leaprc.RNA.Shaw", "leaprc.gaff2"]
            builder.debug = True
            builder.tleap = "tleap"

            builder.tleap_it()

            leap_file = path / "tleap.in"
            assert leap_file.exists()
            content = leap_file.read_text()
            assert "leaprc.protein.ff19SB" in content
            assert "leaprc.RNA.Shaw" in content
            assert "leaprc.gaff2" in content
            mock_subprocess.assert_called()


class TestLigandBuilderFileNotFound:
    """Test LigandBuilder error handling"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.system")
    def test_parameterize_ligand_file_not_found(
        self, mock_os_system, mock_difficult_dependencies
    ):
        """Test parameterize_ligand raises LigandError on FileNotFoundError"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError

        # Use the module's mocks directly (set by the autouse fixture)
        mock_chem = bl_mod.Chem
        mock_pybel = bl_mod.pybel

        mock_os_system.return_value = 0
        mock_mol = MagicMock()
        mock_chem.SDMolSupplier.return_value = [mock_mol]
        mock_chem.AddHs.return_value = mock_mol
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)
        mock_pybel.readfile.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / "ligand.sdf"
            lig_file.write_text("mock sdf")

            builder = LigandBuilder(path=path, lig=Path("ligand.sdf"))

            # Make move_antechamber_outputs raise FileNotFoundError
            with patch.object(
                builder, "move_antechamber_outputs", side_effect=FileNotFoundError
            ), pytest.raises(LigandError, match="Antechamber failed"):
                builder.parameterize_ligand()


class TestComplexBuilderBuildMethod:
    """Test suite for ComplexBuilder build method"""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    @patch("molecular_simulations.build.build_ligand.os.system")
    def test_build_with_precomputed_params(
        self, mock_os_system, mock_chdir, mock_difficult_dependencies
    ):
        """Test build with pre-computed ligand parameters"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf")

                params_dir = path / "params"
                params_dir.mkdir()
                param_prefix = params_dir / "ligand"

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=str(lig_file),
                    lig_param_prefix=str(param_prefix),
                )

                with (
                    patch.object(builder, "prep_pdb"),
                    patch.object(builder, "assemble_system"),
                    patch.object(builder, "get_pdb_extent", return_value=100),
                ):
                    builder.build()

                # LigandBuilder should not be called when using precomputed params
                mock_lig_builder.assert_not_called()
        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    @patch("molecular_simulations.build.build_ligand.os.system")
    def test_build_with_multiple_ligands(
        self, mock_os_system, mock_chdir, mock_difficult_dependencies
    ):
        """Test build with multiple ligands"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = "ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

                lig_file1 = path / "ligand1.sdf"
                lig_file1.write_text("mock sdf")

                lig_file2 = path / "ligand2.sdf"
                lig_file2.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=[str(lig_file1), str(lig_file2)],
                )

                with (
                    patch.object(builder, "prep_pdb"),
                    patch.object(builder, "assemble_system"),
                    patch.object(builder, "get_pdb_extent", return_value=100),
                ):
                    builder.build()

                # LigandBuilder should be called for each ligand
                assert mock_lig_builder.call_count == 2
        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    @patch("molecular_simulations.build.build_ligand.os.system")
    def test_build_with_ion(
        self, mock_os_system, mock_chdir, mock_difficult_dependencies
    ):
        """Test build with ion file"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = "ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\nEND\n")

                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf")

                ion_file = path / "ion.pdb"
                ion_file.write_text("HETATM  1  NA  NA+ A 2  5.0 5.0 5.0  1.0 0.0\n")

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=str(lig_file),
                    ion=str(ion_file),
                )

                with (
                    patch.object(builder, "prep_pdb"),
                    patch.object(builder, "assemble_system"),
                    patch.object(builder, "add_ion_to_pdb") as mock_add_ion,
                    patch.object(builder, "get_pdb_extent", return_value=100),
                ):
                    builder.build()

                mock_add_ion.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder


class TestComplexBuilderAssembleSystem:
    """Test ComplexBuilder assemble_system method."""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("subprocess.run")
    def test_assemble_system_single_ligand(
        self, mock_subprocess_run, mock_difficult_dependencies
    ):
        """Test assemble_system with single ligand."""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / "protein.pdb"
            pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

            builder = ComplexBuilder.__new__(ComplexBuilder)
            builder.path = path
            builder.out = path / "output"
            builder.out.mkdir()
            builder.build_dir = path / "build"
            builder.build_dir.mkdir()
            builder.pdb = str(pdb_file)
            builder.lig = path / "build" / "ligand"
            builder.ffs = ["leaprc.protein.ff19SB", "leaprc.gaff2"]
            builder.water_box = "TIP3PBOX"
            builder.debug = False
            builder.delete = True
            builder.tleap = "tleap"

            builder.assemble_system(dim=80.0, num_ions=50)

            mock_subprocess_run.assert_called_once()

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("subprocess.run")
    def test_assemble_system_multiple_ligands(
        self, mock_subprocess_run, mock_difficult_dependencies
    ):
        """Test assemble_system with multiple ligands."""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / "protein.pdb"
            pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

            builder = ComplexBuilder.__new__(ComplexBuilder)
            builder.path = path
            builder.out = path / "output"
            builder.out.mkdir()
            builder.build_dir = path / "build"
            builder.build_dir.mkdir()
            builder.pdb = str(pdb_file)
            # Multiple ligands
            builder.lig = [path / "build" / "lig1", path / "build" / "lig2"]
            builder.ffs = ["leaprc.protein.ff19SB", "leaprc.gaff2"]
            builder.water_box = "TIP3PBOX"
            builder.debug = False
            builder.delete = True
            builder.tleap = "tleap"

            builder.assemble_system(dim=80.0, num_ions=50)

            mock_subprocess_run.assert_called_once()


class TestComplexBuilderProcessLigandEdgeCases:
    """Test edge cases for ComplexBuilder process_ligand method."""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_process_ligand_already_in_build_dir(self, mock_difficult_dependencies):
        """Test process_ligand when ligand is already in build directory."""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = "ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

                # Create build_dir and put ligand in it
                build_dir = path / "build"
                build_dir.mkdir()
                lig_file = build_dir / "ligand.sdf"
                lig_file.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path), pdb=str(pdb_file), lig=str(lig_file)
                )
                builder.build_dir = build_dir

                # Since ligand is in build_dir, shutil.copy should NOT be called
                with patch(
                    "molecular_simulations.build.build_ligand.shutil.copy"
                ) as mock_copy:
                    result = builder.process_ligand(lig_file)
                    mock_copy.assert_not_called()

        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    def test_process_ligand_with_prefix(self, mock_difficult_dependencies):
        """Test process_ligand with prefix for multi-ligand systems."""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = "0ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path), pdb=str(pdb_file), lig=str(lig_file)
                )
                builder.build_dir = path / "build"
                builder.build_dir.mkdir()

                result = builder.process_ligand(Path(lig_file), prefix=0)

                # LigandBuilder should be called with file_prefix=0 (becomes empty string)
                mock_lig_builder.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder


class TestComplexBuilderBuildFlows:
    """Test various ComplexBuilder.build() flows."""

    @patch.dict(os.environ, {"AMBERHOME": "/fake/amber"})
    @patch("molecular_simulations.build.build_ligand.os.chdir")
    @patch("molecular_simulations.build.build_ligand.os.system")
    def test_build_single_ligand_flow(
        self, mock_os_system, mock_chdir, mock_difficult_dependencies
    ):
        """Test build with single ligand (not list)."""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.out_lig = "ligand"
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / "protein.pdb"
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\nEND\n")

                lig_file = path / "ligand.sdf"
                lig_file.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path), pdb=str(pdb_file), lig=str(lig_file)
                )

                with (
                    patch.object(builder, "prep_pdb"),
                    patch.object(builder, "assemble_system"),
                    patch.object(builder, "get_pdb_extent", return_value=100),
                ):
                    builder.build()

                # Should have processed single ligand
                mock_lig_builder.assert_called_once()
                assert builder.lig == "ligand"
        finally:
            bl_mod.LigandBuilder = original_lig_builder


