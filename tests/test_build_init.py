"""Tests for molecular_simulations.build module __init__.py functions."""

from pathlib import Path

import pytest

# Check for optional dependencies
try:
    from Bio.PDB import PDBIO, MMCIFParser  # noqa: F401

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import gemmi  # noqa: F401

    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False


class TestConvertCifWithBiopython:
    """Tests for convert_cif_with_biopython function."""

    @pytest.mark.skipif(not HAS_BIOPYTHON, reason='Biopython not installed')
    def test_convert_cif_string_input(self, tmp_path):
        """Test that string inputs are converted to Path."""
        from molecular_simulations.build import convert_cif_with_biopython

        # Create a minimal valid CIF file
        cif_file = tmp_path / 'test.cif'
        cif_content = """data_test
#
_cell.length_a           1.000
_cell.length_b           1.000
_cell.length_c           1.000
_cell.angle_alpha        90.00
_cell.angle_beta         90.00
_cell.angle_gamma        90.00
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 0.000 0.000 0.000 1.00 0.00 A 1 ? 1
"""
        cif_file.write_text(cif_content)

        result = convert_cif_with_biopython(str(cif_file))

        # Should return a Path with .pdb extension
        assert str(result).endswith('.pdb')
        assert Path(result).exists()

    @pytest.mark.skipif(not HAS_BIOPYTHON, reason='Biopython not installed')
    def test_convert_cif_path_input(self, tmp_path):
        """Test that Path inputs work correctly."""
        from molecular_simulations.build import convert_cif_with_biopython

        cif_file = tmp_path / 'test2.cif'
        cif_content = """data_test
#
_cell.length_a           1.000
_cell.length_b           1.000
_cell.length_c           1.000
_cell.angle_alpha        90.00
_cell.angle_beta         90.00
_cell.angle_gamma        90.00
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 0.000 0.000 0.000 1.00 0.00 A 1 ? 1
"""
        cif_file.write_text(cif_content)

        result = convert_cif_with_biopython(cif_file)

        assert result == cif_file.with_suffix('.pdb')


class TestConvertCifWithGemmi:
    """Tests for convert_cif_with_gemmi function."""

    @pytest.mark.skipif(not HAS_GEMMI, reason='gemmi not installed')
    def test_convert_cif_with_gemmi(self, tmp_path):
        """Test gemmi-based CIF to PDB conversion."""
        from molecular_simulations.build import convert_cif_with_gemmi

        cif_file = tmp_path / 'test.cif'
        # Create a valid minimal CIF for gemmi
        cif_content = """data_test
_cell.length_a           1.000
_cell.length_b           1.000
_cell.length_c           1.000
_cell.angle_alpha        90.00
_cell.angle_beta         90.00
_cell.angle_gamma        90.00
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
1 N 0.000 0.000 0.000
"""
        cif_file.write_text(cif_content)

        convert_cif_with_gemmi(cif_file)

        assert cif_file.with_suffix('.pdb').exists()


class TestAddChains:
    """Tests for add_chains function."""

    def test_add_chains_default_params(self, tmp_path):
        """Test add_chains with default parameters."""
        from molecular_simulations.build import add_chains

        pdb_file = tmp_path / 'test.pdb'
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       4.458   0.000   0.000  1.00  0.00           C
END
"""
        pdb_file.write_text(pdb_content)

        add_chains(pdb_file)

        # Check output file was created
        output_file = pdb_file.parent / (pdb_file.stem + '_withchains.pdb')
        assert output_file.exists()

    def test_add_chains_with_residue_range(self, tmp_path):
        """Test add_chains with residue range specified."""
        from molecular_simulations.build import add_chains

        pdb_file = tmp_path / 'test2.pdb'
        # Create PDB with multiple residues
        lines = []
        atom_num = 1
        for resid in range(1, 11):
            lines.append(
                f'ATOM  {atom_num:5d}  N   ALA A{resid:4d}       0.000   0.000   0.000  1.00  0.00           N\n'
            )
            atom_num += 1
            lines.append(
                f'ATOM  {atom_num:5d}  CA  ALA A{resid:4d}       1.458   0.000   0.000  1.00  0.00           C\n'
            )
            atom_num += 1
        lines.append('END\n')
        pdb_file.write_text(''.join(lines))

        add_chains(pdb_file, first_res=1, last_res=5)

        output_file = pdb_file.parent / (pdb_file.stem + '_withchains.pdb')
        assert output_file.exists()

    def test_add_chains_single_chain_all_A(self, tmp_path):
        """Default (last_res == -1) assigns every residue to chain A, no chain B."""
        import MDAnalysis as mda

        from molecular_simulations.build import add_chains

        pdb_file = tmp_path / 'single.pdb'
        lines = []
        atom_num = 1
        for resid in range(1, 7):
            for name, elem in (('N', 'N'), ('CA', 'C')):
                lines.append(
                    f'ATOM  {atom_num:5d}  {name:<3s} ALA A{resid:4d}'
                    '       0.000   0.000   0.000  1.00  0.00           '
                    f'{elem}\n'
                )
                atom_num += 1
        lines.append('END\n')
        pdb_file.write_text(''.join(lines))

        output = add_chains(pdb_file)

        u = mda.Universe(str(output))
        # No chain B is created: the whole structure is a single chain A.
        assert set(u.atoms.chainIDs) == {'A'}

    def test_add_chains_split_assigns_chain_B(self, tmp_path):
        """A real last_res split creates chain B for the trailing residues.

        Chain A spans residues below the split point; residues from ``last_res``
        onward (the split boundary is inclusive on the B side) become chain B.
        """
        import MDAnalysis as mda

        from molecular_simulations.build import add_chains

        pdb_file = tmp_path / 'split.pdb'
        lines = []
        atom_num = 1
        for resid in range(1, 11):
            for name, elem in (('N', 'N'), ('CA', 'C')):
                lines.append(
                    f'ATOM  {atom_num:5d}  {name:<3s} ALA A{resid:4d}'
                    '       0.000   0.000   0.000  1.00  0.00           '
                    f'{elem}\n'
                )
                atom_num += 1
        lines.append('END\n')
        pdb_file.write_text(''.join(lines))

        output = add_chains(pdb_file, first_res=1, last_res=5)

        u = mda.Universe(str(output))
        chain_by_resid = {res.resid: res.atoms.chainIDs[0] for res in u.residues}
        # Residues 1-4 are chain A; the split boundary (5) and beyond are chain B.
        assert [chain_by_resid[r] for r in range(1, 5)] == ['A', 'A', 'A', 'A']
        assert [chain_by_resid[r] for r in range(5, 11)] == ['B'] * 6


class TestImports:
    """Test that module imports work correctly."""

    def test_import_explicit_solvent(self):
        """Test that ExplicitSolvent is importable."""
        from molecular_simulations.build import ExplicitSolvent

        assert ExplicitSolvent is not None

    def test_import_implicit_solvent(self):
        """Test that ImplicitSolvent is importable."""
        from molecular_simulations.build import ImplicitSolvent

        assert ImplicitSolvent is not None

    def test_pathlike_type(self):
        """Test PathLike type alias."""

        from molecular_simulations.build import PathLike

        # PathLike should accept both Path and str
        assert PathLike is not None
