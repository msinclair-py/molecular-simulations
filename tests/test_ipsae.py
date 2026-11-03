"""
Unit tests for ipSAE.py module
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from molecular_simulations.analysis.ipSAE import ModelParser, ScoreCalculator, ipSAE


class TestModelParser:
    """Test suite for ModelParser class"""

    def test_init(self):
        """Test ModelParser initialization"""
        parser = ModelParser('test.pdb')
        assert parser.structure == Path('test.pdb')
        assert parser.protein_token_indices == []
        assert parser.residues == []
        assert parser.cb_residues == []
        assert parser.chains == []

    def test_canonical_residue_class_attributes(self):
        """Canonical residues cover amino acids and nucleotides, not ligands"""
        for name in ('DA', 'DC', 'DT', 'DG', 'A', 'C', 'U', 'G'):
            assert name in ModelParser.CANONICAL_NUCLEIC
        assert {'ALA', 'GLY'} <= ModelParser.CANONICAL_PROTEIN
        assert 'SEP' not in ModelParser.CANONICAL_PROTEIN
        assert (
            'LIG' not in ModelParser.CANONICAL_PROTEIN | ModelParser.CANONICAL_NUCLEIC
        )

    def test_read_atoms_pdb(self, tmp_path):
        """PDB columns are read into the atom table"""
        pdb_file = tmp_path / 'test.pdb'
        pdb_file.write_text(
            'ATOM      1  CA  GLY A   1      10.000  20.000  30.000  1.00 20.00           C\n'
        )
        atoms = ModelParser(pdb_file).read_atoms()

        assert atoms.atom_name[0] == 'CA'
        assert atoms.resname[0] == 'GLY'
        assert atoms.chain[0] == 'A'
        assert atoms.resnum[0] == '1'
        assert atoms.element[0] == 'C'
        assert np.allclose(atoms.xyz[0], [10.0, 20.0, 30.0])

    def test_read_atoms_cif(self, tmp_path):
        """mmCIF columns are read by name and quoted atom names unquoted"""
        cif_file = tmp_path / 'test.cif'
        cif_file.write_text(
            '\n'.join(
                [
                    'data_test',
                    'loop_',
                    '_atom_site.group_PDB',
                    '_atom_site.id',
                    '_atom_site.type_symbol',
                    '_atom_site.label_alt_id',
                    '_atom_site.label_atom_id',
                    '_atom_site.label_comp_id',
                    '_atom_site.label_asym_id',
                    '_atom_site.label_seq_id',
                    '_atom_site.Cartn_x',
                    '_atom_site.Cartn_y',
                    '_atom_site.Cartn_z',
                    'ATOM 1 C . CA GLY A 1 10.000 20.000 30.000',
                    'ATOM 2 C . "C1\'" DA B 1 1.0 2.0 3.0',
                ]
            )
            + '\n'
        )
        atoms = ModelParser(cif_file).read_atoms()

        assert list(atoms.atom_name) == ['CA', "C1'"]
        assert list(atoms.chain) == ['A', 'B']
        assert np.allclose(atoms.xyz[0], [10.0, 20.0, 30.0])

    def test_parse_structure_file_pdb(self, tmp_path):
        """Test parsing a real PDB structure file written to disk."""
        pdb_file = tmp_path / 'test.pdb'
        pdb_file.write_text(
            'ATOM      1  CA  GLY A   1      10.000  20.000  30.000  1.00 20.00           C\n'
            'ATOM      2  N   ALA A   2      11.500  21.500  31.500  1.00 20.00           N\n'
            'ATOM      3  CA  ALA A   2      12.000  22.000  32.000  1.00 20.00           C\n'
            'ATOM      4  CB  ALA A   2      13.000  23.000  33.000  1.00 20.00           C\n'
            'END\n'
        )

        parser = ModelParser(str(pdb_file))
        parser.parse_structure_file()

        assert len(parser.residues) == 2  # Only CA atoms
        assert len(parser.cb_residues) == 2  # GLY CA (fallback) + ALA CB
        assert len(parser.chains) == 2
        assert all(chain == 'A' for chain in parser.chains)

    def test_classify_chains(self, tmp_path):
        """Chains are labeled protein or nucleic acid"""
        pdb_file = tmp_path / 'test.pdb'
        pdb_file.write_text(
            'ATOM      1  CA  GLY A   1      10.000  20.000  30.000  1.00 20.00           C\n'
            "ATOM      2  C1' DA  B   1      12.000  20.000  30.000  1.00 20.00           C\n"
        )
        parser = ModelParser(pdb_file)
        parser.parse_structure_file()
        parser.classify_chains()

        assert parser.chain_types == {'A': 'protein', 'B': 'nucleic_acid'}


class TestScoreCalculator:
    """Test suite for ScoreCalculator class"""

    CHAINS = np.array(['A', 'A', 'B', 'B'])
    PLDDT = np.array([90.0, 85.0, 80.0, 75.0])
    PAE = np.array([[0, 2, 8, 10], [2, 0, 9, 8], [8, 9, 0, 5], [10, 8, 5, 0]], float)
    NEAR = np.array([[0, 1, 5, 6], [1, 0, 7, 5], [5, 7, 0, 3], [6, 5, 3, 0]], float)
    FAR = np.array(
        [[0, 1, 100, 100], [1, 0, 100, 100], [100, 100, 0, 1], [100, 100, 1, 0]],
        float,
    )

    def calc(self, chain_types=None):
        return ScoreCalculator(
            self.CHAINS, chain_types or {'A': 'protein', 'B': 'protein'}
        )

    def test_init(self):
        """Test ScoreCalculator initialization"""
        calc = self.calc()
        assert np.array_equal(calc.chains, self.CHAINS)
        assert calc.pdockq_cutoff == 8.0
        assert calc.pae_cutoff == 12.0

    def test_ptm(self):
        """The TM kernel is 1 at zero PAE and 1/2 at PAE == d0"""
        ptm = ScoreCalculator.ptm(np.array([0.0, 10.0]), 10.0)
        assert np.allclose(ptm, [1.0, 0.5])

    def test_d0(self):
        """L is floored at min_length, and d0 at 1 (protein) or 2 (nucleic acid)"""
        assert ScoreCalculator.d0(10, False) == pytest.approx(
            1.24 * 12 ** (1 / 3) - 1.8
        )
        assert ScoreCalculator.d0(10, False, min_length=26) == 1.0
        assert ScoreCalculator.d0(10, True) == 2.0
        d0 = ScoreCalculator.d0(np.array([5, 100, 200]), False)
        assert d0.shape == (3,)
        assert d0[2] > d0[1] > d0[0]

    def test_score_pair(self):
        """All four A-B residue pairs are in contact"""
        pDockQ, pDockQ2, lis, ipTM, ipSAE = self.calc().score_pair(
            'A', 'B', self.NEAR, self.PLDDT, self.PAE
        )
        assert lis == pytest.approx(np.mean([(12 - p) / 12 for p in (8, 10, 9, 8)]))
        assert 0 < pDockQ < 1
        assert 0 < pDockQ2 < 1.5
        assert 0 < ipTM <= 1
        assert 0 < ipSAE <= 1

    def test_no_contacts(self):
        """pDockQ and pDockQ2 are 0 without residue pairs within the cutoff"""
        pDockQ, pDockQ2, *_ = self.calc().score_pair(
            'A', 'B', self.FAR, self.PLDDT, self.PAE
        )
        assert pDockQ == 0.0
        assert pDockQ2 == 0.0

    def test_lis_zero_when_all_pae_above_cutoff(self):
        """No PAE below the cutoff -> LIS 0.0"""
        _, _, lis, _, _ = self.calc().score_pair(
            'A', 'B', self.NEAR, self.PLDDT, np.full((4, 4), 30.0)
        )
        assert lis == 0.0

    def test_zero_when_scored_chain_absent(self):
        """A scored chain without residues gives ipTM and ipSAE of 0.0"""
        calc = ScoreCalculator(np.array(['A', 'A']), {'A': 'protein', 'B': 'protein'})
        *_, ipTM, ipSAE = calc.score_pair(
            'A', 'B', np.zeros((2, 2)), np.full(2, 90.0), np.zeros((2, 2))
        )
        assert ipTM == 0.0
        assert ipSAE == 0.0

    def test_nucleic_acid_uses_larger_d0(self):
        """A nucleic acid partner raises the d0 floor to 2"""
        args = ('A', 'B', self.NEAR, self.PLDDT, self.PAE)
        prot = self.calc().score_pair(*args)
        nuc = self.calc({'A': 'protein', 'B': 'nucleic_acid'}).score_pair(*args)
        assert nuc[3] > prot[3]
        assert nuc[4] > prot[4]

    def test_compute_scores(self):
        """A->B and B->A collapse to the direction with the larger ipSAE"""
        calc = self.calc()
        scores = calc.compute_scores(self.NEAR, self.PLDDT, self.PAE)

        assert scores.height == 1
        both = [
            calc.score_pair(c1, c2, self.NEAR, self.PLDDT, self.PAE)[4]
            for c1, c2 in (('A', 'B'), ('B', 'A'))
        ]
        assert scores['ipSAE'][0] == max(both)


class TestIpSAE:
    """Test suite for ipSAE class (real ModelParser; no mocks)."""

    def test_init(self):
        """Test ipSAE initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ModelParser.__init__ only stores the path; it does not read the
            # file, so an empty placeholder is sufficient here.
            structure_file = Path(tmpdir) / 'test.pdb'
            structure_file.touch()
            plddt_file = Path(tmpdir) / 'plddt.npy'
            pae_file = Path(tmpdir) / 'pae.npy'

            ipsae = ipSAE(structure_file, plddt_file, pae_file)

            assert ipsae.plddt_file == plddt_file
            assert ipsae.pae_file == pae_file
            assert ipsae.path == Path(tmpdir)

    def test_load_pLDDT_file(self):
        """Test loading pLDDT file with values in 0-1 range"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test pLDDT data in 0-1 range (e.g. Boltz)
            plddt_data = np.random.rand(100) * 0.01  # Values between 0 and 0.01
            plddt_file = Path(tmpdir) / 'plddt.npz'
            np.savez(plddt_file, plddt=plddt_data)

            ipsae = ipSAE('test.pdb', plddt_file, 'pae.npy')
            result = ipsae.load_pLDDT_file()

            # Check that values are scaled by 100
            assert np.allclose(result, plddt_data * 100)

    def test_load_pLDDT_file_already_scaled(self):
        """Test loading pLDDT file with values already in 0-100 range"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test pLDDT data already in 0-100 range (e.g. Chai, AF)
            plddt_data = np.random.rand(100) * 100  # Values between 0 and 100
            plddt_file = Path(tmpdir) / 'plddt.npz'
            np.savez(plddt_file, plddt=plddt_data)

            ipsae = ipSAE('test.pdb', plddt_file, 'pae.npy')
            result = ipsae.load_pLDDT_file()

            # Should NOT be rescaled
            assert np.allclose(result, plddt_data)

    def test_load_PAE_file(self):
        """Test loading PAE file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test PAE data
            pae_data = np.random.rand(100, 100) * 30
            pae_file = Path(tmpdir) / 'pae.npz'
            np.savez(pae_file, pae=pae_data)

            ipsae = ipSAE('test.pdb', 'plddt.npy', pae_file)
            result = ipsae.load_PAE_file()

            assert np.array_equal(result, pae_data)

    def test_save_scores(self, tmp_path):
        """Test saving scores to a real parquet file that can be read back."""
        ipsae = ipSAE('test.pdb', 'plddt.npy', 'pae.npy', out_path=str(tmp_path))
        ipsae.scores = pl.DataFrame({'test': [1, 2, 3]})

        ipsae.save_scores()

        out_file = tmp_path / 'ipSAE_scores.parquet'
        assert out_file.exists()
        back = pl.read_parquet(out_file)
        assert back.columns == ['test']
        assert back['test'].to_list() == [1, 2, 3]

    def test_parse_structure_file(self, two_chain_pdb, tmp_path):
        """Test structure file parsing against a real two-chain PDB."""
        ipsae = ipSAE(
            str(two_chain_pdb), 'plddt.npy', 'pae.npy', out_path=str(tmp_path)
        )

        ipsae.parse_structure_file()

        # Two standard residues (LYS in chain A, ASP in chain B) each contribute
        # one anchor and one CB coordinate.
        n_res = len(ipsae.parser.residues)
        assert n_res == 2
        assert ipsae.coordinates.shape == (n_res, 3)
        assert ipsae.cb_coordinates.shape == (n_res, 3)
        assert ipsae.parser.chain_types == {'A': 'protein', 'B': 'protein'}

    def test_prepare_scorer(self, two_chain_pdb, tmp_path):
        """Test scorer preparation."""
        ipsae = ipSAE(
            str(two_chain_pdb), 'plddt.npy', 'pae.npy', out_path=str(tmp_path)
        )
        ipsae.parse_structure_file()

        ipsae.prepare_scorer()

        assert isinstance(ipsae.scorer, ScoreCalculator)
        assert np.array_equal(ipsae.scorer.chains, np.array(['A', 'B']))
        assert ipsae.scorer.chain_types == {'A': 'protein', 'B': 'protein'}


class TestIpSAERun:
    """Test ipSAE.run end-to-end against a real structure (no mocks)."""

    def test_run_full_workflow(self, two_chain_pdb, tmp_path):
        """run() parses a real PDB, scores, and writes a readable parquet."""
        # Real two-chain structure yields two scorable residues (A: LYS, B: ASP).
        plddt_file = tmp_path / 'plddt.npz'
        pae_file = tmp_path / 'pae.npz'
        np.savez(plddt_file, plddt=np.array([0.9, 0.8]))
        np.savez(pae_file, pae=np.array([[0.0, 3.0], [3.0, 0.0]]))

        # The capped (ACE/NME) PDB has no 2-token predictor tokenization, so
        # ipSAE2 is left out; it is covered in test_ipsae2.py.
        obj = ipSAE(
            str(two_chain_pdb),
            plddt_file,
            pae_file,
            out_path=str(tmp_path),
            scores=('ipsae', 'pdockq', 'pdockq2', 'lis'),
        )

        obj.run()

        assert hasattr(obj, 'scores')
        # Real scores DataFrame carries the expected columns.
        for col in ('chain1', 'chain2', 'pDockQ', 'pDockQ2', 'LIS', 'ipTM', 'ipSAE'):
            assert col in obj.scores.columns
        # Undirected A_B pair collapses to a single row.
        assert len(obj.scores) == 1

        out_file = tmp_path / 'ipSAE_scores.parquet'
        assert out_file.exists()
        back = pl.read_parquet(out_file)
        assert 'ipSAE' in back.columns
        assert len(back) == 1


class TestModelParserCIF:
    """Test ModelParser CIF parsing."""

    def test_parse_structure_file_cif(self):
        """Test parsing CIF structure file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_file = Path(tmpdir) / 'test.cif'
            lines = [
                'data_test',
                'loop_',
                '_atom_site.group_PDB',
                '_atom_site.id',
                '_atom_site.type_symbol',
                '_atom_site.label_alt_id',
                '_atom_site.label_atom_id',
                '_atom_site.label_comp_id',
                '_atom_site.label_asym_id',
                '_atom_site.label_seq_id',
                '_atom_site.Cartn_x',
                '_atom_site.Cartn_y',
                '_atom_site.Cartn_z',
                'ATOM 1 C . CA GLY A 1 10.000 20.000 30.000',
                'ATOM 2 C . CA ALA A 2 12.000 22.000 32.000',
                'ATOM 3 C . CB ALA A 2 13.000 23.000 33.000',
            ]
            cif_file.write_text('\n'.join(lines) + '\n')

            parser = ModelParser(str(cif_file))
            parser.parse_structure_file()

            assert len(parser.residues) == 2  # GLY CA + ALA CA
            assert len(parser.cb_residues) == 2  # GLY CA (fallback) + ALA CB
            assert len(parser.chains) == 2
            assert parser.chains == ['A', 'A']


class TestParseStructureCBFallback:
    """Cover the per-residue CA fallback when no CB is present."""

    def test_ca_only_residue_falls_back_to_ca(self, tmp_path):
        """A residue with only a CA atom uses its CA as the distance atom."""
        pdb = tmp_path / 'ca_only.pdb'
        # ALA with a single CA atom: it registers as an anchor residue but has
        # no CB (and is not GLY), so its CA stands in for the CB.
        pdb.write_text(
            'ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 20.00           C\n'
            'END\n'
        )

        obj = ipSAE(str(pdb), 'plddt.npy', 'pae.npy', out_path=str(tmp_path))
        obj.parse_structure_file()

        assert obj.parser.cb_residues[0] == obj.parser.residues[0]
        assert obj.cb_coordinates.shape == (1, 3)
        assert np.array_equal(obj.cb_coordinates, obj.coordinates)


class TestBuildProteinTokenIndices:
    """Cover build_protein_token_indices token-layout inference and errors."""

    def _parser(self, skip_chains):
        return ModelParser('unused.pdb', skip_chains=skip_chains)

    def test_non_contiguous_skipped_runs_raise(self):
        """Skipped chains in two separate runs are unsolvable -> ValueError."""
        parser = self._parser({'B', 'D'})
        parser.chain_order = ['A', 'B', 'C', 'D']
        parser.chain_residue_counts = {'A': 2, 'C': 2}

        with pytest.raises(ValueError, match='non-contiguous'):
            parser.build_protein_token_indices(6)

    def test_kept_exceeds_total_raises(self):
        """More kept residues than total tokens -> negative span -> ValueError."""
        parser = self._parser({'B'})
        parser.chain_order = ['A', 'B']
        parser.chain_residue_counts = {'A': 5}

        with pytest.raises(ValueError, match='exceeds'):
            parser.build_protein_token_indices(3)

    def test_contiguous_skip_infers_indices(self):
        """A single contiguous skip block leaves the correct kept-token indices."""
        parser = self._parser({'B'})
        parser.chain_order = ['A', 'B', 'C']
        parser.chain_residue_counts = {'A': 2, 'C': 2}

        parser.build_protein_token_indices(6)

        # A occupies tokens 0,1; the 2-token skip block spans 2,3; C is 4,5.
        assert parser.protein_token_indices == [0, 1, 4, 5]


class TestReadAtomsAuthFallback:
    """Cover read_atoms' auth_seq_id fallback for a missing label_seq_id."""

    def test_auth_seq_id_used_when_label_missing(self, tmp_path):
        """label_seq_id '.' with auth_seq_id present -> auth used."""
        cif_file = tmp_path / 'test.cif'
        cif_file.write_text(
            '\n'.join(
                [
                    'data_test',
                    'loop_',
                    '_atom_site.group_PDB',
                    '_atom_site.id',
                    '_atom_site.type_symbol',
                    '_atom_site.label_alt_id',
                    '_atom_site.label_atom_id',
                    '_atom_site.label_comp_id',
                    '_atom_site.label_asym_id',
                    '_atom_site.label_seq_id',
                    '_atom_site.Cartn_x',
                    '_atom_site.Cartn_y',
                    '_atom_site.Cartn_z',
                    '_atom_site.auth_seq_id',
                    'HETATM 1 C . C1 NAG A . 10.0 20.0 30.0 5',
                ]
            )
            + '\n'
        )
        atoms = ModelParser(cif_file).read_atoms()

        assert atoms.resnum[0] == '5'
        assert atoms.resname[0] == 'NAG'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
