"""Tests for the ipSAE2 port.

Expected values for the synthetic four-molecule model, the nucleic acid
cutoff, the modified-residue weighting and the quoted-CIF DNA case are the
ones asserted by the upstream ipSAE2 v0.1.5 test suite.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from molecular_simulations.analysis.ipSAE import (
    IpSAE2Calculator,
    IpSAE2Parser,
    MoleculeGeometry,
    TokenTable,
    ipSAE,
)

UPSTREAM_PAE = np.array(
    [
        [25.0, 25.0, 1.0, 2.0, 3.0, 1.0],
        [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        [2.0, 25.0, 25.0, 25.0, 25.0, 1.0],
        [2.0, 25.0, 25.0, 25.0, 25.0, 1.0],
        [2.0, 25.0, 25.0, 25.0, 25.0, 1.0],
        [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
    ]
)

UPSTREAM_ATOMS = [
    # record, name, resname, chain, resnum, xyz, element
    ('ATOM', 'N', 'ALA', 'A', 1, (0, 0, 0), 'N'),
    ('ATOM', 'CA', 'ALA', 'A', 1, (1, 0, 0), 'C'),
    ('ATOM', 'C', 'ALA', 'A', 1, (1, 1, 0), 'C'),
    ('ATOM', 'N', 'ALA', 'B', 1, (50, 0, 0), 'N'),
    ('ATOM', 'CA', 'ALA', 'B', 1, (51, 0, 0), 'C'),
    ('ATOM', 'C', 'ALA', 'B', 1, (51, 1, 0), 'C'),
    ('HETATM', 'C1', 'ATP', 'C', 1, (2, 0, 0), 'C'),
    ('HETATM', 'C2', 'ATP', 'C', 1, (2, 1, 0), 'C'),
    ('HETATM', 'C3', 'ATP', 'C', 1, (2, 0, 1), 'C'),
    ('HETATM', 'MG', 'MG', 'D', 1, (3, 0, 0), 'MG'),
]


def write_pdb(path, atoms):
    """Write atoms as fixed-column PDB records."""
    lines = []
    for serial, (record, name, resname, chain, resnum, xyz, element) in enumerate(
        atoms, start=1
    ):
        x, y, z = xyz
        lines.append(
            f'{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resnum:>4}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00          {element:>2}'
        )
    path.write_text('\n'.join(lines) + '\nEND\n')
    return path


def write_cif(path, atoms):
    """Write atoms as an mmCIF atom_site loop, quoting primed atom names.

    ``label_asym_id`` is left as ``.`` so chains come from ``auth_asym_id``.
    """
    header = [
        'data_test',
        'loop_',
        '_atom_site.group_PDB',
        '_atom_site.id',
        '_atom_site.type_symbol',
        '_atom_site.label_atom_id',
        '_atom_site.label_alt_id',
        '_atom_site.label_comp_id',
        '_atom_site.label_asym_id',
        '_atom_site.label_seq_id',
        '_atom_site.Cartn_x',
        '_atom_site.Cartn_y',
        '_atom_site.Cartn_z',
        '_atom_site.auth_seq_id',
        '_atom_site.auth_asym_id',
        '_atom_site.pdbx_PDB_model_num',
    ]
    rows = []
    for serial, (record, name, resname, chain, resnum, xyz, element) in enumerate(
        atoms, start=1
    ):
        quoted = f'"{name}"' if "'" in name else name
        x, y, z = xyz
        rows.append(
            f'{record} {serial} {element} {quoted} . {resname} . . '
            f'{x} {y} {z} {resnum} {chain} 1'
        )
    path.write_text('\n'.join(header + rows) + '\n')
    return path


def load(path, pae, metadata=None):
    """Parse, tokenize and assign frames for a model."""
    atoms = IpSAE2Parser.read_atoms(path)
    tokens = IpSAE2Parser.infer_tokens(
        atoms, IpSAE2Parser.classify_molecules(atoms), len(pae), metadata
    )
    tokens.frame_valid = IpSAE2Parser.frame_validity(atoms, tokens, metadata)
    return atoms, tokens


def make_tokens(rows):
    """Build a TokenTable from (chain, type, resnum, resname, atom, unit) rows."""
    cols = list(zip(*rows, strict=True))
    n = len(rows)
    return TokenTable(
        chain=np.array(cols[0], dtype=object),
        mol_type=np.array(cols[1], dtype=object),
        resnum=np.array(cols[2], dtype=object),
        resname=np.array(cols[3], dtype=object),
        atom_name=np.array(cols[4], dtype=object),
        tokenization=np.array(['residue'] * n, dtype=object),
        unit_key=np.array(cols[5], dtype=object),
        rep_atom=np.arange(n),
        atom_token=np.arange(n),
        frame_valid=np.ones(n, dtype=bool),
    )


def row_for(scores, chain1, chain2):
    rows = scores.filter(
        (scores['chain1'] == chain1) & (scores['chain2'] == chain2)
    ).to_dicts()
    assert len(rows) == 1
    return rows[0]


class TestGeometry:
    """Contact detection and protein d0."""

    def test_protein_d0(self):
        calc = IpSAE2Calculator()
        d0 = calc._d0(np.array([0, 1, 26, 300]), 'prot')
        assert d0[0] == 1.0 and d0[1] == 1.0 and d0[2] == 1.0
        assert d0[3] > 4
        assert calc._d0(np.array([300]), 'lig')[0] == 2.0

    def test_contact_cutoff_is_inclusive(self):
        def geom(x, key):
            xyz = np.array([[x, 0.0, 0.0]])
            return MoleculeGeometry(
                xyz, np.array([key], object), xyz[0], xyz[0], cKDTree(xyz)
            )

        assert IpSAE2Calculator.contact_units(geom(0, 'a'), geom(5, 'b'), 5.0) == (
            {'a'},
            {'b'},
        )
        assert IpSAE2Calculator.contact_units(geom(0, 'a'), geom(5.01, 'b'), 5.0) == (
            set(),
            set(),
        )


class TestUpstreamExample:
    """Protein A, distant protein B, three-atom ligand C and ion D."""

    @pytest.fixture(params=['pdb', 'cif'])
    def model(self, request, tmp_path):
        writer = write_pdb if request.param == 'pdb' else write_cif
        return writer(tmp_path / f'model.{request.param}', UPSTREAM_ATOMS)

    def test_tokens(self, model):
        atoms, tokens = load(model, UPSTREAM_PAE)
        assert IpSAE2Parser.classify_molecules(atoms) == {
            'A': 'prot',
            'B': 'prot',
            'C': 'lig',
            'D': 'ion',
        }
        assert list(tokens.chain) == ['A', 'B', 'C', 'C', 'C', 'D']
        assert tokens.policy == 'modified=none;ligand=atom'
        assert list(tokens.frame_valid) == [True] * 5 + [False]

    def test_scores(self, model):
        atoms, tokens = load(model, UPSTREAM_PAE)
        scores = IpSAE2Calculator().compute_scores(UPSTREAM_PAE, atoms, tokens)
        pairs = list(zip(scores['chain1'], scores['chain2'], strict=True))
        assert pairs == [('A', 'C'), ('A', 'D'), ('C', 'D')]

        ac = row_for(scores, 'A', 'C')
        assert ac['type1'] == 'prot' and ac['type2'] == 'lig'
        assert ac['ipSAE2'] == pytest.approx(np.mean([0.8, 0.5, 0.3076923076923]))
        assert ac['ipSAE2_source'] == '12'
        assert ac['ipSAE2_n0_12'] == 3 and ac['ipSAE2_n_contact_12'] == 3
        assert ac['ipSAE2_iPAE_12'] == 2.0 and ac['ipSAE2_d0_12'] == 2.0

        ad = row_for(scores, 'A', 'D')
        assert ad['ipSAE2'] == 0.8
        assert ad['ipSAE2_21'] is None and ad['ipSAE2_n0_21'] is None

    def test_all_pairs_and_skip_chains(self, model):
        atoms, tokens = load(model, UPSTREAM_PAE)
        everything = IpSAE2Calculator(all_pairs=True).compute_scores(
            UPSTREAM_PAE, atoms, tokens
        )
        assert len(everything) == 6  # all unordered pairs of four molecules
        skipped = IpSAE2Calculator().compute_scores(
            UPSTREAM_PAE, atoms, tokens, skip_chains={'C'}
        )
        assert set(zip(skipped['chain1'], skipped['chain2'], strict=True)) == {
            ('A', 'D')
        }

    def test_npz_metadata(self, model, tmp_path):
        good = tmp_path / 'good.npz'
        np.savez(good, pae=UPSTREAM_PAE, token_asym_id=[0, 1, 2, 2, 2, 3])
        _, tokens = load(model, UPSTREAM_PAE, IpSAE2Parser.read_pae_metadata(good))
        assert list(tokens.chain) == ['A', 'B', 'C', 'C', 'C', 'D']

        bad = tmp_path / 'bad.npz'
        np.savez(bad, pae=UPSTREAM_PAE, token_asym_id=[0, 1, 1, 2, 2, 3])
        with pytest.raises(ValueError, match='token mapping'):
            load(model, UPSTREAM_PAE, IpSAE2Parser.read_pae_metadata(bad))

    def test_frame_mask_metadata(self, model, tmp_path):
        path = tmp_path / 'pae.npz'
        np.savez(path, pae=UPSTREAM_PAE, frame_mask=[0, 1, 1, 1, 1, 1])
        _, tokens = load(model, UPSTREAM_PAE, IpSAE2Parser.read_pae_metadata(path))
        # Ions never define a frame, even when the predictor says so.
        assert list(tokens.frame_valid) == [False, True, True, True, True, False]

    def test_token_count_mismatch(self, model):
        with pytest.raises(ValueError, match='No token mapping'):
            load(model, np.zeros((7, 7)))


class TestScoreDirection:
    """Unit-level scoring rules."""

    def test_nucleic_acid_cutoff_and_protein_d0(self):
        tokens = make_tokens(
            [
                ('A', 'prot', '1', 'ALA', 'CA', 'A:res:1:ALA'),
                ('B', 'nuc', '1', 'DA', "C1'", 'B:res:1:DA'),
            ]
        )
        pae = np.array([[0, 9], [1, 0]], float)
        calc = IpSAE2Calculator(pae_cutoff=10)
        ab = calc.score_direction(
            pae, tokens, np.array([0]), np.array([1]), 'nuc', set()
        )
        ba = calc.score_direction(
            pae, tokens, np.array([1]), np.array([0]), 'prot', set()
        )
        assert ab.d0 == 2 and ba.d0 == 1
        assert ab.score == pytest.approx(1 / (1 + (9 / 2) ** 2))
        assert ba.score == 0.5

        low = IpSAE2Calculator(pae_cutoff=8).score_direction(
            pae, tokens, np.array([0]), np.array([1]), 'nuc', set()
        )
        assert low.score == 0 and low.ipae is None and low.n0 == 0

    def test_modified_residue_equal_weight(self):
        # Two atom tokens of one modified residue count as one unit.
        tokens = make_tokens(
            [
                ('A', 'prot', '1', 'ALA', 'CA', 'A:res:1:ALA'),
                ('B', 'prot', '7', 'TPO', 'P', 'B:res:7:TPO'),
                ('B', 'prot', '7', 'TPO', 'O1P', 'B:res:7:TPO'),
            ]
        )
        pae = np.array([[0, 1, 3], [1, 0, 0], [3, 0, 0]], float)
        result = IpSAE2Calculator().score_direction(
            pae, tokens, np.array([0]), np.array([1, 2]), 'prot', {'B:res:7:TPO'}
        )
        assert result.n0 == 1 and result.n_contact == 1
        assert result.ipae == 2.0 and result.d0 == 1.0
        assert result.score == pytest.approx((0.5 + 0.1) / 2)

    def test_no_valid_frame_is_none(self):
        tokens = make_tokens(
            [
                ('A', 'prot', '1', 'ALA', 'CA', 'A:res:1:ALA'),
                ('D', 'ion', '1', 'MG', 'MG', 'D:token:2'),
            ]
        )
        tokens.frame_valid = np.array([True, False])
        result = IpSAE2Calculator().score_direction(
            np.zeros((2, 2)), tokens, np.array([1]), np.array([0]), 'prot', set()
        )
        assert result.score is None and result.n0 is None


def test_quoted_cif_dna_frames_and_ion_scores(tmp_path):
    """Primed ribose atom names are quoted in mmCIF and must define frames."""
    atoms = []
    for chain, offset in (('C', 0.0), ('D', 2.0)):
        for name, (x, y, z) in {
            'P': (0.0, 0.0, 0.0),
            "C1'": (0.0, 1.0, 0.0),
            "C3'": (1.0, 0.0, 0.0),
            "C4'": (0.0, 0.0, 1.0),
        }.items():
            atoms.append(('ATOM', name, 'DA', chain, 1, (x + offset, y, z), 'C'))
    atoms.append(('HETATM', 'MG', 'MG', 'O', 1, (1.0, 1.0, 1.0), 'MG'))
    model = write_cif(tmp_path / 'dna_ion.cif', atoms)
    pae = np.array([[0, 1, 2], [3, 0, 4], [9, 9, 0]], float)

    parsed, tokens = load(model, pae)
    assert [n for n in parsed.atom_name if 'C1' in n] == ["C1'", "C1'"]
    assert list(tokens.frame_valid) == [True, True, False]
    assert list(tokens.atom_name) == ["C3'", "C3'", 'MG']

    scores = IpSAE2Calculator().compute_scores(pae, parsed, tokens)
    co = row_for(scores, 'C', 'O')
    assert co['ipSAE2_12'] == pytest.approx(0.5)
    assert co['ipSAE2'] == pytest.approx(0.5)
    assert co['ipSAE2_iPAE_12'] == pytest.approx(2.0)
    assert co['ipSAE2_n0_12'] == 1
    assert co['ipSAE2_21'] is None
    cd = row_for(scores, 'C', 'D')
    assert cd['ipSAE2_12'] == pytest.approx(0.8)
    assert cd['ipSAE2_21'] == pytest.approx(1 / (1 + (3 / 2) ** 2))
    assert cd['ipSAE2'] == pytest.approx(0.8)
    assert row_for(scores, 'D', 'O')['ipSAE2_12'] == pytest.approx(0.2)


def ala(chain, resnum, x):
    return [
        ('ATOM', 'N', 'ALA', chain, resnum, (x, 0.0, 0.0), 'N'),
        ('ATOM', 'CA', 'ALA', chain, resnum, (x + 1.0, 0.0, 0.0), 'C'),
        ('ATOM', 'C', 'ALA', chain, resnum, (x + 1.0, 1.0, 0.0), 'C'),
    ]


def sugar(chain, resnum, x):
    return [
        ('HETATM', 'C1', 'NAG', chain, resnum, (x, 3.0, 0.0), 'C'),
        ('HETATM', 'C2', 'NAG', chain, resnum, (x + 1.0, 3.0, 0.0), 'C'),
        ('HETATM', 'O5', 'NAG', chain, resnum, (x, 4.0, 0.0), 'O'),
    ]


class TestGlycan:
    """A multi-residue glycan chain between two protein chains."""

    @pytest.fixture
    def model(self, tmp_path):
        atoms = ala('A', 1, 0.0) + ala('A', 2, 3.0)
        atoms += sugar('G', 1, 0.0) + sugar('G', 2, 3.0)
        atoms += ala('B', 1, 40.0)
        return write_pdb(tmp_path / 'glycan.pdb', atoms)

    def test_glycan_is_one_atom_tokenized_ligand(self, model):
        n = 2 + 6 + 1
        atoms, tokens = load(model, np.full((n, n), 5.0))
        assert IpSAE2Parser.classify_molecules(atoms)['G'] == 'lig'
        assert list(tokens.chain) == ['A', 'A'] + ['G'] * 6 + ['B']
        assert tokens.policy == 'modified=none;ligand=atom'
        assert len(set(tokens.unit_key[2:8])) == 6

    def test_glycan_contributes_to_scores(self, model):
        n = 9
        pae = np.full((n, n), 25.0)
        pae[:2, 2:8] = 2.0  # protein A aligns the glycan confidently
        atoms, tokens = load(model, pae)
        scores = IpSAE2Calculator().compute_scores(pae, atoms, tokens)
        ag = row_for(scores, 'A', 'G')
        assert ag['type2'] == 'lig'
        assert ag['ipSAE2'] == pytest.approx(0.5)
        assert ag['ipSAE2_n0_12'] == 6
        # Glycan frames see no protein residue within the 10 A PAE cutoff.
        assert ag['ipSAE2_n0_21'] == 0


def test_ambiguous_tokenization_raises(tmp_path):
    # A two-atom modified residue plus a two-atom ligand: residue/atom and
    # atom/residue policies give the same count but different orders.
    atoms = [
        *ala('A', 1, 0.0),
        ('HETATM', 'N', 'SEP', 'A', 2, (3.0, 0.0, 0.0), 'N'),
        ('HETATM', 'CA', 'SEP', 'A', 2, (4.0, 0.0, 0.0), 'C'),
        ('HETATM', 'C1', 'LIG', 'L', 1, (5.0, 0.0, 0.0), 'C'),
        ('HETATM', 'C2', 'LIG', 'L', 1, (6.0, 0.0, 0.0), 'C'),
    ]
    model = write_pdb(tmp_path / 'ambiguous.pdb', atoms)
    with pytest.raises(ValueError, match='Ambiguous token order'):
        load(model, np.zeros((4, 4)))


@pytest.mark.parametrize('n_tokens, policy', [(3, 'residue'), (6, 'atom')])
def test_selenocysteine_is_a_modified_residue(tmp_path, n_tokens, policy):
    # Boltz keeps SEC as one token; AF3/Chai split it into atom tokens.
    atoms = [
        *ala('A', 1, 0.0),
        ('HETATM', 'N', 'SEC', 'A', 2, (3.0, 0.0, 0.0), 'N'),
        ('HETATM', 'CA', 'SEC', 'A', 2, (4.0, 0.0, 0.0), 'C'),
        ('HETATM', 'C', 'SEC', 'A', 2, (4.0, 1.0, 0.0), 'C'),
        ('HETATM', 'SE', 'SEC', 'A', 2, (5.0, 0.0, 0.0), 'SE'),
        *ala('A', 3, 6.0),
    ]
    model = write_pdb(tmp_path / 'sec.pdb', atoms)
    _, tokens = load(model, np.zeros((n_tokens, n_tokens)))
    assert tokens.policy == f'modified={policy};ligand=none'
    assert len(set(tokens.unit_key)) == 3


def test_ion_in_protein_chain_raises(tmp_path):
    atoms = [*ala('A', 1, 0.0), ('HETATM', 'ZN', 'ZN', 'A', 2, (5.0, 0, 0), 'ZN')]
    with pytest.raises(ValueError, match='shares protein chain'):
        IpSAE2Parser.classify_molecules(
            IpSAE2Parser.read_atoms(write_pdb(tmp_path / 'zn.pdb', atoms))
        )


class TestScoreSelection:
    """ipSAE with a protein-glycan-protein model and score selection.

    Tokens: A1, A2 (0-1), six glycan atoms (2-7), B1 (8). B aligns A
    confidently (PAE 1) and A aligns B less so (PAE 2), so the original
    ipSAE reports the pair as B -> A.
    """

    @pytest.fixture
    def files(self, tmp_path):
        atoms = ala('A', 1, 0.0) + ala('A', 2, 3.0)
        atoms += sugar('G', 1, 0.0) + sugar('G', 2, 3.0)
        atoms += ala('B', 1, 6.0)
        model = write_pdb(tmp_path / 'model.pdb', atoms)
        pae = np.full((9, 9), 25.0)
        pae[:2, 8] = 2.0
        pae[8, :2] = 1.0
        pae[:2, 2:8] = 2.0
        np.savez(tmp_path / 'pae.npz', pae=pae)
        np.savez(tmp_path / 'plddt.npz', plddt=np.full(9, 90.0))
        return model, tmp_path / 'plddt.npz', tmp_path / 'pae.npz'

    def test_default_scores_all(self, files):
        scorer = ipSAE(*files)
        scorer.run()
        scores = scorer.scores
        assert set(zip(scores['chain1'], scores['chain2'], strict=True)) == {
            ('B', 'A'),
            ('A', 'G'),
            ('B', 'G'),
        }

        ba = row_for(scores, 'B', 'A')
        # Glycan tokens are dropped, so B maps to token 8, not a glycan atom.
        assert ba['ipSAE'] == pytest.approx(0.5)
        for column in ('pDockQ', 'pDockQ2', 'LIS', 'ipTM'):
            assert ba[column] is not None
        # ipSAE2 directions are reoriented to the merged chain1 -> chain2.
        assert ba['ipSAE2_12'] == pytest.approx(0.5)
        assert ba['ipSAE2_21'] == pytest.approx(0.2)
        assert ba['ipSAE2'] == pytest.approx(0.5)
        assert ba['ipSAE2_source'] == '12'

        ag = row_for(scores, 'A', 'G')
        assert ag['type2'] == 'lig'
        assert ag['ipSAE'] is None and ag['pDockQ'] is None
        assert ag['ipSAE2'] == pytest.approx(0.5)
        bg = row_for(scores, 'B', 'G')
        assert bg['ipSAE2'] == pytest.approx(1 / (1 + (25 / 2) ** 2))

    def test_legacy_only_raises_on_glycan(self, files):
        scorer = ipSAE(*files, scores=('ipsae', 'pDockQ'))
        with pytest.raises(ValueError, match='skip_chains'):
            scorer.run()

    def test_legacy_only_with_skip_chains(self, files):
        scorer = ipSAE(*files, skip_chains=['G'], scores=('ipsae', 'LIS'))
        scorer.run()
        assert scorer.scores.columns == ['chain1', 'chain2', 'LIS', 'ipTM', 'ipSAE']
        row = row_for(scorer.scores, 'B', 'A')
        assert row['ipSAE'] == pytest.approx(0.5)

    def test_ipsae2_only_without_plddt(self, files):
        model, _, pae = files
        scorer = ipSAE(model, None, pae, scores=['ipSAE2'])
        scorer.run()
        assert 'ipSAE' not in scorer.scores.columns
        assert len(scorer.scores) == 3
        assert row_for(scorer.scores, 'A', 'B')['ipSAE2_source'] == '21'

    def test_ipsae2_kwargs(self, files):
        model, _, pae = files
        scorer = ipSAE(
            model, None, pae, scores=['ipsae2'], ipsae2_kwargs={'pae_cutoff': 1.5}
        )
        scorer.run()
        # A -> B fails the tighter cutoff; B -> A (PAE 1) still passes.
        row = row_for(scorer.scores, 'A', 'B')
        assert row['ipSAE2_12'] == 0.0
        assert row['ipSAE2_21'] == pytest.approx(0.5)

    def test_invalid_arguments(self, files):
        model, plddt, pae = files
        with pytest.raises(ValueError, match='Unknown'):
            ipSAE(model, plddt, pae, scores=['ipsae3'])
        with pytest.raises(ValueError, match='plddt_file is required'):
            ipSAE(model, None, pae, scores=['ipsae2', 'lis'])
