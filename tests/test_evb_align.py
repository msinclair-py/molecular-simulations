"""Unit tests for the EVB diabatic-pair aligner (evb_align).

The aligner is AMBER-free -- pure structure manipulation -- so these tests run a
small self-contained hydride transfer end to end. The product acceptor is
deliberately reordered and renamed so the tests exercise the graph-isomorphism
correspondence, not name matching.

Reaction (a minimal hydride transfer):
    reactant:  donor  C1(-O1)(-H1[hydride])(-H2)   +  acceptor  Ca(=Oa)(-Hb)
    product:   donor  C1(=O1)(-H2)                 +  acceptor  Ca(-Oa)(-Hb)(-H1)
The transferring atom is the reactant donor's H1 == the product acceptor's Hc.
"""

from pathlib import Path

import pytest

from molecular_simulations.build.build_evb import _parse_mol2
from molecular_simulations.build.evb_align import (
    AlignmentError,
    DiabaticPairAligner,
    Transfer,
    _element,
    _State,
)

pytestmark = pytest.mark.unit


def _mol2(
    atoms: list[tuple[str, str, float]], bonds: list[tuple[int, int, str]]
) -> str:
    """Render a minimal mol2 from (name, type, charge) atoms and 1-based bonds."""
    lines = [
        '@<TRIPOS>MOLECULE',
        'LIG',
        f'{len(atoms)} {len(bonds)} 1 0 0',
        'SMALL',
        'USER_CHARGES',
        '',
        '@<TRIPOS>ATOM',
    ]
    for i, (name, typ, q) in enumerate(atoms, start=1):
        lines.append(f'{i} {name} 0.0 0.0 0.0 {typ} 1 LIG {q:.6f}')
    lines.append('@<TRIPOS>BOND')
    for i, (a, b, o) in enumerate(bonds, start=1):
        lines.append(f'{i} {a} {b} {o}')
    return '\n'.join(lines) + '\n'


# Reactant state: donor (C1,O1,H1,H2) and acceptor (Ca,Oa,Hb).
_R_DONOR = _mol2(
    [('C1', 'c3', 0.2), ('O1', 'oh', -0.4), ('H1', 'h1', 0.0), ('H2', 'h1', 0.2)],
    [(1, 2, '1'), (1, 3, '1'), (1, 4, '1')],
)
_R_ACCEPTOR = _mol2(
    [('Ca', 'c', 0.3), ('Oa', 'o', -0.4), ('Hb', 'ha', 0.1)],
    [(1, 2, '2'), (1, 3, '1')],
)
# Product state: donor lost H1 (C1=O1); acceptor gained it, REORDERED + RENAMED
# so atom 1 is the transferred H and the carbon is atom 3 -- name matching would
# fail here, graph matching must recover Ca<->C9, Oa<->O9, Hb<->Hd.
_P_DONOR = _mol2(
    [('C1', 'c', 0.5), ('O1', 'o', -0.6), ('H2', 'h1', 0.1)],
    [(1, 2, '2'), (1, 3, '1')],
)
_P_ACCEPTOR = _mol2(
    [('Hc', 'h1', 0.0), ('O9', 'oh', -0.5), ('C9', 'c3', 0.4), ('Hd', 'ha', 0.1)],
    [(3, 2, '1'), (3, 1, '1'), (3, 4, '1')],
)


@pytest.fixture
def states(tmp_path: Path) -> dict[str, list[Path]]:
    """Write the four species mol2s and return reactant/product path lists."""
    files = {}
    for tag, text in [
        ('rd', _R_DONOR),
        ('ra', _R_ACCEPTOR),
        ('pd', _P_DONOR),
        ('pa', _P_ACCEPTOR),
    ]:
        p = tmp_path / f'{tag}.mol2'
        p.write_text(text)
        files[tag] = p
    return {
        'reactant': [files['rd'], files['ra']],
        'product': [files['pd'], files['pa']],
    }


@pytest.fixture
def transfer() -> Transfer:
    return Transfer(reactant_atom=(0, 'H1'), product_atom=(1, 'Hc'))


class TestElement:
    def test_from_name(self):
        assert _element('C1', 'c3') == 'C'
        assert _element('O12', 'os') == 'O'
        assert _element('N1', 'na') == 'N'
        assert _element('H10', 'h1') == 'H'

    def test_two_letter_halogen(self):
        assert _element('Cl1', 'cl') == 'Cl'
        assert _element('Br3', 'br') == 'Br'

    def test_falls_back_to_type(self):
        # An empty-ish/odd name defers to the GAFF type's leading element.
        assert _element('1', 'c3') == 'C'

    def test_unrecognized_raises(self):
        with pytest.raises(AlignmentError, match='cannot perceive element'):
            _element('Xx9', 'zz')


class TestAlignHappyPath:
    def test_resolves_reactive_atoms(self, states, transfer, tmp_path):
        pair = DiabaticPairAligner(
            states['reactant'], states['product'], [transfer]
        ).align(tmp_path / 'out')
        # shared element-leading names: C1,O1,H1,H2 (donor) then C2,O2,H3 (acceptor)
        assert pair.reactive == ['H1']
        assert pair.donor == ['C1']
        assert pair.acceptor == ['C2']
        assert pair.single_transfer
        assert pair.reactant_charge == pytest.approx(pair.product_charge)

    def test_shared_atom_order_and_names(self, states, transfer, tmp_path):
        pair = DiabaticPairAligner(
            states['reactant'], states['product'], [transfer]
        ).align(tmp_path / 'out')
        r = _parse_mol2(pair.reactant_mol2)
        p = _parse_mol2(pair.product_mol2)
        assert r.names == ['C1', 'O1', 'H1', 'H2', 'C2', 'O2', 'H3']
        assert p.names == r.names  # same atoms, same order
        assert r.n_atoms == 7 and p.n_atoms == 7

    def test_product_carries_product_types_and_charges(
        self, states, transfer, tmp_path
    ):
        """The acceptor carbon (slot C2) must take the PRODUCT parameters."""
        pair = DiabaticPairAligner(
            states['reactant'], states['product'], [transfer]
        ).align(tmp_path / 'out')
        r = _parse_mol2(pair.reactant_mol2)
        p = _parse_mol2(pair.product_mol2)
        ca = r.names.index('C2')
        assert r.types[ca] == 'c' and r.charges[ca] == pytest.approx(0.3)  # reactant
        assert p.types[ca] == 'c3' and p.charges[ca] == pytest.approx(0.4)  # product

    def test_only_reactive_bond_differs(self, states, transfer, tmp_path):
        pair = DiabaticPairAligner(
            states['reactant'], states['product'], [transfer]
        ).align(tmp_path / 'out')
        r = _parse_mol2(pair.reactant_mol2)
        p = _parse_mol2(pair.product_mol2)

        def pairs(m):
            return {frozenset((m.names[a - 1], m.names[b - 1])) for a, b, _ in m.bonds}

        only_r = pairs(r) - pairs(p)
        only_p = pairs(p) - pairs(r)
        assert only_r == {frozenset(('C1', 'H1'))}  # donor-hydride breaks
        assert only_p == {frozenset(('C2', 'H1'))}  # acceptor-hydride forms

    def test_writes_alignment_json(self, states, transfer, tmp_path):
        DiabaticPairAligner(states['reactant'], states['product'], [transfer]).align(
            tmp_path / 'out'
        )
        assert (tmp_path / 'out' / 'alignment.json').exists()


class TestFailureModes:
    def test_skeleton_not_isomorphic(self, states, transfer, tmp_path):
        # Add a spurious O9-Hd bond to the product acceptor: after removing the
        # transferring atom the two acceptor skeletons are no longer isomorphic.
        bad = _mol2(
            [
                ('Hc', 'h1', 0.0),
                ('O9', 'oh', -0.5),
                ('C9', 'c3', 0.4),
                ('Hd', 'ha', 0.1),
            ],
            [(3, 2, '1'), (3, 1, '1'), (3, 4, '1'), (2, 4, '1')],
        )
        (tmp_path / 'bad_pa.mol2').write_text(bad)
        product = [states['product'][0], tmp_path / 'bad_pa.mol2']
        with pytest.raises(AlignmentError, match='not isomorphic'):
            DiabaticPairAligner(states['reactant'], product, [transfer]).align(
                tmp_path / 'out'
            )

    def test_charge_not_conserved(self, states, transfer, tmp_path):
        # Perturb only a charge (bonds unchanged): the invariant passes but the
        # net-charge guard must fire.
        bad = _mol2(
            [
                ('Hc', 'h1', 0.0),
                ('O9', 'oh', -0.5),
                ('C9', 'c3', 1.4),
                ('Hd', 'ha', 0.1),
            ],
            [(3, 2, '1'), (3, 1, '1'), (3, 4, '1')],
        )
        (tmp_path / 'bad_pa.mol2').write_text(bad)
        product = [states['product'][0], tmp_path / 'bad_pa.mol2']
        with pytest.raises(AlignmentError, match='net charge'):
            DiabaticPairAligner(states['reactant'], product, [transfer]).align(
                tmp_path / 'out'
            )

    def test_missing_transfer_raises(self, states):
        with pytest.raises(AlignmentError, match='at least one Transfer'):
            DiabaticPairAligner(states['reactant'], states['product'], [])

    def test_unknown_atom_name(self, states, tmp_path):
        bad = Transfer(reactant_atom=(0, 'ZZ'), product_atom=(1, 'Hc'))
        with pytest.raises(AlignmentError, match="atom 'ZZ' not found"):
            DiabaticPairAligner(states['reactant'], states['product'], [bad]).align(
                tmp_path / 'out'
            )


class TestSoleNeighbour:
    def test_infers_single_bond_partner(self):
        state = _State([_parse_mol2_from_text(_R_DONOR)])
        h1 = state.node_by_name(0, 'H1')
        donor = DiabaticPairAligner._sole_neighbour(state, h1, 'donor')
        assert state.atom(donor)['name'] == 'C1'

    def test_multi_bond_atom_raises(self):
        state = _State([_parse_mol2_from_text(_R_DONOR)])
        c1 = state.node_by_name(0, 'C1')  # 3 bonds
        with pytest.raises(AlignmentError, match='cannot infer the donor'):
            DiabaticPairAligner._sole_neighbour(state, c1, 'donor')


class TestExplicitDonorAcceptor:
    def test_explicit_partners_used(self, states, tmp_path):
        # Provide donor/acceptor explicitly; result must match the inferred run.
        t = Transfer(
            reactant_atom=(0, 'H1'),
            product_atom=(1, 'Hc'),
            donor=(0, 'C1'),
            acceptor=(1, 'C9'),
        )
        pair = DiabaticPairAligner(states['reactant'], states['product'], [t]).align(
            tmp_path / 'out'
        )
        assert pair.donor == ['C1'] and pair.acceptor == ['C2']


def _parse_mol2_from_text(text: str):
    """Parse a mol2 from an in-memory string via a temp file-free path."""
    import tempfile

    with tempfile.NamedTemporaryFile('w', suffix='.mol2', delete=False) as fh:
        fh.write(text)
        name = fh.name
    return _parse_mol2(name)
