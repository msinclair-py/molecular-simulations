"""Align two independent state parameterizations into an EVB diabatic pair.

:class:`molecular_simulations.build.EVBBuilder` requires its two inputs to
*already* be a diabatic pair: the same atoms in the same order, on one coordinate
frame, differing only in bonded connectivity around the reactive atom(s) and in
partial charges. Real inputs never arrive that way. A reactant complex and a
product complex are typically equilibrated and parameterized *independently*, so
they disagree on everything mechanical:

  * different atom counts -- the transferring atom lives in a *different* species
    file in each state (e.g. a hydride is part of the substrate in the reactant
    but part of the cofactor in the product);
  * different atom order and naming (an independent antechamber run reorders a
    whole species);
  * different coordinate frames (two separate simulations).

:class:`DiabaticPairAligner` bridges that gap. Given the two states (as mol2
species) and a minimal :class:`Transfer` spec naming each transferring atom in
both states, it:

  1. builds the molecular graph of each state and, with the transferring atoms
     removed, finds the atom correspondence by graph isomorphism (VF2) -- so a
     fully reordered species is matched automatically by connectivity;
  2. assembles one shared atom set in the reactant's order, with element-leading
     unique names (antechamber perceives elements from atom names, and the two
     species share names like ``C1`` once merged into one unit);
  3. emits ``reactant_region.mol2`` and ``product_region.mol2`` -- the same atoms
     in the same order, each carrying its own state's coordinates/types/charges,
     and differing only in the reactive bond(s);
  4. *verifies* the result: the only bond difference between the two mol2s must be
     exactly donor-reactive (reactant) vs acceptor-reactive (product) for every
     transfer, and total charge must be conserved.

The output feeds :class:`EVBBuilder` unchanged. The aligner needs no AMBER
tooling -- it is pure structure manipulation, so it is fast and unit-testable in
isolation.

Scope: the aligner handles any reaction expressible as a set of atom transfers on
a *conserved* heavy-atom skeleton (the graphs are isomorphic once the
transferring atoms are removed). Reactions that reorganize the heavy-atom
skeleton beyond the transfers are out of scope and fail the isomorphism (an
explicit :class:`AlignmentError`, not a silent mis-map). The API takes a *list*
of transfers so multi-atom transfers (e.g. concerted proton+hydride) align
correctly; the single-transfer hand-off to EVBBuilder is what the downstream
Morse/mapping code currently consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .build_evb import EVBBuildError, _Mol2, _parse_mol2

PathLike = str | Path

# Two-letter element symbols matched (case-insensitively) at the start of an atom
# name before the single-letter fallback. Restricted to ligand halogens: metals
# like Ca/Na/Mg would shadow common single-letter atom names (CA, NA, MG) and
# mis-perceive the element. mol2 atom names must be element-leading (the standard
# for these GAFF reactive-region files); pass an ``overrides`` map if not.
_TWO_LETTER = ('Cl', 'Br')
_ONE_LETTER = set('CHONPSFI')


class AlignmentError(EVBBuildError):
    """Raised when two states cannot be aligned into a diabatic pair.

    Subclasses :class:`~molecular_simulations.build.EVBBuildError` so callers can
    catch either with one except.
    """


@dataclass
class Transfer:
    """One transferring atom, anchored in both diabatic states.

    An atom is identified as ``(species_index, atom_name)`` where
    ``species_index`` is the position of the mol2 in the state's species list and
    ``atom_name`` is that atom's name in the *original* file.

    Args:
        reactant_atom: The transferring atom as named in the reactant state.
        product_atom: The *same physical* atom as named in the product state.
        donor: The reactant partner the atom is bonded to (bond that breaks). If
            None, inferred as the transferring atom's sole bonded neighbour in the
            reactant (valid when it has exactly one bond, e.g. a transferring H).
        acceptor: The product partner the atom bonds to (bond that forms). If
            None, inferred as its sole neighbour in the product.
    """

    reactant_atom: tuple[int, str]
    product_atom: tuple[int, str]
    donor: tuple[int, str] | None = None
    acceptor: tuple[int, str] | None = None


@dataclass
class AlignedPair:
    """Result of :meth:`DiabaticPairAligner.align`.

    Attributes:
        reactant_mol2: Path to the aligned reactant-state mol2.
        product_mol2: Path to the aligned product-state mol2 (same atoms/order).
        donor: Resolved donor atom name(s) in the shared unit (one per transfer).
        acceptor: Resolved acceptor atom name(s).
        reactive: Resolved transferring atom name(s).
        name_map: New shared name -> ``[state, original-name]`` audit trail.
        reactant_charge: Total reactant charge (should equal product_charge).
        product_charge: Total product charge.
    """

    reactant_mol2: Path
    product_mol2: Path
    donor: list[str]
    acceptor: list[str]
    reactive: list[str]
    name_map: dict[str, list[str]] = field(default_factory=dict)
    reactant_charge: float = 0.0
    product_charge: float = 0.0

    @property
    def single_transfer(self) -> bool:
        """True if the reaction is a single atom transfer (EVBBuilder-ready)."""
        return len(self.reactive) == 1


# A node in a state graph is (species_index, 1-based local atom index).
_Node = tuple[int, int]


def _element(name: str, atom_type: str) -> str:
    """Perceive the chemical element from an atom name (fallback: GAFF type).

    mol2 atom names here are element-leading (``C1``, ``O2``, ``H10``); the GAFF
    atom-type first letter is the fallback (``c3``->C, ``os``->O).
    """
    for token in (name, atom_type):
        token = token.strip()
        for sym in _TWO_LETTER:
            if token[:2].capitalize() == sym:
                return sym
        if token and token[0].upper() in _ONE_LETTER:
            return token[0].upper()
    raise AlignmentError(
        f'cannot perceive element from name={name!r} type={atom_type!r}'
    )


class _State:
    """A parsed diabatic state: its species mol2s and their union graph."""

    def __init__(self, mols: list[_Mol2]):
        self.mols = mols
        # element per node and adjacency (as node-sets), keyed by (species, idx1)
        self.element: dict[_Node, str] = {}
        self.adj: dict[_Node, set[_Node]] = {}
        for si, m in enumerate(mols):
            for j, (nm, ty) in enumerate(zip(m.names, m.types, strict=True), start=1):
                node = (si, j)
                self.element[node] = _element(nm, ty)
                self.adj[node] = set()
        for si, m in enumerate(mols):
            for a, b, _order in m.bonds:
                self.adj[(si, a)].add((si, b))
                self.adj[(si, b)].add((si, a))

    def node_by_name(self, species: int, name: str) -> _Node:
        m = self.mols[species]
        for j, nm in enumerate(m.names, start=1):
            if nm == name:
                return (species, j)
        raise AlignmentError(
            f'atom {name!r} not found in species {species} '
            f'(available: {sorted(set(m.names))})'
        )

    def atom(self, node: _Node) -> dict:
        si, j = node
        m = self.mols[si]
        return dict(
            name=m.names[j - 1],
            type=m.types[j - 1],
            coord=m.coords[j - 1],
            charge=m.charges[j - 1],
        )


class DiabaticPairAligner:
    """Align two independent state parameterizations into an EVB diabatic pair.

    Args:
        reactant: mol2 path or list of paths for the reactant state's species.
        product: mol2 path or list of paths for the product state's species.
        transfers: One :class:`Transfer` per atom that moves between states.
        overrides: Optional explicit atom pins to disambiguate chemically
            symmetric atoms, as ``{(react_species, react_name): (prod_species,
            prod_name)}``. Rarely needed -- symmetric swaps are harmless because
            such atoms share type/charge -- but available for full control.

    Example:
        >>> pair = DiabaticPairAligner(
        ...     reactant=['reactant_glucose.mol2', 'reactant_PQQ.mol2'],
        ...     product=['product_glucose.mol2', 'product_PQQ.mol2'],
        ...     transfers=[Transfer(reactant_atom=(0, 'H7'), product_atom=(1, 'H4'))],
        ... ).align('build')
        >>> EVBBuilder(
        ...     out_path='evb',
        ...     reactant=pair.reactant_mol2,
        ...     product=pair.product_mol2,
        ...     protein='enzyme.pdb',
        ...     donor=pair.donor[0],
        ...     acceptor=pair.acceptor[0],
        ...     reactive=pair.reactive[0],
        ... ).build()
    """

    def __init__(
        self,
        reactant: PathLike | list[PathLike],
        product: PathLike | list[PathLike],
        transfers: list[Transfer],
        overrides: dict[tuple[int, str], tuple[int, str]] | None = None,
    ):
        self.reactant = _State([_parse_mol2(p) for p in _as_list(reactant)])
        self.product = _State([_parse_mol2(p) for p in _as_list(product)])
        if not transfers:
            raise AlignmentError('at least one Transfer is required')
        self.transfers = transfers
        self.overrides = overrides or {}

    # -- public API ---------------------------------------------------------

    def align(self, out_dir: PathLike, resname: str = 'LIG') -> AlignedPair:
        """Align the two states and write the paired mol2s.

        Args:
            out_dir: Directory for ``reactant_region.mol2`` / ``product_region.mol2``
                (created if absent).
            resname: Residue name for the shared unit.

        Returns:
            An :class:`AlignedPair`.

        Raises:
            AlignmentError: If the atom counts differ, the skeletons are not
                isomorphic (reaction reorganizes more than the transfers), the
                single-reactive-bond invariant fails, or charge is not conserved.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        r_transfer_nodes, p_transfer_nodes = self._resolve_transfer_nodes()
        node_map = self._correspondence(r_transfer_nodes, p_transfer_nodes)

        # Shared order = reactant species order, then local atom order.
        order = [n for n in self.reactant.adj]  # dict preserves insertion order
        names = self._assign_names(order)

        react_bonds = self._state_bonds(
            self.reactant, {n: i for i, n in enumerate(order)}
        )
        # product bonds map product-node -> slot via the inverse correspondence.
        prod_to_slot = {node_map[n]: i for i, n in enumerate(order)}
        prod_bonds = self._state_bonds(self.product, prod_to_slot)

        donor, acceptor, reactive = self._resolve_reactive_names(
            order, names, r_transfer_nodes, p_transfer_nodes, node_map
        )
        self._verify_invariant(
            names, react_bonds, prod_bonds, donor, acceptor, reactive
        )

        qr = self._write(
            out / 'reactant_region.mol2',
            order,
            names,
            react_bonds,
            self.reactant,
            node_from_slot=None,
            resname=resname,
        )
        qp = self._write(
            out / 'product_region.mol2',
            order,
            names,
            prod_bonds,
            self.product,
            node_from_slot=node_map,
            resname=resname,
        )
        if abs(qr - qp) > 0.05:
            raise AlignmentError(
                f'diabatic states differ in net charge ({qr:+.4f} vs {qp:+.4f}); '
                'the two states must describe the same nuclei -- check the inputs.'
            )

        name_map = {names[i]: self._audit(order[i]) for i in range(len(order))}
        pair = AlignedPair(
            reactant_mol2=out / 'reactant_region.mol2',
            product_mol2=out / 'product_region.mol2',
            donor=donor,
            acceptor=acceptor,
            reactive=reactive,
            name_map=name_map,
            reactant_charge=qr,
            product_charge=qp,
        )
        (out / 'alignment.json').write_text(
            json.dumps(
                {
                    'donor': donor,
                    'acceptor': acceptor,
                    'reactive': reactive,
                    'reactant_charge': qr,
                    'product_charge': qp,
                    'name_map': name_map,
                },
                indent=2,
            )
        )
        return pair

    # -- correspondence -----------------------------------------------------

    def _resolve_transfer_nodes(self) -> tuple[list[_Node], list[_Node]]:
        """Resolve each Transfer's reactant/product nodes (by name)."""
        r_nodes, p_nodes = [], []
        for t in self.transfers:
            r_nodes.append(self.reactant.node_by_name(*t.reactant_atom))
            p_nodes.append(self.product.node_by_name(*t.product_atom))
        return r_nodes, p_nodes

    def _correspondence(
        self, r_transfer: list[_Node], p_transfer: list[_Node]
    ) -> dict[_Node, _Node]:
        """Full reactant-node -> product-node map.

        Non-transferring atoms are matched by graph isomorphism on the two
        skeletons with the transferring atoms removed; transferring atoms are
        matched by their Transfer anchors. Overrides pin symmetric atoms.
        """
        import networkx as nx
        from networkx.algorithms import isomorphism

        r_skip, p_skip = set(r_transfer), set(p_transfer)

        # Build override "colors": paired reactant/product atoms get a unique tag
        # so VF2 is forced to map them together.
        r_tag: dict[_Node, int] = {}
        p_tag: dict[_Node, int] = {}
        for k, (r_key, p_key) in enumerate(self.overrides.items(), start=1):
            r_tag[self.reactant.node_by_name(*r_key)] = k
            p_tag[self.product.node_by_name(*p_key)] = k

        def build(state: _State, skip: set[_Node], tag: dict[_Node, int]) -> nx.Graph:
            g = nx.Graph()
            for node, el in state.element.items():
                if node in skip:
                    continue
                g.add_node(node, color=(el, tag.get(node)))
            for node, nbrs in state.adj.items():
                if node in skip:
                    continue
                for nb in nbrs:
                    if nb not in skip:
                        g.add_edge(node, nb)
            return g

        gr = build(self.reactant, r_skip, r_tag)
        gp = build(self.product, p_skip, p_tag)
        if gr.number_of_nodes() != gp.number_of_nodes():
            raise AlignmentError(
                f'non-transferring atom counts differ ({gr.number_of_nodes()} vs '
                f'{gp.number_of_nodes()}); the transfers do not account for the '
                'atom-count change between states.'
            )

        gm = isomorphism.GraphMatcher(
            gr, gp, node_match=lambda a, b: a['color'] == b['color']
        )
        if not gm.is_isomorphic():
            raise AlignmentError(
                'reactant/product skeletons are not isomorphic once the '
                'transferring atoms are removed; the reaction reorganizes the '
                'heavy-atom skeleton beyond the declared transfers (out of scope) '
                'or a Transfer is mis-specified.'
            )
        mapping = dict(gm.mapping)
        for rn, pn in zip(r_transfer, p_transfer, strict=True):
            mapping[rn] = pn
        return mapping

    # -- assembly helpers ---------------------------------------------------

    def _assign_names(self, order: list[_Node]) -> list[str]:
        """Element-leading unique names (C1..Cn, O1.., N1.., H1..) in slot order."""
        names: list[str] = []
        counts: dict[str, int] = {}
        for node in order:
            el = self.reactant.element[node]
            counts[el] = counts.get(el, 0) + 1
            nm = f'{el}{counts[el]}'
            if len(nm) > 4:
                raise AlignmentError(f'atom name {nm!r} exceeds 4 characters')
            names.append(nm)
        if len(set(names)) != len(names):
            raise AlignmentError('failed to generate unique atom names')
        return names

    @staticmethod
    def _state_bonds(
        state: _State, node_to_slot: dict[_Node, int]
    ) -> list[tuple[int, int, str]]:
        """Translate a state's intra-species bonds into shared-slot index pairs."""
        out = []
        for si, m in enumerate(state.mols):
            for a, b, order in m.bonds:
                out.append((node_to_slot[(si, a)], node_to_slot[(si, b)], order))
        return out

    def _resolve_reactive_names(self, order, names, r_transfer, p_transfer, node_map):
        slot_of = {n: i for i, n in enumerate(order)}
        p_slot_of = {node_map[n]: i for i, n in enumerate(order)}
        donor, acceptor, reactive = [], [], []
        for t, rn, pn in zip(self.transfers, r_transfer, p_transfer, strict=True):
            reactive.append(names[slot_of[rn]])
            d_node = (
                self.reactant.node_by_name(*t.donor)
                if t.donor
                else self._sole_neighbour(self.reactant, rn, 'donor')
            )
            a_node = (
                self.product.node_by_name(*t.acceptor)
                if t.acceptor
                else self._sole_neighbour(self.product, pn, 'acceptor')
            )
            donor.append(names[slot_of[d_node]])
            acceptor.append(names[p_slot_of[a_node]])
        return donor, acceptor, reactive

    @staticmethod
    def _sole_neighbour(state: _State, node: _Node, role: str) -> _Node:
        nbrs = state.adj[node]
        if len(nbrs) != 1:
            atom = state.atom(node)
            raise AlignmentError(
                f'cannot infer the {role} for transferring atom '
                f'{atom["name"]!r}: it has {len(nbrs)} bonds, not 1; specify the '
                f'{role} explicitly in the Transfer.'
            )
        return next(iter(nbrs))

    def _verify_invariant(
        self, names, react_bonds, prod_bonds, donor, acceptor, reactive
    ):
        """The only bond difference must be donor-reactive vs acceptor-reactive."""

        def pairs(bl):
            return {frozenset((names[a], names[b])) for a, b, _o in bl}

        only_r = pairs(react_bonds) - pairs(prod_bonds)
        only_p = pairs(prod_bonds) - pairs(react_bonds)
        expect_r = {frozenset((d, x)) for d, x in zip(donor, reactive, strict=True)}
        expect_p = {frozenset((a, x)) for a, x in zip(acceptor, reactive, strict=True)}
        if only_r != expect_r or only_p != expect_p:
            raise AlignmentError(
                'diabatic bond-difference check failed. Expected the only changed '
                f'bonds to be reactant {sorted(map(sorted, expect_r))} and product '
                f'{sorted(map(sorted, expect_p))}, but found reactant-only '
                f'{sorted(map(sorted, only_r))} and product-only '
                f'{sorted(map(sorted, only_p))}. The correspondence is wrong '
                '(try an override) or the reaction is out of scope.'
            )

    def _audit(self, node: _Node) -> list[str]:
        si, j = node
        return ['reactant', self.reactant.mols[si].names[j - 1]]

    def _write(self, path, order, names, bonds, state, node_from_slot, resname):
        """Write one aligned mol2; returns the total charge."""
        total = 0.0
        lines = [
            '@<TRIPOS>MOLECULE',
            resname,
            f'{len(order)} {len(bonds)} 1 0 0',
            'SMALL',
            'USER_CHARGES',
            '',
            '@<TRIPOS>ATOM',
        ]
        for i, node in enumerate(order):
            src = node_from_slot[node] if node_from_slot is not None else node
            a = state.atom(src)
            x, y, z = a['coord']
            total += a['charge']
            lines.append(
                f'{i + 1:>7} {names[i]:<6} {x:>10.4f} {y:>10.4f} {z:>10.4f} '
                f'{a["type"]:<6} 1 {resname} {a["charge"]:>10.6f}'
            )
        lines.append('@<TRIPOS>BOND')
        for i, (a, b, order_str) in enumerate(bonds, start=1):
            lines.append(f'{i:>6} {a + 1:>6} {b + 1:>6} {order_str}')
        lines.append('')
        Path(path).write_text('\n'.join(lines))
        return total


def _as_list(x: PathLike | list[PathLike]) -> list[PathLike]:
    if isinstance(x, (str, Path)):
        return [x]
    return list(x)
