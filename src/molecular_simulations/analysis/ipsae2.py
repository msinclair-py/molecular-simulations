"""ipSAE2: directional PAE-based interface scores for any molecule type.

Port of the scoring algorithm from ipSAE2 v0.1.5-alpha by the Dunbrack
Laboratory (https://github.com/DunbrackLab/IPSAE2, MIT license). Unlike the
original ipSAE, proteins, nucleic acids, ligands and ions are all scored as
molecules: the predictor's token order (atom-tokenized ligands and modified
residues) is reconstructed from the structure and checked against the PAE
dimension, and the final pair score is chosen by molecule type.

Deviations from upstream:
    * A non-polymer chain with several residues (e.g. a glycan) is scored as
      one ligand molecule instead of raising an error.
    * PAE is read from ``.npz`` only, and predictor-native ipTM values are
      not imported.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

PathLike = Path | str

PROT = frozenset(
    [
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'CYS',
        'GLN',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LEU',
        'LYS',
        'MET',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
        'SEC',
        'PYL',
    ]
)
NUC = frozenset(['A', 'C', 'G', 'U', 'I', 'DA', 'DC', 'DG', 'DT', 'DI', 'DU'])
IONS = frozenset(
    [
        'LI',
        'NA',
        'K',
        'RB',
        'CS',
        'MG',
        'CA',
        'SR',
        'BA',
        'MN',
        'FE',
        'FE2',
        'FE3',
        'CO',
        'NI',
        'CU',
        'CU1',
        'CU2',
        'ZN',
        'CD',
        'AL',
        'GA',
        'HG',
        'PB',
        'F',
        'CL',
        'BR',
        'IOD',
        'I',
    ]
)
WATER = frozenset(['HOH', 'WAT', 'DOD'])
TYPE_ORDER = {'prot': 0, 'nuc': 1, 'lig': 2, 'ion': 3}

PAE_METADATA_KEYS = (
    'frame_mask',
    'token_has_frame',
    'asym_id',
    'token_asym_id',
)

_CIF_TOKEN = re.compile(
    r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+"""
)


@dataclass
class AtomTable:
    """Heavy atoms of a structure model, in file order.

    Attributes:
        record: ``ATOM`` or ``HETATM`` per atom.
        chain: Chain ID per atom (``label_asym_id`` for mmCIF).
        resnum: Residue number per atom, as a string.
        resname: Residue name per atom.
        atom_name: Atom name per atom (CIF quotes removed).
        element: Upper-case element symbol per atom.
        xyz: Coordinates, shape ``(n_atoms, 3)``.
    """

    record: np.ndarray
    chain: np.ndarray
    resnum: np.ndarray
    resname: np.ndarray
    atom_name: np.ndarray
    element: np.ndarray
    xyz: np.ndarray

    def __len__(self) -> int:
        """Return the number of atoms."""
        return len(self.chain)

    def residues(self) -> 'OrderedDict[tuple[str, str, str], list[int]]':
        """Group atom indices by ``(chain, resnum, resname)``.

        Returns:
            Ordered mapping from residue key to atom indices, in order of
            first appearance.
        """
        groups: OrderedDict[tuple[str, str, str], list[int]] = OrderedDict()
        for i, key in enumerate(
            zip(self.chain, self.resnum, self.resname, strict=True)
        ):
            groups.setdefault(key, []).append(i)
        return groups


@dataclass
class TokenTable:
    """Predictor tokens reconstructed to match the PAE matrix.

    Attributes:
        chain: Chain ID per token.
        resnum: Residue number per token.
        resname: Residue name per token.
        atom_name: Name of the token's representative atom.
        mol_type: Molecule type per token (prot, nuc, lig or ion).
        tokenization: ``atom`` or ``residue`` per token.
        unit_key: Scoring unit per token. Atom tokens of one modified
            polymer residue share a unit; ligand tokens are their own unit.
        rep_atom: Index of each token's representative atom.
        atom_token: Token index for every atom in the AtomTable.
        policy: Tokenization policy that matched the PAE dimension.
        frame_valid: Whether each token defines an alignment frame.
    """

    chain: np.ndarray
    resnum: np.ndarray
    resname: np.ndarray
    atom_name: np.ndarray
    mol_type: np.ndarray
    tokenization: np.ndarray
    unit_key: np.ndarray
    rep_atom: np.ndarray
    atom_token: np.ndarray
    policy: str = ''
    frame_valid: np.ndarray = field(default_factory=lambda: np.array([], bool))

    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(self.chain)


@dataclass
class DirectionResult:
    """ipSAE2 score for one direction (align on one molecule, score other).

    Attributes:
        score: Directional ipSAE2, or None without a valid frame.
        ipae: Mean PAE of the scored units in the maximizing frame.
        n0: Number of scored units passing the PAE cutoff.
        d0: d0 used in the maximizing frame.
        n_contact: Scored units in heavy-atom contact with the partner.
        align_token: Label of the maximizing alignment token.
    """

    score: float | None = None
    ipae: float | None = None
    n0: int | None = None
    d0: float | None = None
    n_contact: int | None = None
    align_token: str | None = None


def _split_cif_row(line: str) -> list[str]:
    """Split an mmCIF data row, removing quotes around values like "C1'"."""
    values = []
    for token in _CIF_TOKEN.findall(line):
        if len(token) >= 2 and token[0] == token[-1] and token[0] in '\'"':
            token = token[1:-1]
        values.append(token)
    return values


def _read_pdb_rows(path: Path) -> list[tuple]:
    rows = []
    with open(path) as f:
        for line in f:
            record = line[:6].strip()
            if record == 'ENDMDL':
                break
            if record not in ('ATOM', 'HETATM'):
                continue
            if line[16:17].strip() not in ('', 'A', '1'):
                continue
            element = line[76:78].strip() if len(line) >= 78 else ''
            rows.append(
                (
                    record,
                    line[21:22].strip() or '_',
                    line[22:27].strip() or '1',
                    line[17:20].strip(),
                    line[12:16].strip(),
                    element,
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return rows


def _read_cif_rows(path: Path) -> list[tuple]:
    fields: dict[str, int] = {}
    rows = []
    first_model = None
    with open(path) as f:
        for line in f:
            if line.startswith('_atom_site.'):
                fields[line.strip().split('.', 1)[1].strip()] = len(fields)
                continue
            if not fields or not line.startswith(('ATOM', 'HETATM')):
                continue
            values = _split_cif_row(line)

            def get(name: str, values: list[str] = values) -> str:
                idx = fields.get(name)
                if idx is None:
                    return ''
                value = values[idx]
                return '' if value in ('.', '?') else value

            model = get('pdbx_PDB_model_num')
            if first_model is None:
                first_model = model
            elif model != first_model:
                break
            if get('label_alt_id') not in ('', 'A', '1'):
                continue
            rows.append(
                (
                    get('group_PDB'),
                    get('label_asym_id') or get('auth_asym_id') or '_',
                    get('label_seq_id') or get('auth_seq_id') or '1',
                    get('label_comp_id'),
                    get('label_atom_id'),
                    get('type_symbol'),
                    float(get('Cartn_x')),
                    float(get('Cartn_y')),
                    float(get('Cartn_z')),
                )
            )
    return rows


def _infer_element(atom_name: str, resname: str) -> str:
    name = atom_name.upper().lstrip('0123456789')
    if name.startswith(('H', 'D')):
        return 'H'
    if resname.upper() in IONS:
        return resname.upper()
    return name[:1]


def read_atoms(path: PathLike) -> AtomTable:
    """Read heavy atoms from a PDB or mmCIF model.

    Waters, hydrogens and alternate locations other than the first are
    dropped, and only the first model is read.

    Args:
        path: Path to a ``.pdb`` or ``.cif``/``.mmcif`` file.

    Returns:
        AtomTable of the model's heavy atoms.

    Raises:
        ValueError: If the file contains no atoms.
    """
    path = Path(path)
    if path.suffix.lower() == '.pdb':
        rows = _read_pdb_rows(path)
    else:
        rows = _read_cif_rows(path)

    kept = []
    for row in rows:
        record, chain, resnum, resname, atom_name, element = row[:6]
        if resname in WATER:
            continue
        element = (element or _infer_element(atom_name, resname)).upper()
        if element in ('H', 'D'):
            continue
        kept.append((record, chain, resnum, resname, atom_name, element, *row[6:]))
    if not kept:
        raise ValueError(f'No coordinate atoms in {path}')

    columns = list(zip(*kept, strict=True))
    return AtomTable(
        record=np.array(columns[0], dtype=object),
        chain=np.array(columns[1], dtype=object),
        resnum=np.array(columns[2], dtype=object),
        resname=np.array(columns[3], dtype=object),
        atom_name=np.array(columns[4], dtype=object),
        element=np.array(columns[5], dtype=object),
        xyz=np.array(columns[6:9], dtype=float).T,
    )


def classify_molecules(atoms: AtomTable) -> dict[str, str]:
    """Assign each chain a molecule type: prot, nuc, lig or ion.

    A chain with any canonical amino acid is ``prot`` and one with any
    canonical nucleotide is ``nuc``. A chain of a single ion atom is
    ``ion``; every other non-polymer chain, including multi-residue chains
    such as glycans, is ``lig``.

    Args:
        atoms: Parsed heavy atoms.

    Returns:
        Mapping of chain ID to molecule type, in order of first appearance.

    Raises:
        ValueError: If a chain mixes protein and nucleic acid residues, or
            an ion shares a protein chain.
    """
    residues = atoms.residues()
    chain_residues: OrderedDict[str, list[tuple[str, str, str]]] = OrderedDict()
    for key in residues:
        chain_residues.setdefault(key[0], []).append(key)

    chain_types = {}
    for chain, keys in chain_residues.items():
        names = [key[2] for key in keys]
        has_prot = any(name in PROT for name in names)
        has_nuc = any(name in NUC for name in names)
        if has_prot and has_nuc:
            raise ValueError(f'Chain {chain} mixes protein and nucleic acid residues')
        if has_prot:
            chain_types[chain] = 'prot'
        elif has_nuc:
            chain_types[chain] = 'nuc'
        else:
            idx = residues[keys[0]]
            is_ion = (
                len(keys) == 1
                and len(idx) == 1
                and (names[0].upper() in IONS or atoms.element[idx[0]] in IONS)
            )
            chain_types[chain] = 'ion' if is_ion else 'lig'

    # Folding an ion into a protein would silently misassign its token.
    for key, idx in residues.items():
        if chain_types[key[0]] != 'prot' or key[2] in PROT:
            continue
        if (
            len(idx) == 1
            and atoms.record[idx[0]] == 'HETATM'
            and atoms.element[idx[0]] in IONS
        ):
            raise ValueError(
                f'Ion {key[2]} shares protein chain {key[0]}; use unique '
                'per-molecule chain IDs'
            )
    return chain_types


def read_pae_metadata(path: PathLike) -> dict[str, np.ndarray]:
    """Read optional token metadata stored next to the PAE in an npz file.

    Args:
        path: Path to the PAE ``.npz`` file.

    Returns:
        Arrays for any of :data:`PAE_METADATA_KEYS` present in the file.
    """
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in PAE_METADATA_KEYS if k in data}


def _canonical(mol_type: str) -> frozenset:
    return PROT if mol_type == 'prot' else NUC


def _build_tokens(
    atoms: AtomTable,
    residues: 'OrderedDict[tuple[str, str, str], list[int]]',
    chain_types: dict[str, str],
    mod_mode: str,
    lig_mode: str,
) -> TokenTable:
    chunks: list[tuple[tuple[str, str, str], str, list[int], bool]] = []
    lig_chain_chunk: dict[str, int] = {}
    for key, idx in residues.items():
        chain, _, resname = key
        mol_type = chain_types[chain]
        if mol_type in ('lig', 'ion'):
            if lig_mode == 'atom':
                chunks.extend((key, mol_type, [i], True) for i in idx)
            elif lig_mode == 'residue':
                chunks.append((key, mol_type, list(idx), False))
            elif chain in lig_chain_chunk:
                chunks[lig_chain_chunk[chain]][2].extend(idx)
            else:
                lig_chain_chunk[chain] = len(chunks)
                chunks.append((key, mol_type, list(idx), False))
        elif resname not in _canonical(mol_type) and mod_mode == 'atom':
            chunks.extend((key, mol_type, [i], True) for i in idx)
        else:
            chunks.append((key, mol_type, list(idx), False))

    rep_names = {'prot': ('CA', 'C', 'N'), 'nuc': ("C3'", 'C3*', 'P')}
    atom_token = np.empty(len(atoms), dtype=int)
    rep_atom = np.empty(len(chunks), dtype=int)
    columns: dict[str, list] = {
        k: []
        for k in (
            'chain',
            'resnum',
            'resname',
            'atom_name',
            'mol_type',
            'tokenization',
            'unit_key',
        )
    }
    for t, (key, mol_type, idx, atom_tokenized) in enumerate(chunks):
        chain, resnum, resname = key
        rep = idx[0]
        for name in rep_names.get(mol_type, ()):
            match = [i for i in idx if atoms.atom_name[i] == name]
            if match:
                rep = match[0]
                break
        single = len(idx) == 1 and (atom_tokenized or mol_type in ('lig', 'ion'))
        if mol_type in ('prot', 'nuc'):
            unit = f'{chain}:res:{resnum}:{resname}'
        else:
            unit = f'{chain}:token:{t + 1}'
        columns['chain'].append(chain)
        columns['resnum'].append(resnum)
        columns['resname'].append(resname)
        columns['atom_name'].append(atoms.atom_name[rep])
        columns['mol_type'].append(mol_type)
        columns['tokenization'].append('atom' if single else 'residue')
        columns['unit_key'].append(unit)
        rep_atom[t] = rep
        atom_token[idx] = t

    return TokenTable(
        **{k: np.array(v, dtype=object) for k, v in columns.items()},
        rep_atom=rep_atom,
        atom_token=atom_token,
    )


def _metadata_chains(metadata: dict[str, np.ndarray], n: int) -> list | None:
    for key in ('token_asym_id', 'asym_id'):
        values = metadata.get(key)
        if values is not None and values.ndim == 1 and len(values) == n:
            return [str(v) for v in values]
    return None


def _chains_match(token_chains: np.ndarray, reference: list) -> bool:
    """Check token chains against numeric per-token chain assignments."""
    seen = list(dict.fromkeys(reference))
    chains = list(dict.fromkeys(token_chains))
    if len(seen) != len(chains):
        return False
    expected = dict(zip(seen, chains, strict=True))
    return [expected[c] for c in reference] == list(token_chains)


def infer_tokens(
    atoms: AtomTable,
    chain_types: dict[str, str],
    n_tokens: int,
    metadata: dict[str, np.ndarray] | None = None,
) -> TokenTable:
    """Reconstruct the predictor's token order to match the PAE matrix.

    Canonical residues are one token each. Modified residues may be atom-
    or residue-tokenized and ligands atom-, residue- or chain-tokenized;
    every combination whose token count equals ``n_tokens`` (and agrees
    with any per-token chain metadata) is a candidate.

    Args:
        atoms: Parsed heavy atoms.
        chain_types: Output of :func:`classify_molecules`.
        n_tokens: PAE matrix dimension.
        metadata: Optional per-token arrays from :func:`read_pae_metadata`.

    Returns:
        The unique matching TokenTable.

    Raises:
        ValueError: If no tokenization matches, or several distinct ones do.
    """
    metadata = metadata or {}
    residues = atoms.residues()
    n_canonical = mod_count = mod_atoms = lig_count = lig_atoms = 0
    lig_chains = set()
    for (chain, _, resname), idx in residues.items():
        mol_type = chain_types[chain]
        if mol_type in ('lig', 'ion'):
            lig_count += 1
            lig_atoms += len(idx)
            lig_chains.add(chain)
        elif resname in _canonical(mol_type):
            n_canonical += 1
        else:
            mod_count += 1
            mod_atoms += len(idx)

    mod_tokens = {'atom': mod_atoms, 'residue': mod_count}
    lig_tokens = {'atom': lig_atoms, 'residue': lig_count, 'chain': len(lig_chains)}
    reference = _metadata_chains(metadata, n_tokens)

    descriptions = []
    candidates: dict[tuple[str, str], TokenTable] = {}
    for mod_mode, n_mod in mod_tokens.items():
        for lig_mode, n_lig in lig_tokens.items():
            estimate = n_canonical + n_mod + n_lig
            descriptions.append(
                f'modified={mod_mode},ligand={lig_mode}: {estimate} tokens'
            )
            if estimate != n_tokens:
                continue
            # Policies for absent PTMs or ligands are identical.
            effective = (
                mod_mode if mod_count else 'none',
                lig_mode if lig_count else 'none',
            )
            if effective in candidates:
                continue
            tokens = _build_tokens(atoms, residues, chain_types, mod_mode, lig_mode)
            if reference is not None and not _chains_match(tokens.chain, reference):
                continue
            candidates[effective] = tokens

    if not candidates:
        raise ValueError(
            f'No token mapping matches the {n_tokens}-token PAE matrix: '
            + '; '.join(descriptions)
        )
    signatures = {
        tuple(zip(t.chain, t.resnum, t.resname, t.atom_name, strict=True))
        for t in candidates.values()
    }
    if len(signatures) > 1:
        raise ValueError(
            f'Ambiguous token order: {len(signatures)} count-compatible '
            'mappings. Provide per-token chain metadata in the PAE file.'
        )
    (mod_mode, lig_mode), tokens = next(iter(candidates.items()))
    tokens.policy = f'modified={mod_mode};ligand={lig_mode}'
    return tokens


def frame_validity(
    atoms: AtomTable,
    tokens: TokenTable,
    metadata: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Decide which tokens define a valid alignment frame.

    A predictor-supplied ``token_has_frame``/``frame_mask`` is used when
    present. Otherwise residue tokens need non-collinear N/CA/C (protein)
    or C1'/C3'/C4' (nucleotide) atoms, and other tokens need two
    non-collinear neighbouring atoms in the same residue. Ions never
    define a frame.

    Args:
        atoms: Parsed heavy atoms.
        tokens: Output of :func:`infer_tokens`.
        metadata: Optional per-token arrays from :func:`read_pae_metadata`.

    Returns:
        Boolean array with one entry per token.
    """
    not_ion = tokens.mol_type != 'ion'
    for key in ('token_has_frame', 'frame_mask'):
        mask = (metadata or {}).get(key)
        if mask is not None and mask.ndim == 1 and len(mask) == len(tokens):
            return mask.astype(bool) & not_ion

    residues = atoms.residues()
    required = {'prot': ('N', 'CA', 'C'), 'nuc': ("C1'", "C3'", "C4'")}
    valid = np.zeros(len(tokens), dtype=bool)
    for t in range(len(tokens)):
        mol_type = tokens.mol_type[t]
        if mol_type == 'ion':
            continue
        idx = residues[(tokens.chain[t], tokens.resnum[t], tokens.resname[t])]
        if tokens.tokenization[t] == 'residue' and mol_type in required:
            positions = {atoms.atom_name[i]: atoms.xyz[i] for i in idx}
            if not all(name in positions for name in required[mol_type]):
                continue
            p0, p1, p2 = (positions[name] for name in required[mol_type])
        else:
            p0 = atoms.xyz[tokens.rep_atom[t]]
            xyz = atoms.xyz[idx]
            dist = np.linalg.norm(xyz - p0, axis=1)
            others = xyz[dist > 1e-5]
            if len(others) < 2:
                continue
            nearest = np.argsort(dist[dist > 1e-5])[:2]
            p1, p2 = others[nearest[0]], others[nearest[1]]
        valid[t] = bool(np.linalg.norm(np.cross(p1 - p0, p2 - p0)) > 1e-4)
    return valid


@dataclass
class MoleculeGeometry:
    """Heavy atoms of one molecule prepared for contact queries."""

    xyz: np.ndarray
    unit_ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    tree: cKDTree


def molecule_geometry(
    atoms: AtomTable, tokens: TokenTable, chain: str
) -> MoleculeGeometry:
    """Collect one chain's heavy atoms and their scoring units.

    Args:
        atoms: Parsed heavy atoms.
        tokens: Token table mapping atoms to scoring units.
        chain: Chain ID.

    Returns:
        MoleculeGeometry with a KD-tree over the chain's atoms.
    """
    mask = atoms.chain == chain
    xyz = atoms.xyz[mask]
    if not len(xyz):
        raise ValueError(f'Molecule {chain} has no heavy-atom coordinates')
    return MoleculeGeometry(
        xyz=xyz,
        unit_ids=tokens.unit_key[tokens.atom_token[mask]],
        lo=xyz.min(axis=0),
        hi=xyz.max(axis=0),
        tree=cKDTree(xyz),
    )


def contact_units(
    a: MoleculeGeometry, b: MoleculeGeometry, cutoff: float = 5.0
) -> tuple[set, set]:
    """Find scoring units of each molecule within ``cutoff`` of the other.

    Args:
        a: First molecule.
        b: Second molecule.
        cutoff: Inclusive heavy-atom distance cutoff in Angstroms.

    Returns:
        Tuple of (units in ``a`` touching ``b``, units in ``b`` touching
        ``a``).
    """
    if np.any(a.hi + cutoff < b.lo) or np.any(b.hi + cutoff < a.lo):
        return set(), set()
    # nextafter keeps the boundary distance inclusive.
    r = float(np.nextafter(float(cutoff), np.inf))
    da, _ = b.tree.query(a.xyz, distance_upper_bound=r)
    if not np.isfinite(da).any():
        return set(), set()
    db, _ = a.tree.query(b.xyz, distance_upper_bound=r)
    return set(a.unit_ids[np.isfinite(da)]), set(b.unit_ids[np.isfinite(db)])


class IpSAE2Calculator:
    """Compute directional ipSAE2 scores for all contacting molecule pairs.

    For each pair, ``12`` aligns on the first molecule and scores the
    second. Each direction takes the maximum over valid alignment frames of
    the mean TM-style kernel over scored units passing the PAE filter. The
    final score is the larger direction for same-type pairs, otherwise the
    direction aligned on the more complex molecule (prot > nuc > lig > ion).

    Attributes:
        scores: Polars DataFrame of scores after compute_scores().

    Args:
        pae_cutoff: PAE cutoff (inclusive) for scored protein and nucleic
            acid units. Ligand and ion units have no cutoff. Defaults to 10.
        contact_cutoff: Heavy-atom distance defining a contacting pair.
            Defaults to 5.
        fixed_d0: d0 for scored nucleic acid, ligand and ion units.
            Defaults to 2.
        protein_min_d0: Floor of the length-dependent protein d0.
            Defaults to 1.
        all_pairs: Also score pairs without a heavy-atom contact.

    Example:
        >>> calc = IpSAE2Calculator()
        >>> calc.compute_scores(pae, atoms, tokens)
        >>> print(calc.scores)
    """

    DIRECTIONAL = ('ipSAE2', 'iPAE', 'n0', 'd0', 'n_contact', 'align_token')

    def __init__(
        self,
        pae_cutoff: float = 10.0,
        contact_cutoff: float = 5.0,
        fixed_d0: float = 2.0,
        protein_min_d0: float = 1.0,
        all_pairs: bool = False,
    ):
        """Initialize the IpSAE2Calculator.

        Args:
            pae_cutoff: PAE cutoff for scored protein/nucleic acid units.
            contact_cutoff: Heavy-atom contact distance.
            fixed_d0: d0 for scored nucleic acid, ligand and ion units.
            protein_min_d0: Floor of the protein d0.
            all_pairs: Also score non-contacting pairs.
        """
        self.pae_cutoff = pae_cutoff
        self.contact_cutoff = contact_cutoff
        self.fixed_d0 = fixed_d0
        self.protein_min_d0 = protein_min_d0
        self.all_pairs = all_pairs

    def compute_scores(
        self,
        pae: np.ndarray,
        atoms: AtomTable,
        tokens: TokenTable,
        skip_chains: set[str] | None = None,
    ) -> pl.DataFrame:
        """Score every molecule pair.

        Args:
            pae: Full PAE matrix, one row per token.
            atoms: Parsed heavy atoms.
            tokens: Token table with ``frame_valid`` populated.
            skip_chains: Chains to leave out of the reported pairs. Their
                tokens still count toward the PAE dimension.

        Returns:
            DataFrame with one row per scored pair.

        Raises:
            ValueError: If the token and PAE dimensions disagree.
        """
        if len(tokens) != len(pae):
            raise ValueError(
                f'PAE has {len(pae)} tokens but structure has {len(tokens)}'
            )
        skip_chains = skip_chains or set()
        types = dict(zip(tokens.chain, tokens.mol_type, strict=True))
        molecules = sorted(
            (c for c in dict.fromkeys(tokens.chain) if c not in skip_chains),
            key=lambda c: TYPE_ORDER[types[c]],
        )
        token_idx = {c: np.flatnonzero(tokens.chain == c) for c in molecules}
        geometry = {c: molecule_geometry(atoms, tokens, c) for c in molecules}

        rows = []
        for i, chain1 in enumerate(molecules):
            for chain2 in molecules[i + 1 :]:
                type1, type2 = types[chain1], types[chain2]
                if type1 == 'ion' and type2 == 'ion':
                    continue
                c1, c2 = contact_units(
                    geometry[chain1], geometry[chain2], self.contact_cutoff
                )
                if not (c1 and c2) and not self.all_pairs:
                    continue
                d12 = self.score_direction(
                    pae, tokens, token_idx[chain1], token_idx[chain2], type2, c2
                )
                d21 = self.score_direction(
                    pae, tokens, token_idx[chain2], token_idx[chain1], type1, c1
                )
                use_21 = (
                    type1 == type2
                    and d21.score is not None
                    and (d12.score is None or d21.score > d12.score)
                )
                if use_21:
                    score, source = d21.score, '21'
                else:
                    score = d12.score
                    source = '12' if score is not None else None

                row = {
                    'chain1': chain1,
                    'chain2': chain2,
                    'type1': type1,
                    'type2': type2,
                    'ipSAE2': score,
                    'ipSAE2_source': source,
                }
                for suffix, result in (('12', d12), ('21', d21)):
                    row[f'ipSAE2_{suffix}'] = result.score
                    row[f'ipSAE2_iPAE_{suffix}'] = result.ipae
                    row[f'ipSAE2_n0_{suffix}'] = result.n0
                    row[f'ipSAE2_d0_{suffix}'] = result.d0
                    row[f'ipSAE2_n_contact_{suffix}'] = result.n_contact
                    row[f'ipSAE2_align_token_{suffix}'] = result.align_token
                rows.append(row)

        self.scores = pl.DataFrame(rows, schema=self.schema())
        return self.scores

    @staticmethod
    def schema() -> dict[str, pl.DataType]:
        """Return the polars schema of the score table."""
        schema = {
            'chain1': pl.String,
            'chain2': pl.String,
            'type1': pl.String,
            'type2': pl.String,
            'ipSAE2': pl.Float64,
            'ipSAE2_source': pl.String,
        }
        for suffix in ('12', '21'):
            schema[f'ipSAE2_{suffix}'] = pl.Float64
            schema[f'ipSAE2_iPAE_{suffix}'] = pl.Float64
            schema[f'ipSAE2_n0_{suffix}'] = pl.Int64
            schema[f'ipSAE2_d0_{suffix}'] = pl.Float64
            schema[f'ipSAE2_n_contact_{suffix}'] = pl.Int64
            schema[f'ipSAE2_align_token_{suffix}'] = pl.String
        return schema

    def score_direction(
        self,
        pae: np.ndarray,
        tokens: TokenTable,
        aligned_idx: np.ndarray,
        scored_idx: np.ndarray,
        scored_type: str,
        scored_contact_units: set,
    ) -> DirectionResult:
        """Score one direction: max over frames of the mean over scored units.

        Args:
            pae: Full PAE matrix.
            tokens: Token table with ``frame_valid`` populated.
            aligned_idx: Token indices of the aligned molecule.
            scored_idx: Token indices of the scored molecule.
            scored_type: Molecule type of the scored molecule.
            scored_contact_units: Scored units touching the aligned molecule.

        Returns:
            DirectionResult for the maximizing frame, or an all-None result
            if the aligned molecule has no valid frame.

        Raises:
            ValueError: If a scored ligand/ion has non-finite PAE.
        """
        frames = aligned_idx[tokens.frame_valid[aligned_idx]]
        if not len(frames):
            return DirectionResult()

        use_cutoff = scored_type in ('prot', 'nuc')
        sub = np.asarray(pae[np.ix_(frames, scored_idx)], dtype=float)
        if use_cutoff:
            mask = np.isfinite(sub) & (sub <= self.pae_cutoff)
        elif not np.isfinite(sub).all():
            raise ValueError('Non-finite ligand/ion PAE: cannot score the molecule')
        else:
            mask = np.ones_like(sub, dtype=bool)

        units = tokens.unit_key[scored_idx]
        unique_units, unit_of = np.unique(units, return_inverse=True)
        if len(unique_units) == len(units):
            n0 = mask.sum(axis=1)
            unit_pae_sum = None
        else:
            # Atom-tokenized modified residues: average each residue's
            # passing atom tokens first so every residue weighs the same.
            n_units = len(unique_units)
            onehot = np.zeros((len(units), n_units))
            onehot[np.arange(len(units)), unit_of] = 1.0
            n_pass = mask.astype(float) @ onehot
            unit_pae_sum = np.where(mask, sub, 0.0) @ onehot
            n0 = (n_pass > 0).sum(axis=1)

        d0 = self._d0(n0, scored_type)
        tm = 1.0 / (1.0 + (sub / d0[:, None]) ** 2)
        if unit_pae_sum is None:
            unit_tm = np.where(mask, tm, 0.0)
            unit_pae = np.where(mask, sub, 0.0)
            unit_pass = mask
        else:
            unit_tm_sum = np.where(mask, tm, 0.0) @ onehot
            with np.errstate(invalid='ignore', divide='ignore'):
                unit_tm = np.where(n_pass > 0, unit_tm_sum / n_pass, 0.0)
                unit_pae = np.where(n_pass > 0, unit_pae_sum / n_pass, 0.0)
            unit_pass = n_pass > 0

        means = np.divide(
            unit_tm.sum(axis=1),
            n0,
            out=np.zeros(len(frames)),
            where=n0 > 0,
        )
        best = int(np.argmax(means))
        best_n0 = int(n0[best])
        passing = unit_pass[best]
        if unit_pae_sum is None:
            passing_units = units[passing]
        else:
            passing_units = unique_units[passing]
        frame = frames[best]
        return DirectionResult(
            score=float(means[best]),
            ipae=float(unit_pae[best, passing].mean()) if best_n0 else None,
            n0=best_n0,
            d0=float(d0[best]),
            n_contact=int(sum(u in scored_contact_units for u in passing_units)),
            align_token=(
                f'{tokens.resname[frame]}:{tokens.resnum[frame]}:'
                f'{tokens.atom_name[frame]}'
            ),
        )

    def _d0(self, n0: np.ndarray, scored_type: str) -> np.ndarray:
        if scored_type != 'prot':
            return np.full(len(n0), float(self.fixed_d0))
        n = np.maximum(n0, 1).astype(float)
        return np.maximum(self.protein_min_d0, 1.24 * np.cbrt(n - 15.0) - 1.8)
