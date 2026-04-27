# ruff: noqa: N801, N999
"""Interface prediction Score from Aligned Errors (ipSAE) module.

This module computes interaction prediction scores from pLDDT and PAE data,
adapted from https://doi.org/10.1101/2025.02.10.637595. Supports outputs
from structure prediction tools like Boltz and AlphaFold.
"""

from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

PathLike = Path | str
OptPath = Path | str | None

class ipSAE:
    """Compute interaction prediction Score from Aligned Errors.

    Computes various model quality scores including pDockQ, pDockQ2,
    LIS, ipTM, and ipSAE for structure predictions.

    Attributes:
        parser: ModelParser instance for structure file.
        plddt_file: Path to pLDDT data file.
        pae_file: Path to PAE data file.
        path: Output directory path.
        scores: Polars DataFrame of computed scores after run().

    Args:
        structure_file: Path to PDB/CIF model file.
        plddt_file: Path to pLDDT numpy file (.npz with 'plddt' key).
        pae_file: Path to PAE numpy file (.npz with 'pae' key).
        out_path: Output directory path. If None, uses parent directory
            of plddt_file.

    Example:
        >>> scorer = ipSAE("model.pdb", "plddt.npz", "pae.npz")
        >>> scorer.run()
        >>> print(scorer.scores)
    """

    def __init__(
        self,
        structure_file: PathLike,
        plddt_file: PathLike,
        pae_file: PathLike,
        out_path: OptPath = None,
        skip_chains: set[str] | list[str] | None = None,
    ):
        """Initialize the ipSAE scorer.

        Args:
            structure_file: Path to structure file.
            plddt_file: Path to pLDDT data file.
            pae_file: Path to PAE data file.
            out_path: Output directory path.
            skip_chains: Chain IDs to exclude from scoring (e.g. glycan
                or ligand chains). Their pLDDT/PAE tokens are inferred
                from remaining tokens and dropped before scoring.
        """
        self.parser = ModelParser(structure_file, skip_chains=skip_chains)
        self.plddt_file = Path(plddt_file)
        self.pae_file = Path(pae_file)

        self.path = Path(out_path) if out_path is not None else self.plddt_file.parent
        self.path.mkdir(exist_ok=True)

    def parse_structure_file(self) -> None:
        """Parse the structure file and extract relevant details.

        Runs the parser to read the structure file and classifies
        chains as protein or nucleic acid.
        """
        self.parser.parse_structure_file()
        self.parser.classify_chains()
        self.coordinates = np.vstack([res['coor'] for res in self.parser.residues])
        if self.parser.cb_residues:
            self.cb_coordinates = np.vstack(
                [res['coor'] for res in self.parser.cb_residues]
            )
        else:
            self.cb_coordinates = self.coordinates

    def prepare_scorer(self) -> None:
        """Initialize the ScoreCalculator for computing scores.

        Creates a ScoreCalculator instance with chain information
        extracted from the parsed structure.
        """
        chains = np.array(self.parser.chains)
        chain_types = self.parser.chain_types

        self.scorer = ScoreCalculator(
            chains=chains,
            chain_pair_type=chain_types,
            n_residues=len(self.parser.residues),
        )

    def run(self) -> None:
        """Execute the complete ipSAE scoring workflow.

        Parses structure, computes distogram, loads pLDDT and PAE data,
        runs the scorer, and saves results.
        """
        self.parse_structure_file()

        distances = cdist(self.cb_coordinates, self.cb_coordinates)
        pLDDT = self.load_pLDDT_file()
        PAE = self.load_PAE_file()

        n_residues = len(self.parser.residues)
        if pLDDT.shape[0] != n_residues or PAE.shape[0] != n_residues:
            self.parser.build_protein_token_indices(pLDDT.shape[0])
            indices = np.array(self.parser.protein_token_indices, dtype=int)
            pLDDT = pLDDT[indices]
            PAE = PAE[np.ix_(indices, indices)]

        self.prepare_scorer()
        self.scorer.compute_scores(distances, pLDDT, PAE)

        self.scores = self.scorer.scores
        self.save_scores()

    def save_scores(self) -> None:
        """Save scores DataFrame to a Parquet file."""
        self.scores.write_parquet(self.path / 'ipSAE_scores.parquet')

    def load_pLDDT_file(self) -> np.ndarray:
        """Load and scale pLDDT data.

        Returns:
            pLDDT array scaled to 0-100 range.
        """
        data = np.load(self.plddt_file)
        pLDDT_arr = data['plddt']
        if np.max(pLDDT_arr) <= 1.0:
            pLDDT_arr = pLDDT_arr * 100.0
        return pLDDT_arr

    def load_PAE_file(self) -> np.ndarray:
        """Load PAE data from file.

        Returns:
            PAE array from the 'pae' key in the npz file.
        """
        data = np.load(self.pae_file)['pae']
        return data


class ScoreCalculator:
    """Calculate model quality scores from structure predictions.

    Computes pDockQ, pDockQ2, LIS, ipTM, and ipSAE scores for all
    chain pairs in a structure.

    Attributes:
        chains: Array of chain IDs for each residue.
        unique_chains: Unique chain IDs in the structure.
        chain_pair_type: Dictionary mapping chain ID to type.
        n_res: Array of residue types.
        permuted: List of all chain pairs to evaluate.
        scores: DataFrame of computed scores after compute_scores().

    Args:
        chains: Array of chain IDs.
        chain_pair_type: Dictionary mapping chain ID to chain type
            ('protein' or 'nucleic_acid').
        n_residues: Number of residues per chain.
        pdockq_cutoff: Distance cutoff for pDockQ in Angstroms.
            Defaults to 8.0.
        pae_cutoff: PAE cutoff for ipSAE in Angstroms. Defaults to 12.0.

    Example:
        >>> calc = ScoreCalculator(chains, chain_types, n_residues)
        >>> calc.compute_scores(distances, plddt, pae)
        >>> print(calc.scores)
    """

    def __init__(
        self,
        chains: np.ndarray,
        chain_pair_type: dict[str, str],
        n_residues: int,
        pdockq_cutoff: float = 8.0,
        pae_cutoff: float = 12.0,
    ):
        """Initialize the ScoreCalculator.

        Args:
            chains: Array of chain IDs.
            chain_pair_type: Chain ID to type mapping.
            n_residues: Residue type array.
            pdockq_cutoff: pDockQ distance cutoff.
            pae_cutoff: PAE cutoff.
        """
        self.chains = chains
        self.unique_chains = np.unique(chains)
        self.chain_pair_type = chain_pair_type
        self.n_res = n_residues
        self.pDockQ_cutoff = pdockq_cutoff
        self.PAE_cutoff = pae_cutoff

        self.permute_chains()

    def compute_scores(
        self, distances: np.ndarray, pLDDT: np.ndarray, PAE: np.ndarray
    ) -> None:
        """Compute all scores for all chain pairs.

        Calculates pDockQ, pDockQ2, LIS, ipTM, and ipSAE scores for
        each permutation of chain pairs.

        Args:
            distances: Pairwise distance matrix between all residues.
            pLDDT: Per-residue pLDDT values (0-100 scale).
            PAE: Predicted aligned error matrix.
        """
        self.distances = distances
        self.pLDDT = pLDDT
        self.PAE = PAE

        results = []
        for chain1, chain2 in self.permuted:
            pDockQ, pDockQ2 = self.compute_pDockQ_scores(chain1, chain2)
            LIS = self.compute_LIS(chain1, chain2)
            ipTM, ipSAE = self.compute_ipTM_ipSAE(chain1, chain2)

            results.append([chain1, chain2, pDockQ, pDockQ2, LIS, ipTM, ipSAE])

        self.df = pl.DataFrame(
            np.array(results),
            schema={
                'chain1': str,
                'chain2': str,
                'pDockQ': float,
                'pDockQ2': float,
                'LIS': float,
                'ipTM': float,
                'ipSAE': float,
            },
        )
        self.get_max_values()

    def compute_pDockQ_scores(self, chain1: str, chain2: str) -> tuple[float, float]:
        """Compute pDockQ and pDockQ2 scores for a chain pair.

        pDockQ depends solely on pLDDT, while pDockQ2 depends on both
        pLDDT and PAE.

        Args:
            chain1: First chain identifier.
            chain2: Second chain identifier.

        Returns:
            Tuple of (pDockQ, pDockQ2) scores.
        """
        mask_c1 = self.chains == chain1
        mask_c2 = self.chains == chain2

        dist_sub = self.distances[np.ix_(mask_c1, mask_c2)]
        contact = dist_sub <= self.pDockQ_cutoff
        n_pairs = np.sum(contact)

        if n_pairs == 0:
            return 0.0, 0.0

        c1_idx = np.where(mask_c1)[0]
        c2_idx = np.where(mask_c2)[0]
        c1_contact = c1_idx[contact.any(axis=1)]
        c2_contact = c2_idx[contact.any(axis=0)]
        residues = np.concatenate([c1_contact, c2_contact])

        mean_pLDDT = self.pLDDT[residues].mean()
        x = mean_pLDDT * np.log10(n_pairs)
        pDockQ = self.pDockQ_score(x)

        pae_sub = self.PAE[np.ix_(mask_c1, mask_c2)]
        pae_contacts = pae_sub[contact]
        pae_ptm = self.compute_pTM(pae_contacts, 10.0)
        mean_pTM = pae_ptm.mean()
        x = mean_pLDDT * mean_pTM
        pDockQ2 = self.pDockQ2_score(x)

        return pDockQ, pDockQ2

    def compute_LIS(self, chain1: str, chain2: str) -> float:
        """Compute Local Interaction Score (LIS) for a chain pair.

        LIS is based on a subset of the predicted aligned error using
        a cutoff of 12 Å. Values range in (0, 1] where 1 indicates
        perfect accuracy.

        Adapted from: https://doi.org/10.1101/2024.02.19.580970

        Args:
            chain1: First chain identifier.
            chain2: Second chain identifier.

        Returns:
            LIS value for the chain pair.
        """
        mask = (self.chains[:, None] == chain1) & (self.chains[None, :] == chain2)
        selected_pae = self.PAE[mask]

        LIS = 0.0
        if selected_pae.size:
            valid_pae = selected_pae[selected_pae < self.PAE_cutoff]
            if valid_pae.size:
                scores = (self.PAE_cutoff - valid_pae) / self.PAE_cutoff
                LIS = np.mean(scores)

        return LIS

    def compute_ipTM_ipSAE(self, chain1: str, chain2: str) -> tuple[Any, Any]:
        """Compute ipTM and ipSAE scores for a chain pair.

        ipTM uses d0 based on total chain pair length and averages over
        all chain2 residues. ipSAE uses a per-residue d0 based on the
        count of chain2 residues with PAE below the cutoff for each
        aligned residue in chain1, averaging only over those valid
        residues.

        Args:
            chain1: First chain identifier (aligned chain).
            chain2: Second chain identifier (scored chain).

        Returns:
            Tuple of (ipTM, ipSAE) scores.
        """
        pair_type = 'protein'
        if (
            self.chain_pair_type[chain1] == 'nucleic_acid'
            or self.chain_pair_type[chain2] == 'nucleic_acid'
        ):
            pair_type = 'nucleic_acid'

        mask_c1 = self.chains == chain1
        mask_c2 = self.chains == chain2

        # ipTM: d0 from total chain pair length
        L = np.sum(mask_c1) + np.sum(mask_c2)
        d0_chain = self.compute_d0(L, pair_type)

        pae_sub = self.PAE[np.ix_(mask_c1, mask_c2)]

        # ipTM: mean pTM over all chain2 residues per chain1 residue
        ptm_sub_chain = self.compute_pTM(pae_sub, d0_chain)
        ipTM_byres = np.array([0.0])
        if mask_c2.any():
            ipTM_byres = np.mean(ptm_sub_chain, axis=1)

        # ipSAE: PAE filtering + per-residue d0
        valid_mask = pae_sub < self.PAE_cutoff
        n_valid_per_row = valid_mask.sum(axis=1)

        d0_per_res = self.compute_d0_array(n_valid_per_row, pair_type)
        ptm_sub_res = 1.0 / (1 + (pae_sub / d0_per_res[:, None]) ** 2)

        ipSAE_byres = np.zeros(mask_c1.sum())
        rows_with_valid = n_valid_per_row > 0
        if rows_with_valid.any():
            masked_ptm = np.where(valid_mask, ptm_sub_res, 0.0)
            ipSAE_byres[rows_with_valid] = (
                masked_ptm[rows_with_valid].sum(axis=1)
                / n_valid_per_row[rows_with_valid]
            )

        ipTM = np.max(ipTM_byres)
        ipSAE = np.max(ipSAE_byres) if ipSAE_byres.size > 0 else 0.0

        return ipTM, ipSAE

    def get_max_values(self) -> None:
        """Extract maximum scores for undirected chain pairs.

        Because some scores like ipSAE are asymmetric (A->B != B->A),
        takes the maximum score for either direction as the undirected
        score.
        """
        self.scores = (
            self.df.with_columns(
                pl.when(pl.col('chain1') < pl.col('chain2'))
                .then(pl.concat_str(['chain1', 'chain2'], separator='_'))
                .otherwise(pl.concat_str(['chain2', 'chain1'], separator='_'))
                .alias('pair_key')
            )
            .sort('ipSAE', descending=True)
            .unique(subset=['pair_key'], keep='first')
            .drop('pair_key')
        )

    def permute_chains(self) -> None:
        """Generate all permutations of chain pairs.

        Creates all unique ordered pairs of chains, excluding self-pairs.
        """
        self.permuted = list(permutations(self.unique_chains, 2))

    @staticmethod
    def pDockQ_score(x: float) -> float:
        """Compute pDockQ score.

        Formula: pDockQ = 0.724 / (1 + exp(-0.052 * (x - 152.611))) + 0.018

        Reference: https://doi.org/10.1038/s41467-022-28865-w

        Args:
            x: Mean pLDDT scaled by log10 of the number of residue pairs
                meeting pLDDT and distance cutoffs.

        Returns:
            pDockQ score.
        """
        return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

    @staticmethod
    def pDockQ2_score(x: float) -> float:
        """Compute pDockQ2 score.

        Formula: pDockQ2 = 1.31 / (1 + exp(-0.075 * (x - 84.733))) + 0.005

        Reference: https://doi.org/10.1093/bioinformatics/btad424

        Args:
            x: Mean pLDDT scaled by mean PAE score.

        Returns:
            pDockQ2 score.
        """
        return 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005

    @staticmethod
    def compute_pTM(x: np.ndarray, d0: float) -> np.ndarray:
        """Compute pTM score.

        Formula: pTM = 1.0 / (1 + (x / d0)^2)

        Args:
            x: PAE values (scalar or array).
            d0: Distance parameter from compute_d0.

        Returns:
            pTM score(s).
        """
        return 1.0 / (1 + (x / d0) ** 2)

    @staticmethod
    def compute_d0(L: int, pair_type: str) -> float:
        """Compute d0 parameter for pTM calculation.

        Formula: d0 = max(min_value, 1.24 * (L - 15)^(1/3) - 1.8)

        Args:
            L: Sequence length (minimum 27).
            pair_type: 'protein' or 'nucleic_acid'.

        Returns:
            d0 parameter value.
        """
        L = max(27, L)

        min_value = 1.0
        if pair_type == 'nucleic_acid':
            min_value = 2.0

        return max(min_value, 1.24 * (L - 15) ** (1 / 3) - 1.8)

    @staticmethod
    def compute_d0_array(L: np.ndarray, pair_type: str) -> np.ndarray:
        """Compute d0 parameter for an array of sequence lengths.

        Vectorized version of compute_d0 for per-residue d0 calculation
        used in ipSAE scoring.

        Args:
            L: Array of sequence lengths.
            pair_type: 'protein' or 'nucleic_acid'.

        Returns:
            Array of d0 parameter values.
        """
        L = np.asarray(L, dtype=float)
        L = np.maximum(26.0, L)

        min_value = 1.0
        if pair_type == 'nucleic_acid':
            min_value = 2.0

        return np.maximum(min_value, 1.24 * (L - 15) ** (1 / 3) - 1.8)


class ModelParser:
    """Parse structure files to extract residue and atom information.

    Handles both PDB and CIF format files, extracting C-alpha, C-beta,
    and nucleic acid backbone atom coordinates.

    Attributes:
        structure: Path to the structure file.
        protein_token_indices: pLDDT/PAE indices for kept-chain anchor
            tokens; populated by :meth:`build_protein_token_indices`.
        residues: List of dictionaries containing residue information.
        chains: List of chain IDs for each residue.
        chain_types: Dictionary mapping chain ID to type after
            classify_chains().

    Args:
        structure: Path to PDB or CIF file.

    Example:
        >>> parser = ModelParser("model.pdb")
        >>> parser.parse_structure_file()
        >>> parser.classify_chains()
    """

    def __init__(
        self,
        structure: PathLike,
        skip_chains: set[str] | list[str] | None = None,
    ):
        """Initialize the ModelParser.

        Args:
            structure: Path to PDB or CIF file.
            skip_chains: Chain IDs to exclude (e.g. glycan/ligand chains
                whose per-atom token counts in the pLDDT/PAE arrays
                aren't recoverable from the structure file alone).
        """
        self.structure = Path(structure)
        self.skip_chains = set(skip_chains) if skip_chains else set()

        self.residues = []
        self.cb_residues = []
        self.chains = []
        self.chain_order: list[str] = []
        self.chain_residue_counts: dict[str, int] = {}
        self.protein_token_indices: list[int] = []

    def parse_structure_file(self) -> None:
        """Parse the structure file and extract atom/residue data.

        Atoms in ``skip_chains`` are ignored. Chain order is preserved
        so that :meth:`build_protein_token_indices` can later infer the
        token span of skipped chains by arithmetic.
        """
        if self.structure.suffix == '.pdb':
            line_parser = self.parse_pdb_line
        else:
            line_parser = self.parse_cif_line

        field_num = 0
        fields = {}
        with open(self.structure) as f:
            lines = f.readlines()

        chain_residues: dict[str, list[dict[str, Any]]] = {}
        chain_cb: dict[str, list[dict[str, Any]]] = {}
        seen_chains: set[str] = set()

        for line in lines:
            if line.startswith('_atom_site.'):
                _, field_name = line.strip().split('.')
                fields[field_name] = field_num
                field_num += 1

            if line.startswith(('ATOM', 'HETATM')):
                atom = line_parser(line, fields, allow_missing_seq_id=True)
                if atom is None:
                    continue

                cid = atom['chain_id']
                if cid not in seen_chains:
                    seen_chains.add(cid)
                    self.chain_order.append(cid)
                    chain_residues[cid] = []
                    chain_cb[cid] = []

                if cid in self.skip_chains:
                    continue

                name = atom['atom_name']
                is_standard = atom['res'] in self.STANDARD_RESIDUES
                if is_standard and (name == 'CA' or 'C1' in name):
                    chain_residues[cid].append(atom)

                if is_standard and (
                    name == 'CB'
                    or 'C3' in name
                    or (atom['res'] == 'GLY' and name == 'CA')
                ):
                    chain_cb[cid].append(atom)

        for cid in self.chain_order:
            if cid in self.skip_chains:
                continue
            residues = chain_residues[cid]
            self.residues.extend(residues)
            self.cb_residues.extend(chain_cb[cid])
            self.chains.extend([cid] * len(residues))
            self.chain_residue_counts[cid] = len(residues)

    def build_protein_token_indices(self, total_tokens: int) -> None:
        """Derive pLDDT/PAE indices for kept-chain anchor tokens.

        Each kept chain contributes one token per residue; any run of
        consecutive skipped chains occupies the token span left over
        between kept chains. Solvable when skipped chains form at most
        one contiguous block in ``chain_order`` — the default layout
        emitted by Chai, Boltz, and AlphaFold (polymers first, then
        non-polymer chains).

        Args:
            total_tokens: Length of the pLDDT array (== PAE dim).

        Raises:
            ValueError: If skipped chains appear in multiple
                non-contiguous runs, making per-run sizes ambiguous.
        """
        runs: list[tuple[bool, list[str]]] = []
        for cid in self.chain_order:
            is_kept = cid not in self.skip_chains
            if runs and runs[-1][0] == is_kept:
                runs[-1][1].append(cid)
            else:
                runs.append((is_kept, [cid]))

        skipped_runs = [r for r in runs if not r[0]]
        total_kept_tokens = sum(self.chain_residue_counts.values())
        skipped_span = total_tokens - total_kept_tokens

        if len(skipped_runs) > 1:
            raise ValueError(
                f'Cannot infer token layout: skipped chains appear in '
                f'{len(skipped_runs)} non-contiguous runs. Reorder input '
                'so non-polymer chains are grouped.'
            )
        if skipped_span < 0:
            raise ValueError(
                f'Kept residue count ({total_kept_tokens}) exceeds '
                f'total tokens ({total_tokens}); structure and score '
                'files are inconsistent.'
            )

        indices: list[int] = []
        offset = 0
        for is_kept, chain_ids in runs:
            if is_kept:
                for cid in chain_ids:
                    n = self.chain_residue_counts[cid]
                    indices.extend(range(offset, offset + n))
                    offset += n
            else:
                offset += skipped_span

        self.protein_token_indices = indices

    def classify_chains(self) -> None:
        """Classify chains as protein or nucleic acid.

        Reads through residue data to assign chain identity based on
        whether nucleic acid residues are detected.
        """
        self.residue_types = np.array([res['res'] for res in self.residues])
        unique_chains = np.unique(self.chains)
        chains_array = np.array(self.chains)
        self.chain_types = {chain: 'protein' for chain in unique_chains}
        for chain in unique_chains:
            indices = np.where(chains_array == chain)[0]
            chain_residues = self.residue_types[indices]
            if set(chain_residues) & self.NUCLEIC_ACIDS:
                self.chain_types[chain] = 'nucleic_acid'

    NUCLEIC_ACIDS = frozenset(['DA', 'DC', 'DT', 'DG', 'A', 'C', 'U', 'G'])

    STANDARD_RESIDUES = frozenset([
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
        'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'DA', 'DC', 'DT', 'DG', 'A', 'C', 'U', 'G',
    ])

    @staticmethod
    def parse_pdb_line(line: str, *args, **kwargs) -> dict[str, Any]:
        """Parse a single line of a PDB file.

        Args:
            line: Line from the PDB file.
            *args: Unused, for API compatibility with parse_cif_line.

        Returns:
            Dictionary with atom/residue information.
        """
        atom_num = line[6:11].strip()
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_id = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()

        return ModelParser.package_line(
            atom_num, atom_name, residue_name, chain_id, residue_id, x, y, z
        )

    @staticmethod
    def parse_cif_line(
        line: str,
        fields: dict[str, int],
        allow_missing_seq_id: bool = False,
    ) -> dict[str, Any] | None:
        """Parse a single line of a CIF file.

        Args:
            line: Line from the CIF file.
            fields: Dictionary mapping field names to column indices.
            allow_missing_seq_id: If True, fall back to auth_seq_id when
                label_seq_id is '.'. Used for non-polymer residues
                (ligands, glycans) which lack a label_seq_id.

        Returns:
            Dictionary with atom/residue information, or None if
            residue_id is missing and fallback is disabled.
        """
        _split = line.split()
        atom_num = _split[fields['id']]
        atom_name = _split[fields['label_atom_id']]
        residue_name = _split[fields['label_comp_id']]
        chain_id = _split[fields['label_asym_id']]
        residue_id = _split[fields['label_seq_id']]
        x = _split[fields['Cartn_x']]
        y = _split[fields['Cartn_y']]
        z = _split[fields['Cartn_z']]

        if residue_id == '.':
            if not allow_missing_seq_id or 'auth_seq_id' not in fields:
                return None
            residue_id = _split[fields['auth_seq_id']]

        return ModelParser.package_line(
            atom_num, atom_name, residue_name, chain_id, residue_id, x, y, z
        )

    @staticmethod
    def package_line(
        atom_num: str,
        atom_name: str,
        residue_name: str,
        chain_id: str,
        residue_id: str,
        x: str,
        y: str,
        z: str,
    ) -> dict[str, Any]:
        """Package parsed line data into a dictionary.

        Args:
            atom_num: Atom index.
            atom_name: Atom name (e.g., 'CA', 'CB').
            residue_name: Residue name (e.g., 'ALA').
            chain_id: Chain identifier.
            residue_id: Residue sequence number.
            x: X coordinate as string.
            y: Y coordinate as string.
            z: Z coordinate as string.

        Returns:
            Dictionary containing parsed atom/residue data.
        """
        return {
            'atom_num': int(atom_num),
            'atom_name': atom_name,
            'coor': np.array([x, y, z], dtype=float),
            'res': residue_name,
            'chain_id': chain_id,
            'resid': int(residue_id),
        }
