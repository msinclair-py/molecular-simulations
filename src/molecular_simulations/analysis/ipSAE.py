# ruff: noqa: N801, N999
"""Interface prediction Score from Aligned Errors (ipSAE) module.

This module computes interaction prediction scores from pLDDT and PAE data,
adapted from https://doi.org/10.1101/2025.02.10.637595. Supports outputs
from structure prediction tools like Boltz and AlphaFold.

It also ports the ipSAE2 scoring algorithm from ipSAE2 v0.1.5-alpha by the
Dunbrack Laboratory (https://github.com/DunbrackLab/IPSAE2, MIT license).
Unlike the original ipSAE, ipSAE2 scores proteins, nucleic acids, ligands
and ions as molecules: the predictor's token order (atom-tokenized ligands
and modified residues) is reconstructed from the structure and checked
against the PAE dimension, and the final pair score is chosen by molecule
type.

ipSAE2 deviations from upstream:
    * A non-polymer chain with several residues (e.g. a glycan) is scored as
      one ligand molecule instead of raising an error.
    * Selenocysteine and pyrrolysine are treated as modified residues, so
      both atom and residue tokenization are tried for them.
    * PAE is read from ``.npz`` only, and predictor-native ipTM values are
      not imported.
"""

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

PathLike = Path | str
OptPath = Path | str | None


class ipSAE:
    """Compute interaction prediction Score from Aligned Errors.

    Computes any of pDockQ, pDockQ2, LIS, ipSAE (with ipTM) and ipSAE2 for
    structure predictions, one row per unordered chain pair.

    pDockQ, pDockQ2, LIS and ipSAE only score protein and nucleic acid
    chains. Chains without standard residues (ligands, glycans, ions) must
    be listed in ``skip_chains`` unless ipSAE2 is also requested, in which
    case they are left out of those scores automatically while still being
    scored by ipSAE2.

    Attributes:
        parser: ModelParser instance for structure file.
        plddt_file: Path to pLDDT data file.
        pae_file: Path to PAE data file.
        path: Output directory path.
        score_types: Requested scores, lower case.
        scores: Polars DataFrame of computed scores after run().

    Args:
        structure_file: Path to PDB/CIF model file.
        plddt_file: Path to pLDDT numpy file (.npz with 'plddt' key).
            May be None when only ipSAE2 is requested.
        pae_file: Path to PAE numpy file (.npz with 'pae' key).
        out_path: Output directory path. If None, uses parent directory
            of plddt_file (or pae_file).
        skip_chains: Chain IDs to exclude from all scores. Their
            pLDDT/PAE tokens are inferred from remaining tokens and
            dropped before scoring.
        scores: Scores to compute, any of 'ipsae', 'ipsae2', 'pDockQ',
            'pDockQ2' and 'LIS' (case-insensitive). Defaults to all.
        ipsae2_kwargs: Keyword arguments for
            :class:`IpSAE2Calculator`
            (pae_cutoff, contact_cutoff, fixed_d0, protein_min_d0,
            all_pairs).

    Example:
        >>> scorer = ipSAE('model.pdb', 'plddt.npz', 'pae.npz')
        >>> scorer.run()
        >>> print(scorer.scores)
    """

    SCORE_TYPES = ('ipsae', 'ipsae2', 'pdockq', 'pdockq2', 'lis')
    LEGACY_COLUMNS: ClassVar[dict[str, tuple[str, ...]]] = {
        'pdockq': ('pDockQ',),
        'pdockq2': ('pDockQ2',),
        'lis': ('LIS',),
        'ipsae': ('ipTM', 'ipSAE'),
    }

    def __init__(
        self,
        structure_file: PathLike,
        plddt_file: OptPath,
        pae_file: PathLike,
        out_path: OptPath = None,
        skip_chains: set[str] | list[str] | None = None,
        scores: Iterable[str] = SCORE_TYPES,
        ipsae2_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the ipSAE scorer.

        Args:
            structure_file: Path to structure file.
            plddt_file: Path to pLDDT data file, or None for ipSAE2 only.
            pae_file: Path to PAE data file.
            out_path: Output directory path.
            skip_chains: Chain IDs to exclude from all scores.
            scores: Scores to compute. Defaults to all.
            ipsae2_kwargs: Keyword arguments for IpSAE2Calculator.

        Raises:
            ValueError: If a score name is unknown, or plddt_file is None
                while a score other than ipSAE2 is requested.
        """
        self.score_types = {name.lower() for name in scores}
        unknown = self.score_types - set(self.SCORE_TYPES)
        if unknown or not self.score_types:
            raise ValueError(
                f'Unknown or empty scores {sorted(unknown)}; choose from {self.SCORE_TYPES}'
            )
        if plddt_file is None and self.score_types - {'ipsae2'}:
            raise ValueError('plddt_file is required for scores other than ipsae2')

        self.skip_chains = set(skip_chains) if skip_chains else set()
        self.ipsae2_kwargs = ipsae2_kwargs or {}
        self.parser = ModelParser(structure_file, skip_chains=skip_chains)
        self.plddt_file = Path(plddt_file) if plddt_file is not None else None
        self.pae_file = Path(pae_file)

        if out_path is not None:
            self.path = Path(out_path)
        else:
            self.path = (self.plddt_file or self.pae_file).parent
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
        """Execute the complete scoring workflow.

        Computes the requested scores, merges them into one row per chain
        pair, and saves results.
        """
        PAE = self.load_PAE_file()
        legacy = None
        if self.score_types & self.LEGACY_COLUMNS.keys():
            legacy = self.run_legacy(PAE)
        ipsae2 = self.run_ipsae2(PAE) if 'ipsae2' in self.score_types else None

        self.scores = self.merge_scores(legacy, ipsae2)
        self.save_scores()

    def run_legacy(self, PAE: np.ndarray) -> pl.DataFrame:
        """Compute pDockQ, pDockQ2, LIS, ipTM and ipSAE.

        All four are computed in one pass, since the reported direction of
        each pair is the one with the larger ipSAE; unrequested columns are
        then dropped.

        Args:
            PAE: Full PAE matrix.

        Returns:
            DataFrame with chain1, chain2 and the requested columns.

        Raises:
            ValueError: If chains without standard residues are present,
                not in ``skip_chains``, and ipSAE2 was not requested.
        """
        self.parse_structure_file()
        nonpolymer = self.parser.nonpolymer_chains()
        if nonpolymer and 'ipsae2' not in self.score_types:
            raise ValueError(
                f'Chains {nonpolymer} have no standard residues (ligand, glycan '
                'or ion) and cannot be scored by pDockQ, pDockQ2, LIS or ipSAE. '
                "Pass them in skip_chains, or request 'ipsae2' to score them."
            )
        for chain in nonpolymer:
            self.parser.skip_chain(chain)

        distances = cdist(self.cb_coordinates, self.cb_coordinates)
        pLDDT = self.load_pLDDT_file()

        n_residues = len(self.parser.residues)
        if pLDDT.shape[0] != n_residues or PAE.shape[0] != n_residues:
            self.parser.build_protein_token_indices(pLDDT.shape[0])
            indices = np.array(self.parser.protein_token_indices, dtype=int)
            pLDDT = pLDDT[indices]
            PAE = PAE[np.ix_(indices, indices)]

        self.prepare_scorer()
        self.scorer.compute_scores(distances, pLDDT, PAE)

        columns = [
            column
            for name, cols in self.LEGACY_COLUMNS.items()
            if name in self.score_types
            for column in cols
        ]
        return self.scorer.scores.select('chain1', 'chain2', *columns)

    def run_ipsae2(self, PAE: np.ndarray) -> pl.DataFrame:
        """Compute ipSAE2 for every contacting molecule pair.

        Args:
            PAE: Full PAE matrix.

        Returns:
            DataFrame of ipSAE2 scores and directional diagnostics.
        """
        atoms = IpSAE2Parser.read_atoms(self.parser.structure)
        metadata = IpSAE2Parser.read_pae_metadata(self.pae_file)
        chain_types = IpSAE2Parser.classify_molecules(atoms)
        tokens = IpSAE2Parser.infer_tokens(atoms, chain_types, len(PAE), metadata)
        tokens.frame_valid = IpSAE2Parser.frame_validity(atoms, tokens, metadata)
        calculator = IpSAE2Calculator(**self.ipsae2_kwargs)
        return calculator.compute_scores(PAE, atoms, tokens, self.skip_chains)

    @staticmethod
    def merge_scores(
        legacy: pl.DataFrame | None, ipsae2: pl.DataFrame | None
    ) -> pl.DataFrame:
        """Join legacy and ipSAE2 scores on the unordered chain pair.

        ipSAE2 directional columns are reoriented so that ``12`` always
        means align on ``chain1`` and score ``chain2`` of the merged row.
        Pairs reported by only one method have nulls for the other.

        Args:
            legacy: Output of run_legacy, or None.
            ipsae2: Output of run_ipsae2, or None.

        Returns:
            Merged DataFrame.
        """
        if ipsae2 is None or legacy is None:
            return legacy if ipsae2 is None else ipsae2

        def pair_key(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                pl.min_horizontal('chain1', 'chain2').alias('_lo'),
                pl.max_horizontal('chain1', 'chain2').alias('_hi'),
            )

        legacy = pair_key(legacy)
        ipsae2 = pair_key(ipsae2).join(
            legacy.select('_lo', '_hi', pl.col('chain1').alias('_legacy_chain1')),
            on=['_lo', '_hi'],
            how='left',
        )
        swap = pl.col('_legacy_chain1').is_not_null() & (
            pl.col('_legacy_chain1') != pl.col('chain1')
        )
        pairs = [('chain1', 'chain2'), ('type1', 'type2')] + [
            (c, c[:-2] + '21') for c in ipsae2.columns if c.endswith('_12')
        ]
        swapped = []
        for a, b in pairs:
            swapped.append(pl.when(swap).then(pl.col(b)).otherwise(pl.col(a)).alias(a))
            swapped.append(pl.when(swap).then(pl.col(a)).otherwise(pl.col(b)).alias(b))
        source = pl.col('ipSAE2_source')
        flipped = source.replace({'12': '21', '21': '12'})
        swapped.append(
            pl.when(swap).then(flipped).otherwise(source).alias('ipSAE2_source')
        )
        ipsae2 = ipsae2.with_columns(swapped).drop('_legacy_chain1')

        merged = legacy.join(ipsae2, on=['_lo', '_hi'], how='full', coalesce=True)
        merged = merged.with_columns(
            pl.coalesce('chain1', 'chain1_right').alias('chain1'),
            pl.coalesce('chain2', 'chain2_right').alias('chain2'),
        )
        legacy_columns = [
            c for c in legacy.columns if c not in ('chain1', 'chain2', '_lo', '_hi')
        ]
        ipsae2_columns = [
            c
            for c in ipsae2.columns
            if c not in ('chain1', 'chain2', 'type1', 'type2', '_lo', '_hi')
        ]
        return merged.sort('_lo', '_hi').select(
            'chain1', 'chain2', 'type1', 'type2', *legacy_columns, *ipsae2_columns
        )

    def save_scores(self) -> None:
        """Save scores DataFrame to a Parquet file."""
        self.scores.write_parquet(self.path / 'ipSAE_scores.parquet')

    def load_pLDDT_file(self) -> np.ndarray:
        """Load and scale pLDDT data.

        Returns:
            pLDDT array scaled to 0-100 range.
        """
        if self.plddt_file is None:
            raise ValueError('No plddt_file was provided')
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
        n_res: Total number of residues across all chains (integer).
        permuted: List of all chain pairs to evaluate.
        scores: DataFrame of computed scores after compute_scores().

    Args:
        chains: Array of chain IDs.
        chain_pair_type: Dictionary mapping chain ID to chain type
            ('protein' or 'nucleic_acid').
        n_residues: Total number of residues across all chains.
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
            n_residues: Total number of residues across all chains.
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
                within the distance cutoff.

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
            x: Mean pLDDT scaled by the mean pTM derived from the PAE
                values of contacting residue pairs.

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
        skip_chains: Chain IDs to exclude (e.g. glycan/ligand chains
            whose per-atom token counts in the pLDDT/PAE arrays
            aren't recoverable from the structure file alone).

    Example:
        >>> parser = ModelParser('model.pdb')
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

    def nonpolymer_chains(self) -> list[str]:
        """Return kept chains without standard residues.

        These are ligand, glycan or ion chains that the residue-based
        scores cannot handle and that occupy PAE tokens.

        Returns:
            Chain IDs in file order.
        """
        return [
            cid
            for cid in self.chain_order
            if cid not in self.skip_chains and self.chain_residue_counts.get(cid) == 0
        ]

    def skip_chain(self, chain: str) -> None:
        """Mark an already-parsed chain without residues as skipped.

        Args:
            chain: Chain ID with no standard residues.
        """
        self.skip_chains.add(chain)
        self.chain_residue_counts.pop(chain, None)

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

    STANDARD_RESIDUES = frozenset(
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
            'DA',
            'DC',
            'DT',
            'DG',
            'A',
            'C',
            'U',
            'G',
        ]
    )

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


@dataclass
class MoleculeGeometry:
    """Heavy atoms of one molecule prepared for contact queries."""

    xyz: np.ndarray
    unit_ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    tree: cKDTree


class IpSAE2Parser:
    """Read structures and rebuild predictor token order for ipSAE2.

    Canonical residues are the ones every predictor tokenizes as a single
    residue token. Any other residue in a protein or nucleic acid chain
    (phosphorylations, MSE, selenocysteine, modified nucleotides, ...) is a
    modified residue; whether it is atom- or residue-tokenized is inferred
    from the PAE dimension.

    Attributes:
        CANONICAL_PROTEIN: Standard amino acid residue names.
        CANONICAL_NUCLEIC: Standard RNA and DNA residue names.
        IONS: Residue or element names of single-atom ions.
        WATER: Water residue names, dropped on read.
        PAE_METADATA_KEYS: Optional per-token arrays read from the PAE file.
        CIF_TOKEN: Regex splitting an mmCIF row into values.
    """

    CANONICAL_PROTEIN = frozenset(
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
        ]
    )
    CANONICAL_NUCLEIC = frozenset(
        ['A', 'C', 'G', 'U', 'I', 'DA', 'DC', 'DG', 'DT', 'DI', 'DU']
    )
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

    PAE_METADATA_KEYS = (
        'frame_mask',
        'token_has_frame',
        'asym_id',
        'token_asym_id',
    )

    CIF_TOKEN = re.compile(
        r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+"""
    )

    @classmethod
    def _split_cif_row(cls, line: str) -> list[str]:
        """Split an mmCIF data row, removing quotes around values like "C1'"."""
        values = []
        for token in cls.CIF_TOKEN.findall(line):
            if len(token) >= 2 and token[0] == token[-1] and token[0] in '\'"':
                token = token[1:-1]
            values.append(token)
        return values

    @staticmethod
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

    @classmethod
    def _read_cif_rows(cls, path: Path) -> list[tuple]:
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
                values = cls._split_cif_row(line)

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

    @classmethod
    def _infer_element(cls, atom_name: str, resname: str) -> str:
        name = atom_name.upper().lstrip('0123456789')
        if name.startswith(('H', 'D')):
            return 'H'
        if resname.upper() in cls.IONS:
            return resname.upper()
        return name[:1]

    @classmethod
    def read_atoms(cls, path: PathLike) -> AtomTable:
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
            rows = cls._read_pdb_rows(path)
        else:
            rows = cls._read_cif_rows(path)

        kept = []
        for row in rows:
            record, chain, resnum, resname, atom_name, element = row[:6]
            if resname in cls.WATER:
                continue
            element = (element or cls._infer_element(atom_name, resname)).upper()
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

    @classmethod
    def classify_molecules(cls, atoms: AtomTable) -> dict[str, str]:
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
            has_prot = any(name in cls.CANONICAL_PROTEIN for name in names)
            has_nuc = any(name in cls.CANONICAL_NUCLEIC for name in names)
            if has_prot and has_nuc:
                raise ValueError(
                    f'Chain {chain} mixes protein and nucleic acid residues'
                )
            if has_prot:
                chain_types[chain] = 'prot'
            elif has_nuc:
                chain_types[chain] = 'nuc'
            else:
                idx = residues[keys[0]]
                is_ion = (
                    len(keys) == 1
                    and len(idx) == 1
                    and (
                        names[0].upper() in cls.IONS
                        or atoms.element[idx[0]] in cls.IONS
                    )
                )
                chain_types[chain] = 'ion' if is_ion else 'lig'

        # Folding an ion into a protein would silently misassign its token.
        for key, idx in residues.items():
            if chain_types[key[0]] != 'prot' or key[2] in cls.CANONICAL_PROTEIN:
                continue
            if (
                len(idx) == 1
                and atoms.record[idx[0]] == 'HETATM'
                and atoms.element[idx[0]] in cls.IONS
            ):
                raise ValueError(
                    f'Ion {key[2]} shares protein chain {key[0]}; use unique '
                    'per-molecule chain IDs'
                )
        return chain_types

    @classmethod
    def read_pae_metadata(cls, path: PathLike) -> dict[str, np.ndarray]:
        """Read optional token metadata stored next to the PAE in an npz file.

        Args:
            path: Path to the PAE ``.npz`` file.

        Returns:
            Arrays for any of ``PAE_METADATA_KEYS`` present in the file.
        """
        with np.load(path, allow_pickle=False) as data:
            return {k: np.asarray(data[k]) for k in cls.PAE_METADATA_KEYS if k in data}

    @classmethod
    def _canonical(cls, mol_type: str) -> frozenset:
        return cls.CANONICAL_PROTEIN if mol_type == 'prot' else cls.CANONICAL_NUCLEIC

    @classmethod
    def _build_tokens(
        cls,
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
            elif resname not in cls._canonical(mol_type) and mod_mode == 'atom':
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

    @staticmethod
    def _metadata_chains(metadata: dict[str, np.ndarray], n: int) -> list | None:
        for key in ('token_asym_id', 'asym_id'):
            values = metadata.get(key)
            if values is not None and values.ndim == 1 and len(values) == n:
                return [str(v) for v in values]
        return None

    @staticmethod
    def _chains_match(token_chains: np.ndarray, reference: list) -> bool:
        """Check token chains against numeric per-token chain assignments."""
        seen = list(dict.fromkeys(reference))
        chains = list(dict.fromkeys(token_chains))
        if len(seen) != len(chains):
            return False
        expected = dict(zip(seen, chains, strict=True))
        return [expected[c] for c in reference] == list(token_chains)

    @classmethod
    def infer_tokens(
        cls,
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
            chain_types: Output of :meth:`classify_molecules`.
            n_tokens: PAE matrix dimension.
            metadata: Optional per-token arrays from :meth:`read_pae_metadata`.

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
            elif resname in cls._canonical(mol_type):
                n_canonical += 1
            else:
                mod_count += 1
                mod_atoms += len(idx)

        mod_tokens = {'atom': mod_atoms, 'residue': mod_count}
        lig_tokens = {'atom': lig_atoms, 'residue': lig_count, 'chain': len(lig_chains)}
        reference = cls._metadata_chains(metadata, n_tokens)

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
                tokens = cls._build_tokens(
                    atoms, residues, chain_types, mod_mode, lig_mode
                )
                if reference is not None and not cls._chains_match(
                    tokens.chain, reference
                ):
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

    @staticmethod
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
            tokens: Output of :meth:`infer_tokens`.
            metadata: Optional per-token arrays from :meth:`read_pae_metadata`.

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
    TYPE_ORDER: ClassVar[dict[str, int]] = {'prot': 0, 'nuc': 1, 'lig': 2, 'ion': 3}

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

    @staticmethod
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

    @staticmethod
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
            key=lambda c: self.TYPE_ORDER[types[c]],
        )
        token_idx = {c: np.flatnonzero(tokens.chain == c) for c in molecules}
        geometry = {c: self.molecule_geometry(atoms, tokens, c) for c in molecules}

        rows = []
        for i, chain1 in enumerate(molecules):
            for chain2 in molecules[i + 1 :]:
                type1, type2 = types[chain1], types[chain2]
                if type1 == 'ion' and type2 == 'ion':
                    continue
                c1, c2 = self.contact_units(
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
