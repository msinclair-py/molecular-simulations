"""Covariance-based protein-protein interaction analysis.

This module provides tools for analyzing protein-protein interactions based
on covariance analysis of molecular dynamics trajectories. Adapted from
https://www.biorxiv.org/content/10.1101/2025.03.24.644990v1.full.pdf
"""

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import polars as pl
import seaborn as sns
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.util import convert_aa_code

PathLike = Path | str
Results = dict[str, dict[str, dict[str, float]]]
TaskTree = tuple[list[Callable[..., int | float]], list[str]]


class PPInteractions:
    """Analyze protein-protein interactions using covariance analysis.

    Takes an input topology and trajectory file, computes the covariance
    matrix between two selections, filters interactions by distance
    (11Å for positive covariance, 13Å for negative covariance), and
    evaluates each based on distance and angle cutoffs for various
    interaction types.

    Attributes:
        u: MDAnalysis Universe object.
        n_frames: Number of frames in the trajectory.
        out: Output path for results.
        mapping: Residue index to resID mapping for both selections.

    Args:
        top: Path to topology file.
        traj: Path to trajectory file.
        out: Path to output results file.
        sel1: MDAnalysis selection string for the first selection.
            Defaults to 'chainID A'.
        sel2: MDAnalysis selection string for the second selection.
            Defaults to 'chainID B'.
        cov_cutoff: Distance cutoffs for positive and negative covariance
            filtering respectively. Defaults to (11.0, 13.0) Angstroms.
        sb_cutoff: Distance cutoff for salt bridges. Defaults to 6.0 Å.
        hbond_cutoff: Distance cutoff for hydrogen bonds. Defaults to 3.5 Å.
        hbond_angle: Angle cutoff for hydrogen bonds in degrees.
            Defaults to 30.0 degrees.
        hydrophobic_cutoff: Distance cutoff for hydrophobic interactions.
            Defaults to 8.0 Å.
        plot: Whether to generate and save plots. Defaults to True.

    Example:
        >>> ppi = PPInteractions("complex.prmtop", "traj.dcd", "results.json")
        >>> ppi.run()
    """

    def __init__(
        self,
        top: PathLike,
        traj: PathLike,
        out: PathLike,
        sel1: str = 'chainID A',
        sel2: str = 'chainID B',
        cov_cutoff: tuple[float, float] = (11.0, 13.0),
        sb_cutoff: float = 6.0,
        hbond_cutoff: float = 3.5,
        hbond_angle: float = 30.0,
        hydrophobic_cutoff: float = 8.0,
        plot: bool = True,
    ):
        """Initialize the protein-protein interaction analyzer.

        Args:
            top: Path to topology file.
            traj: Path to trajectory file.
            out: Path to output results file.
            sel1: MDAnalysis selection string for first selection.
            sel2: MDAnalysis selection string for second selection.
            cov_cutoff: Tuple of distance cutoffs for (positive, negative)
                covariance.
            sb_cutoff: Salt bridge distance cutoff in Angstroms.
            hbond_cutoff: Hydrogen bond distance cutoff in Angstroms.
            hbond_angle: Hydrogen bond angle cutoff in degrees.
            hydrophobic_cutoff: Hydrophobic interaction cutoff in Angstroms.
            plot: Whether to generate plots.
        """
        self.u = mda.Universe(top, traj)
        self.n_frames = len(self.u.trajectory)
        self.out = out
        self.sel1 = sel1
        self.sel2 = sel2
        self.cov_cutoff = cov_cutoff
        self.sb = sb_cutoff
        self.hb_d = hbond_cutoff
        self.hb_a = hbond_angle * 180 / np.pi
        self.hydr = hydrophobic_cutoff
        self.plot = plot

    def run(self) -> None:
        """Execute the full interaction analysis workflow.

        Obtains a covariance matrix, screens for close interactions,
        evaluates each pairwise interaction, and reports contact
        probabilities. Optionally generates plots.
        """
        cov = self.get_covariance()
        positive, negative = self.interpret_covariance(cov)

        results = {'positive': {}, 'negative': {}}
        for res1, res2 in positive:
            data = self.compute_interactions(res1, res2)
            results['positive'].update(data)

        for res1, res2 in negative:
            data = self.compute_interactions(res1, res2)
            results['negative'].update(data)

        self.save(results)

        if self.plot:
            self.plot_results(results)

    def compute_interactions(self, res1: int, res2: int) -> dict[str, dict[str, float]]:
        """Compute interaction probabilities between two residues.

        Generates MDAnalysis AtomGroups for each residue, identifies
        relevant non-bonded interactions (hydrogen bonds, salt bridges,
        hydrophobic), and computes the fraction of simulation time each
        interaction is engaged.

        Args:
            res1: ResID for a residue in sel1.
            res2: ResID for a residue in sel2.

        Returns:
            Nested dictionary containing the results of each interaction
            type. Keys are residue pair names, values are dictionaries
            mapping interaction type to probability.
        """
        grp1 = self.u.select_atoms(f'{self.sel1} and resid {res1}')
        grp2 = self.u.select_atoms(f'{self.sel2} and resid {res2}')
        r1 = convert_aa_code(grp1.resnames[0])
        r2 = convert_aa_code(grp2.resnames[0])
        name = f'A_{r1}{res1}-B_{r2}{res2}'

        data = {name: {label: 0.0 for label in ['hydrophobic', 'hbond', 'saltbridge']}}
        function_calls, labels = self.identify_interaction_type(
            grp1.resnames[0], grp2.resnames[0]
        )

        for call, label in zip(function_calls, labels, strict=True):
            data[name][label] = call(grp1, grp2)

        return data

    def get_covariance(self) -> np.ndarray:
        """Compute the positional covariance matrix between selections.

        Loops over all C-alpha atoms and computes the positional
        covariance using the functional form::

            C = <(R1 - <R1>)(R2 - <R2>)^T>

        where each element corresponds to the ensemble average movement::

            C_ij = <deltaR_i * deltaR_j>

        The magnitude indicates correlation strength and the sign
        indicates positive or negative correlation.

        Returns:
            Covariance matrix with shape (N_residues_sel1, N_residues_sel2).
        """
        p1_ca = self.u.select_atoms('chainID A and name CA')
        N = p1_ca.n_residues

        p2_ca = self.u.select_atoms('chainID B and name CA')
        M = p2_ca.n_residues

        self.res_map(p1_ca, p2_ca)

        R1_avg = np.zeros((N, 3))
        R2_avg = np.zeros((M, 3))

        for _ts in self.u.trajectory:
            R1_avg += p1_ca.positions
            R2_avg += p2_ca.positions

        R1_avg /= self.n_frames
        R2_avg /= self.n_frames

        C = np.zeros((N, M))

        for _ts in self.u.trajectory:
            R1 = p1_ca.positions
            R2 = p2_ca.positions

            dR1 = R1 - R1_avg
            dR2 = R2 - R2_avg

            for i in range(N):
                for j in range(M):
                    C[i, j] += np.dot(dR1[i], dR2[j])

        C /= self.n_frames

        for i in range(N):
            for j in range(M):
                dist = np.linalg.norm(R1_avg[i] - R2_avg[j])
                if C[i, j] > 0:
                    if dist > self.cov_cutoff[0]:
                        C[i, j] = 0.0
                elif dist > self.cov_cutoff[1]:
                    C[i, j] = 0.0

        return C

    def res_map(self, ag1: mda.AtomGroup, ag2: mda.AtomGroup) -> None:
        """Create mapping from covariance matrix indices to resIDs.

        Maps covariance matrix indices to AtomGroup resIDs to ensure
        correct residue pairs are examined.

        Args:
            ag1: AtomGroup of the first selection.
            ag2: AtomGroup of the second selection.
        """
        mapping = {'ag1': {}, 'ag2': {}}
        for i, resid in enumerate(ag1.resids):
            mapping['ag1'][i] = resid

        for i, resid in enumerate(ag2.resids):
            mapping['ag2'][i] = resid

        self.mapping = mapping

    def interpret_covariance(self, cov_mat: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Identify residue pairs with positive or negative correlations.

        Args:
            cov_mat: Covariance matrix from get_covariance().

        Returns:
            Tuple of two lists: (positive_pairs, negative_pairs).
            Each pair is a tuple of (resID_sel1, resID_sel2).
        """
        pos_corr = np.where(cov_mat > 0.0)
        neg_corr = np.where(cov_mat < 0.0)

        seen = set()
        positive = list()
        for i in range(len(pos_corr[0])):
            res1 = self.mapping['ag1'][pos_corr[0][i]]
            res2 = self.mapping['ag2'][pos_corr[1][i]]
            if (res1, res2) not in seen:
                positive.append((res1, res2))
                seen.add((res1, res2))
                seen.add((res2, res1))

        negative = list()
        for i in range(len(neg_corr[0])):
            res1 = self.mapping['ag1'][neg_corr[0][i]]
            res2 = self.mapping['ag2'][neg_corr[1][i]]
            if (res1, res2) not in seen:
                negative.append((res1, res2))
                seen.add((res1, res2))
                seen.add((res2, res1))

        return positive, negative

    def identify_interaction_type(self, res1: str, res2: str) -> TaskTree:
        """Determine which analyses to compute for a residue pair.

        Identifies what analyses to compute based on residue types
        (hydrophobic interactions, hydrogen bonds, salt bridges).

        Args:
            res1: 3-letter code resname for a residue from selection 1.
            res2: 3-letter code resname for a residue from selection 2.

        Returns:
            Tuple containing (list of function calls, list of labels).
        """
        int_types = {
            'TYR': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'HIS': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'HID': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'HIE': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'SER': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'THR': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'ASN': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'GLN': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'ASP': {
                'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                'label': ['hbond', 'saltbridge'],
            },
            'GLU': {
                'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                'label': ['hbond', 'saltbridge'],
            },
            'LYS': {
                'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                'label': ['hbond', 'saltbridge'],
            },
            'ARG': {
                'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                'label': ['hbond', 'saltbridge'],
            },
            'HIP': {
                'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                'label': ['hbond', 'saltbridge'],
            },
        }

        funcs = defaultdict(lambda: [[], []])
        for res, calls in int_types.items():
            funcs[res] = [calls['funcs'], calls['label']]

        functions = [self.analyze_hydrophobic]
        labels = ['hydrophobic']
        for func, lab in zip(*funcs[res1], strict=True):
            if func in funcs[res2][0]:
                functions.append(func)
                labels.append(lab)

        return functions, labels

    def analyze_saltbridge(self, res1: mda.AtomGroup, res2: mda.AtomGroup) -> float:
        """Calculate salt bridge occupancy between two residues.

        Uses a simple distance cutoff to determine salt bridge formation.

        Args:
            res1: AtomGroup for a residue from selection 1.
            res2: AtomGroup for a residue from selection 2.

        Returns:
            Proportion of simulation time spent in salt bridge contact.
        """
        pos = ['LYS', 'ARG']
        neg = ['ASP', 'GLU']
        name1 = res1.resnames[0]
        name2 = res2.resnames[0]
        if name1 not in pos + neg or name2 not in pos + neg or (name1 in pos and name2 in pos) or (name1 in neg and name2 in neg):
            return 0.0

        atom_names = ['NZ', 'NH1', 'NH2', 'OD1', 'OD2', 'OE1', 'OE2']

        grp1 = self.u.select_atoms('resname DUMMY')
        for atom in res1.atoms:
            if atom.name in atom_names:
                grp1 += atom

        grp2 = self.u.select_atoms('resname DUMMY')
        for atom in res2.atoms:
            if atom.name in atom_names:
                grp2 += atom

        n_frames = 0
        for _ts in self.u.trajectory:
            dist = np.linalg.norm(grp1.positions - grp2.positions)
            if dist < self.sb:
                n_frames += 1

        return n_frames / self.n_frames

    def analyze_hbond(self, res1: mda.AtomGroup, res2: mda.AtomGroup) -> float:
        """Calculate hydrogen bond occupancy between two residues.

        Identifies all potential donor/acceptor atoms, filters by distance,
        then evaluates each pair over the trajectory using distance and
        angle cutoffs.

        Args:
            res1: AtomGroup for a residue from selection 1.
            res2: AtomGroup for a residue from selection 2.

        Returns:
            Proportion of simulation time spent in hydrogen bond contact.
        """
        donors, acceptors = self.survey_donors_acceptors(res1, res2)

        n_frames = 0
        for _ts in self.u.trajectory:
            n_frames += self.evaluate_hbond(donors, acceptors)

        return n_frames / self.n_frames

    def analyze_hydrophobic(self, res1: mda.AtomGroup, res2: mda.AtomGroup) -> float:
        """Calculate hydrophobic interaction occupancy between residues.

        Uses a simple distance cutoff between carbon atoms to determine
        hydrophobic contact.

        Args:
            res1: AtomGroup for a residue from selection 1.
            res2: AtomGroup for a residue from selection 2.

        Returns:
            Proportion of simulation time spent in hydrophobic contact.
        """
        h1 = self.u.select_atoms('resname DUMMY')
        h2 = self.u.select_atoms('resname DUMMY')

        for atom in res1.atoms:
            if 'C' in atom.type:
                h1 += atom

        for atom in res2.atoms:
            if 'C' in atom.type:
                h2 += atom

        n_frames = 0
        for _ts in self.u.trajectory:
            da = distance_array(h1, h2)
            if np.min(da) < self.hydr:
                n_frames += 1

        return n_frames / self.n_frames

    def survey_donors_acceptors(
        self, res1: mda.AtomGroup, res2: mda.AtomGroup
    ) -> tuple[mda.AtomGroup, mda.AtomGroup]:
        """Identify potential hydrogen bond donors and acceptors.

        First-pass distance threshold to identify potential hydrogen bonds.
        Should be followed by querying H-bond angles but this serves to
        reduce the search space.

        Args:
            res1: AtomGroup for a residue from selection 1.
            res2: AtomGroup for a residue from selection 2.

        Returns:
            Tuple of (donors, acceptors) AtomGroups containing atoms
            that pass the crude distance cutoff.
        """
        donors = self.u.select_atoms('resname DUMMY')
        acceptors = self.u.select_atoms('resname DUMMY')

        for atom in res1.atoms:
            if any([a in atom.type for a in ['O', 'N']]):
                if any(['H' in bond for bond in atom.bonded_atoms.types]):
                    donors += atom
                acceptors += atom

        for atom in res2.atoms:
            if any([a in atom.type for a in ['O', 'N']]):
                if any(['H' in bond for bond in atom.bonded_atoms.types]):
                    donors += atom
                acceptors += atom

        distances = distance_array(donors, acceptors)
        contacts = np.where(distances < self.hb_d)
        don_contacts = np.unique(contacts[0])
        acc_contacts = np.unique(contacts[1])

        return donors[don_contacts], acceptors[acc_contacts]

    def evaluate_hbond(self, donor: mda.AtomGroup, acceptor: mda.AtomGroup) -> int:
        """Evaluate hydrogen bond formation in the current frame.

        Checks whether there is a defined hydrogen bond between any
        donor and acceptor atoms using distance and angle criteria.
        Returns early when a valid H-bond is detected.

        Args:
            donor: AtomGroup of potential H-bond donors.
            acceptor: AtomGroup of potential H-bond acceptors.

        Returns:
            1 if a valid hydrogen bond is found, else 0.
        """
        for d in donor.atoms:
            pos1 = d.position
            hpos = [atom.position for atom in d.bonded_atoms if 'H' in atom.type]
            for a in acceptor.atoms:
                pos3 = a.position

                if np.linalg.norm(pos3 - pos1) <= self.hb_d:
                    for pos2 in hpos:
                        v1 = pos2 - pos1
                        v2 = pos3 - pos2

                        v1 /= np.linalg.norm(v1)
                        v2 /= np.linalg.norm(v2)

                        if np.arccos(np.dot(v1, v2)) <= self.hb_a:
                            return 1

        return 0

    def save(self, results: Results) -> None:
        """Save results to a JSON file.

        Args:
            results: Dictionary of results to be saved.
        """
        with open(self.out, 'w') as fout:
            json.dump(results, fout, indent=4)

    def plot_results(self, results: Results) -> None:
        """Generate and save plots of the results.

        Creates bar plots for each combination of covariance type
        (positive/negative) and interaction type (hydrophobic,
        hydrogen bond, salt bridge).

        Args:
            results: Dictionary of results to be plotted.
        """
        df = self.parse_results(results)

        # Nothing met the covariance/interaction thresholds; no plots to make.
        if df.is_empty():
            return

        plot = Path('plots')
        plot.mkdir(exist_ok=True)
        for cov_type in ['positive', 'negative']:
            for int_type in ['Hydrophobic', 'Hydrogen Bond', 'Salt Bridge']:
                data = df.filter(
                    (pl.col('Covariance') == cov_type) & (pl.col(int_type) > 0.0)
                )

                if not data.is_empty():
                    name = f'{cov_type.capitalize()}_Covariance_'
                    name += f"{'_'.join(int_type.split(' '))}.png"

                    self.make_plot(data, int_type, plot / name)

    def parse_results(self, results: Results) -> pl.DataFrame:
        """Prepare results for plotting.

        Removes entries with all-zero interactions and converts to
        a Polars DataFrame for easier plotting.

        Args:
            results: Dictionary of results to be prepped.

        Returns:
            Polars DataFrame with columns for residue pair, interaction
            probabilities, and covariance type.
        """
        data_rows = []
        for cov_type, pair_dict in results.items():
            for pair, data in pair_dict.items():
                if any(val > 0.0 for val in data.values()):
                    row = {
                        'Residue Pair': pair,
                        'Hydrophobic': data['hydrophobic'],
                        'Hydrogen Bond': data['hbond'],
                        'Salt Bridge': data['saltbridge'],
                        'Covariance': cov_type,
                    }

                    data_rows.append(row)

        return pl.DataFrame(data_rows)

    def make_plot(
        self, data: pl.DataFrame, column: str, name: PathLike, fs: int = 15
    ) -> None:
        """Generate a bar plot for a specified interaction type.

        Args:
            data: Polars DataFrame of data.
            column: Column name for the interaction type to plot.
            name: Path to save the plot.
            fs: Font size for plot labels. Defaults to 15.
        """
        _fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        sns.barplot(data=data, x='Residue Pair', y=column, ax=ax)

        ax.set_xlabel('Residue Pair', fontsize=fs)
        ax.set_ylabel('Probability', fontsize=fs)
        ax.set_title(column, fontsize=fs + 2)
        ax.tick_params(labelsize=fs)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(str(name), dpi=300)
