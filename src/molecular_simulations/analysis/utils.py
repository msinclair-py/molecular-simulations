"""Analysis utility functions and classes.

This module provides utility classes for embedding analysis data into
PDB files, particularly for visualization of per-residue properties.
"""

import shutil
from pathlib import Path

import MDAnalysis as mda
import numpy as np

OptPath = Path | str | None


class EmbedData:
    """Embed data into the beta-factor column of a PDB file.

    Writes out to the same path as input PDB (backing up the original)
    unless an output path is explicitly provided. Embedding data should
    be provided as a dictionary where keys are MDAnalysis selection
    strings and values are numpy arrays.

    Attributes:
        pdb: Path to the PDB file.
        embeddings: Dictionary of selections and data to embed.
        out: Output path for the modified PDB.
        u: MDAnalysis Universe object.

    Args:
        pdb: Path to PDB file to load. Also serves as output if one
            is not provided.
        embedding_dict: Dictionary with MDAnalysis selections as keys
            and data arrays as values. Arrays should have shape
            (n_frames, n_residues, n_datapoints) or (n_residues, n_datapoints).
        out: Output path. If None, uses the input PDB path.

    Example:
        >>> data = {'protein': np.random.rand(100)}  # 100 residues
        >>> embedder = EmbedData('structure.pdb', data)
        >>> embedder.embed()
    """

    def __init__(
        self, pdb: Path, embedding_dict: dict[str, np.ndarray], out: OptPath = None
    ):
        """Initialize the EmbedData instance.

        Args:
            pdb: Path to input PDB file.
            embedding_dict: Selection to data mapping.
            out: Optional output path.
        """
        self.pdb = pdb if isinstance(pdb, Path) else Path(pdb)
        self.embeddings = embedding_dict
        self.out: Path = Path(out) if out is not None else self.pdb

        self.u = mda.Universe(str(self.pdb))

    def embed(self) -> None:
        """Embed all data and write the modified PDB.

        Unpacks the embedding dictionary, embeds data into each
        selection, and writes out the new PDB file.
        """
        for sel, data in self.embeddings.items():
            self.embed_selection(sel, data)

        self.write_new_pdb()

    def embed_selection(self, selection: str, data: np.ndarray) -> None:
        """Embed data into a specific selection's beta column.

        Args:
            selection: MDAnalysis selection string.
            data: Array of data to embed. Shape should be
                (n_residues_in_selection,) or compatible.
        """
        sel = self.u.select_atoms(selection)

        for residue, datum in zip(sel.residues, data, strict=True):
            residue.atoms.tempfactors = np.full(residue.atoms.tempfactors.shape, datum)

    def write_new_pdb(self) -> None:
        """Write the modified PDB file.

        If output path exists and equals the input path, backs up the
        original PDB with '.orig.pdb' extension (only if backup doesn't
        already exist to prevent overwriting the true original).
        """
        if self.out.exists() and not self.pdb.with_suffix('.orig.pdb').exists():
            shutil.copyfile(str(self.pdb), str(self.pdb.with_suffix('.orig.pdb')))

        with mda.Writer(str(self.out)) as W:
            W.write(self.u.atoms)


class EmbedEnergyData(EmbedData):
    """Embed energy data into PDB beta-factor column.

    Special case of EmbedData for non-bonded energy data with both
    LJ and Coulombic terms. Sums the energy terms and rescales to
    handle negative values (which many visualization tools don't
    support in beta factors).

    Args:
        pdb: Path to PDB file to load.
        embedding_dict: Dictionary with MDAnalysis selections as keys
            and energy data arrays as values.
        out: Output path. If None, uses the input PDB path.

    Example:
        >>> energies = {'chainA': energy_array}  # shape (n_frames, n_res, 2)
        >>> embedder = EmbedEnergyData('structure.pdb', energies)
        >>> embedder.embed()
    """

    def __init__(
        self, pdb: Path, embedding_dict: dict[str, np.ndarray], out: OptPath = None
    ):
        """Initialize the EmbedEnergyData instance.

        Args:
            pdb: Path to input PDB file.
            embedding_dict: Selection to energy data mapping.
            out: Optional output path.
        """
        super().__init__(pdb, embedding_dict, out)
        self.embeddings = self.preprocess()

    def preprocess(self) -> dict[str, np.ndarray]:
        """Process embeddings data for PDB embedding.

        Reduces multi-dimensional energy data to 1D per-residue values
        and rescales to ensure non-negative values while preserving
        relative differences.

        Returns:
            Processed data dictionary ready for embedding.
        """
        new_embeddings = dict()
        all_data = []
        for data in self.embeddings.values():
            sanitized = self.sanitize_data(data)
            all_data.append(sanitized)

        rescaling_factor = np.min(np.concatenate(all_data))
        for sel, data in self.embeddings.items():
            sanitized = self.sanitize_data(data)
            rescaled = sanitized / rescaling_factor
            rescaled[np.where(rescaled > 1.0)] = 1.0
            new_embeddings[sel] = rescaled

        return new_embeddings

    @staticmethod
    def sanitize_data(data: np.ndarray) -> np.ndarray:
        """Reduce data to one-dimensional per-residue values.

        Takes data of shape (n_frames, n_residues, n_terms) and
        returns array of shape (n_residues,) by averaging over
        frames and summing energy terms.

        Args:
            data: Input data array with multiple dimensions.

        Returns:
            One-dimensional processed data array.
        """
        if len(data.shape) > 2:
            data = np.mean(data, axis=0)

        if data.shape[1] > 1:
            data = np.sum(data, axis=1)

        return data
