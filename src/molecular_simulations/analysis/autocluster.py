"""Automated clustering module for molecular dynamics data.

This module provides tools for automatic clustering of molecular dynamics
trajectory data using KMeans++ with dimensionality reduction.
"""

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

PathLike = Path | str


@runtime_checkable
class DataloaderProtocol(Protocol):
    """Protocol for dataloader classes used by AutoKMeans."""

    files: list[PathLike]

    @property
    def data(self) -> np.ndarray: ...

    @property
    def shape(self) -> tuple[int, ...] | list[tuple[int, ...]]: ...


class GenericDataloader:
    """Loads generic data stored in numpy arrays.

    Stores the full dataset and is capable of loading data with variable
    row lengths but must be consistent in the columnar dimension.

    Attributes:
        files: List of paths to the loaded data files.
        data_array: The concatenated data array.
        shapes: List of shapes of the original data files.

    Args:
        data_files: List of paths to input data files (.npy format).

    Example:
        >>> loader = GenericDataloader(["data1.npy", "data2.npy"])
        >>> print(loader.data.shape)
    """

    def __init__(self, data_files: list[PathLike]):
        """Initialize the dataloader with a list of data files.

        Args:
            data_files: List of paths to input data files (.npy format).
        """
        self.files = data_files
        self.data_array: np.ndarray = np.array([])
        self.shapes: list[tuple[int, ...]] = []
        self.load_data()

    def load_data(self) -> None:
        """Load and concatenate data from all files into one array.

        Lumps data into one large array by vertical stacking. If the
        stacked array has more than 2 rows along its first axis, it is
        flattened to 2D by collapsing all trailing axes into a single
        dimension.
        """
        data_list: list[np.ndarray] = []
        self.shapes = []
        for f in self.files:
            temp = np.load(str(f))
            self.shapes.append(temp.shape)
            data_list.append(temp)

        self.data_array = np.vstack(data_list)
        if len(self.data_array) > 2:
            x, *y = self.data_array.shape
            shape2 = 1
            for shape in y:
                shape2 *= shape

            self.data_array = self.data_array.reshape(x, shape2)

    @property
    def data(self) -> np.ndarray:
        """Return the internal data array.

        Returns:
            The concatenated and reshaped data array.
        """
        return self.data_array

    @property
    def shape(self) -> tuple[int, ...] | list[tuple[int, ...]]:
        """Return the shape(s) of the input data.

        Returns:
            If all input files have the same shape, returns that shape.
            Otherwise, returns a list of shapes in the order the files
            were provided.
        """
        if len(set(self.shapes)) == 1:
            return self.shapes[0]

        return self.shapes


class PeriodicDataloader(GenericDataloader):
    """Dataloader that decomposes periodic data using sin and cos.

    Extends GenericDataloader to handle periodic data by decomposing
    each feature into sin and cos components, effectively doubling
    the number of features.

    Args:
        data_files: List of paths to input data files containing
            periodic data (e.g., dihedral angles).

    Example:
        >>> loader = PeriodicDataloader(["dihedrals.npy"])
        >>> # Original 10 features become 20 features
    """

    def __init__(self, data_files: list[PathLike]):
        """Initialize the periodic dataloader.

        Args:
            data_files: List of paths to input data files.
        """
        super().__init__(data_files)

    def load_data(self) -> None:
        """Load periodic data and remove periodicity.

        Loads each file, applies the periodicity removal transformation,
        and stores the results.
        """
        data_list: list[np.ndarray] = []
        self.shapes = []
        for f in self.files:
            temp = self.remove_periodicity(np.load(str(f)))

            self.shapes.append(temp.shape)
            data_list.append(temp)

        self.data_array = np.vstack(data_list)

    def remove_periodicity(self, arr: np.ndarray) -> np.ndarray:
        """Remove periodicity from each feature using sin and cos.

        Each column is expanded into two columns such that the indices
        become i -> 2*i, 2*i + 1. This preserves the circular nature
        of periodic variables like angles.

        Args:
            arr: Data array with periodic features. Shape should be
                (n_samples, n_features).

        Returns:
            New array with shape (arr.shape[0], arr.shape[1] * 2) where
            each original feature is replaced by its cos and sin values.
        """
        n_features = arr.shape[1]
        return_arr = np.zeros((arr.shape[0], n_features * 2))

        for i in range(n_features):
            return_arr[:, 2 * i] = np.cos(arr[:, i])
            return_arr[:, 2 * i + 1] = np.sin(arr[:, i])

        return return_arr


class AutoKMeans:
    """Automatic clustering using KMeans++ with dimensionality reduction.

    Performs automatic clustering including dimensionality reduction of
    the feature space and parameter sweep over number of clusters using
    silhouette score optimization.

    Attributes:
        data: The loaded data array.
        shape: Shape of the input data.
        reduced: Dimensionality-reduced data.
        centers: Cluster centers in reduced space.
        labels: Cluster assignments for each data point.
        cluster_centers: Mapping of cluster index to (replica, frame) tuple.

    Args:
        data_directory: Directory where data files can be found.
        pattern: Optional filename pattern to select subset of npy files
            using glob. Defaults to empty string (all .npy files).
        dataloader: Which dataloader class to use. Defaults to
            GenericDataloader.
        max_clusters: Maximum number of clusters to test during parameter
            sweep. Defaults to 10.
        stride: Linear stride of number of clusters during parameter sweep.
            Helps avoid testing too many values. Defaults to 1.
        reduction_algorithm: Which dimensionality reduction algorithm to
            use. Currently only 'PCA' is supported. Defaults to 'PCA'.
        reduction_kws: Keyword arguments for the reduction algorithm.
            Defaults to {'n_components': 2}.

    Example:
        >>> clusterer = AutoKMeans("data/", max_clusters=15)
        >>> clusterer.run()
        >>> print(clusterer.cluster_centers)
    """

    def __init__(
        self,
        data_directory: PathLike,
        pattern: str = '',
        dataloader: type[GenericDataloader] = GenericDataloader,
        max_clusters: int = 10,
        stride: int = 1,
        reduction_algorithm: str = 'PCA',
        reduction_kws: dict[str, Any] = {'n_components': 2}, # noqa: B006
    ):
        """Initialize the automatic clustering workflow.

        Args:
            data_directory: Directory where data files can be found.
            pattern: Optional filename pattern for glob matching.
            dataloader: Dataloader class to use for loading data.
            max_clusters: Maximum number of clusters to test.
            stride: Step size for cluster number sweep.
            reduction_algorithm: Dimensionality reduction method.
            reduction_kws: Arguments for the reduction algorithm.
        """
        self.data_dir = Path(data_directory)
        self.dataloader: DataloaderProtocol = dataloader(
            list(self.data_dir.glob(f'{pattern}*.npy'))
        )
        self.data = self.dataloader.data
        self.shape = self.dataloader.shape

        self.n_clusters = max_clusters
        self.stride = stride

        self.decomposition = Decomposition(reduction_algorithm, **reduction_kws)

    def run(self) -> None:
        """Run the complete automated clustering workflow.

        Executes dimensionality reduction, parameter sweep, center mapping,
        and saves results to disk.
        """
        self.reduce_dimensionality()
        self.sweep_n_clusters([n for n in range(2, self.n_clusters, self.stride)])
        self.map_centers_to_frames()
        self.save_centers()
        self.save_labels()

    def reduce_dimensionality(self) -> None:
        """Perform dimensionality reduction on the data.

        Uses the configured decomposition algorithm to reduce the
        feature space dimensionality.
        """
        self.reduced = self.decomposition.fit_transform(self.data)

    def sweep_n_clusters(self, n_clusters: list[int]) -> None:
        """Sweep over number of clusters to find optimal clustering.

        Uses silhouette score to perform a parameter sweep over number
        of clusters. Stores the cluster centers and labels for the best
        performing parameterization.

        Args:
            n_clusters: List of cluster numbers to test.
        """
        best_centers = None
        best_labels = None
        best_score = 0.0
        for n in tqdm(
            n_clusters,
            total=len(n_clusters),
            position=0,
            leave=False,
            desc='Sweeping `n_clusters`',
        ):
            clusterer = KMeans(n_clusters=n)
            labels = clusterer.fit_predict(self.reduced)
            average_score = silhouette_score(self.reduced, labels)

            if average_score > best_score:
                best_centers = clusterer.cluster_centers_
                best_labels = labels
                best_score = average_score

        self.centers = best_centers
        self.labels = best_labels

    def map_centers_to_frames(self) -> None:
        """Map cluster centers to the closest actual data points.

        Finds and stores the data point which lies closest to each
        cluster center, recording the replica and frame indices.
        """
        if self.centers is None:
            msg = 'No cluster centers found. Run sweep_n_clusters first.'
            raise ValueError(msg)
        shape = self.shape
        if not isinstance(shape, tuple):
            msg = 'Cannot map centers when input files have different shapes.'
            raise ValueError(msg)
        n_frames: int = shape[0]
        cluster_centers = {i: None for i in range(len(self.centers))}
        for i, center in enumerate(self.centers):
            closest = 100.0
            for p, point in enumerate(self.reduced):
                if (dist := np.linalg.norm(point - center)) < closest:
                    rep = p // n_frames
                    frame = p % n_frames
                    cluster_centers[i] = (rep, frame)
                    closest = dist

        self.cluster_centers = cluster_centers

    def save_centers(self) -> None:
        """Save cluster centers to a JSON file.

        Writes the cluster_centers dictionary to 'cluster_centers.json'
        in the data directory.
        """
        with open(str(self.data_dir / 'cluster_centers.json'), 'w') as fout:
            json.dump(self.cluster_centers, fout, indent=4)

    def save_labels(self) -> None:
        """Save cluster labels to a Parquet file.

        Generates a Polars DataFrame containing system, frame, and cluster
        label assignments and saves to 'cluster_assignments.parquet'.
        """
        files = self.dataloader.files
        if isinstance(self.dataloader.shape, tuple):
            shapes = [self.dataloader.shape[0]] * len(files)
        else:
            shapes = [shape[0] for shape in self.dataloader.shape]

        df = pl.DataFrame()
        for file, shape in zip(files, shapes, strict=True):
            temp = pl.DataFrame(
                {'system': [file.name] * shape, 'frame': np.arange(shape)}
            )
            df = pl.concat([df, temp], how='vertical')

        df = df.with_columns(pl.Series(self.labels).alias('cluster'))

        df.write_parquet(str(self.data_dir / 'cluster_assignments.parquet'))


class Decomposition:
    """Wrapper for dimensionality reduction algorithms.

    Provides a thin wrapper around various dimensionality reduction
    algorithms using scikit-learn style methods.

    Attributes:
        decomposer: The underlying decomposition algorithm instance.

    Args:
        algorithm: Which algorithm to use. Options are 'PCA', 'TICA',
            and 'UMAP'. Currently only 'PCA' is fully supported.
        **kwargs: Algorithm-specific keyword arguments passed to the
            decomposer constructor.

    Example:
        >>> decomp = Decomposition("PCA", n_components=3)
        >>> reduced_data = decomp.fit_transform(data)
    """

    def __init__(self, algorithm: str, **kwargs):
        """Initialize the decomposition wrapper.

        Args:
            algorithm: Name of the decomposition algorithm.
            **kwargs: Arguments passed to the algorithm constructor.

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        algorithms: dict[str, type[PCA] | None] = {
            'PCA': PCA,
            'TICA': None,
            'UMAP': None,
        }

        algo = algorithms[algorithm]
        if algo is None:
            msg = f'Algorithm {algorithm!r} is not yet supported.'
            raise ValueError(msg)
        self.decomposer = algo(**kwargs)

    def fit(self, X: np.ndarray) -> None:
        """Fit the decomposer with data.

        Args:
            X: Array of input data with shape (n_samples, n_features).
        """
        self.decomposer.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted decomposer.

        Args:
            X: Array of input data with shape (n_samples, n_features).

        Returns:
            Reduced dimension data with shape (n_samples, n_components).

        Raises:
            sklearn.exceptions.NotFittedError: If called before fit().
        """
        return self.decomposer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the decomposer and transform data in one step.

        Args:
            X: Array of input data with shape (n_samples, n_features).

        Returns:
            Reduced dimension data with shape (n_samples, n_components).
        """
        return self.decomposer.fit_transform(X)
