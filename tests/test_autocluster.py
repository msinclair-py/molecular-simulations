"""
Unit tests for autocluster.py module
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from molecular_simulations.analysis.autocluster import (
    AutoKMeans,
    Decomposition,
    GenericDataloader,
    PeriodicDataloader,
)


class TestGenericDataloader:
    """Test suite for GenericDataloader class"""

    def test_init(self):
        """Test initialization of GenericDataloader"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test numpy files
            test_data1 = np.random.rand(10, 5)
            test_data2 = np.random.rand(15, 5)

            file1 = Path(tmpdir) / 'data1.npy'
            file2 = Path(tmpdir) / 'data2.npy'

            np.save(file1, test_data1)
            np.save(file2, test_data2)

            # Initialize dataloader - expects list, not np.array
            loader = GenericDataloader([file1, file2])

            # Check that data is loaded and vstacked correctly
            assert loader.data.shape == (25, 5)
            assert len(loader.shapes) == 2
            assert loader.shapes[0] == (10, 5)
            assert loader.shapes[1] == (15, 5)

    def test_data_property(self):
        """Test data property returns correct array"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            loader = GenericDataloader([file])
            assert np.array_equal(loader.data, test_data)

    def test_shape_property_uniform(self):
        """Test shape property when all files have same shape"""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                data = np.random.rand(10, 5)
                file = Path(tmpdir) / f'data{i}.npy'
                np.save(file, data)
                files.append(file)

            loader = GenericDataloader(files)
            assert loader.shape == (10, 5)

    def test_shape_property_mixed(self):
        """Test shape property when files have different shapes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = np.random.rand(10, 5)
            data2 = np.random.rand(15, 5)

            file1 = Path(tmpdir) / 'data1.npy'
            file2 = Path(tmpdir) / 'data2.npy'

            np.save(file1, data1)
            np.save(file2, data2)

            loader = GenericDataloader([file1, file2])
            assert loader.shape == [(10, 5), (15, 5)]

    def test_multidimensional_reshape(self):
        """Test reshaping of multidimensional data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 3, 4)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            loader = GenericDataloader([file])
            # Should reshape to (10, 12) since 3*4 = 12
            assert loader.data.shape == (10, 12)

    def test_single_row_not_reshaped(self):
        """A single-row multi-dim array is not flattened (load_data 80->exit).

        vstack of one file yields a single row (``len == 1``), so the
        ``len(self.data_array) > 2`` reshape guard is skipped and the trailing
        dimensions survive instead of being erroneously collapsed to (1, 12).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(1, 3, 4)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            loader = GenericDataloader([file])

            # The >2-row reshape did not run: shape and values are preserved.
            assert loader.data.shape == (1, 3, 4)
            assert np.array_equal(loader.data, test_data)


class TestPeriodicDataloader:
    """Test suite for PeriodicDataloader class"""

    def test_remove_periodicity_method(self):
        """Test the remove_periodicity method directly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file so __init__ doesn't fail
            dummy_data = np.array([[0, 0]])
            dummy_file = Path(tmpdir) / 'dummy.npy'
            np.save(dummy_file, dummy_data)

            loader = PeriodicDataloader([dummy_file])

            # Now test remove_periodicity with our test data
            test_data = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
            result = loader.remove_periodicity(test_data)

            # Check that features are doubled
            assert result.shape[1] == test_data.shape[1] * 2

            # Verify sin/cos decomposition
            # For angle 0: cos(0)=1, sin(0)=0
            assert np.isclose(result[0, 0], 1.0)  # cos(0)
            assert np.isclose(result[0, 1], 0.0)  # sin(0)

            # For angle π/2: cos(π/2)≈0, sin(π/2)=1
            assert np.isclose(result[0, 2], 0.0, atol=1e-10)  # cos(π/2)
            assert np.isclose(result[0, 3], 1.0)  # sin(π/2)

            # For angle π: cos(π)=-1, sin(π)≈0
            assert np.isclose(result[1, 0], -1.0)  # cos(π)
            assert np.isclose(result[1, 1], 0.0, atol=1e-10)  # sin(π)

    def test_load_data_with_periodicity(self):
        """Test loading data with periodicity removal applied"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with known values
            test_data = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
            file = Path(tmpdir) / 'periodic.npy'
            np.save(file, test_data)

            loader = PeriodicDataloader([file])

            # Data should be vstacked into a numpy array
            assert isinstance(loader.data_array, np.ndarray)
            assert loader.data.shape[1] == test_data.shape[1] * 2

            # Verify periodicity was removed
            # For angle 0: cos(0)=1, sin(0)=0
            assert np.isclose(loader.data[0, 0], 1.0)  # cos(0)
            assert np.isclose(loader.data[0, 1], 0.0)  # sin(0)

            # For angle π/2: cos(π/2)≈0, sin(π/2)=1
            assert np.isclose(loader.data[0, 2], 0.0, atol=1e-10)  # cos(π/2)
            assert np.isclose(loader.data[0, 3], 1.0)  # sin(π/2)

    def test_inheritance_from_generic(self):
        """Test that PeriodicDataloader properly inherits from GenericDataloader"""
        assert issubclass(PeriodicDataloader, GenericDataloader)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 3)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            loader = PeriodicDataloader([file])

            # Should have inherited attributes
            assert hasattr(loader, 'files')
            assert hasattr(loader, 'shapes')
            assert hasattr(loader, 'data_array')


class TestAutoKMeans:
    """Test suite for AutoKMeans class"""

    def test_initialization(self):
        """Test AutoKMeans initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data file
            test_data = np.random.rand(20, 5)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            # Initialize AutoKMeans
            auto_km = AutoKMeans(tmpdir, max_clusters=5, stride=1)

            assert auto_km.data_dir == Path(tmpdir)
            assert auto_km.n_clusters == 5
            assert auto_km.stride == 1
            assert auto_km.dataloader is not None
            assert auto_km.decomposition is not None

    def test_sweep_n_clusters(self):
        """sweep_n_clusters runs real KMeans/silhouette and is reproducible."""
        # Two well-separated 2D blobs: silhouette is maximized at k=2.
        rng = np.random.default_rng(0)
        blobs = np.vstack(
            [rng.normal(0.0, 0.1, (10, 2)), rng.normal(5.0, 0.1, (10, 2))]
        )

        def run_sweep():
            with tempfile.TemporaryDirectory() as tmpdir:
                np.save(Path(tmpdir) / 'data.npy', np.random.rand(20, 5))
                auto_km = AutoKMeans(tmpdir, max_clusters=5, stride=1, random_state=42)
                auto_km.reduced = blobs.copy()
                auto_km.sweep_n_clusters([2, 3, 4])
                return auto_km

        auto_km = run_sweep()

        assert auto_km.centers is not None
        assert auto_km.labels is not None
        # The two clean blobs are best described by 2 clusters
        assert len(np.unique(auto_km.labels)) == 2

        # Fixed random_state -> identical labels across runs (issue #13)
        repeat = run_sweep()
        np.testing.assert_array_equal(auto_km.labels, repeat.labels)

    def test_map_centers_to_frames(self):
        """Test mapping cluster centers to frames"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            auto_km = AutoKMeans(tmpdir)
            # Set up reduced data and centers
            auto_km.reduced = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
            auto_km.centers = np.array([[0.1, 0.1], [3.9, 3.9]])
            auto_km.shape = (10, 5)

            auto_km.map_centers_to_frames()

            # Should have one entry per cluster
            assert len(auto_km.cluster_centers) == 2
            # Each value should be a tuple of (replica, frame)
            assert all(
                isinstance(v, tuple) and len(v) == 2
                for v in auto_km.cluster_centers.values()
            )

    def test_save_centers(self):
        """Test saving cluster centers to JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / 'data.npy'
            np.save(file, test_data)

            auto_km = AutoKMeans(tmpdir)
            auto_km.cluster_centers = {0: (0, 1), 1: (0, 2)}

            auto_km.save_centers()

            # Check that file was created
            output_file = Path(tmpdir) / 'cluster_centers.json'
            assert output_file.exists()

            # Verify contents
            import json

            with open(output_file) as f:
                loaded = json.load(f)
            assert '0' in loaded  # Keys become strings in JSON
            assert '1' in loaded

    def test_save_labels(self):
        """Test saving cluster labels to parquet"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file1 = Path(tmpdir) / 'test1.npy'
            file2 = Path(tmpdir) / 'test2.npy'
            np.save(file1, test_data[:5])
            np.save(file2, test_data[5:])

            auto_km = AutoKMeans(tmpdir)
            auto_km.dataloader.files = [file1, file2]
            # Set uniform shape
            auto_km.dataloader.shapes = [(5, 5), (5, 5)]
            auto_km.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

            auto_km.save_labels()

            # Check that file was created
            output_file = Path(tmpdir) / 'cluster_assignments.parquet'
            assert output_file.exists()

            # Verify it can be read
            df = pl.read_parquet(output_file)
            assert 'system' in df.columns
            assert 'frame' in df.columns
            assert 'cluster' in df.columns
            assert len(df) == 10


class TestAutoKMeansGuards:
    """Test AutoKMeans guard clauses."""

    def test_map_centers_without_centers_raises(self):
        """map_centers_to_frames guards against an absent clustering result.

        When sweep_n_clusters has not produced any centers (``self.centers`` is
        None -- the state left behind by a sweep that found no valid
        clustering), map_centers_to_frames raises a clear ValueError rather than
        iterating over nothing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(Path(tmpdir) / 'data.npy', np.random.rand(10, 5))
            auto_km = AutoKMeans(tmpdir)
            auto_km.centers = None

            with pytest.raises(ValueError, match='sweep_n_clusters'):
                auto_km.map_centers_to_frames()


class TestAutoKMeansRun:
    """Test AutoKMeans.run end-to-end against real sklearn (no mocks)."""

    def test_run_end_to_end(self, tmp_path):
        """run() reduces, clusters, maps centers and writes real outputs."""
        # Two well-separated 5D blobs stored as two "replica" files. PCA + KMeans
        # over a silhouette sweep should recover exactly two clusters.
        rng = np.random.default_rng(0)
        blob0 = rng.normal(0.0, 0.1, (10, 5))
        blob1 = rng.normal(5.0, 0.1, (10, 5))
        np.save(tmp_path / 'rep0.npy', blob0)
        np.save(tmp_path / 'rep1.npy', blob1)

        auto_km = AutoKMeans(str(tmp_path), max_clusters=5, stride=1, random_state=42)
        auto_km.run()

        # Real cluster results were assigned.
        assert auto_km.labels is not None
        assert auto_km.centers is not None
        assert len(auto_km.labels) == 20
        # The two clean blobs are best described by 2 clusters.
        assert len(np.unique(auto_km.labels)) == 2

        # Centers were mapped to real (replica, frame) coordinates.
        assert len(auto_km.cluster_centers) == 2
        for value in auto_km.cluster_centers.values():
            assert isinstance(value, tuple) and len(value) == 2
            rep, frame = value
            assert 0 <= rep < 2
            assert 0 <= frame < 10

        # Output files were written to disk.
        assert (tmp_path / 'cluster_centers.json').exists()
        assignments = tmp_path / 'cluster_assignments.parquet'
        assert assignments.exists()

        df = pl.read_parquet(assignments)
        assert set(df.columns) == {'system', 'frame', 'cluster'}
        assert len(df) == 20
        assert set(df['cluster'].to_list()) == set(np.unique(auto_km.labels).tolist())


class TestAutoKMeansSaveLabelsNonUniform:
    """Test save_labels with non-uniform shapes (line 311)."""

    def test_save_labels_mixed_shapes(self):
        """Test save_labels when files have different shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = np.random.rand(5, 3)
            data2 = np.random.rand(10, 3)
            file1 = Path(tmpdir) / 'a.npy'
            file2 = Path(tmpdir) / 'b.npy'
            np.save(file1, data1)
            np.save(file2, data2)

            auto_km = AutoKMeans(tmpdir)
            auto_km.labels = np.array([0] * 15)

            auto_km.save_labels()

            output_file = Path(tmpdir) / 'cluster_assignments.parquet'
            assert output_file.exists()
            df = pl.read_parquet(output_file)
            assert len(df) == 15


class TestDecomposition:
    """Test suite for Decomposition class"""

    def test_pca_initialization(self):
        """Test PCA decomposition initialization"""
        decomp = Decomposition('PCA', n_components=2)
        assert decomp.decomposer is not None
        # Should be a PCA instance
        from sklearn.decomposition import PCA

        assert isinstance(decomp.decomposer, PCA)

    def test_fit_transform(self):
        """Test fit_transform method"""
        X = np.random.rand(100, 10)
        decomp = Decomposition('PCA', n_components=2)

        X_reduced = decomp.fit_transform(X)

        assert X_reduced.shape == (100, 2)
        assert isinstance(X_reduced, np.ndarray)

    def test_separate_fit_and_transform(self):
        """Test separate fit and transform methods"""
        X = np.random.rand(100, 10)
        decomp = Decomposition('PCA', n_components=2)

        decomp.fit(X)
        X_reduced = decomp.transform(X)

        assert X_reduced.shape == (100, 2)
        assert isinstance(X_reduced, np.ndarray)

    def test_fit_transform_consistency(self):
        """Test that fit_transform gives same result as fit then transform"""
        X = np.random.rand(100, 10)

        # Method 1: fit_transform
        decomp1 = Decomposition('PCA', n_components=2, random_state=42)
        X_reduced1 = decomp1.fit_transform(X)

        # Method 2: separate fit and transform
        decomp2 = Decomposition('PCA', n_components=2, random_state=42)
        decomp2.fit(X)
        X_reduced2 = decomp2.transform(X)

        # Should give same results
        assert np.allclose(X_reduced1, X_reduced2)

    def test_unsupported_algorithm(self):
        """Test that unsupported algorithms raise appropriate errors"""
        with pytest.raises(ValueError):
            # TICA and UMAP are not implemented (None in the dict)
            decomp = Decomposition('TICA')
            decomp.fit_transform(np.random.rand(10, 5))

    def test_pca_with_custom_params(self):
        """Test PCA with custom parameters"""
        X = np.random.rand(100, 10)

        # Test with different n_components
        decomp3 = Decomposition('PCA', n_components=3)
        X_reduced3 = decomp3.fit_transform(X)
        assert X_reduced3.shape == (100, 3)

        decomp5 = Decomposition('PCA', n_components=5)
        X_reduced5 = decomp5.fit_transform(X)
        assert X_reduced5.shape == (100, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
