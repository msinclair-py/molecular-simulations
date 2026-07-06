"""
Unit tests for utils/parsl_settings.py module
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider, PBSProProvider


class TestBaseSettings:
    """Test suite for BaseSettings class"""

    def test_base_settings_dump_yaml(self):
        """Test dumping settings to YAML file"""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        settings = LocalSettings(
            available_accelerators=2,
            retries=3,
            label='test_htex',
            worker_port_range=(10000, 15000),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'settings.yaml'
            settings.dump_yaml(yaml_path)

            assert yaml_path.exists()

            # Verify contents
            with open(yaml_path) as f:
                loaded = yaml.safe_load(f)

            assert loaded['available_accelerators'] == 2
            assert loaded['retries'] == 3
            assert loaded['label'] == 'test_htex'

    def test_base_settings_from_yaml(self):
        """Test loading settings from YAML file"""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'settings.yaml'

            yaml_content = {
                'available_accelerators': 4,
                'retries': 2,
                'label': 'loaded_htex',
                'worker_port_range': [11000, 12000],
            }

            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)

            settings = LocalSettings.from_yaml(yaml_path)

            assert settings.available_accelerators == 4
            assert settings.retries == 2
            assert settings.label == 'loaded_htex'


class TestLocalSettings:
    """Test suite for LocalSettings class"""

    def test_local_settings_defaults(self):
        """Test LocalSettings default values"""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        settings = LocalSettings()

        assert settings.available_accelerators == 4
        assert settings.retries == 1
        assert settings.label == 'gpu'
        assert settings.worker_port_range == (10000, 20000)

    def test_local_settings_custom_values(self):
        """Test LocalSettings with custom values"""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        settings = LocalSettings(
            available_accelerators=['0', '1'],
            retries=5,
            label='custom',
            worker_port_range=(15000, 16000),
        )

        assert settings.available_accelerators == ['0', '1']
        assert settings.retries == 5
        assert settings.label == 'custom'
        assert settings.worker_port_range == (15000, 16000)

    def test_local_settings_config_factory(self):
        """config_factory builds a real Parsl Config with a local GPU executor."""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        settings = LocalSettings(available_accelerators=2, retries=3, label='gpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            config = settings.config_factory(Path(tmpdir))

        # Real Config object, not a mock: inspect its actual structure.
        assert isinstance(config, Config)
        assert config.retries == 3
        assert len(config.executors) == 1
        executor = config.executors[0]
        assert isinstance(executor, HighThroughputExecutor)
        assert executor.label == 'gpu'
        assert isinstance(executor.provider, LocalProvider)
        # parsl expands an integer accelerator count into per-GPU id strings.
        assert executor.available_accelerators == ['0', '1']


class TestPolarisSettings:
    """Test suite for PolarisSettings class"""

    def test_polaris_settings_init(self):
        """Test PolarisSettings initialization"""
        from molecular_simulations.utils.parsl_settings import PolarisSettings

        settings = PolarisSettings(
            account='test_account', queue='debug', walltime='01:00:00'
        )

        assert settings.account == 'test_account'
        assert settings.queue == 'debug'
        assert settings.walltime == '01:00:00'
        assert settings.num_nodes == 1
        assert settings.cpus_per_node == 64
        assert settings.available_accelerators == 4

    def test_polaris_settings_full_config(self):
        """Test PolarisSettings with all parameters"""
        from molecular_simulations.utils.parsl_settings import PolarisSettings

        settings = PolarisSettings(
            label='custom_htex',
            num_nodes=4,
            worker_init='module load conda',
            scheduler_options='#PBS -l select=4',
            account='production',
            queue='compute',
            walltime='24:00:00',
            cpus_per_node=32,
            strategy='htex_auto_scale',
            available_accelerators=['0', '1', '2', '3'],
        )

        assert settings.label == 'custom_htex'
        assert settings.num_nodes == 4
        assert settings.worker_init == 'module load conda'
        assert settings.cpus_per_node == 32
        assert settings.strategy == 'htex_auto_scale'

    def test_polaris_settings_config_factory(self):
        """config_factory builds a real PBSPro-backed Config for Polaris."""
        from molecular_simulations.utils.parsl_settings import PolarisSettings

        settings = PolarisSettings(
            account='test_account', queue='debug', walltime='01:00:00'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = settings.config_factory(run_dir=tmpdir)

        assert isinstance(config, Config)
        assert len(config.executors) == 1
        executor = config.executors[0]
        assert isinstance(executor, HighThroughputExecutor)
        assert executor.label == 'htex'
        assert isinstance(executor.provider, PBSProProvider)
        # The real PBSPro provider carries the account/queue/walltime through.
        assert executor.provider.account == 'test_account'
        assert executor.provider.queue == 'debug'
        assert executor.provider.walltime == '01:00:00'


class TestAuroraSettings:
    """Test suite for AuroraSettings class"""

    def test_aurora_settings_init(self):
        """Test AuroraSettings initialization"""
        from molecular_simulations.utils.parsl_settings import AuroraSettings

        settings = AuroraSettings(
            account='aurora_account', queue='workq', walltime='02:00:00'
        )

        assert settings.account == 'aurora_account'
        assert settings.queue == 'workq'
        assert settings.walltime == '02:00:00'
        assert settings.num_nodes == 1
        assert settings.cpus_per_node == 48
        assert len(settings.available_accelerators) == 12

    def test_aurora_settings_available_accelerators(self):
        """Test AuroraSettings default accelerators"""
        from molecular_simulations.utils.parsl_settings import AuroraSettings

        settings = AuroraSettings(account='test', queue='debug', walltime='00:30:00')

        # Should have 12 accelerators (0-11)
        assert settings.available_accelerators == [str(i) for i in range(12)]

    def test_aurora_settings_config_factory(self):
        """config_factory builds a real PBSPro-backed Config for Aurora."""
        from molecular_simulations.utils.parsl_settings import AuroraSettings

        settings = AuroraSettings(
            account='test_account', queue='debug', walltime='01:00:00'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = settings.config_factory(run_dir=tmpdir)

        assert isinstance(config, Config)
        assert len(config.executors) == 1
        executor = config.executors[0]
        assert isinstance(executor, HighThroughputExecutor)
        assert isinstance(executor.provider, PBSProProvider)
        # Aurora exposes its 12 GPU tiles as accelerator ids.
        assert len(executor.available_accelerators) == 12


class TestGetNodeCount:
    """Test suite for get_node_count scheduler-environment inference."""

    def test_get_node_count_slurm(self, monkeypatch):
        """SLURM_JOB_NUM_NODES takes precedence and is returned as an int."""
        from molecular_simulations.utils.parsl_settings import get_node_count

        monkeypatch.delenv('PBS_NODEFILE', raising=False)
        monkeypatch.setenv('SLURM_JOB_NUM_NODES', '4')

        assert get_node_count() == 4

    def test_get_node_count_pbs_nodefile(self, monkeypatch, tmp_path):
        """With no SLURM var, the PBS nodefile line count gives the node count."""
        from molecular_simulations.utils.parsl_settings import get_node_count

        nodefile = tmp_path / 'nodes'
        nodefile.write_text('nodeA\nnodeB\nnodeC\n')

        monkeypatch.delenv('SLURM_JOB_NUM_NODES', raising=False)
        monkeypatch.setenv('PBS_NODEFILE', str(nodefile))

        # Three real lines in the nodefile -> three nodes.
        assert get_node_count() == 3

    def test_get_node_count_default(self, monkeypatch):
        """With neither SLURM nor PBS set, the count defaults to 1."""
        from molecular_simulations.utils.parsl_settings import get_node_count

        monkeypatch.delenv('SLURM_JOB_NUM_NODES', raising=False)
        monkeypatch.delenv('PBS_NODEFILE', raising=False)

        assert get_node_count() == 1


class TestLocalCPUSettings:
    """Test suite for LocalCPUSettings.config_factory."""

    def test_local_cpu_settings_config_factory(self):
        """config_factory builds a real CPU-only local Parsl Config."""
        from molecular_simulations.utils.parsl_settings import LocalCPUSettings

        settings = LocalCPUSettings(
            max_workers_per_node=3, cores_per_worker=2.0, retries=2
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = settings.config_factory(Path(tmpdir))

        assert isinstance(config, Config)
        assert config.retries == 2
        assert len(config.executors) == 1
        executor = config.executors[0]
        assert isinstance(executor, HighThroughputExecutor)
        assert executor.label == 'cpu'
        assert isinstance(executor.provider, LocalProvider)
        # CPU executor carries worker/core counts, not GPU accelerators.
        assert executor.max_workers_per_node == 3
        assert executor.cores_per_worker == 2.0
        # No accelerators are configured for a CPU-only executor.
        assert executor.available_accelerators == []
        assert settings.available_accelerators == []


class TestHeterogeneousSettings:
    """Test suite for HeterogeneousSettings.config_factory."""

    def test_heterogeneous_settings_config_factory(self):
        """config_factory builds a real Config with separate GPU and CPU executors."""
        from molecular_simulations.utils.parsl_settings import HeterogeneousSettings

        settings = HeterogeneousSettings(
            max_workers_per_node=2,
            cores_per_worker=4.0,
            available_accelerators=4,
            retries=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = settings.config_factory(Path(tmpdir))

        assert isinstance(config, Config)
        # Two executors: one GPU-pinned, one CPU.
        assert len(config.executors) == 2
        gpu_executor, cpu_executor = config.executors

        assert isinstance(gpu_executor, HighThroughputExecutor)
        assert isinstance(cpu_executor, HighThroughputExecutor)
        assert gpu_executor.label == 'gpu'
        assert cpu_executor.label == 'cpu'
        assert isinstance(gpu_executor.provider, LocalProvider)
        assert isinstance(cpu_executor.provider, LocalProvider)

        # GPU executor: the integer accelerator count expands to per-GPU ids and
        # drives the worker count.
        assert gpu_executor.available_accelerators == ['0', '1', '2', '3']
        assert gpu_executor.max_workers_per_node == 4

        # CPU executor: worker/core counts, no accelerators.
        assert cpu_executor.max_workers_per_node == 2
        assert cpu_executor.cores_per_worker == 4.0
        assert cpu_executor.available_accelerators == []


class TestSettingsRoundTrip:
    """Test round-trip serialization for settings classes"""

    def test_local_settings_round_trip(self):
        """Test LocalSettings YAML round-trip"""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        original = LocalSettings(
            available_accelerators=8, retries=3, label='roundtrip_test'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'settings.yaml'

            original.dump_yaml(yaml_path)
            loaded = LocalSettings.from_yaml(yaml_path)

            assert loaded.available_accelerators == original.available_accelerators
            assert loaded.retries == original.retries
            assert loaded.label == original.label

    def test_polaris_settings_round_trip(self):
        """Test PolarisSettings YAML round-trip"""
        from molecular_simulations.utils.parsl_settings import PolarisSettings

        original = PolarisSettings(
            account='round_trip', queue='test_queue', walltime='10:00:00', num_nodes=8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'polaris.yaml'

            original.dump_yaml(yaml_path)
            loaded = PolarisSettings.from_yaml(yaml_path)

            assert loaded.account == original.account
            assert loaded.queue == original.queue
            assert loaded.walltime == original.walltime
            assert loaded.num_nodes == original.num_nodes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
