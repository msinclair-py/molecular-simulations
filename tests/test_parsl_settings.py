"""
Unit tests for utils/parsl_settings.py module
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


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

    @patch('molecular_simulations.utils.parsl_settings.Config')
    @patch('molecular_simulations.utils.parsl_settings.HighThroughputExecutor')
    @patch('molecular_simulations.utils.parsl_settings.LocalProvider')
    def test_local_settings_config_factory(
        self, mock_provider, mock_executor, mock_config
    ):
        """Test LocalSettings config_factory method"""
        from molecular_simulations.utils.parsl_settings import LocalSettings

        settings = LocalSettings()
        settings.config_factory(Path('.'))


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

    @patch('molecular_simulations.utils.parsl_settings.Config')
    @patch('molecular_simulations.utils.parsl_settings.HighThroughputExecutor')
    @patch('molecular_simulations.utils.parsl_settings.PBSProProvider')
    @patch('molecular_simulations.utils.parsl_settings.MpiExecLauncher')
    @patch('molecular_simulations.utils.parsl_settings.address_by_hostname')
    def test_polaris_settings_config_factory(
        self, mock_addr, mock_launcher, mock_provider, mock_executor, mock_config
    ):
        """Test PolarisSettings config_factory method"""
        from molecular_simulations.utils.parsl_settings import PolarisSettings

        mock_addr.return_value = '192.168.1.1'

        settings = PolarisSettings(
            account='test_account', queue='debug', walltime='01:00:00'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.config_factory(run_dir=tmpdir)

        # Should have created config with executor
        mock_config.assert_called_once()
        mock_executor.assert_called_once()


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

    @patch('molecular_simulations.utils.parsl_settings.Config')
    @patch('molecular_simulations.utils.parsl_settings.HighThroughputExecutor')
    @patch('molecular_simulations.utils.parsl_settings.PBSProProvider')
    @patch('molecular_simulations.utils.parsl_settings.MpiExecLauncher')
    def test_aurora_settings_config_factory(
        self, mock_launcher, mock_provider, mock_executor, mock_config
    ):
        """Test AuroraSettings config_factory method"""
        from molecular_simulations.utils.parsl_settings import AuroraSettings

        settings = AuroraSettings(
            account='test_account', queue='debug', walltime='01:00:00'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.config_factory(run_dir=tmpdir)

        # Should have created config
        mock_config.assert_called_once()


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
