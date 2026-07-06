"""Unit tests for build/build_calvados.py module.

CGBuilder construction, ``from_dict`` and parameter/attribute storage need no
CALVADOS objects (the ``calvados.cfg`` import is performed lazily inside
``write_config`` / ``write_components``), so those tests run REAL in CI. The
methods that instantiate ``calvados.cfg.Config`` / ``Components`` and write the
YAML files are skip-gated on CALVADOS availability and execute only where the
module is installed.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

_have_calvados = importlib.util.find_spec('calvados') is not None
requires_calvados = pytest.mark.skipif(
    not _have_calvados, reason='calvados not available'
)


@pytest.fixture
def builder_inputs(tmp_path):
    """Create real input files for a CGBuilder."""
    pdb_path = tmp_path / 'test.pdb'
    pdb_path.write_text(
        'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
    )
    residues_file = tmp_path / 'residues.csv'
    residues_file.write_text('resname,sigma\nALA,0.5\n')
    domains_file = tmp_path / 'domains.yaml'
    domains_file.write_text('domains: []\n')

    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    return {
        'pdb': pdb_path,
        'residues': residues_file,
        'domains': domains_file,
        'output': output_dir,
        'root': tmp_path,
    }


# ============================================================================
# REAL construction / parameter logic (no CALVADOS objects needed)
# ============================================================================


class TestCGBuilder:
    """CGBuilder construction and attribute storage."""

    def test_cgbuilder_init(self, builder_inputs):
        from molecular_simulations.build.build_calvados import CGBuilder

        builder = CGBuilder(
            path=builder_inputs['output'],
            input_pdb=builder_inputs['pdb'],
            residues_file=builder_inputs['residues'],
            domains_file=builder_inputs['domains'],
            box_dim=[100.0, 100.0, 100.0],
            temp=310.0,
            ion_conc=0.15,
            pH=7.4,
            topol='center',
            dcd_freq=2000,
            n_steps=1000000,
            platform='CUDA',
            restart='checkpoint',
            frestart='restart.chk',
            verbose=True,
            molecule_type='protein',
            nmol=1,
            restraint=True,
            charge_termini='end-capped',
            restraint_type='harmonic',
            use_com=True,
            colabfold=0,
            k_harmonic=700.0,
        )

        assert builder.path == builder_inputs['output']
        assert builder.input_pdb == builder_inputs['pdb']
        assert builder.temp == 310.0
        assert builder.ion_conc == 0.15
        assert builder.pH == 7.4
        assert builder.box_dim == [100.0, 100.0, 100.0]
        assert builder.platform == 'CUDA'
        assert builder.nmol == 1
        assert builder.restraint is True
        assert builder.k_harmonic == 700.0

    def test_cgbuilder_converts_str_paths_to_path(self, builder_inputs):
        """String path arguments are coerced to Path objects."""
        from molecular_simulations.build.build_calvados import CGBuilder

        builder = CGBuilder(
            path=str(builder_inputs['output']),
            input_pdb=str(builder_inputs['pdb']),
            residues_file=str(builder_inputs['residues']),
            domains_file=str(builder_inputs['domains']),
            box_dim=[50.0, 50.0, 50.0],
        )

        assert isinstance(builder.path, Path)
        assert isinstance(builder.input_pdb, Path)
        assert isinstance(builder.residues_file, Path)
        assert isinstance(builder.domains_file, Path)
        assert builder.input_pdb == builder_inputs['pdb']

    def test_cgbuilder_from_dict(self, builder_inputs):
        from molecular_simulations.build.build_calvados import CGBuilder

        cg_params = {
            'config': {
                'path': str(builder_inputs['output']),
                'input_pdb': str(builder_inputs['pdb']),
                'box_dim': [80.0, 80.0, 80.0],
                'temp': 300.0,
                'ion_conc': 0.1,
                'pH': 7.0,
                'topol': 'center',
                'dcd_freq': 1000,
                'n_steps': 500000,
                'platform': 'CPU',
                'restart': 'checkpoint',
                'frestart': 'restart.chk',
                'verbose': False,
            },
            'components': {
                'residues_file': str(builder_inputs['residues']),
                'domains_file': str(builder_inputs['domains']),
                'molecule_type': 'protein',
                'nmol': 2,
                'restraint': False,
                'charge_termini': 'both',
                'restraint_type': 'go',
                'use_com': False,
                'colabfold': 1,
                'k_harmonic': 500.0,
            },
        }

        builder = CGBuilder.from_dict(cg_params)

        # Every field threaded through from_dict, read back from the real object.
        assert builder.path == builder_inputs['output']
        assert builder.input_pdb == builder_inputs['pdb']
        assert builder.box_dim == [80.0, 80.0, 80.0]
        assert builder.temp == 300.0
        assert builder.ion_conc == 0.1
        assert builder.pH == 7.0
        assert builder.dcd_freq == 1000
        assert builder.n_steps == 500000
        assert builder.platform == 'CPU'
        assert builder.verbose is False
        assert builder.molecule_type == 'protein'
        assert builder.nmol == 2
        assert builder.restraint is False
        assert builder.charge_termini == 'both'
        assert builder.restraint_type == 'go'
        assert builder.use_com is False
        assert builder.colabfold == 1
        assert builder.k_harmonic == 500.0


class TestCGBuilderParameters:
    """CGBuilder default parameter handling."""

    def test_default_parameters(self, builder_inputs):
        from molecular_simulations.build.build_calvados import CGBuilder

        builder = CGBuilder(
            path=builder_inputs['root'],
            input_pdb=builder_inputs['pdb'],
            residues_file=builder_inputs['residues'],
            domains_file=builder_inputs['domains'],
            box_dim=[50.0, 50.0, 50.0],
        )

        assert builder.temp == 310.0
        assert builder.ion_conc == 0.15
        assert builder.pH == 7.4
        assert builder.topol == 'center'
        assert builder.dcd_freq == 2000
        assert builder.n_steps == 10_000_000
        assert builder.platform == 'CUDA'
        assert builder.restart == 'checkpoint'
        assert builder.frestart == 'restart.chk'
        assert builder.verbose is True
        assert builder.molecule_type == 'protein'
        assert builder.nmol == 1
        assert builder.restraint is True
        assert builder.charge_termini == 'end-capped'
        assert builder.restraint_type == 'harmonic'
        assert builder.use_com is True
        assert builder.colabfold == 0
        assert builder.k_harmonic == 700.0


def test_module_imports_without_calvados():
    """build_calvados imports without CALVADOS installed (lazy cfg import)."""
    import molecular_simulations.build.build_calvados as m

    assert hasattr(m, 'CGBuilder')


# ============================================================================
# Skip-gated REAL CALVADOS object usage (config / components YAML writing)
# ============================================================================


@requires_calvados
class TestCGBuilderWriteRealCalvados:
    """Exercise the real calvados.cfg Config/Components YAML writers."""

    def _builder(self, builder_inputs):
        from molecular_simulations.build.build_calvados import CGBuilder

        return CGBuilder(
            path=builder_inputs['output'],
            input_pdb=builder_inputs['pdb'],
            residues_file=builder_inputs['residues'],
            domains_file=builder_inputs['domains'],
            box_dim=[100.0, 100.0, 100.0],
        )

    def test_write_config(self, builder_inputs):
        builder = self._builder(builder_inputs)
        builder.write_config()

        config_file = builder_inputs['output'] / 'config.yaml'
        assert config_file.exists()
        # Real YAML written to disk and re-loadable.
        loaded = yaml.safe_load(config_file.read_text())
        assert isinstance(loaded, dict)

    def test_write_components(self, builder_inputs):
        builder = self._builder(builder_inputs)
        builder.write_components()

        components_file = builder_inputs['output'] / 'components.yaml'
        assert components_file.exists()
        loaded = yaml.safe_load(components_file.read_text())
        assert loaded is not None

    def test_build_writes_both(self, builder_inputs):
        builder = self._builder(builder_inputs)
        builder.build()

        assert (builder_inputs['output'] / 'config.yaml').exists()
        assert (builder_inputs['output'] / 'components.yaml').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
