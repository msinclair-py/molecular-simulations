"""Unit tests for simulate/multires_simulator.py module.

The module's optional CALVADOS import is performed lazily inside
``run_rounds``, so the dataclass, constructor, ``from_toml``, ``strip_solvent``
and ``sander_minimize`` logic all import and run REAL in CI without CALVADOS or
any AmberTools binary. Paths that genuinely shell out to ``sander`` are
skip-gated on ``shutil.which`` and run only where the binary is installed.

No ``subprocess``, ``calvados`` or ``cg2all`` is mocked in this file. Return-code
branches are driven by the real ``true``/``false`` shell builtins; ``strip_solvent``
is exercised against a real OpenMM ``Simulation`` built from committed AMBER
fixtures with real ParmEd.
"""

import shutil
from pathlib import Path

import pytest

_have_sander = shutil.which('sander') is not None
requires_sander = pytest.mark.skipif(
    not _have_sander, reason='sander binary not available'
)


def _build_real_simulation(prmtop_path: Path, inpcrd_path: Path, explicit: bool):
    """Build a real OpenMM Simulation from AMBER files (Reference platform)."""
    import openmm as mm
    from openmm import app, unit

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
    if explicit:
        system = prmtop.createSystem(
            nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer
        )
    else:
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond
    )
    sim = app.Simulation(
        prmtop.topology,
        system,
        integrator,
        mm.Platform.getPlatformByName('Reference'),
    )
    sim.context.setPositions(inpcrd.positions)
    return sim


# ============================================================================
# SanderMinDefaults dataclass - pure logic, no deps
# ============================================================================


class TestSanderMinDefaults:
    """Test suite for SanderMinDefaults dataclass."""

    def test_sander_min_defaults_values(self):
        from molecular_simulations.simulate.multires_simulator import SanderMinDefaults

        defaults = SanderMinDefaults()

        assert defaults.imin == 1
        assert defaults.maxcyc == 5000
        assert defaults.ncyc == 2500
        assert defaults.ntb == 0
        assert defaults.ntr == 0
        assert defaults.cut == 10.0
        assert defaults.ntpr == 10000
        assert defaults.ntwr == 5000
        assert defaults.ntxo == 1

    def test_sander_min_defaults_mdin_contents(self):
        from molecular_simulations.simulate.multires_simulator import SanderMinDefaults

        defaults = SanderMinDefaults()

        assert 'Minimization input' in defaults.mdin_contents
        assert 'imin=1' in defaults.mdin_contents
        assert 'maxcyc=5000' in defaults.mdin_contents
        assert 'ncyc=2500' in defaults.mdin_contents
        assert 'ntb=0' in defaults.mdin_contents
        assert 'cut=10.0' in defaults.mdin_contents
        assert '&cntrl' in defaults.mdin_contents

    def test_sander_min_defaults_attributes_modifiable(self):
        from molecular_simulations.simulate.multires_simulator import SanderMinDefaults

        defaults = SanderMinDefaults()
        defaults.maxcyc = 10000
        defaults.ncyc = 5000
        defaults.cut = 12.0
        defaults.__post_init__()

        assert defaults.maxcyc == 10000
        assert defaults.ncyc == 5000
        assert defaults.cut == 12.0
        assert 'maxcyc=10000' in defaults.mdin_contents
        assert 'ncyc=5000' in defaults.mdin_contents
        assert 'cut=12.0' in defaults.mdin_contents


# ============================================================================
# sander_minimize - real subprocess via true/false builtins
# ============================================================================


class TestSanderMinimize:
    """sander_minimize input/command/return-code logic, driven really."""

    def test_sander_minimize_success_returncode(self, tmp_path):
        """A zero-exit command ('true') completes without raising."""
        from molecular_simulations.simulate.multires_simulator import sander_minimize

        (tmp_path / 'system.inpcrd').write_text('coords')
        (tmp_path / 'system.prmtop').write_text('topology')

        # 'true' ignores its args and exits 0 -> no RuntimeError.
        sander_minimize(
            path=tmp_path,
            inpcrd_file='system.inpcrd',
            prmtop_file='system.prmtop',
            sander_cmd='true',
        )

    def test_sander_minimize_failure_raises(self, tmp_path):
        """A nonzero-exit command ('false') raises RuntimeError."""
        from molecular_simulations.simulate.multires_simulator import sander_minimize

        (tmp_path / 'system.inpcrd').write_text('coords')
        (tmp_path / 'system.prmtop').write_text('topology')

        with pytest.raises(RuntimeError, match='sander error'):
            sander_minimize(
                path=tmp_path,
                inpcrd_file='system.inpcrd',
                prmtop_file='system.prmtop',
                sander_cmd='false',
            )

    @requires_sander
    def test_sander_minimize_real(self, real_amber_system_files):
        """Run real sander minimization on a real (vacuum) AMBER system."""
        from molecular_simulations.simulate.multires_simulator import sander_minimize

        path = real_amber_system_files['path']
        # sander_minimize hardcodes the output as <inpcrd>.min.inpcrd.
        shutil.copy(real_amber_system_files['prmtop'], path / 'system.prmtop')
        shutil.copy(real_amber_system_files['inpcrd'], path / 'system.inpcrd')

        sander_minimize(
            path=path,
            inpcrd_file='system.inpcrd',
            prmtop_file='system.prmtop',
            sander_cmd='sander',
        )

        out = path / 'system.min.inpcrd'
        assert out.exists()
        assert out.stat().st_size > 0


# ============================================================================
# MultiResolutionSimulator construction / config parsing - real, no deps
# ============================================================================


class TestMultiResolutionSimulator:
    """Construction and from_toml parsing."""

    def _aa_params(self, scheme='implicit'):
        return {
            'solvation_scheme': scheme,
            'protein': True,
            'rna': False,
            'dna': False,
            'phos_protein': False,
            'use_amber': True,
            'out': 'system.pdb',
            'equilibration_steps': 1000,
            'production_steps': 10000,
            'device_ids': [0],
        }

    def test_multires_init(self, tmp_path):
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        (tmp_path / 'protein.pdb').write_text(
            'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
        )

        sim = MultiResolutionSimulator(
            path=tmp_path,
            input_pdb='protein.pdb',
            n_rounds=3,
            cg_params={'config': {}, 'components': {}},
            aa_params=self._aa_params(),
            cg2all_bin='convert_cg2all',
            cg2all_ckpt=None,
            amberhome='/fake/amber',
        )

        assert sim.path == tmp_path
        assert sim.input_pdb == 'protein.pdb'
        assert sim.n_rounds == 3
        assert sim.amberhome == Path('/fake/amber')

    def test_multires_init_no_amberhome(self, tmp_path):
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        (tmp_path / 'protein.pdb').write_text(
            'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
        )

        sim = MultiResolutionSimulator(
            path=tmp_path,
            input_pdb='protein.pdb',
            n_rounds=1,
            cg_params={},
            aa_params={},
            amberhome=None,
        )

        assert sim.amberhome is None

    def test_multires_from_toml(self, tmp_path):
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        toml_content = f"""
[settings]
path = "{tmp_path}"
input_pdb = "protein.pdb"
n_rounds = 2
cg2all_bin = "convert_cg2all"
cg2all_ckpt = "/path/to/ckpt"
amberhome = "/fake/amber"

[[cg_params]]
test = "value"

[aa_params]
solvation_scheme = "explicit"
"""
        config_path = tmp_path / 'config.toml'
        config_path.write_text(toml_content)

        sim = MultiResolutionSimulator.from_toml(config_path)

        assert sim.n_rounds == 2
        assert sim.cg2all_bin == 'convert_cg2all'
        assert sim.cg2all_ckpt == '/path/to/ckpt'
        assert sim.amberhome == Path('/fake/amber')
        # cg_params picks the first element of the [[cg_params]] array.
        assert sim.cg_params == {'test': 'value'}
        assert sim.aa_params == {'solvation_scheme': 'explicit'}

    def test_multires_from_toml_minimal(self, tmp_path):
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        toml_content = f"""
[settings]
path = "{tmp_path}"
input_pdb = "protein.pdb"
n_rounds = 1

[[cg_params]]
test = "value"

[aa_params]
solvation_scheme = "implicit"
"""
        config_path = tmp_path / 'config.toml'
        config_path.write_text(toml_content)

        sim = MultiResolutionSimulator.from_toml(config_path)

        assert sim.n_rounds == 1
        assert sim.cg2all_bin == 'convert_cg2all'  # default
        assert sim.cg2all_ckpt is None  # default
        assert sim.amberhome is None  # default


# ============================================================================
# strip_solvent - REAL ParmEd over a real OpenMM Simulation
# ============================================================================


class TestStripSolvent:
    """strip_solvent exercised with real ParmEd, no mocks."""

    def test_strip_solvent_writes_protein_pdb(
        self, real_amber_system_files, skip_without_openmm, tmp_path
    ):
        """A vacuum protein system writes a valid PDB (nothing to strip)."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        sim = _build_real_simulation(
            real_amber_system_files['prmtop'],
            real_amber_system_files['inpcrd'],
            explicit=False,
        )
        out = tmp_path / 'protein.pdb'

        MultiResolutionSimulator.strip_solvent(sim, output_pdb=str(out))

        assert out.exists() and out.stat().st_size > 0
        text = out.read_text()
        assert 'ATOM' in text
        assert 'ALA' in text

    def test_strip_solvent_removes_water_and_ions(
        self, real_amber_explicit_files, skip_without_openmm, tmp_path
    ):
        """A solvated system has all waters/ions stripped from the output."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        sim = _build_real_simulation(
            real_amber_explicit_files['prmtop'],
            real_amber_explicit_files['inpcrd'],
            explicit=True,
        )
        n_solvated = sim.topology.getNumAtoms()
        out = tmp_path / 'protein.pdb'

        MultiResolutionSimulator.strip_solvent(sim, output_pdb=str(out))

        text = out.read_text()
        # The real solvent mask must remove every water residue.
        assert 'HOH' not in text
        assert 'WAT' not in text
        n_remaining = sum(
            1 for ln in text.splitlines() if ln.startswith(('ATOM', 'HETATM'))
        )
        assert 0 < n_remaining < n_solvated


# ============================================================================
# run_rounds - real no-binary control flow
# ============================================================================


class TestRunRounds:
    """run_rounds branches reachable without external binaries."""

    def test_run_rounds_invalid_solvation_scheme(self, tmp_path):
        """Invalid solvation_scheme raises AttributeError before any tool runs."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        (tmp_path / 'protein.pdb').write_text(
            'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
        )

        aa_params = {
            'solvation_scheme': 'invalid_scheme',
            'protein': True,
            'rna': False,
            'dna': False,
            'phos_protein': False,
            'use_amber': True,
            'out': 'system.pdb',
            'equilibration_steps': 1000,
            'production_steps': 10000,
            'device_ids': [0],
        }

        sim = MultiResolutionSimulator(
            path=tmp_path,
            input_pdb='protein.pdb',
            n_rounds=1,
            cg_params={'config': {}, 'components': {}},
            aa_params=aa_params,
            amberhome=None,
        )

        with pytest.raises(AttributeError, match='solvation_scheme must be'):
            sim.run_rounds()


def test_module_imports_without_calvados_or_cg2all():
    """Module imports in CI without CALVADOS / cg2all (lazy calvados import)."""
    import molecular_simulations.simulate.multires_simulator as m

    assert hasattr(m, 'MultiResolutionSimulator')
    assert hasattr(m, 'sander_minimize')
    assert hasattr(m, 'SanderMinDefaults')
    # The calvados import is deferred into run_rounds, so it is not a module attr.
    assert not hasattr(m, 'sim')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
