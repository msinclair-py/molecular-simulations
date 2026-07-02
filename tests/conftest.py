"""
Test configuration and fixtures.

This module provides shared fixtures for the molecular-simulations test suite.
Fixtures are designed to reduce mocking by providing real test data where possible.
"""

import os
from pathlib import Path

import pytest

# Disable numba JIT compilation to avoid path resolution issues during testing.
# This must be set before numba is imported.
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Force a non-interactive matplotlib backend so plotting code can run headless
# (CI, no display) and write real figure files. Must precede any pyplot import.
os.environ.setdefault('MPLBACKEND', 'Agg')


# ---------------------------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------------------------


def get_test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / 'data'


# ---------------------------------------------------------------------------
# Environment Detection Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='session')
def real_openmm_available() -> bool:
    """
    Session-scoped check for OpenMM availability.

    Returns True if OpenMM is properly installed and functional,
    False otherwise. This allows tests to conditionally skip
    when OpenMM is not available.

    Usage:
        def test_simulation(real_openmm_available):
            if not real_openmm_available:
                pytest.skip("OpenMM not available")
            # ... test code ...
    """
    try:
        import openmm  # noqa: F401
        from openmm import Platform

        # Verify we can access at least one platform
        num_platforms = Platform.getNumPlatforms()
        return num_platforms > 0
    except ImportError:
        return False
    except Exception:
        return False


@pytest.fixture(scope='session')
def real_amber_available() -> bool:
    """
    Session-scoped check for AmberTools availability.

    Returns True if AmberTools (tleap) is properly installed,
    False otherwise.
    """
    import shutil

    amberhome = os.environ.get('AMBERHOME')
    if amberhome:
        tleap_path = Path(amberhome) / 'bin' / 'tleap'
        if tleap_path.exists():
            return True
    # Also check if tleap is in PATH
    return shutil.which('tleap') is not None


@pytest.fixture(scope='session')
def real_rdkit_available() -> bool:
    """
    Session-scoped check for RDKit availability.

    Returns True if RDKit is properly installed and functional,
    False otherwise.
    """
    try:
        from rdkit import Chem

        # Verify basic functionality
        mol = Chem.MolFromSmiles('C')
        return mol is not None
    except ImportError:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PDB Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdb_path(tmp_path: Path) -> Path:
    """
    Creates a valid minimal PDB file for testing.

    This provides a simple alanine-glycine dipeptide structure that is
    valid for testing with AMBER forcefields and MDAnalysis.

    Returns:
        Path to the created PDB file in a temporary directory.
    """
    pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.989  -0.744   1.232  1.00  0.00           C
ATOM      6  N   GLY A   2       3.331   1.539   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       4.021   2.826   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.528   2.661   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.089   1.563   0.000  1.00  0.00           O
TER
END
"""
    pdb_file = tmp_path / 'test_structure.pdb'
    pdb_file.write_text(pdb_content)
    return pdb_file


@pytest.fixture
def alanine_dipeptide_pdb() -> Path:
    """
    Returns the path to the alanine dipeptide PDB test file.

    This is a standard test system (Ace-Ala-Nme) commonly used in
    molecular dynamics simulations.

    Returns:
        Path to the static alanine dipeptide PDB file.
    """
    return get_test_data_dir() / 'pdb' / 'alanine_dipeptide.pdb'


@pytest.fixture
def two_chain_pdb() -> Path:
    """Return a real two-chain PDB with a charged interface pair.

    Chain A is Ace-Lys-Nme, chain B is Ace-Asp-Nme, positioned so the Lys NZ
    and Asp carboxylate sit ~3.4 A apart -- within the default salt-bridge
    (6.0 A) and hydrogen-bond (3.5 A) cutoffs. This exercises the
    ``chainID A`` / ``chainID B`` interface logic in cov_ppi and ipSAE that the
    single-chain alanine dipeptide cannot.

    The PDB carries CONECT bond records, so analyses that need connectivity
    (e.g. cov_ppi hydrogen-bond donor/acceptor surveys) work directly.

    Returns:
        Path to the static two-chain PDB file.
    """
    return get_test_data_dir() / 'pdb' / 'two_chain_saltbridge.pdb'


@pytest.fixture
def two_chain_trajectory() -> dict:
    """Return the two-chain PDB topology plus a short matching trajectory.

    The 5-frame DCD drifts chain B away from chain A so the salt bridge is
    present in the first frame and broken in later ones -- giving non-trivial
    occupancy fractions for trajectory analyses (DynamicInteractionEnergy,
    cov_ppi over a trajectory) rather than a single static frame.

    Returns:
        Dictionary with ``top`` (the two-chain PDB) and ``traj`` (the DCD).
    """
    pdb_dir = get_test_data_dir() / 'pdb'
    return {
        'top': pdb_dir / 'two_chain_saltbridge.pdb',
        'traj': pdb_dir / 'two_chain_saltbridge.dcd',
    }


# ---------------------------------------------------------------------------
# AMBER System File Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def amber_system_files(tmp_path: Path) -> dict:
    """
    Creates minimal prmtop and inpcrd files for testing.

    These files contain the minimal valid structure needed for
    testing AMBER-related functionality without running tleap.

    Returns:
        Dictionary with keys 'prmtop', 'inpcrd', and 'path' containing
        the paths to the created files and the base directory.
    """
    # Minimal prmtop file structure - this is a simplified version
    # that contains the essential sections for parsing
    prmtop_content = """\
%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/00  00:00:00
%FLAG TITLE
%FORMAT(20a4)
Test system for unit testing
%FLAG POINTERS
%FORMAT(10I8)
       9       2       6       2      12       1      18       0       0       0
      33       1       2       1       0       1       1       1       2       0
       0       0       0       0       0       0       0       0       9       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
N   CA  C   O   CB  N   CA  C   O
%FLAG CHARGE
%FORMAT(5E16.8)
 -0.41570000E+01  0.33760000E+00  0.59730000E+01 -0.56790000E+01 -0.18250000E+01
 -0.41570000E+01  0.33760000E+00  0.59730000E+01 -0.56790000E+01
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       7       6       6       8       6       7       6       6       8
%FLAG MASS
%FORMAT(5E16.8)
  0.14010000E+02  0.12010000E+02  0.12010000E+02  0.16000000E+02  0.12010000E+02
  0.14010000E+02  0.12010000E+02  0.12010000E+02  0.16000000E+02
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1       1       2       1       1       1       1       2
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       6       5       4       3       2       3       2       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       2       3
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
ALA GLY
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1       6
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  0.31700000E+03  0.52600000E+03
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  0.15220000E+01  0.12290000E+01
%FLAG ANGLE_FORCE_CONSTANT
%FORMAT(5E16.8)
  0.63000000E+02  0.80000000E+02
%FLAG ANGLE_EQUIL_VALUE
%FORMAT(5E16.8)
  0.19480000E+01  0.21230000E+01
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  0.15000000E+02  0.00000000E+00
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  0.20000000E+01  0.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.31415927E+01  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  0.12000000E+01  0.00000000E+00
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  0.20000000E+01  0.00000000E+00
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  0.10610000E+07  0.51280000E+06  0.10000000E+01
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  0.61400000E+03  0.49340000E+03  0.10000000E+01
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)

%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       6       9       2
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)

%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1       3       6       9       2
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)

%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1      12       1
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2       3       4       5       6       7       3       4       5       6
       7       4       5       6       7       5       6       7       6       7
       7       8       9       8       9       9       0
%FLAG RADII
%FORMAT(5E16.8)
  0.17000000E+01  0.17000000E+01  0.17000000E+01  0.15000000E+01  0.17000000E+01
  0.17000000E+01  0.17000000E+01  0.17000000E+01  0.15000000E+01
%FLAG SCREEN
%FORMAT(5E16.8)
  0.79000000E+00  0.72000000E+00  0.72000000E+00  0.85000000E+00  0.72000000E+00
  0.79000000E+00  0.72000000E+00  0.72000000E+00  0.85000000E+00
"""

    # Minimal inpcrd file structure
    inpcrd_content = """\
Test system coordinates
    9
   0.0000000   0.0000000   0.0000000   1.4580000   0.0000000   0.0000000
   2.0090000   1.4200000   0.0000000   1.2510000   2.3900000   0.0000000
   1.9890000  -0.7440000   1.2320000   3.3310000   1.5390000   0.0000000
   4.0210000   2.8260000   0.0000000   5.5280000   2.6610000   0.0000000
   6.0890000   1.5630000   0.0000000
"""

    prmtop_file = tmp_path / 'system.prmtop'
    inpcrd_file = tmp_path / 'system.inpcrd'

    prmtop_file.write_text(prmtop_content)
    inpcrd_file.write_text(inpcrd_content)

    return {
        'prmtop': prmtop_file,
        'inpcrd': inpcrd_file,
        'path': tmp_path,
    }


@pytest.fixture
def real_amber_system_files(tmp_path: Path) -> dict:
    """Provide a real, complete AMBER system (Ace-Ala-Nme) for testing.

    Unlike :func:`amber_system_files` (a hand-written minimal prmtop that
    fails real parsing), these files were generated with tleap
    (``leaprc.protein.ff19SB``) and load cleanly via OpenMM's
    ``AmberPrmtopFile.createSystem`` (explicit and ``implicitSolvent=GBn2``)
    and via MDAnalysis (``backbone`` -> indices [4, 5, 6, 8, 14, 15, 16, 18]).
    The committed copies are copied into a per-test ``tmp_path`` so tests may
    write outputs alongside them without mutating the fixtures.

    Returns:
        Dictionary with ``prmtop``, ``inpcrd``, ``pdb``, ``dcd`` (a short 5-frame
        trajectory matching the topology) and ``path`` (the temp directory).
    """
    import shutil

    src = get_test_data_dir() / 'amber'
    files = {}
    for key, name in (
        ('prmtop', 'ala_dipeptide.prmtop'),
        ('inpcrd', 'ala_dipeptide.inpcrd'),
        ('pdb', 'ala_dipeptide.pdb'),
        ('dcd', 'ala_dipeptide.dcd'),
    ):
        dest = tmp_path / name
        shutil.copy(src / name, dest)
        files[key] = dest

    files['path'] = tmp_path
    return files


@pytest.fixture
def real_amber_explicit_files(tmp_path: Path) -> dict:
    """Provide a real, solvated (boxed) AMBER system for PME/NPT testing.

    Unlike :func:`real_amber_system_files` (an implicit/vacuum Ace-Ala-Nme with
    no periodic box), this is the same dipeptide solvated in a small TIP3P box
    (913 atoms) generated with tleap (ff19SB + tip3p). It carries periodic box
    vectors, so it loads under ``AmberPrmtopFile.createSystem(nonbondedMethod=PME)``
    and supports a ``MonteCarloBarostat`` (NPT) -- the paths the boxless fixture
    cannot exercise. Copied into a per-test ``tmp_path``.

    Returns:
        Dictionary with ``prmtop``, ``inpcrd`` and ``path`` (the temp directory).
    """
    import shutil

    src = get_test_data_dir() / 'amber'
    files = {}
    for key, name in (
        ('prmtop', 'ala_dipeptide_solv.prmtop'),
        ('inpcrd', 'ala_dipeptide_solv.inpcrd'),
    ):
        dest = tmp_path / name
        shutil.copy(src / name, dest)
        files[key] = dest

    files['path'] = tmp_path
    return files


@pytest.fixture
def real_amber_titratable_files(tmp_path: Path) -> dict:
    """Provide a real AMBER system containing titratable residues.

    Unlike :func:`real_amber_system_files` (Ace-Ala-Nme, which has no titratable
    residues), this is a capped Ace-Lys-Asp-Nme tetrapeptide generated with tleap
    (``leaprc.protein.ff19SB``). The OpenMM topology has residues
    ``[ACE, LYS, ASP, NME]`` at indices 0-3 (46 atoms) and MDAnalysis identifies
    the same protein residues, so ``ConstantPHEnsemble.build_dicts`` finds LYS at
    index 1 and ASP at index 2 (ACE/NME are excluded as termini). The files are
    copied into a per-test ``tmp_path`` and named ``system.prmtop`` /
    ``system.inpcrd`` so callers that hardcode those names (build_dicts) work
    directly.

    Returns:
        Dictionary with ``prmtop``, ``inpcrd``, ``pdb`` and ``path`` (the temp
        directory). ``prmtop``/``inpcrd`` are the ``system.*`` copies.
    """
    import shutil

    src = get_test_data_dir() / 'amber'
    shutil.copy(src / 'lys_asp.prmtop', tmp_path / 'system.prmtop')
    shutil.copy(src / 'lys_asp.inpcrd', tmp_path / 'system.inpcrd')
    shutil.copy(src / 'lys_asp.pdb', tmp_path / 'lys_asp.pdb')

    return {
        'prmtop': tmp_path / 'system.prmtop',
        'inpcrd': tmp_path / 'system.inpcrd',
        'pdb': tmp_path / 'lys_asp.pdb',
        'path': tmp_path,
    }


@pytest.fixture
def real_amber_titratable_solvated_files(tmp_path: Path) -> dict:
    """Provide a real, SOLVATED titratable AMBER system for full ConstantPH runs.

    Unlike :func:`real_amber_titratable_files` (the same Ace-Lys-Asp-Nme peptide
    in vacuum, used for the lightweight ``build_dicts`` residue-identification
    logic), this is that peptide solvated in a ~3.5 nm TIP3P box (2620 atoms,
    net-neutral, generated with tleap ``leaprc.protein.ff19SB`` +
    ``leaprc.water.tip3p``). It carries periodic box vectors, so the explicit
    PME system that ``ConstantPH.__init__`` builds (nonbondedCutoff 0.9 nm) is
    valid, and ParmEd can strip the water/ions down to the 46-atom implicit
    system. This lets the WHOLE ``ConstantPH.__init__`` pipeline run for real in
    CI with no AmberTools at runtime (the fixture was pre-built with tleap).

    Titratable residues are LYS (index 1) and ASP (index 2); ACE/NME are the
    excluded termini.

    Returns:
        Dictionary with ``prmtop``, ``inpcrd`` (copied to ``system.*``) and
        ``path`` (the temp directory).
    """
    import shutil

    src = get_test_data_dir() / 'amber'
    shutil.copy(src / 'lys_asp_solv.prmtop', tmp_path / 'system.prmtop')
    shutil.copy(src / 'lys_asp_solv.inpcrd', tmp_path / 'system.inpcrd')

    return {
        'prmtop': tmp_path / 'system.prmtop',
        'inpcrd': tmp_path / 'system.inpcrd',
        'path': tmp_path,
    }


# ---------------------------------------------------------------------------
# Ligand/SDF Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_sdf_path(tmp_path: Path) -> Path:
    """
    Creates a valid minimal SDF ligand file (methane) for testing.

    This provides the simplest valid SDF structure for testing
    ligand-related functionality.

    Returns:
        Path to the created SDF file in a temporary directory.
    """
    sdf_content = """\
methane
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6276    0.6276    0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6276   -0.6276    0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6276    0.6276   -0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6276   -0.6276   -0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$$$$
"""
    sdf_file = tmp_path / 'ligand.sdf'
    sdf_file.write_text(sdf_content)
    return sdf_file


@pytest.fixture
def benzene_sdf() -> Path:
    """
    Returns the path to the benzene SDF test file.

    This is a simple aromatic ligand commonly used in testing.

    Returns:
        Path to the static benzene SDF file.
    """
    return get_test_data_dir() / 'sdf' / 'benzene.sdf'


# ---------------------------------------------------------------------------
# Skip Markers
# ---------------------------------------------------------------------------


@pytest.fixture
def skip_without_openmm(real_openmm_available):
    """Skip test if OpenMM is not available."""
    if not real_openmm_available:
        pytest.skip('OpenMM not available')


@pytest.fixture
def skip_without_amber(real_amber_available):
    """Skip test if AmberTools is not available."""
    if not real_amber_available:
        pytest.skip('AmberTools not available')


@pytest.fixture
def skip_without_rdkit(real_rdkit_available):
    """Skip test if RDKit is not available."""
    if not real_rdkit_available:
        pytest.skip('RDKit not available')


# ---------------------------------------------------------------------------
# Stub AmberTools (no real binaries required)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_amberhome(tmp_path: Path, monkeypatch) -> Path:
    """Provide a fake ``AMBERHOME`` whose ``bin/`` holds stub executables.

    The build classes (:class:`ImplicitSolvent`, :class:`ExplicitSolvent`)
    locate ``tleap``/``cpptraj`` under ``AMBERHOME/bin`` and invoke them through
    real ``subprocess.run`` calls. These stubs are genuine executables -- not
    mocks -- so the build code's REAL input-file generation and command
    construction run end to end without an AmberTools install. They simply do
    not produce valid topology/coordinate output, so tests that need real AMBER
    output stay gated behind :func:`skip_without_amber`.

    Each stub records what it received so tests can read it back:

    * ``tleap`` copies the file passed via ``-f`` to ``AMBERHOME/tleap_input.txt``
    * ``cpptraj`` writes the stdin it is piped to ``AMBERHOME/cpptraj_input.txt``
    * ``pdb4amber`` writes its argv to ``AMBERHOME/pdb4amber_args.txt``

    Returns:
        Path to the fake ``AMBERHOME`` directory.
    """
    home = tmp_path / 'amber'
    bindir = home / 'bin'
    bindir.mkdir(parents=True)

    stubs = {
        'tleap': (
            '#!/bin/sh\n'
            f'out="{home}/tleap_input.txt"\n'
            'while [ $# -gt 0 ]; do\n'
            '  case "$1" in\n'
            '    -f) shift; cp "$1" "$out" ;;\n'
            '  esac\n'
            '  shift\n'
            'done\n'
            'exit 0\n'
        ),
        'cpptraj': (f'#!/bin/sh\ncat > "{home}/cpptraj_input.txt"\nexit 0\n'),
        'pdb4amber': (f'#!/bin/sh\necho "$@" > "{home}/pdb4amber_args.txt"\nexit 0\n'),
    }
    for name, script in stubs.items():
        exe = bindir / name
        exe.write_text(script)
        exe.chmod(0o755)

    monkeypatch.setenv('AMBERHOME', str(home))
    return home


# ---------------------------------------------------------------------------
# Parsl Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def local_parsl_config(tmp_path):
    """A real, lightweight Parsl Config backed by an in-process thread pool.

    ThreadPoolExecutor needs no worker processes, ports, or scheduler, so a real
    DataFlowKernel can be loaded and cleaned up inside a test -- letting code
    that submits Parsl apps actually run and return real futures instead of
    being mocked. The fixture clears any leftover global Parsl state on teardown
    so loading Parsl in one test cannot leak into the next.
    """
    import parsl
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor

    config = Config(
        run_dir=str(tmp_path / 'runinfo'),
        executors=[ThreadPoolExecutor(max_threads=2, label='local_threads')],
    )
    yield config

    # Ensure a clean global Parsl state regardless of how the test exited.
    import contextlib

    with contextlib.suppress(Exception):
        parsl.clear()
