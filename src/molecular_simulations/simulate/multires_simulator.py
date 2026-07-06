import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import parmed as pmd
from openmm.app import Simulation

try:
    import tomllib  # type: ignore[import-not-found]  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10
from typing import TypeVar

from ..build import ExplicitSolvent, ImplicitSolvent
from ..build.build_calvados import CGBuilder
from .omm_simulator import ImplicitSimulator, Simulator

_T = TypeVar('_T')
OptPath = Path | str | None
PathLike = Path | str


@dataclass
class SanderMinDefaults:
    """Dataclass with default values for sander minimization.

    Creates the contents of a sander input file during initialization.

    Attributes:
        imin: Minimization flag. Set to 1 to perform energy minimization.
        maxcyc: Maximum number of minimization cycles.
        ncyc: Number of steepest descent steps before switching to
            conjugate gradient.
        ntb: Periodic boundary conditions flag. 0 for no periodicity.
        ntr: Restraint flag. 0 for no restraints.
        cut: Nonbonded cutoff distance in Angstroms.
        ntpr: Frequency of energy printing in steps.
        ntwr: Frequency of restart file writing in steps.
        ntxo: Output restart file format. 1 for ASCII format.
    """

    imin = 1  # Perform energy minimization
    maxcyc = 5000  # Maximum number of minimization cycles
    ncyc = (
        2500  # Switch from steepest descent to conjugate gradient after this many steps
    )
    ntb = 0  # Periodic boundary conditions (constant volume)
    ntr = 0  # No restraints
    cut = 10.0  # Nonbonded cutoff in Angstroms
    ntpr = 10000  # Print energy every 10000 steps (don't print it)
    ntwr = 5000  # Write restart file every 5000 steps (only once)
    ntxo = 1  # Output restart file format (ASCII)

    def __post_init__(self):
        self.mdin_contents = f"""Minimization input
 &cntrl
  imin={self.imin},
  maxcyc={self.maxcyc},
  ncyc={self.ncyc},
  ntb={self.ntb},
  ntr={self.ntr},
  cut={self.cut:.1f},
  ntpr={self.ntpr},
  ntwr={self.ntwr},
  ntxo={self.ntxo}
 /
 """


def sander_minimize(
    path: Path, inpcrd_file: str, prmtop_file: str, sander_cmd: str
) -> None:
    """
    Minimize MD system with sander and output new inpcrd file.

    Args:
        path (Path): Path to directory containing inpcrd and prmtop. New inpcrd will be
            written here as well.
        inpcrd_file (str): Name of inpcrd file in path
        prmtop_file (str): Name of prmtop file in path
        sander_cmd (str): Command for sander
    """
    defaults = SanderMinDefaults()
    mdin = defaults.mdin_contents
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.in', dir=str(path)) as tmp_in:
        tmp_in.write(mdin)
        tmp_in.flush()
        outfile = Path(inpcrd_file).with_suffix('.min.inpcrd')
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.out', dir=str(path)
        ) as tmp_out:
            command = [
                sander_cmd,
                '-O',
                '-i',
                tmp_in.name,
                '-o',
                tmp_out.name,
                '-p',
                str(path / prmtop_file),
                '-c',
                str(path / inpcrd_file),
                '-r',
                str(path / outfile),
                '-inf',
                str(path / 'min.mdinfo'),
            ]
            result = subprocess.run(
                command, shell=False, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f'sander error!\n{result.stderr}\n{result.stdout}')


class MultiResolutionSimulator:
    """
    Class for performing multi-resolution simulations with switching between CG and AA
    representations. Utilizes CALVADOS for CG simulations and omm_simulator.py for AA
    simulations.

    Args:
        path (PathLike): Path to simulation input files, also serves as output path.
        input_pdb (str): Input pdb for simulations, must exist in path.
        n_rounds (int): Number of rounds of CG/AA simulation to perform.
        cg_params (dict): Parameters for CG simulations. Initializes CGBuilder.
        aa_params (dict): Parameters for AA simulations. Initializes omm_simulator.
        cg2all_bin (str): Defaults to 'convert_cg2all'. Path to cg2all binary. Must
            be provided if cg2all is installed in a separate environment.
        cg2all_ckpt (OptPath): Path to cg2all checkpoint file.
        amberhome (str | None): Defaults to None. Path to amberhome (excluding bin).
            Used for sander and pdb4amber. If None, assumes AmberTools binaries are
            available in the current $PATH.

    Usage:
        sim = MultiResolutionSimulator.from_toml('config.toml')
        sim.run()
    """

    def __init__(
        self,
        path: PathLike,
        input_pdb: str,
        n_rounds: int,
        cg_params: dict,
        aa_params: dict,
        cg2all_bin: str = 'convert_cg2all',
        cg2all_ckpt: OptPath = None,
        amberhome: str | None = None,
    ):
        self.path = Path(path)
        self.input_pdb = input_pdb
        self.n_rounds = n_rounds
        self.cg_params = cg_params
        self.aa_params = aa_params
        self.cg2all_bin = cg2all_bin
        self.cg2all_ckpt = cg2all_ckpt
        self.amberhome = Path(amberhome) if amberhome is not None else None

    @classmethod
    def from_toml(cls: type[_T], config: PathLike) -> _T:
        """Construct MultiResolutionSimulator from a TOML configuration file.

        This is the recommended method for instantiating MultiResolutionSimulator
        as it allows all parameters to be specified in a single configuration file.

        Args:
            config: Path to the TOML configuration file containing settings,
                cg_params, and aa_params sections.

        Returns:
            Configured MultiResolutionSimulator instance.
        """
        with open(config, 'rb') as f:
            cfg = tomllib.load(f)
        settings = cfg['settings']
        cg_params = cfg['cg_params'][0]
        aa_params = cfg['aa_params']
        path = settings['path']
        input_pdb = settings['input_pdb']
        n_rounds = settings['n_rounds']

        kwargs = {}
        if 'cg2all_bin' in settings:
            kwargs['cg2all_bin'] = settings['cg2all_bin']
        if 'cg2all_ckpt' in settings:
            kwargs['cg2all_ckpt'] = settings['cg2all_ckpt']
        if 'amberhome' in settings:
            kwargs['amberhome'] = settings['amberhome']

        return cls(
            path,
            input_pdb,
            n_rounds,
            cg_params,
            aa_params,
            **kwargs,
        )

    @staticmethod
    def strip_solvent(
        simulation: Simulation, output_pdb: PathLike = 'protein.pdb'
    ) -> None:
        """Strip solvent and ions from an OpenMM simulation and write PDB.

        Uses ParmEd to remove water molecules and common ions from the
        simulation, then saves the remaining structure to a PDB file.

        Args:
            simulation: OpenMM Simulation object containing the solvated system.
            output_pdb: Path for the output PDB file. Defaults to 'protein.pdb'.
        """
        struc = pmd.openmm.load_topology(
            simulation.topology,
            simulation.system,
            xyz=simulation.context.getState(getPositions=True).getPositions(),
        )
        solvent_resnames = [
            'WAT',
            'HOH',
            'TIP3',
            'TIP3P',
            'SOL',
            'OW',
            'H2O',
            'NA',
            'K',
            'CL',
            'MG',
            'CA',
            'ZN',
            'MN',
            'FE',
            'Na+',
            'K+',
            'Cl-',
            'Mg2+',
            'Ca2+',
            'Zn2+',
            'Mn2+',
            'Fe2+',
            'Fe3+',
            'SOD',
            'POT',
            'CLA',
        ]
        mask = ':' + ','.join(solvent_resnames)
        struc.strip(mask)
        struc.save(output_pdb)

    def run_rounds(self) -> None:
        """Execute the multi-resolution simulation workflow.

        Runs alternating cycles of all-atom (AA) and coarse-grained (CG)
        simulations for the specified number of rounds. Each round consists of:
        1. Building the AA system from input or previous CG structure
        2. Minimizing with sander to resolve clashes
        3. Running AA equilibration and production
        4. Stripping solvent and converting to CG representation
        5. Running CG simulation with CALVADOS
        6. Back-mapping CG to AA using cg2all for the next round

        Note:
            Does not currently handle restart runs.
        """
        for r in range(self.n_rounds):
            aa_path = self.path / f'aa_round{r}'
            aa_path.mkdir()

            if r == 0:
                input_pdb = str(self.path / self.input_pdb)
            else:
                input_pdb = str(self.path / f'cg_round{r - 1}/last_frame.amber.pdb')

            match self.aa_params['solvation_scheme']:
                case 'implicit':
                    _aa_builder = ImplicitSolvent
                    _aa_simulator = ImplicitSimulator
                case 'explicit':
                    _aa_builder = ExplicitSolvent
                    _aa_simulator = Simulator
                case _:
                    raise AttributeError(
                        "solvation_scheme must be 'implicit' or 'explicit'"
                    )

            aa_builder = _aa_builder(
                aa_path,
                input_pdb,
                protein=self.aa_params['protein'],
                rna=self.aa_params['rna'],
                dna=self.aa_params['dna'],
                phos_protein=self.aa_params['phos_protein'],
                use_amber=self.aa_params['use_amber'],
                out=self.aa_params['out'],
            )

            aa_builder.build()

            # cg2all may create clashes which OpenMM minimization does not address.
            # Therefore, we want to minimize all cg2all-created structures with sander instead.
            if self.amberhome is None:
                sander = 'sander'
            else:
                sander = str(self.amberhome / 'bin/sander')
            sander_minimize(aa_path, 'system.inpcrd', 'system.prmtop', sander)

            aa_simulator = _aa_simulator(
                aa_path,
                coor_name='system.min.inpcrd',
                ff='amber',
                equil_steps=int(self.aa_params['equilibration_steps']),
                prod_steps=int(self.aa_params['production_steps']),
                n_equil_cycles=1,
                device_ids=self.aa_params['device_ids'],
            )

            aa_simulator.run()

            # strip solvent and output AA structure for next step (CG)
            self.strip_solvent(aa_simulator.simulation, str(aa_path / 'protein.pdb'))

            # build CG
            cg_path = self.path / f'cg_round{r}'
            cg_path.mkdir()
            cg_params = self.cg_params
            cg_params['config']['path'] = str(cg_path)
            cg_params['config']['input_pdb'] = str(aa_path / 'protein.pdb')

            cg_builder = CGBuilder.from_dict(cg_params)
            cg_builder.build()  # writes config and components yamls

            # run CG
            from calvados import sim  # ty: ignore[unresolved-import]

            sim.run(
                path=str(cg_path), fconfig='config.yaml', fcomponents='components.yaml'
            )

            # convert CG to AA for next round using cg2all
            command = [
                self.cg2all_bin,
                '-p',
                str(cg_path / 'top.pdb'),
                '-d',
                str(cg_path / 'protein.dcd'),
                '-o',
                str(cg_path / 'traj_aa.dcd'),
                '-opdb',
                str(cg_path / 'last_frame.pdb'),
                '--cg',
                'ResidueBasedModel',
                '--standard-name',
                '--device',
                'cuda',
                '--proc',
                '1',
            ]
            if self.cg2all_ckpt is not None:
                command += ['--ckpt', self.cg2all_ckpt]

            result = subprocess.run(
                command, shell=False, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f'cg2all error!\n{result.stderr}')

            # use pdb4amber to fix cg2all-generated pdb
            if self.amberhome is None:
                command = ['pdb4amber']
            else:
                command = [str(self.amberhome / 'bin/pdb4amber')]
            command += [str(cg_path / 'last_frame.pdb'), '-y']
            result = subprocess.run(
                command, shell=False, capture_output=True, text=True
            )
            if result.returncode == 0:
                with open(str(cg_path / 'last_frame.amber.pdb'), 'w') as f:
                    f.write(result.stdout)
            else:
                raise RuntimeError(f'pdb4amber error!\n{result.stderr}')
