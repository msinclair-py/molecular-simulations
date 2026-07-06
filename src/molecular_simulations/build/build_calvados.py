"""CALVADOS coarse-grained system building module.

This module provides the CGBuilder class for building coarse-grained
molecular dynamics systems using the CALVADOS force field.

Classes:
    CGBuilder: Build CALVADOS simulation systems from PDB structures.
"""

from pathlib import Path
from typing import TypeVar

import yaml

_T = TypeVar('_T')
OptPath = Path | str | None
PathLike = Path | str


class CGBuilder:
    """Build CALVADOS coarse-grained simulation systems from PDB files.

    Creates configuration and component YAML files required to run
    CALVADOS simulations with secondary structure restraints.

    Args:
        path: Directory path to contain the simulation run.
        input_pdb: Path to input structure for simulation.
        residues_file: Path to CALVADOS residues force field file.
        domains_file: Path to file containing domain definitions.
        box_dim: Dimensions [x, y, z] for orthonormal periodic boundary
            conditions in nm.
        temp: Simulation temperature in Kelvin. Defaults to 310.0.
        ion_conc: Ion concentration in Molar. Defaults to 0.15.
        pH: Solution pH. Defaults to 7.4.
        topol: Initial placement of protein chains. Defaults to 'center'.
        dcd_freq: Write frequency for output DCD trajectory in steps.
            Defaults to 2000.
        n_steps: Total number of integration steps (10 fs timestep).
            Defaults to 10,000,000.
        platform: OpenMM platform for running simulations.
            Defaults to 'CUDA'.
        restart: Style of OpenMM restart. Defaults to 'checkpoint'.
        frestart: Name of output restart files. Defaults to 'restart.chk'.
        verbose: Whether to enable verbose output. Defaults to True.
        molecule_type: Type of molecule. Defaults to 'protein'.
        nmol: Total number of molecules. Defaults to 1.
        restraint: Whether to use secondary structure restraints.
            Defaults to True.
        charge_termini: Terminus patching style. Options: 'N', 'C', 'both',
            or 'end-capped'. Defaults to 'end-capped'.
        restraint_type: Type of restraint, either 'harmonic' or 'go'.
            Defaults to 'harmonic'.
        use_com: Whether to apply restraints to center of mass (True)
            or CA atoms (False). Defaults to True.
        colabfold: Predicted beta-column confidence style. Use 0 for
            AlphaFold EBI, 1 or 2 for ColabFold. Defaults to 0.
        k_harmonic: Harmonic spring constant for restraints in kJ/mol.
            Defaults to 700.0.

    Attributes:
        path: Path object for simulation directory.
        input_pdb: Path object for input structure.
        residues_file: Path object for residues force field file.
        domains_file: Path object for domain definitions.

    Example:
        >>> builder = CGBuilder(
        ...     path='./simulation',
        ...     input_pdb='protein.pdb',
        ...     residues_file='residues.csv',
        ...     domains_file='domains.yaml',
        ...     box_dim=[50.0, 50.0, 50.0],
        ... )
        >>> builder.build()
    """

    def __init__(
        self,
        path: PathLike,
        input_pdb: PathLike,
        residues_file: PathLike,
        domains_file: PathLike,
        box_dim: list[float],
        temp: float = 310.0,
        ion_conc: float = 0.15,
        pH: float = 7.4,
        topol: str = 'center',
        dcd_freq: int = 2000,
        n_steps: int = 10_000_000,
        platform: str = 'CUDA',
        restart: str = 'checkpoint',
        frestart: str = 'restart.chk',
        verbose: bool = True,
        molecule_type: str = 'protein',
        nmol: int = 1,
        restraint: bool = True,
        charge_termini: str = 'end-capped',
        restraint_type: str = 'harmonic',
        use_com: bool = True,
        colabfold: int = 0,
        k_harmonic: float = 700.0,
    ):
        """Initialize the CGBuilder."""
        self.path = Path(path)
        self.input_pdb = Path(input_pdb)
        self.residues_file = Path(residues_file)
        self.domains_file = Path(domains_file)
        self.box_dim = box_dim
        self.temp = temp
        self.ion_conc = ion_conc
        self.pH = pH
        self.topol = topol
        self.dcd_freq = dcd_freq
        self.n_steps = n_steps
        self.platform = platform
        self.restart = restart
        self.frestart = frestart
        self.verbose = verbose
        self.molecule_type = molecule_type
        self.nmol = nmol
        self.restraint = restraint
        self.charge_termini = charge_termini
        self.restraint_type = restraint_type
        self.use_com = use_com
        self.colabfold = colabfold
        self.k_harmonic = k_harmonic

    @classmethod
    def from_dict(cls: type[_T], cg_params: dict) -> _T:
        """Create a CGBuilder instance from a configuration dictionary.

        This is the recommended method for instantiating CGBuilder,
        typically used with TOML configuration files.

        Args:
            cg_params: Dictionary containing 'config' and 'components'
                sub-dictionaries with all required parameters.

        Returns:
            New CGBuilder instance configured from the dictionary.

        Example:
            >>> import tomllib
            >>> with open('config.toml', 'rb') as f:
            ...     params = tomllib.load(f)
            >>> builder = CGBuilder.from_dict(params)
        """
        conf_args = cg_params['config']
        comp_args = cg_params['components']

        path = Path(conf_args['path'])
        input_pdb = conf_args['input_pdb']
        residues_file = comp_args['residues_file']
        domains_file = comp_args['domains_file']
        box_dim = conf_args['box_dim']
        temp = conf_args['temp']
        ion_conc = conf_args['ion_conc']
        pH = conf_args['pH']
        topol = conf_args['topol']
        dcd_freq = conf_args['dcd_freq']
        n_steps = conf_args['n_steps']
        platform = conf_args['platform']
        restart = conf_args['restart']
        frestart = conf_args['frestart']
        verbose = conf_args['verbose']

        molecule_type = comp_args['molecule_type']
        nmol = comp_args['nmol']
        restraint = comp_args['restraint']
        charge_termini = comp_args['charge_termini']
        restraint_type = comp_args['restraint_type']
        use_com = comp_args['use_com']
        colabfold = comp_args['colabfold']
        k_harmonic = comp_args['k_harmonic']

        return cls(
            path,
            input_pdb,
            residues_file,
            domains_file,
            box_dim=box_dim,
            temp=temp,
            ion_conc=ion_conc,
            pH=pH,
            topol=topol,
            dcd_freq=dcd_freq,
            n_steps=n_steps,
            platform=platform,
            restart=restart,
            frestart=frestart,
            verbose=verbose,
            molecule_type=molecule_type,
            nmol=nmol,
            restraint=restraint,
            charge_termini=charge_termini,
            restraint_type=restraint_type,
            use_com=use_com,
            colabfold=colabfold,
            k_harmonic=k_harmonic,
        )

    def build(self) -> None:
        """Prepare the system for CALVADOS simulation.

        Writes both the config.yaml and components.yaml files required
        to run a CALVADOS simulation.
        """
        self.write_config()
        self.write_components()

    def write_config(self) -> None:
        """Write the CALVADOS configuration YAML file.

        Creates config.yaml in the simulation directory with all
        simulation parameters.
        """
        from calvados.cfg import Config  # ty: ignore[unresolved-import]

        config = Config(
            sysname=self.input_pdb.stem,
            box=self.box_dim,
            temp=self.temp,
            ionic=self.ion_conc,
            pH=self.pH,
            topol=self.topol,
            wfreq=self.dcd_freq,
            steps=self.n_steps,
            platform=self.platform,
            restart=self.restart,
            frestart=self.frestart,
            verbose=self.verbose,
        )

        with open(f'{self.path}/config.yaml', 'w') as f:
            yaml.dump(config.config, f)

    def write_components(self) -> None:
        """Write the CALVADOS components YAML file.

        Creates components.yaml in the simulation directory with
        molecule and restraint specifications.
        """
        from calvados.cfg import Components  # ty: ignore[unresolved-import]

        components = Components(
            molecule_type=self.molecule_type,
            nmol=self.nmol,
            restraint=self.restraint,
            charge_termini=self.charge_termini,
            fresidues=str(self.residues_file),
            fdomains=str(self.domains_file),
            pdb_folder=str(self.input_pdb.parent),
            restraint_type=self.restraint_type,
            use_com=self.use_com,
            colabfold=self.colabfold,
            k_harmonic=self.k_harmonic,
        )
        components.add(name=self.input_pdb.stem)

        with open(f'{self.path}/components.yaml', 'w') as f:
            yaml.dump(components.components, f)
