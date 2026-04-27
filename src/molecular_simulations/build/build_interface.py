"""Interface-based system building module for DeepDriveMD.

This module provides the InterfaceBuilder class for building molecular
systems targeting specific protein-protein interfaces, with support for
DeepDriveMD enhanced sampling simulations.

Classes:
    InterfaceBuilder: Build systems for interface-targeted simulations.
"""

from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import yaml

from .build_amber import ExplicitSolvent

PathLike = Path | str
Config = dict[str, Any]


class InterfaceBuilder(ExplicitSolvent):
    """Build systems for driving binding to specific protein interfaces.

    For a given target/binder pair, builds systems for driving binding to
    each of the supplied interfaces using DeepDriveMD. Includes writing
    the required YAML configuration files for running DeepDrive simulations.

    Args:
        path: Base directory path for output files.
        pdb: Path to input PDB file.
        interfaces: Dictionary of interface configurations, keyed by site
            name (e.g., 'site0'). Each interface should contain:
            'contact_sel', 'distance_sel', 'vector', and 'com'.
        target: Path to target protein structure.
        binder: Path to binder protein structure.
        padding: Padding around solute in Angstroms. Defaults to 10.0.
        protein: Whether to load protein force field. Defaults to True.
        rna: Whether to load RNA force field. Defaults to False.
        dna: Whether to load DNA force field. Defaults to False.
        polarizable: Whether to use polarizable force field.
            Defaults to False.

    Attributes:
        interfaces: Dictionary of interface site configurations.
        target: MDAnalysis AtomGroup for the target protein.
        binder: Path to binder structure file.
        root: Root output directory based on target name.
        com: Center of mass of the target protein.

    Example:
        >>> interfaces = {
        ...     "site0": {
        ...         "contact_sel": "resid 10-50",
        ...         "distance_sel": "resid 10-50",
        ...         "vector": [10.0, 0.0, 0.0],
        ...         "com": [50.0, 50.0, 50.0],
        ...     }
        ... }
        >>> builder = InterfaceBuilder(
        ...     path="./build",
        ...     pdb="complex.pdb",
        ...     interfaces=interfaces,
        ...     target="target.pdb",
        ...     binder="binder.pdb",
        ... )
        >>> builder.build_all()
    """

    def __init__(
        self,
        path: PathLike,
        pdb: str,
        interfaces: Config,
        target: PathLike,
        binder: PathLike,
        padding: float = 10.0,
        protein: bool = True,
        rna: bool = False,
        dna: bool = False,
        polarizable: bool = False,
    ):
        """Initialize the InterfaceBuilder."""
        super().__init__(
            path=path, 
            pdb=pdb, 
            padding=padding, 
            protein=protein, 
            rna=rna, 
            dna=dna, 
            polarizable=polarizable
        )
        self.interfaces = interfaces
        self.target = mda.Universe(target).select_atoms('all')
        self.binder = binder
        self.root = self.path / Path(target).name[:-4]
        self.com = self.target.center_of_mass()

    def build_all(self) -> None:
        """Build systems for all interface sites.

        Iterates through each interface site for the target and builds
        the corresponding solvated system with the binder positioned
        near the interface.
        """
        for site in self.interfaces:
            # set pathing for this target/binder/site
            self.yaml_out = self.root / site / Path(self.binder).name[:-4]
            self.out = self.yaml_out / 'ddmd'
            self.out.mkdir(parents=True, exist_ok=True)
            self.out = self.out / 'system'
            self.build_dir = self.yaml_out / 'build'
            self.build_dir.mkdir(parents=True, exist_ok=True)
            self.pdb = self.build_dir / 'protein.pdb'

            cont_sel, dist_sel, vector, com, input_shape = self.parse_interface(site)

            binder = self.place_binder(
                np.array(vector, dtype=np.float32), np.array(com, dtype=np.float32)
            )
            self.merge_proteins(binder)

            self.path = self.build_dir
            self.build()

            self.write_ddmd_yaml(cont_sel, dist_sel)
            self.write_cvae_yaml(input_shape)

    def place_binder(self, vector: np.ndarray, com: np.ndarray) -> mda.AtomGroup:
        """Position the binder near the target interface.

        Translates the binder structure to be positioned near the
        interface as defined by the displacement vector.

        Args:
            vector: Displacement vector from interface center of mass.
            com: Center of mass of the interface region.

        Returns:
            MDAnalysis AtomGroup containing the repositioned binder.
        """
        u = mda.Universe(self.binder)
        sel = u.select_atoms('all')

        sel.positions -= com
        sel.positions += vector

        return sel

    def merge_proteins(self, binder: mda.AtomGroup) -> None:
        """Merge target and binder into a single structure.

        Combines the target and binder AtomGroups and writes a unified
        PDB file for subsequent system building.

        Args:
            binder: MDAnalysis AtomGroup for the positioned binder.
        """
        merged_atoms = mda.Merge(self.target, binder)

        with mda.Writer(self.pdb) as W:
            W.write(merged_atoms)

    def parse_interface(self, site: str = 'site0') -> list:
        """Extract configuration data for an interface site.

        Args:
            site: Name of the interface site to parse. Defaults to 'site0'.

        Returns:
            Tuple containing (contact_sel, distance_sel, vector, com,
            input_shape) for the specified interface site.
        """
        s = self.interfaces[site]
        N = len(s['contact_sel'][18:].split())
        inp_shape = (1, N, N)
        ret = [data for data in s.values()]
        ret.append(inp_shape)
        return ret  # contact_sel, distance_sel, vector, com, inp_shape

    def write_ddmd_yaml(self, contact_selection: str, distance_selection: str) -> None:
        """Write the DeepDriveMD simulation configuration YAML.

        Creates the prod.yaml file containing all settings for running
        a DeepDriveMD enhanced sampling simulation.

        Args:
            contact_selection: MDAnalysis selection string for contact
                calculations.
            distance_selection: MDAnalysis selection string for distance
                calculations.
        """
        yaml_settings = {
            'simulation_input_dir': 'ddmd',
            'num_workers': 4,
            'simulations_per_train': 6,
            'simulations_per_inference': 1,
            'num_total_simulations': 1000,
            'compute_settings': {
                'name': 'polaris',
                'num_nodes': 1,
                'worker_init': f'module use /soft/modulefiles; module load conda; \
                    conda activate deepdrive; cd {self.out}',
                'scheduler_options': '#PBS -l filesystems=home:eagle',
                'account': 'FoundEpidem',
                'queue': 'preemptable',
                'walltime': '72:00:00',
            },
            'simulation_settings': {
                'solvent_type': 'explicit',
                'dt_ps': 0.004,
                'mda_selection': contact_selection,
                'mda_selection_resid_list': None,
                'simulation_length_ns': 10,
                'report_interval_ps': 10,
                'temperature_kelvin': 300,
                'rmsd_reference_pdb': None,
                'distance_sels': distance_selection,
            },
            'train_settings': {
                'cvae_settings_yaml': 'cvae-prod-settings.yaml',
            },
        }

        with open(self.yaml_out / 'prod.yaml', 'w') as f:
            yaml.dump(yaml_settings, f)

    def write_cvae_yaml(self, input_shape: tuple[int, ...]) -> None:
        """Write the CVAE model configuration YAML.

        Creates the cvae-prod-settings.yaml file containing neural
        network architecture and training hyperparameters.

        Args:
            input_shape: Shape of input tensor for the CVAE model,
                typically (1, N, N) for contact matrices.
        """
        yaml_settings = {
            'input_shape': input_shape,
            'filters': [16, 16, 16, 16],
            'kernels': [3, 3, 3, 3],
            'strides': [1, 1, 1, 2],
            'affine_widths': [128],
            'affine_dropouts': [0.5],
            'latent_dim': 3,
            'lambda_rec': 1.0,
            'num_data_workers': 4,
            'prefetch_factor': 2,
            'batch_size': 64,
            'device': 'cuda',
            'optimizer_name': 'RMSprop',
            'optimizer_hparams': {
                'lr': 0.001,
                'weight_decay': 0.00001,
            },
            'epochs': 20,
            'checkpoint_log_every': 20,
            'plot_log_every': 20,
            'plot_n_samples': 5000,
            'plot_method': 'raw',
        }

        with open(self.yaml_out / 'cvae-prod-settings.yaml', 'w') as f:
            yaml.dump(yaml_settings, f)
