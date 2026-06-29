"""Ligand parameterization and complex building module.

This module provides classes for parameterizing small molecule ligands
using GAFF2 force fields and building protein-ligand complex systems
for molecular dynamics simulations.

Classes:
    LigandError: Custom exception for ligand parameterization failures.
    LigandBuilder: Parameterize ligands with GAFF2 force field.
    PLINDERBuilder: Build complexes from PLINDER database entries.
    ComplexBuilder: Build general protein-ligand complex systems.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from MDAnalysis.lib.util import convert_aa_code
from openbabel import pybel  # ty: ignore[unresolved-import]
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import Sequence
from rdkit import Chem

from .build_amber import ExplicitSolvent, ImplicitSolvent

PathLike = str | Path
Sequences = list[Sequence]

class LigandError(Exception):
    """Custom exception for ligand parameterization errors.

    Raised when antechamber or SQM fails to parameterize a ligand,
    or when other ligand-related issues occur during system building.

    Args:
        message: Error message describing the failure. Defaults to a
            generic message about ligand modeling failure.

    Example:
        >>> raise LigandError('Antechamber failed for molecule XYZ')
    """

    def __init__(self, message='This system contains ligands which we cannot model!'):
        """Initialize the LigandError."""
        self.message = message
        super().__init__(self.message)


class LigandBuilder:
    """Parameterize a ligand molecule with GAFF2 force field.

    Generates all relevant force field files (.frcmod, .lib, .mol2)
    for running tleap with small molecule ligands.

    Args:
        path: Directory path for input/output files.
        lig: Ligand filename (SDF or PDB format).
        lig_number: Numeric identifier for the ligand residue name.
            Creates residue names like 'LG0', 'LG1', etc. Defaults to 0.
        file_prefix: Optional prefix for output files. Defaults to ''.

    Attributes:
        path: Path object for working directory.
        lig: Path to input ligand file.
        ln: Ligand number for residue naming.
        out_lig: Path for output ligand files (without extension).

    Example:
        >>> builder = LigandBuilder(path='./build', lig='ligand.sdf', lig_number=0)
        >>> builder.parameterize_ligand()
    """

    def __init__(
        self, 
        path: PathLike, 
        lig: PathLike, 
        lig_number: int = 0, 
        file_prefix: str = ''
    ):
        """Initialize the LigandBuilder."""
        self.path = Path(path)
        self.lig = self.path / str(lig)
        self.ln = lig_number
        self.out_lig = self.path / f'{file_prefix}{Path(lig).stem}'

        if 'AMBERHOME' in os.environ:
            amberhome = Path(os.environ['AMBERHOME'])
        else:
            raise ValueError('AMBERHOME is not set in env vars!')

        self.antechamber = str(amberhome / 'bin' / 'antechamber')
        self.parmchk2 = str(amberhome / 'bin' / 'parmchk2')
        self.tleap = str(amberhome / 'bin' / 'tleap')

    def parameterize_ligand(self) -> None:
        """Generate GAFF2 parameters for the ligand.

        Ensures consistent treatment of all ligand files by:
        1. Adding hydrogens using RDKit
        2. Converting to mol2 format
        3. Running antechamber for GAFF2 atom typing and AM1-BCC charges
        4. Running parmchk2 to generate missing parameters
        5. Creating a tleap library file

        Raises:
            LigandError: If antechamber fails to parameterize the ligand.
        """
        lig_path = Path(self.lig)
        ext = lig_path.suffix
        self.lig = lig_path.stem

        convert_to_gaff = f'{self.antechamber} -i {self.lig}_prep.mol2 -fi mol2 -o \
                {self.out_lig}.mol2 -fo mol2 -at gaff2 -c bcc -s 0 -pf y -rn LG{self.ln}'
        parmchk2_cmd = (
            f'{self.parmchk2} -i {self.out_lig}.mol2 -f mol2 -o {self.out_lig}.frcmod'
        )

        tleap_ligand = f"""source leaprc.gaff2
        LG{self.ln} = loadmol2 {self.out_lig}.mol2
        loadamberparams {self.out_lig}.frcmod
        saveoff LG{self.ln} {self.out_lig}.lib
        quit
        """

        self.process_input(ext)
        self.convert_to_mol2()

        os.system(convert_to_gaff)
        try:
            self.move_antechamber_outputs()
            self.check_sqm()
            os.system(parmchk2_cmd)
            leap_file, leap_log = self.write_leap(tleap_ligand)
            os.system(f'{self.tleap} -f {leap_file} > {leap_log}')
        except FileNotFoundError as exc:
            raise LigandError(f'Antechamber failed! {self.lig}') from exc

    def process_input(self,
                      extension: str) -> None:
        """Process input ligand of filetypes mol2, pdb or sdf.

        Adds hydrogens using RDKit and writes the result to a new SDF file.
        Note that incorrect atom hybridization in SDF may lead to incorrect
        hydrogen placement.

        Args:
            extension: File extension of the input ligand. One of '.mol2',
                '.pdb', or '.sdf'.

        Raises:
            LigandError: If the extension is not a supported filetype.
        """
        match extension:
            case '.mol2':
                mol = Chem.MolFromMolFile(f'{self.lig}{extension}')
            case '.pdb':
                mol = Chem.MolFromPDBFile(f'{self.lig}{extension}')
            case '.sdf':
                mol = Chem.SDMolSupplier(f'{self.lig}{extension}')[0]
            case _:
                raise LigandError()

        molH = Chem.AddHs(mol, addCoords=True)

        with Chem.SDWriter(f'{self.lig}_H.sdf') as w:
            w.write(molH)

    def convert_to_mol2(self) -> None:
        """Convert SDF to mol2 format using OpenBabel."""
        mol = next(iter(pybel.readfile('sdf', f'{self.lig}_H.sdf')))
        mol.write('mol2', f'{self.lig}_prep.mol2', True)

    def move_antechamber_outputs(self) -> None:
        """Clean up antechamber output files.

        Removes unnecessary outputs and renames sqm.out for later
        verification that antechamber completed successfully.
        """
        os.remove('sqm.in')
        os.remove('sqm.pdb')
        shutil.move('sqm.out', f'{self.lig}_sqm.out')

    def check_sqm(self) -> None:
        """Verify that SQM calculations completed successfully.

        Checks the sqm.out file for completion message. If absent,
        indicates parameter generation failed.

        Raises:
            LigandError: If SQM calculations did not complete.
        """
        with open(f'{self.lig}_sqm.out') as f:
            line = f.readlines()[-2]

        if 'Calculation Completed' not in line:
            raise LigandError(f'SQM failed for ligand {self.lig}!')

    def write_leap(self, inp: str) -> tuple[str, str]:
        """Write a tleap input file.

        Args:
            inp: The tleap input file contents as a string.

        Returns:
            Tuple of (input_file_path, log_file_path).
        """
        leap_file = f'{self.path}/tleap.in'
        leap_log = f'{self.path}/leap.log'
        with open(leap_file, 'w') as outfile:
            outfile.write(inp)

        return leap_file, leap_log


class ComplexBuilder(ExplicitSolvent):
    """Build protein-ligand complexes with explicit solvent.

    Extends ExplicitSolvent to handle ligand parameterization and
    complex assembly. Supports both automatic parameterization via
    antechamber and pre-computed parameter files.

    Args:
        path: Directory path for output files.
        pdb: Path to protein PDB file.
        lig: Path to ligand file(s). Can be a single path or list of paths.
        padding: Box padding in Angstroms. Defaults to 10.0.
        lig_param_prefix: Optional path prefix to pre-computed ligand
            parameters (.frcmod, .lib, .mol2). If None, parameters are
            generated automatically. Defaults to None.
        **kwargs: Additional attributes (e.g., 'ion' for ion PDB path).

    Attributes:
        lig: Path(s) to ligand file(s).
        build_dir: Directory for intermediate build files.
        lig_param_prefix: Prefix for pre-computed parameter files.

    Example:
        >>> builder = ComplexBuilder(
        ...     path='./build', pdb='protein.pdb', lig='ligand.sdf', padding=12.0
        ... )
        >>> builder.build()
    """

    def __init__(
        self,
        path: str,
        pdb: str,
        lig: str | list[str],
        padding: float = 10.0,
        lig_param_prefix: str | None = None,
        **kwargs,
    ):
        """Initialize the ComplexBuilder."""
        super().__init__(path=path, pdb=pdb, padding=padding)
        self.lig = (
            Path(lig).resolve()
            if isinstance(lig, str)
            else [Path(_lig).resolve() for _lig in lig]
        )
        self.ffs.append('leaprc.gaff2')
        self.build_dir = self.out.parent / 'build'

        if lig_param_prefix is None:
            self.lig_param_prefix = lig_param_prefix
        else:
            prefix = Path(lig_param_prefix)
            self.lig_param_prefix = prefix.parent / prefix.stem

        self.ion: str | Path | None = kwargs.pop('ion', None)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def build(self) -> None:
        """Build the solvated protein-ligand complex.

        Parameterizes ligands (if needed), prepares the protein,
        and assembles the solvated system.
        """
        self.build_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(self.build_dir)  # necessary for antechamber outputs

        if self.lig_param_prefix is None:
            if isinstance(self.lig, list):
                lig_paths: list[Path] = []
                for i, lig in enumerate(self.lig):
                    lig_paths.append(self.process_ligand(lig, i))

                self.lig = lig_paths

            else:
                self.lig = self.process_ligand(self.lig)
        else:
            self.lig = self.lig_param_prefix

        if self.ion is not None:
            self.add_ion_to_pdb()

        self.prep_pdb()
        dim = self.get_pdb_extent()
        num_ions = self.get_ion_numbers(dim**3)
        self.assemble_system(dim, num_ions)

    def process_ligand(self, lig: PathLike, prefix: int | None = None) -> Path:
        """Process and parameterize a single ligand.

        Args:
            lig: Path to ligand file.
            prefix: Optional numeric prefix for multi-ligand systems.

        Returns:
            Path to parameterized ligand (without extension).
        """
        lig_path = Path(lig)
        if lig_path.parent != self.build_dir:
            shutil.copy(lig_path, self.build_dir)

        file_prefix = '' if prefix is None else str(prefix)

        lig_builder = LigandBuilder(self.build_dir, str(lig_path.name), file_prefix=file_prefix)
        lig_builder.parameterize_ligand()

        return lig_builder.out_lig

    def add_ion_to_pdb(self) -> None:
        """Add ion coordinates to the protein PDB file.

        Reads ion coordinates from a separate file and appends them
        to the protein PDB before the END record.
        """
        ion = []
        assert self.ion is not None
        with open(self.ion) as f:
            for line in f.readlines():
                if any(['ATOM' in line, 'HETATM' in line]):
                    ion.append(line)

        with open(self.pdb) as f:
            pdb = f.readlines()

        out_pdb = []
        for line in pdb:
            if 'END' in line:
                out_pdb.extend(ion)
                out_pdb.append(line)
            else:
                out_pdb.append(line)

        with open(self.pdb, 'w') as f:
            f.write(''.join(out_pdb))

    def assemble_system(self, dim: float, num_ions: int) -> None:
        """Assemble the solvated protein-ligand complex.

        Loads ligand parameters, combines with protein, solvates,
        and ionizes the system.

        Args:
            dim: Box dimension in Angstroms.
            num_ions: Number of Na+/Cl- pairs for 150mM concentration.
        """
        tleap_ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        tleap_complex = [
            tleap_ffs,
            'source leaprc.gaff2',
        ]

        if not isinstance(self.lig, list):
            self.lig = [self.lig]

        LABELS = []
        for i, lig in enumerate(self.lig):
            tleap_complex += [
                f'loadamberparams {lig}.frcmod',
                f'loadoff {lig}.lib',
                f'LG{i} = loadmol2 {lig}.mol2',
            ]

            LABELS.append(f'LG{i}')

        LABELS.append('PROT')
        LABELS = ' '.join(LABELS)

        out_top = self.out.with_suffix('.prmtop')
        out_coor = self.out.with_suffix('.inpcrd')

        tleap_complex += [
            f'PROT = loadpdb {self.pdb}',
            f'COMPLEX = combine {{{LABELS}}}',
            'setbox COMPLEX centers',
            f'set COMPLEX box {{{dim} {dim} {dim}}}',
            f'solvatebox COMPLEX {self.water_box} {{0 0 0}}',
            'addions COMPLEX Na+ 0',
            'addions COMPLEX Cl- 0',
            f'addIonsRand COMPLEX Na+ {num_ions} Cl- {num_ions}',
            f'savepdb COMPLEX {self.out}',
            f'saveamberparm COMPLEX {out_top} {out_coor}',
            'quit',
        ]

        if self.debug:
            self.debug_tleap('\n'.join(tleap_complex))
        else:
            self.temp_tleap('\n'.join(tleap_complex))
