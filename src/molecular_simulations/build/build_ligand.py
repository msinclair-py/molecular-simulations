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
        self, path: PathLike, lig: PathLike, lig_number: int = 0, file_prefix: str = ''
    ):
        """Initialize the LigandBuilder."""
        self.path = Path(path)
        self.lig: Path | str = self.path / str(lig)
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

        if ext == '.sdf':
            self.process_sdf()
        else:
            self.process_pdb()

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

    def process_sdf(self) -> None:
        """Process an SDF file and add hydrogens.

        Uses RDKit to add hydrogens based on atom hybridization from
        the input SDF file. Note that incorrect hybridization in the
        input will result in incorrect hydrogen placement.
        """
        mol = Chem.SDMolSupplier(f'{self.lig}.sdf')[0]
        molH = Chem.AddHs(mol, addCoords=True)
        with Chem.SDWriter(f'{self.lig}_H.sdf') as w:
            w.write(molH)

    def process_pdb(self) -> None:
        """Process a PDB file and add hydrogens.

        Reads a small molecule PDB, adds hydrogens using RDKit,
        and writes the result to an SDF file.
        """
        mol = Chem.MolFromPDBFile(f'{self.lig}.pdb')
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


class PLINDERBuilder(ImplicitSolvent):
    """Build protein-ligand complexes from PLINDER database entries.

    Extends ImplicitSolvent to handle PLINDER-format input directories
    containing receptor PDB, ligand SDF files, and sequence information.

    Args:
        path: Base path to PLINDER system directory.
        system_id: PLINDER system identifier.
        out: Output directory path.
        **kwargs: Additional arguments passed to ImplicitSolvent.

    Attributes:
        system_id: PLINDER system identifier.
        build_dir: Directory for intermediate build files.
        ions: List of ions extracted from ligand files.
        ligs: List of parameterized ligand names.

    Example:
        >>> builder = PLINDERBuilder(
        ...     path='./plinder_data', system_id='1abc_A_B_ligand', out='./output'
        ... )
        >>> builder.build()
    """

    def __init__(self, path: PathLike, system_id: str, out: PathLike, **kwargs):
        """Initialize the PLINDERBuilder."""
        super().__init__(
            Path(path) / system_id,
            'receptor.pdb',
            protein=True,
            rna=True,
            dna=True,
            phos_protein=True,
            mod_protein=True,
            out=Path(out) / system_id,
            **kwargs,
        )
        self.system_id = system_id
        self.ffs.append('leaprc.gaff2')
        self.build_dir = self.out / 'build'
        self.ions: list[list[list[Any]]] | None = None

    def build(self) -> None:
        """Build the protein-ligand complex system.

        Migrates files, parameterizes ligands, and assembles the
        final system with topology and coordinate files.

        Raises:
            LigandError: If no ligands are found or parameterization fails.
        """
        ligs = self.migrate_files()

        if not ligs:
            print(f'No ligands!\n\n{self.pdb}')
            raise LigandError

        self.ligs = self.ligand_handler(ligs)
        self.assemble_system()

    def ligand_handler(self, ligs: list[str]) -> list[str]:
        """Parameterize all ligands in the system.

        Args:
            ligs: List of ligand file paths to parameterize.

        Returns:
            List of parameterized ligand names (without extensions).
        """
        ligands = []
        for i, lig in enumerate(ligs):
            lig_builder = LigandBuilder(self.build_dir, lig, i)
            lig_builder.parameterize_ligand()
            ligands.append(os.path.basename(lig)[:-4])

        return ligands

    def migrate_files(self) -> list[str]:
        """Prepare input files for system building.

        Copies sequence files, fixes and moves the receptor PDB,
        and processes ligand files. Handles ions separately.

        Returns:
            List of ligand filenames to parameterize.
        """
        os.makedirs(str(self.build_dir), exist_ok=True)
        os.chdir(self.build_dir)  # necessary for antechamber outputs

        # grab the sequence file to complete protein modeling
        shutil.copy(str(self.path / 'sequences.fasta'), str(self.build_dir))
        self.fasta = str(self.build_dir / 'sequences.fasta')

        # fix and move pdb
        self.prep_protein()

        # move ligand(s)
        ligands = []
        lig_files = self.path / 'ligand_files'
        for lig in lig_files.glob('*.sdf'):
            shutil.copy(str(lig), str(self.build_dir))

            if self.check_ligand(lig):
                ligands.append(lig.name)

        # handle any potential ions
        if self.ions is not None:
            self.ffs.append('leaprc.water.tip3p')
            self.place_ions()

        return ligands

    def place_ions(self) -> None:
        """Add ion records to the PDB file.

        Appends ATOM records for extracted ions to the receptor PDB.
        Handles various ion naming conventions for AMBER compatibility.

        Note:
            This method handles complex PDB formatting requirements.
            Proceed with caution if modifications are needed.
        """
        with open(self.pdb) as f:
            pdb_lines = f.readlines()[:-1]

        if 'END' in pdb_lines[-1]:
            ln = -3 if 'TER' in pdb_lines[-2] else -2
        elif 'TER' in pdb_lines[-1]:
            ln = -2
        else:
            ln = -1

        try:
            next_atom_num = int(pdb_lines[ln][6:12].strip()) + 1
            next_resid = int(pdb_lines[ln][22:26].strip()) + 1
        except ValueError as exc:
            print(f'ERROR: {self.pdb}')
            raise LigandError from exc

        assert self.ions is not None
        for ion in self.ions:
            for atom in ion:
                ion_line = f'ATOM  {next_atom_num:>5}'

                if atom[0].lower() in ['na', 'k', 'cl']:
                    ionname = atom[0] + atom[1]
                    ion_line += f'{ionname:>4}  {ionname:<3} '
                else:
                    ionname = atom[0].upper()
                    ion_line += f'{ionname:>3}   {ionname:<3}'

                coords = ''.join([f'{x:>8.3f}' for x in atom[2:]])
                ion_line += f'{next_resid:>5}    {coords}  0.00  0.00\n'

                pdb_lines.append(ion_line)
                pdb_lines.append('TER\n')

                next_atom_num += 1
                next_resid += 1

        pdb_lines.append('END')

        with open(self.pdb, 'w') as f:
            f.write(''.join(pdb_lines))

    def assemble_system(self) -> None:
        """Assemble the protein-ligand complex with tleap.

        Creates topology and coordinate files for the complex,
        including all parameterized ligands and ions.
        """
        tleap_complex = [f'source {ff}' for ff in self.ffs]
        structs = [f'PROT = loadpdb {self.pdb}']
        combine = 'COMPLEX = combine{PROT'
        for i, lig in enumerate(self.ligs):
            ligand = self.build_dir / lig
            tleap_complex += [
                f'loadamberparams {ligand}.frcmod',
                f'loadoff {ligand}.lib',
            ]
            structs += [f'LG{i} = loadmol2 {ligand}.mol2']
            combine += f' LG{i}'

        combine += '}'
        tleap_complex += structs
        tleap_complex.append(combine)
        tleap_complex += [
            'set default PBRadii mbondi3',
            f'savepdb COMPLEX {self.out}/system.pdb',
            f'saveamberparm COMPLEX {self.out}/system.prmtop {self.out}/system.inpcrd',
            'quit',
        ]

        tleap_complex = '\n'.join(tleap_complex)
        leap_file = self.build_dir / 'tleap.in'
        with open(str(leap_file), 'w') as outfile:
            outfile.write(tleap_complex)

        tleap = f'tleap -f {leap_file}'
        os.system(tleap)

    def prep_protein(self) -> None:
        """Prepare the receptor protein structure.

        Runs PDBFixer to repair missing residues, then pdb4amber
        to remove hydrogens and waters for clean tleap input.
        """
        raw_pdb = self.path / self.pdb
        prep_pdb = self.build_dir / 'prepped.pdb'
        self.pdb = self.build_dir / 'protein.pdb'

        # complex workflow for modeling missing residues
        self.triage_pdb(raw_pdb, prep_pdb)

        # remove hydrogens (-y) and waters (-d) from the input PDB
        pdb4amber = f'pdb4amber -i {prep_pdb} -o {self.pdb} -y -d'
        os.system(pdb4amber)

    def triage_pdb(self, broken_pdb: PathLike, repaired_pdb: PathLike) -> None:
        """Repair a PDB structure using PDBFixer.

        Runs PDBFixer to add missing loops and atoms. Uses FASTA
        sequence from PLINDER to properly model missing residues,
        including non-canonical residues like phosphorylations.

        Args:
            broken_pdb: Path to input PDB requiring repairs.
            repaired_pdb: Path for output repaired PDB.
        """
        fixer = PDBFixer(filename=str(broken_pdb))
        chains = [chain for chain in fixer.topology.chains()]
        chain_map = {chain.id: [res for res in chain.residues()] for chain in chains}

        # non-databank models do not have SEQRES and therefore no
        # sequence data to model missing residues
        fixer.sequences = self.inject_fasta(chain_map)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        with open(str(repaired_pdb), 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

    def inject_fasta(self, chain_map: dict[str, list[Any]]) -> Sequences:
        """Inject FASTA sequence data for PDBFixer.

        Checks FASTA against actual sequence and modifies to correctly
        handle non-canonical residues (e.g., SER -> SEP for phosphoserine).

        Args:
            chain_map: Dictionary mapping chain IDs to lists of residues.

        Returns:
            List of PDBFixer Sequence objects for all chains.

        Raises:
            LigandError: If unknown residues are found in the FASTA.
        """
        with open(self.fasta) as f:
            fasta = f.readlines()

        with open(self.path / 'chain_mapping.json', 'rb') as f:
            remapping = json.load(f)

        sequences = []
        for i in range(len(fasta) // 2):
            seq_chain = fasta[2 * i].strip()[1:]  # strip off > and \n
            chain = remapping[seq_chain]
            one_letter_seq = fasta[2 * i + 1].strip()
            try:
                three_letter_seq = [convert_aa_code(aa) for aa in one_letter_seq]
            except ValueError as exc:
                print(f'\nUnknown residue in fasta!\n\n{self.pdb}')
                raise LigandError from exc

            try:
                three_letter_seq = self.check_ptms(three_letter_seq, chain_map[chain])
                sequences.append(Sequence(chainId=chain, residues=three_letter_seq))
            except KeyError as exc:
                print(f'\nUnknown ligand error!\n\n{self.pdb}')
                raise LigandError from exc

        return sequences

    def check_ptms(self, sequence: list[str], chain_residues: list[Any]) -> list[str]:
        """Check for post-translational modifications in the sequence.

        Compares the full sequence from FASTA against the partial
        sequence from the structural model and updates non-canonical
        residue names appropriately.

        Args:
            sequence: List of three-letter residue codes from FASTA.
            chain_residues: List of residue objects from the structure.

        Returns:
            Updated sequence list with PTM residue names.

        Raises:
            LigandError: If sequence length mismatches occur.
        """
        for residue in chain_residues:
            resID = int(residue.id) - 1  # since 0-indexed in list

            try:
                if sequence[resID] != residue.name:
                    sequence[resID] = residue.name
            except IndexError as exc:
                print(f'Sequence length is messed up!\n\n{self.pdb}')
                raise LigandError from exc

        return sequence

    def check_ligand(self, ligand: PathLike) -> bool:
        """Check if a ligand file contains ions or valid small molecules.

        Identifies species that are ions based on element type and
        formal charge. True ligands are returned for parameterization
        while ions are stored separately.

        Args:
            ligand: Path to ligand SDF file.

        Returns:
            True if ligand should be parameterized, False if it's an ion.
        """
        ion = False
        mol = Chem.SDMolSupplier(str(ligand))[0]

        ion_atoms: list[list[Any]] = []
        for atom, position in zip(mol.GetAtoms(),
                                  mol.GetConformer().GetPositions(),
                                  strict=True):
            symbol = atom.GetSymbol()
            if symbol.lower() in self.cation_list + self.anion_list:
                charge = atom.GetFormalCharge()
                if charge != 0:
                    ion = True
                    sign = '+' if charge > 0 else '-'
                    if abs(charge) > 1:
                        sign = f'{charge}{sign}'

                    ion_atoms.append([symbol, sign] + [x for x in position])

        if ion:
            if self.ions is None:
                self.ions = [ion_atoms]
            else:
                self.ions.append(ion_atoms)
            return False

        return True

    @property
    def cation_list(self) -> list[str]:
        """List of common cation element symbols (lowercase)."""
        return [
            'na',
            'k',
            'ca',
            'mn',
            'mg',
            'li',
            'rb',
            'cs',
            'cu',
            'ag',
            'au',
            'ti',
            'be',
            'sr',
            'ba',
            'ra',
            'v',
            'cr',
            'fe',
            'co',
            'zn',
            'ni',
            'pd',
            'cd',
            'sn',
            'pt',
            'hg',
            'pb',
            'al',
        ]

    @property
    def anion_list(self) -> list[str]:
        """List of common anion element symbols (lowercase)."""
        return ['cl', 'br', 'i', 'f']


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
