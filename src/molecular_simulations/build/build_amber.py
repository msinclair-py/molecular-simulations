# ruff: noqa: W293
"""AMBER system building module.

This module provides classes for building molecular systems using AmberTools.
Supports both implicit and explicit solvent setups with automatic ionization
and neutralization.

Classes:
    ImplicitSolvent: Build implicit solvent systems with AMBER force fields.
    ExplicitSolvent: Build explicit solvent cubic boxes with 150mM NaCl.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

PathLike = str | Path
OptPath = str | Path | None

logger = logging.getLogger(__name__)


class ImplicitSolvent:
    """Build an implicit solvent system using AmberTools.

    Produces topology and coordinate files for implicit solvent simulations
    using tleap with user-specified force fields.

    Args:
        path: Directory path for output files. If None, uses parent of pdb.
        pdb: Path to input PDB file.
        protein: Whether to load protein force field (ff19SB). Defaults to True.
        rna: Whether to load RNA force field (Shaw). Defaults to False.
        dna: Whether to load DNA force field (OL21). Defaults to False.
        phos_protein: Whether to load phosphorylated protein force field.
            Defaults to False.
        mod_protein: Whether to load modified amino acid force field.
            Defaults to False.
        out: Output filename. If None, uses 'system.pdb'. Defaults to None.
        delete_temp_file: Whether to delete temporary tleap input files.
            Defaults to True.
        amberhome: Path to AMBER installation. If None, uses AMBERHOME
            environment variable. Defaults to None.
        **kwargs: Additional attributes to set on the instance.

    Attributes:
        path: Resolved Path object for output directory.
        pdb: Resolved Path to input PDB file.
        out: Resolved Path for output files.
        tleap: Path to tleap executable.
        pdb4amber: Path to pdb4amber executable.
        ffs: List of force field files to load.

    Raises:
        ValueError: If AMBERHOME is not set and amberhome is None.

    Example:
        >>> builder = ImplicitSolvent(path="./build", pdb="protein.pdb", protein=True)
        >>> builder.build()
    """

    def __init__(
        self,
        path: OptPath,
        pdb: str,
        protein: bool = True,
        glycans: bool = True,
        rna: bool = False,
        dna: bool = False,
        phos_protein: bool = False,
        mod_protein: bool = False,
        out: OptPath = None,
        delete_temp_file: bool = True,
        amberhome: str | None = None,
        debug: bool = False,
        **kwargs,
    ):
        """Initialize the ImplicitSolvent builder."""
        if path is None:
            self.path = Path(pdb).parent
        elif isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path

        self.path = self.path.resolve()
        self.path.mkdir(exist_ok=True, parents=True)

        self.pdb = Path(pdb).resolve()

        if out is not None:
            self.out = self.path / out
        else:
            self.out = self.path / 'system.pdb'

        self.out = self.out.resolve()
        self.delete = delete_temp_file

        if amberhome is None:
            if 'AMBERHOME' in os.environ:
                amberhome = os.environ['AMBERHOME']
            else:
                raise ValueError('AMBERHOME is not set in env vars!')

        self.amberhome = Path(amberhome)

        self.tleap = str(self.amberhome / 'bin' / 'tleap')
        self.pdb4amber = str(self.amberhome / 'bin' / 'pdb4amber')

        switches = [protein, glycans, rna, dna, phos_protein, mod_protein]
        ffs = [
            'leaprc.protein.ff19SB',
            'leaprc.GLYCAM_06j-1',
            'leaprc.RNA.Shaw',
            'leaprc.DNA.OL21',
            'leaprc.phosaa19SB',
            'leaprc.protein.ff14SB_modAA',
        ]

        self.ffs = [ff for ff, switch in zip(ffs, switches, strict=True) if switch]

        self.debug = debug

        for key, val in kwargs.items():
            setattr(self, key, val)

    def build(self) -> None:
        """Orchestrate the implicit solvent system build.

        Runs tleap to produce topology (.prmtop) and coordinate (.inpcrd)
        files for the input structure.
        """
        logger.info(
            'Build start: implicit solvent',
            extra={'pdb': str(self.pdb), 'out': str(self.out)},
        )
        self.tleap_it()
        logger.info('Build finished')

    def tleap_it(self) -> None:
        """Run tleap to build the system.

        Runs the input PDB through tleap with FF19SB protein force field
        and any other enabled force fields. Sets mbondi3 radii for
        implicit solvent calculations.
        """
        ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        tleap_in = f"""
        {ffs}
        prot = loadpdb {self.pdb}
        set default pbradii mbondi3
        savepdb prot {self.out}
        saveamberparm prot {self.out.with_suffix(".prmtop")} {self.out.with_suffix(".inpcrd")}
        quit
        """

        if self.debug:
            self.debug_tleap(tleap_in)
        else:
            self.temp_tleap(tleap_in)

    def debug_tleap(self, inp: str) -> None:
        """Write a tleap input file.

        Args:
            inp: The tleap input file contents as a string.

        Returns:
            Path to the written tleap input file.
        """
        leap_file = f'{self.path}/tleap.in'
        with open(leap_file, 'w') as outfile:
            outfile.write(inp)

        tleap_command = f'{self.tleap} -f {leap_file}'
        subprocess.run(
            tleap_command,
            shell=True,
            cwd=str(self.path),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def temp_tleap(self, inp: str) -> None:
        """Run tleap with a temporary input file.

        Writes a temporary file for tleap and executes it. This approach
        simplifies parallel tleap runs by avoiding input file conflicts
        between different workers.

        Args:
            inp: The tleap input file contents as a string.
        """
        with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.in', delete=self.delete, dir=str(self.path)
        ) as temp_file:
            temp_file.write(inp)
            temp_file.flush()
            tleap_command = f'{self.tleap} -f {temp_file.name}'
            subprocess.run(
                tleap_command,
                shell=True,
                cwd=str(self.path),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


class ExplicitSolvent(ImplicitSolvent):
    """Build an explicit solvent system using AmberTools.

    Produces an explicit solvent cubic box with user-specified padding,
    neutralized and ionized with 150mM NaCl.

    Args:
        path: Directory path for output files.
        pdb: Path to input PDB file.
        padding: Padding around solute in Angstroms. Defaults to 10.0.
        protein: Whether to load protein force field. Defaults to True.
        rna: Whether to load RNA force field. Defaults to False.
        dna: Whether to load DNA force field. Defaults to False.
        phos_protein: Whether to load phosphorylated protein force field.
            Defaults to False.
        mod_protein: Whether to load modified amino acid force field.
            Defaults to False.
        polarizable: Whether to use polarizable force field (ff15ipq/SPC-Eb).
            Defaults to False.
        delete_temp_file: Whether to delete temporary files. Defaults to True.
        amberhome: Path to AMBER installation. Defaults to None.
        **kwargs: Additional attributes to set on the instance.

    Attributes:
        pad: Padding value in Angstroms.
        water_box: Water box type ('OPCBOX' or 'SPCBOX').

    Example:
        >>> builder = ExplicitSolvent(path="./build", pdb="protein.pdb", padding=12.0)
        >>> builder.build()
    """

    def __init__(
        self,
        path: PathLike,
        pdb: PathLike,
        disulfide_residues: list[int] | None = None,
        padding: float = 10.0,
        protein: bool = True,
        glycans: bool = False,
        rna: bool = False,
        dna: bool = False,
        phos_protein: bool = False,
        mod_protein: bool = False,
        polarizable: bool = False,
        delete_temp_file: bool = True,
        amberhome: str | None = None,
        debug: bool = False,
        **kwargs,
    ):
        """Initialize the ExplicitSolvent builder."""
        super().__init__(
            path=path,
            pdb=str(pdb),
            protein=protein,
            glycans=glycans,
            rna=rna,
            dna=dna,
            phos_protein=phos_protein,
            mod_protein=mod_protein,
            out=None,
            delete_temp_file=delete_temp_file,
            amberhome=amberhome,
            debug=debug,
            **kwargs,
        )
        self.pad = padding
        self.ffs.extend(['leaprc.water.opc'])
        self.water_box = 'OPCBOX'
        
        if disulfide_residues is not None:
            self.disulfides = '\n'.join([f'protein.{resid} = CYX' for resid in disulfide_residues])
        else:
            self.disulfides = '\n'

        if polarizable:
            self.ffs[0] = 'leaprc.protein.ff15ipq'
            self.ffs[-1] = 'leaprc.water.spceb'
            self.water_box = 'SPCBOX'

    def build(self) -> None:
        """Orchestrate the explicit solvent system build.

        Runs pdb4amber to prepare the structure, computes box dimensions,
        calculates ion numbers for 150mM concentration, and runs tleap
        to assemble the final solvated system.
        """
        logger.info(
            'Build started: explicit solvent',
            extra={'pdb': str(self.pdb), 'out': str(self.out)},
        )
        self.prep_pdb()
        dim = self.get_pdb_extent()
        num_ions = self.get_ion_numbers(dim**3)
        self.assemble_system(dim, num_ions)
        self.clean_up_directory()
        logger.info('Build finished')

    def prep_pdb(self) -> None:
        """Prepare the input PDB using cpptraj.

        While pdb4amber has been the standard for some time, it sometimes
        makes decisions such as distance-based disulfide formation which
        cannot be stopped and is sometimes undesirable (such as in binder
        design use cases). For that reason, we are utilizing the more
        feature-rich prepareforleap function in cpptraj.
        """
        self.ss_bonds_leap = self.path  / 'ss_bonds.leap'
        prepared_pdb = self.path / 'protein.pdb'
        cpptraj_in = [
            f'parm {self.pdb}',
            f'loadcrd {self.pdb} name IN',
            (
                'prepareforleap crdset IN name OUT '
                f'pdbout {prepared_pdb} noh existingdisulfides '
                f'leapunitname PROT out {self.ss_bonds_leap}'
            ),
            'quit'
        ]

        cpptraj = str(self.amberhome / 'bin' / 'cpptraj')
        subprocess.run(
            [cpptraj], 
            input='\n'.join(cpptraj_in),
            text=True, 
            cwd=str(self.path), 
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.pdb = str(prepared_pdb)

    def _prep_pdb(self) -> None:
        """Prepare the input PDB using pdb4amber.

        Runs pdb4amber to ensure tleap compatibility. Removes explicit
        hydrogens from the input to avoid naming mismatches.
        """
        cmd = f'{self.pdb4amber} -i {self.pdb} -o {self.path}/protein.pdb -y'
        subprocess.run(
            cmd,
            shell=True,
            cwd=str(self.path),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.pdb = f'{self.path}/protein.pdb'

    def assemble_system(self, dim: float, num_ions: int) -> None:
        """Build the solvated system in tleap.

        Args:
            dim: Box dimension (longest axis + padding) in Angstroms.
            num_ions: Number of Na+/Cl- ion pairs for 150mM concentration.
        """
        tleap_ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        out_pdb = self.out
        out_top = self.out.with_suffix('.prmtop')
        out_coor = self.out.with_suffix('.inpcrd')

        tleap_complex = f"""{tleap_ffs}
        PROT = loadpdb {self.pdb}
        {self.disulfides}
        
        setbox PROT centers
        set PROT box {{{dim} {dim} {dim}}}
        solvatebox PROT {self.water_box} {{0 0 0}}
        
        addions PROT Na+ 0
        addions PROT Cl- 0
        
        addIonsRand PROT Na+ {num_ions} Cl- {num_ions}
        
        savepdb PROT {out_pdb}
        saveamberparm PROT {out_top} {out_coor}
        quit
        """

        if self.debug:
            self.debug_tleap(tleap_complex)
        else:
            self.temp_tleap(tleap_complex)

    def get_pdb_extent(self) -> int:
        """Calculate the required box dimension.

        Identifies the longest axis of the protein based on X/Y/Z
        coordinate projections. Not highly accurate but sufficient
        for determining periodic box size.

        Returns:
            Longest dimension plus twice the padding, in Angstroms.
        """
        lines = []
        with open(self.pdb) as f:
            for line in f.readlines():
                if 'ATOM' in line:
                    lines.append(line)

        xs, ys, zs = [], [], []

        for line in lines:
            xs.append(float(line[30:38].strip()))
            ys.append(float(line[38:46].strip()))
            zs.append(float(line[46:54].strip()))

        xtent = max(xs) - min(xs)
        ytent = max(ys) - min(ys)
        ztent = max(zs) - min(zs)

        return int(max([xtent, ytent, ztent]) + 2 * self.pad)

    def clean_up_directory(self) -> None:
        """Organize output directory.

        Moves intermediate files to a 'build' subdirectory, keeping
        only the final .prmtop and .inpcrd files in the main directory.
        """
        (self.path / 'build').mkdir(exist_ok=True)
        for f in self.path.glob('*'):
            if not any([ext in f.name for ext in ['.prmtop', '.inpcrd', 'build']]):
                f.rename(f.parent / 'build' / f.name)

    @staticmethod
    def get_ion_numbers(volume: float) -> int:
        """Calculate ion count for 150mM NaCl concentration.

        Args:
            volume: Box volume in cubic Angstroms.

        Returns:
            Number of each ion type (Na+ and Cl-) needed for 150mM.
        """
        return round(volume * 10e-6 * 9.03)

