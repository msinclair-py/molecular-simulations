from __future__ import annotations

# ruff: noqa: SIM115
"""MM-P(G)BSA calculation module.

This module provides a reconstructed implementation of MM-P(G)BSA from AMBER,
with improved documentation and parallelization options. Supports frame-level
parallelization for efficient processing of long trajectories.
"""

import json
import MDAnalysis as mda
import os
import re
import subprocess
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

# Thread settings for higher level parallelism (Parsl/Academy)
# Must be set BEFORE importing numpy
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

PathLike = Path | str


def _run_energy_calculation(
    args: tuple[str, str, str, str, str, str, str], max_retries: int = 3
) -> tuple[Path, bool, str]:
    """Worker function for parallel energy calculations.

    Must be module-level for ThreadPoolExecutor pickling.

    Args:
        args: Tuple of (mmpbsa_binary, mdin_path, prmtop, pdb,
            traj_chunk, output_path, cwd).
        max_retries: Number of retry attempts for failed calculations.

    Returns:
        Tuple of (output_path, success_bool, error_message).
    """
    mmpbsa_binary, mdin, prm, pdb, trj, out, cwd = args
    cmd = f'{mmpbsa_binary} -O -i {mdin} -p {prm} -c {pdb} -y {trj} -o {out}'

    expected_output = Path(cwd) / out

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(cwd), capture_output=True, text=True
            )

            if expected_output.exists() and expected_output.stat().st_size > 0:
                with open(expected_output) as f:
                    content = f.read()
                    if ' BOND' in content:
                        return (Path(out), True, '')
                    else:
                        error = 'Output file exists but contains no energy data'
            else:
                error = 'Output file missing or empty after subprocess complete'

            if result.returncode != 0:
                error = f'Return code: {result.returncode}: {result.stderr or result.stdout}'

        except subprocess.TimeoutExpired:
            error = 'Calculation timed out'
        except Exception as e:
            error = f'Exception: {e}'

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    return (Path(out), False, error)


def _run_sasa_calculation(
    args: tuple[str, str, str], max_retries: int = 3
) -> tuple[Path, bool, str]:
    """Worker function for parallel SASA calculations.

    Must be module-level for ThreadPoolExecutor pickling.

    Args:
        args: Tuple of (cpptraj_binary, sasa_script_path, cwd).
        max_retries: Number of retry attempts.

    Returns:
        Tuple of (script_path, success_bool, error_message).
    """
    cpptraj_binary, sasa_script, cwd = args

    # Parse expected output from script
    script_path = Path(sasa_script)
    with open(script_path) as f:
        script_content = f.read()

    match = re.search(r'molsurf\s+.*?\s+out\s+(\S+)', script_content)

    expected_output = Path(cwd) / match.group(1) if match else None

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                f'{cpptraj_binary} -i {sasa_script}',
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
            if expected_output and expected_output.exists():
                if expected_output.stat().st_size > 0:
                    with open(expected_output) as f:
                        lines = f.readlines()
                        data_lines = [
                            line
                            for line in lines
                            if line.strip() and not line.strip().startswith('#')
                        ]
                        if len(data_lines) > 0:
                            return (Path(sasa_script), True, '')
                        else:
                            error = 'Output file has no data lines'
                else:
                    error = 'Expected output file is empty'
            else:
                error = f'Expected output file not found: {expected_output}'

            if result.returncode != 0:
                error = (
                    f'Return code {result.returncode}: {result.stderr or result.stdout}'
                )

        except subprocess.TimeoutExpired:
            error = 'Calculation timed out'
        except Exception as e:
            error = f'Exception: {e}'

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    return (script_path, False, error)


@dataclass
class MMPBSASettings:
    """Settings dataclass for MMPBSA calculations.

    Attributes:
        top: Path to topology file.
        dcd: Path to trajectory file.
        selections: List of residue selections for receptor and ligand.
        first_frame: First frame to analyze.
        last_frame: Last frame to analyze (-1 for all).
        stride: Frame stride.
        n_cpus: Number of parallel processes.
        out: Output prefix/path.
        solvent_probe: Probe radius for SASA.
        offset: SASA offset parameter.
        gb_surften: GB surface tension parameter.
        gb_surfoff: GB surface offset parameter.
    """

    top: PathLike
    dcd: PathLike
    selections: list[str] | None
    first_frame: int = 0
    last_frame: int = -1
    stride: int = 1
    n_cpus: int = 1
    out: str = 'mmpbsa'
    solvent_probe: float = 1.4
    offset: int = 0
    gb_surften: float = 0.0072
    gb_surfoff: float = 0.0


class MMPBSA(MMPBSASettings):
    """Perform MM-P(G)BSA calculations with parallelization support.

    A reconstructed implementation of AMBER's MM-P(G)BSA workflow with
    improved documentation and parallelization options. Supports both
    Generalized Born (GB) and Poisson-Boltzmann (PB) solvation methods.

    Attributes:
        path: Working directory for calculations.
        fh: FileHandler instance for file management.
        analyzer: OutputAnalyzer instance for parsing results.
        free_energy: Final binding free energy after run().

    Args:
        top: Path to solvated system topology (prmtop).
        dcd: Path to trajectory file (DCD or MDCRD).
        selections: List of two residue ID selections for receptor and
            ligand, formatted for cpptraj (e.g., ':1-10').
        first_frame: First frame to analyze. Defaults to 0.
        last_frame: Last frame (-1 for all). Defaults to -1.
        stride: Frame stride. Defaults to 1.
        n_cpus: Number of parallel processes. Defaults to 1.
        out: Output prefix or path. Defaults to 'mmpbsa'.
        solvent_probe: Probe radius for SASA in Angstroms. Defaults to 1.4.
        offset: SASA offset. Defaults to 0.
        gb_surften: GB surface tension. Defaults to 0.0072.
        gb_surfoff: GB surface offset. Defaults to 0.0.
        amberhome: Path to AMBERHOME. If None, uses $AMBERHOME env var.
        parallel_mode: 'frame' for frame-level parallelization (recommended),
            'serial' for sequential processing.
        **kwargs: Additional keyword arguments stored as attributes.

    Example:
        >>> mmpbsa = MMPBSA(
        ...     "system.prmtop", "traj.dcd", [":1-100", ":101-150"], n_cpus=4
        ... )
        >>> mmpbsa.run()
        >>> print(mmpbsa.free_energy)
    """

    def __init__(
        self,
        top: PathLike,
        dcd: PathLike,
        selections: list[str],
        first_frame: int = 0,
        last_frame: int = -1,
        stride: int = 1,
        n_cpus: int = 1,
        out: str = 'mmpbsa',
        solvent_probe: float = 1.4,
        offset: int = 0,
        gb_surften: float = 0.0072,
        gb_surfoff: float = 0.0,
        amberhome: str | None = None,
        parallel_mode: Literal['frame', 'serial'] = 'frame',
        **kwargs,
    ):
        """Initialize the MMPBSA calculator.

        Args:
            top: Path to topology file.
            dcd: Path to trajectory file.
            selections: Receptor and ligand selections.
            first_frame: Starting frame.
            last_frame: Ending frame.
            stride: Frame stride.
            n_cpus: Number of CPUs.
            out: Output path/prefix.
            solvent_probe: SASA probe radius.
            offset: SASA offset.
            gb_surften: GB surface tension.
            gb_surfoff: GB surface offset.
            amberhome: AMBER installation path.
            parallel_mode: Parallelization strategy.
            **kwargs: Additional attributes.
        """
        super().__init__(
            top=top,
            dcd=dcd,
            selections=selections,
            first_frame=first_frame,
            last_frame=last_frame,
            stride=stride,
            n_cpus=n_cpus,
            out=out,
            solvent_probe=solvent_probe,
            offset=offset,
            gb_surften=gb_surften,
            gb_surfoff=gb_surfoff,
        )
        self.parallel_mode = parallel_mode
        self.top = Path(self.top).resolve()
        self.traj = Path(self.dcd).resolve()
        self.path = self.top.parent
        if out == 'mmpbsa':
            self.path = self.path / 'mmpbsa'
        else:
            self.path = Path(out).resolve()

        self.path.mkdir(exist_ok=True, parents=True)

        self.cpptraj = 'cpptraj'
        self.mmpbsa_py_energy = 'mmpbsa_py_energy'
        if amberhome is None:
            if 'AMBERHOME' in os.environ:
                amberhome = os.environ['AMBERHOME']
            else:
                raise ValueError('AMBERHOME not set in env vars!')

        self.cpptraj = Path(amberhome) / 'bin' / self.cpptraj
        self.mmpbsa_py_energy = Path(amberhome) / 'bin' / self.mmpbsa_py_energy

        self.fh = FileHandler(
            top=self.top,
            traj=self.traj,
            path=self.path,
            sels=self.selections,
            first=self.first_frame,
            last=self.last_frame,
            stride=self.stride,
            cpptraj_binary=self.cpptraj,
            n_chunks=self.n_cpus,
        )

        self.analyzer = OutputAnalyzer(
            path=self.path, surface_tension=self.gb_surften, sasa_offset=self.gb_surfoff
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.selections is None:
            self.get_selections()

    def run(self) -> None:
        """Execute the complete MM-PBSA workflow.

        Runs file preparation, energy calculations, and result analysis
        using the configured parallelization mode.
        """
        gb_mdin, pb_mdin = self.write_mdins()

        if self.parallel_mode == 'frame':
            self._run_frame_parallel(gb_mdin, pb_mdin)
        else:
            self._run_serial(gb_mdin, pb_mdin)

        self.analyzer.parse_outputs()

        self.free_energy = self.analyzer.free_energy

    def _run_serial(self, gb_mdin: Path, pb_mdin: Path) -> None:
        """Run calculations sequentially.

        Args:
            gb_mdin: Path to GB input file.
            pb_mdin: Path to PB input file.
        """
        for prefix, top, traj, pdb in self.fh.files:
            self.calculate_sasa(prefix, top, traj)
            self.calculate_energy(prefix, top, traj, pdb, gb_mdin, 'gb')
            self.calculate_energy(prefix, top, traj, pdb, pb_mdin, 'pb')

    def _run_frame_parallel(self, gb_mdin: Path, pb_mdin: Path) -> None:
        """Run calculations with frame-level parallelization.

        Splits trajectory into chunks and processes in parallel,
        providing the best speedup for long trajectories.

        Args:
            gb_mdin: Path to GB input file.
            pb_mdin: Path to PB input file.

        Raises:
            RuntimeError: If any calculations fail.
        """
        # Collect all calculation tasks
        energy_tasks = []
        sasa_tasks = []

        for prefix, top, traj_chunks, pdb in self.fh.files_chunked:
            system_name = prefix.name

            # SASA calculations for each chunk
            for i, traj_chunk in enumerate(traj_chunks):
                sasa_script = self._write_sasa_script(
                    prefix, top, traj_chunk, chunk_idx=i
                )
                sasa_tasks.append((str(self.cpptraj), str(sasa_script), str(self.path)))

            # Energy calculations for each chunk
            for i, traj_chunk in enumerate(traj_chunks):
                # GB calculation
                out_gb = f'{system_name}_chunk{i}_gb.mdout'
                energy_tasks.append(
                    (
                        str(self.mmpbsa_py_energy),
                        str(gb_mdin),
                        str(top),
                        str(pdb),
                        str(traj_chunk),
                        out_gb,
                        str(self.path),
                    )
                )
                # PB calculation
                out_pb = f'{system_name}_chunk{i}_pb.mdout'
                energy_tasks.append(
                    (
                        str(self.mmpbsa_py_energy),
                        str(pb_mdin),
                        str(top),
                        str(pdb),
                        str(traj_chunk),
                        out_pb,
                        str(self.path),
                    )
                )

        # Run SASA calculations in parallel
        sasa_failures = []
        with ThreadPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = []
            for task in sasa_tasks:
                futures.append(executor.submit(_run_sasa_calculation, task))

            done, _ = wait(futures, return_when=ALL_COMPLETED)

            for future in done:
                script, success, error = future.result()
                if not success:
                    sasa_failures.append((script, error))

        if sasa_failures:
            failed_scripts = [f[0] for f in sasa_failures]
            raise RuntimeError(
                f'{len(sasa_failures)} SASA calculations failed: {failed_scripts}'
            )

        # Combine SASA results
        self._combine_sasa_chunks()

        # Run Energy calculations in parallel
        energy_failures = []
        with ThreadPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = []
            for task in energy_tasks:
                futures.append(executor.submit(_run_energy_calculation, task))

            done, _ = wait(futures, return_when=ALL_COMPLETED)

            for future in done:
                script, success, error = future.result()
                if not success:
                    energy_failures.append((script, error))

        if energy_failures:
            failed_scripts = [f[0] for f in energy_failures]
            raise RuntimeError(
                f'{len(energy_failures)} Energy calculations failed: {failed_scripts}'
            )

        # Combine Energy results
        self._combine_energy_chunks()
        self._verify_combined_outputs()

    def _write_sasa_script(
        self, prefix: Path, prm: Path, trj: Path, chunk_idx: int = 0
    ) -> Path:
        """Write a SASA calculation script for a trajectory chunk.

        Args:
            prefix: Output prefix path.
            prm: Path to topology file.
            trj: Path to trajectory chunk.
            chunk_idx: Index of this chunk.

        Returns:
            Path to the generated script file.
        """
        sasa = self.path / f'sasa_{prefix.name}_chunk{chunk_idx}.in'
        out_file = f'{prefix.name}_chunk{chunk_idx}_surf.dat'
        sasa_in = [
            f'parm {prm}',
            f'trajin {trj}',
            f'molsurf :* out {out_file} probe {self.solvent_probe} offset {self.offset}',
            'run',
            'quit',
        ]
        self.fh.write_file(sasa_in, sasa)
        return sasa

    def _combine_sasa_chunks(self) -> None:
        """Combine SASA results from all chunks in correct frame order."""

        def extract_chunk_idx(filepath: Path) -> int:
            match = re.search(r'_chunk(\d+)_', filepath.name)
            return int(match.group(1)) if match else 0

        for system in ['complex', 'receptor', 'ligand']:
            combined_data = []
            chunk_files = list(self.path.glob(f'{system}_chunk*_surf.dat'))
            chunk_files.sort(key=extract_chunk_idx)

            for chunk_file in chunk_files:
                with open(chunk_file) as f:
                    lines = f.readlines()
                    if combined_data:
                        combined_data.extend(lines[1:])
                    else:
                        combined_data.extend(lines)

            output = self.path / f'{system}_surf.dat'
            with open(output, 'w') as f:
                f.writelines(combined_data)

            for chunk_file in chunk_files:
                chunk_file.unlink()

    def _combine_energy_chunks(self) -> None:
        """Combine energy results from all chunks in correct frame order."""

        def extract_chunk_idx(filepath: Path) -> int:
            match = re.search(r'_chunk(\d+)_', filepath.name)
            return int(match.group(1)) if match else 0

        for system in ['complex', 'receptor', 'ligand']:
            for level in ['gb', 'pb']:
                combined_data = []
                chunk_files = list(self.path.glob(f'{system}_chunk*_{level}.mdout'))
                chunk_files.sort(key=extract_chunk_idx)

                for chunk_file in chunk_files:
                    with open(chunk_file) as f:
                        content = f.read()
                        combined_data.append(content)

                output = self.path / f'{system}_{level}.mdout'
                with open(output, 'w') as f:
                    f.write('\n'.join(combined_data))

                for chunk_file in chunk_files:
                    chunk_file.unlink()

    def calculate_sasa(self, pre: PathLike, prm: PathLike, trj: PathLike) -> None:
        """Calculate SASA using cpptraj's molsurf command.

        Args:
            pre: Prefix for output SASA file.
            prm: Path to prmtop file.
            trj: Path to CRD trajectory file.
        """
        sasa = self.fh.path / 'sasa.in'
        sasa_in = [
            f'parm {prm}',
            f'trajin {trj}',
            f'molsurf :* out {pre}_surf.dat probe {self.solvent_probe} offset {self.offset}',
            'run',
            'quit',
        ]

        self.fh.write_file(sasa_in, sasa)

        subprocess.run(
            f'{self.cpptraj} -i {sasa}',
            shell=True,
            cwd=str(self.path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sasa.unlink()

    def calculate_energy(
        self,
        pre: PathLike,
        prm: PathLike,
        trj: PathLike,
        pdb: PathLike,
        mdin: PathLike,
        suf: str,
    ) -> None:
        """Run mmpbsa_py_energy to compute system energy.

        Args:
            pre: Prefix for output file.
            prm: Path to prmtop file.
            trj: Path to CRD trajectory file.
            pdb: Path to PDB file.
            mdin: Path to configuration file.
            suf: Suffix for output file ('gb' or 'pb').
        """
        cmd = (
            f'{self.mmpbsa_py_energy} -O -i {mdin} -p {prm} '
            f'-c {pdb} -y {trj} -o {pre}_{suf}.mdout'
        )
        subprocess.run(
            cmd,
            shell=True,
            cwd=str(self.path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_mdins(self) -> tuple[Path, Path]:
        """Write configuration files for mmpbsa_py_energy.

        Returns:
            Tuple of (gb_mdin_path, pb_mdin_path).
        """
        gb = self.fh.path / 'gb_mdin'
        gb_mdin = [
            'GB',
            'igb = 2',
            'extdiel = 78.3',
            'saltcon = 0.10',
            f'surften = {self.gb_surften}',
            'rgbmax = 25.0',
        ]
        self.fh.write_file(gb_mdin, gb)

        pb = self.fh.path / 'pb_mdin'
        pb_mdin = [
            'PB',
            'inp = 2',
            'smoothopt = 1',
            'radiopt = 0',
            'npbopt = 0',
            'solvopt = 1',
            'maxitn = 1000',
            'nfocus = 2',
            'bcopt = 5',
            'eneopt = 2',
            'fscale = 8',
            'epsin = 1.0',
            'epsout = 80.0',
            'istrng = 0.10',
            'dprob = 1.4',
            'iprob = 2.0',
            'accept = 0.001',
            'fillratio = 4.0',
            'space = 0.5',
            'cutnb = 0',
            'sprob = 0.557',
            'cavity_surften = 0.0378',
            'cavity_offset = -0.5692',
        ]
        self.fh.write_file(pb_mdin, pb)

        return gb, pb

    def _verify_combined_outputs(self) -> None:
        """Verify all required output files exist with valid content.

        Raises:
            RuntimeError: If any files are missing or invalid.
        """
        missing_files = []
        empty_files = []
        invalid_files = []

        for system in ['complex', 'receptor', 'ligand']:
            sasa_file = self.path / f'{system}_surf.dat'
            if not sasa_file.exists():
                missing_files.append(str(sasa_file))
            elif sasa_file.stat().st_size == 0:
                empty_files.append(str(sasa_file))
            else:
                with open(sasa_file) as f:
                    data_lines = [
                        line for line in f if line.strip() and not line.strip().startswith('#')
                    ]
                    if len(data_lines) == 0:
                        invalid_files.append(f'{sasa_file} (no data lines)')

            for level in ['gb', 'pb']:
                energy_file = self.path / f'{system}_{level}.mdout'
                if not energy_file.exists():
                    missing_files.append(str(energy_file))
                elif energy_file.stat().st_size == 0:
                    empty_files.append(str(energy_file))
                else:
                    with open(energy_file) as f:
                        content = f.read()
                        if ' BOND' not in content:
                            invalid_files.append(f'{energy_file} (no energy data)')

        errors = []
        if missing_files:
            errors.append(f'Missing files: {missing_files}')
        if empty_files:
            errors.append(f'Empty files: {empty_files}')
        if invalid_files:
            errors.append(f'Invalid files: {invalid_files}')

        if errors:
            raise RuntimeError(f"Output verification failed: {'; '.join(errors)}")

    def get_selections(self) -> list[str]:
        """Infer receptor and ligand selections when none are provided.

        If AMBER-style selections are not provided, they are naively inferred to
        be all chains up to the last (receptor) against the last chain (ligand).
        This involves identifying all OXT termini atoms to demarcate the
        separation of chains.

        Returns:
            List of two cpptraj-formatted selection strings, [receptor, ligand].
        """
        u = mda.Universe(self.top)
        protein = u.select_atoms('protein').residues.resids
        oxts = u.select_atoms('name OXT').residues.resids
        last_oxt = oxts[-2] # if we don't have two chains we should be crashing

        sel1 = [resid for resid in protein if resid <= last_oxt]
        sel2 = [resid for resid in protein if resid > last_oxt]

        return [self.format_for_cpptraj(sel1), self.format_for_cpptraj(sel2)]
    
    @staticmethod
    def format_for_cpptraj(resids) -> str:
        """Format a list of resids into an AMBER-style selection.

        The general format is a colon followed by stretches of continuous amino
        acids represented by a dash (e.g. `:1-10`; `:1-10,25-43`).

        Args:
            resids: Sorted list of residue IDs to format.

        Returns:
            cpptraj-formatted selection string.
        """
        string = ':'
        cur = resids[0] - 1
        start = resids[0]
        end = None
        for resid in resids:
            if resid - cur > 1:
                end = cur
                string += f'{start}-{end},'
                start = resid

            cur = resid

        end = resids[-1]
        string += f'{start}-{end}'

        return string


class OutputAnalyzer:
    """Analyze and summarize MM-PBSA output files.

    Parses energy output files and computes binding free energies
    for both GB and PB solvation methods.

    Attributes:
        path: Working directory path.
        surften: Surface tension parameter.
        offset: SASA offset parameter.
        free_energy: Final [mean, std] after parse_outputs().
        gb: Polars DataFrame of GB results.
        pb: Polars DataFrame of PB results.

    Args:
        path: Path to output directory.
        surface_tension: GB surface tension. Defaults to 0.0072.
        sasa_offset: SASA offset. Defaults to 0.0.
        _tolerance: Tolerance for sanity checks. Defaults to 0.005.
        log: Whether to log results. Defaults to True.
    """

    def __init__(
        self,
        path: PathLike,
        surface_tension: float = 0.0072,
        sasa_offset: float = 0.0,
        _tolerance: float = 0.005,
        log: bool = True,
    ):
        """Initialize the OutputAnalyzer.

        Args:
            path: Output directory path.
            surface_tension: GB surface tension.
            sasa_offset: SASA offset.
            _tolerance: Sanity check tolerance.
            log: Whether to log results.
        """
        self.path = Path(path)
        self.surften = surface_tension
        self.offset = sasa_offset
        self.tolerance = _tolerance
        self.log = log

        self.free_energy = None

        self.systems = ['receptor', 'ligand', 'complex']
        self.levels = ['gb', 'pb']

        self.solvent_contributions = ['EGB', 'ESURF', 'EPB', 'ECAVITY']

    def parse_outputs(self) -> None:
        """Parse all output files and compute binding free energies."""
        self.gb = pl.DataFrame()
        self.pb = pl.DataFrame()

        for system in self.systems:
            E_sasa = self.read_sasa(self.path / f'{system}_surf.dat')
            E_gb = self.read_GB(self.path / f'{system}_gb.mdout', system)
            E_pb = self.read_PB(self.path / f'{system}_pb.mdout', system)

            E_gb = E_gb.drop('ESURF').with_columns(E_sasa)

            self.gb = pl.concat([self.gb, E_gb], how='vertical')
            self.pb = pl.concat([self.pb, E_pb], how='vertical')

        all_cols = list(set(self.gb.columns + self.pb.columns))
        self.contributions = {
            'G gas': [col for col in all_cols if col not in self.solvent_contributions],
            'G solv': [col for col in all_cols if col in self.solvent_contributions],
        }

        self.check_bonded_terms()
        self.generate_summary()
        self.compute_dG()

    def read_sasa(self, _file: PathLike) -> pl.Series:
        """Read cpptraj SASA calculation results.

        Args:
            _file: Path to SASA data file.

        Returns:
            Polars Series of rescaled SASA energies.
        """
        df = pd.read_csv(_file, sep=r'\s+')
        sasa = df.iloc[:, -1].to_numpy(dtype=float) * self.surften + self.offset
        return pl.Series('ESURF', sasa)

    def read_GB(self, _file: PathLike, system: str) -> pl.DataFrame:
        """Read GB mdout file and parse energy terms.

        Args:
            _file: Path to GB energy file.
            system: System label ('complex', 'receptor', or 'ligand').

        Returns:
            Polars DataFrame of parsed energy data.
        """
        gb_terms = [
            'BOND',
            'ANGLE',
            'DIHED',
            'VDWAALS',
            'EEL',
            'EGB',
            '1-4 VDW',
            '1-4 EEL',
            'RESTRAINT',
            'ESURF',
        ]
        data = {gb_term: [] for gb_term in gb_terms}

        lines = open(_file).readlines()
        return self.parse_energy_file(lines, data, system)

    def read_PB(self, _file: PathLike, system: str) -> pl.DataFrame:
        """Read PB mdout file and parse energy terms.

        Args:
            _file: Path to PB energy file.
            system: System label ('complex', 'receptor', or 'ligand').

        Returns:
            Polars DataFrame of parsed energy data.
        """
        pb_terms = [
            'BOND',
            'ANGLE',
            'DIHED',
            'VDWAALS',
            'EEL',
            'EPB',
            '1-4 VDW',
            '1-4 EEL',
            'RESTRAINT',
            'ECAVITY',
            'EDISPER',
        ]
        data = {pb_term: [] for pb_term in pb_terms}

        lines = open(_file).readlines()
        return self.parse_energy_file(lines, data, system)

    def parse_energy_file(
        self, file_contents: list[str], data: dict[str, list], system: str
    ) -> pl.DataFrame:
        """Parse energy file contents.

        Args:
            file_contents: Lines from the energy file.
            data: Dictionary to populate with energy values.
            system: System label.

        Returns:
            Polars DataFrame of energy data.
        """
        for line in file_contents:
            if '=' in line and any(key in line for key in data):
                parsed = self.parse_line(line)
                for key, val in parsed:
                    if key in data:
                        data[key].append(val)

        df = pl.DataFrame(data)
        df = df.with_columns(pl.lit(system).alias('system'))
        return df

    def check_bonded_terms(self) -> None:
        """Verify bonded terms cancel correctly between systems.

        Raises:
            ValueError: If bonded terms don't cancel.
        """
        bonded = ['BOND', 'ANGLE', 'DIHED', '1-4 VDW', '1-4 EEL']

        for theory_level in (self.gb, self.pb):
            a = theory_level.filter(pl.col('system') == 'receptor')
            b = theory_level.filter(pl.col('system') == 'ligand')
            c = theory_level.filter(pl.col('system') == 'complex')

            a = a.select(pl.col([col for col in a.columns if col in bonded])).to_numpy()
            b = b.select(pl.col([col for col in b.columns if col in bonded])).to_numpy()
            c = c.select(pl.col([col for col in c.columns if col in bonded])).to_numpy()

            diffs = np.array(c - b - a)
            if np.where(diffs >= self.tolerance)[0].size > 0:
                raise ValueError('Bonded terms for receptor + ligand != complex!')

        remove = ['RESTRAINT', 'EDISPER']
        self.gb = self.gb.select(
            pl.col([col for col in self.gb.columns if col not in remove])
        )
        self.pb = self.pb.select(
            pl.col([col for col in self.pb.columns if col not in remove])
        )

        self.n_frames = self.gb.height
        self.square_root_N = np.sqrt(self.n_frames)

    def generate_summary(self) -> None:
        """Generate and save summary statistics to JSON file."""
        full_statistics = {sys: {} for sys in self.systems}
        for theory, level in zip([self.gb, self.pb], self.levels, strict=True):
            for system in self.systems:
                sys = theory.filter(pl.col('system') == system).drop('system')

                stats = {}
                for col in sys.columns:
                    mean = sys.select(pl.mean(col)).item()
                    stdev = sys.select(pl.std(col)).item()

                    err = stdev
                    if err is not None:
                        err /= self.square_root_N

                    stats[col] = {'mean': mean, 'std': stdev, 'err': err}

                for energy, contributors in self.contributions.items():
                    pooled_data = (
                        sys.select(
                            pl.col([col for col in sys.columns if col in contributors])
                        )
                        .to_numpy()
                        .flatten()
                    )

                    stats[energy] = {
                        'mean': np.mean(pooled_data),
                        'std': np.std(pooled_data),
                        'err': np.std(pooled_data) / self.square_root_N,
                    }

                total_data = sys.to_numpy().flatten()
                stats['total'] = {
                    'mean': np.mean(total_data),
                    'std': np.std(total_data),
                    'err': np.std(total_data) / self.square_root_N,
                }

                full_statistics[system][level] = stats

        with open('statistics.json', 'w') as fout:
            json.dump(full_statistics, fout, indent=4)

    def compute_dG(self) -> None:
        """Compute binding free energy (ΔG) for GB and PB methods."""
        differences = []
        for theory in [self.gb, self.pb]:
            diff_cols = [col for col in theory.columns if col != 'system']
            diff_arr = (
                theory.filter(pl.col('system') == 'complex').drop('system').to_numpy()
            )
            for system in self.systems[:2]:
                diff_arr -= (
                    theory.filter(pl.col('system') == system).drop('system').to_numpy()
                )

            means = np.mean(diff_arr, axis=0)
            stds = np.std(diff_arr, axis=0)
            errs = stds / self.square_root_N

            gas_solv_phase = []
            for energy, contributors in self.contributions.items():
                indices = [
                    i
                    for i, diff_col in enumerate(diff_cols)
                    if diff_col in contributors
                ]
                contribution = np.sum(diff_arr[:, indices], axis=1)
                gas_solv_phase.append(contribution)

                diff_cols.append(energy)
                means = np.concatenate((means, [np.mean(contribution)]))
                stds = np.concatenate((stds, [np.std(contribution)]))
                errs = np.concatenate(
                    (errs, [np.std(contribution) / self.square_root_N])
                )

            diff_cols.append('∆G Binding')
            total = np.sum(np.vstack(gas_solv_phase), axis=0)

            means = np.concatenate((means, [np.mean(total)]))
            stds = np.concatenate((stds, [np.std(total)]))
            errs = np.concatenate((errs, [np.std(total) / self.square_root_N]))

            data = np.vstack((means, stds, errs))

            differences.append(
                pl.DataFrame({diff_cols[i]: data[:, i] for i in range(len(diff_cols))})
            )

        self.pretty_print(differences)

    def pretty_print(self, dfs: list[pl.DataFrame]) -> None:
        """Print and save formatted energy results.

        Args:
            dfs: List of DataFrames for GB and PB results.
        """
        print_statement = []
        for df, level in zip(dfs, ['Generalized Born ', 'Poisson Boltzmann'], strict=True):
            print_statement += [
                f"{' ':<20}=========================",
                f"{' ':<20}=== {level} ===",
                f"{' ':<20}=========================",
                'Energy Component    Average         Std. Dev.       Std. Err. of Mean',
                '---------------------------------------------------------------------',
            ]
            for col in df.columns:
                mean, std, err = [x.item() for x in df.select(pl.col(col)).to_numpy()]
                report = f'{col:<20}{mean:<16.3f}{std:<16.3f}{err:<16.3f}'
                if abs(mean) <= self.tolerance:
                    continue

                if col in ['G gas', '∆G Binding']:
                    print_statement.append('')

                if col == '∆G Binding':
                    if level == 'Poisson Boltzmann':
                        self.free_energy = [mean, std]

                print_statement.append(report)

        print_statement = '\n'.join(print_statement)
        with open(self.path / 'deltaG.txt', 'w') as fout:
            fout.write(print_statement)

        if not self.log:
            print(print_statement)

    @staticmethod
    def parse_line(line: str) -> zip[tuple[str, float]]:
        """Parse an energy line from mmpbsa_energy output.

        Args:
            line: Line from the energy output file.

        Returns:
            Zip iterator of (term_names, values).
        """
        eq_split = line.split('=')

        if len(eq_split) == 2:
            splits = [eq_spl.strip() for eq_spl in eq_split]
        else:
            splits = [eq_split[0].strip()]

            for i in range(1, len(eq_split) - 1):
                splits += [spl.strip() for spl in eq_split[i].strip().split('  ')]

            splits += [eq_split[-1].strip()]

        keys = splits[::2]
        vals = np.array(splits[1::2], dtype=float)

        return zip(keys, vals, strict=True)


class FileHandler:
    """Handle file preprocessing for MM-PBSA calculations.

    Manages trajectory conversion, topology slicing, and file organization.

    Attributes:
        top: Path to topology file.
        traj: Path to trajectory file.
        path: Working directory.
        selections: Receptor and ligand selections.
        topologies: List of generated topology paths.
        trajectories: List of generated trajectory paths.
        pdbs: List of generated PDB paths.
        trajectory_chunks: Dictionary of chunked trajectories.

    Args:
        top: Path to topology file.
        traj: Path to trajectory file.
        path: Working directory.
        sels: Residue selections.
        first: First frame.
        last: Last frame.
        stride: Frame stride.
        cpptraj_binary: Path to cpptraj executable.
        n_chunks: Number of trajectory chunks.
    """

    def __init__(
        self,
        top: Path,
        traj: Path,
        path: Path,
        sels: list[str],
        first: int,
        last: int,
        stride: int,
        cpptraj_binary: PathLike,
        n_chunks: int = 1,
    ):
        """Initialize the FileHandler.

        Args:
            top: Topology file path.
            traj: Trajectory file path.
            path: Working directory.
            sels: Selections list.
            first: First frame.
            last: Last frame.
            stride: Frame stride.
            cpptraj_binary: cpptraj path.
            n_chunks: Number of chunks.
        """
        self.top = top
        self.traj = traj
        self.path = path
        self.selections = sels
        self.ff = first
        self.lf = last
        self.stride = stride
        self.cpptraj = cpptraj_binary
        self.n_chunks = n_chunks

        self.prepare_topologies()
        self.prepare_trajectories()

        self.trajectory_chunks = {}

        if n_chunks > 1:
            self._count_frames()
            self._split_trajectories()
        else:
            for system, traj in zip(
                ['complex', 'receptor', 'ligand'], self.trajectories, strict=True
            ):
                self.trajectory_chunks[system] = [traj]

    def prepare_topologies(self) -> None:
        """Slice sub-topologies for complex, receptor, and ligand."""
        self.topologies = [
            self.path / 'complex.prmtop',
            self.path / 'receptor.prmtop',
            self.path / 'ligand.prmtop',
        ]

        cpptraj_in = [
            f'parm {self.top}',
            'parmstrip :Na+,Cl-,WAT',
            'parmbox nobox',
            f'parmwrite out {self.topologies[0]}',
            'run',
            'clear all',
            f'parm {self.topologies[0]}',
            f'parmstrip {self.selections[0]}',
            f'parmwrite out {self.topologies[1]}',
            'run',
            'clear all',
            f'parm {self.topologies[0]}',
            f'parmstrip {self.selections[1]}',
            f'parmwrite out {self.topologies[2]}',
            'run',
            'quit',
        ]

        script = self.path / 'cpptraj.in'
        self.write_file('\n'.join(cpptraj_in), script)
        subprocess.call(
            f'{self.cpptraj} -i {script}',
            shell=True,
            cwd=str(self.path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        script.unlink()

    def prepare_trajectories(self) -> None:
        """Convert DCD trajectory to AMBER CRD format."""
        self.trajectories = [path.with_suffix('.crd') for path in self.topologies]
        self.pdbs = [path.with_suffix('.pdb') for path in self.topologies]

        frame_control = f'start {self.ff}'
        if self.lf > -1:
            frame_control += f' stop {self.lf}'
        frame_control += f' offset {self.stride}'

        cpptraj_in = [
            f'parm {self.top}',
            f'trajin {self.traj}',
            'strip :WAT,Na+,Cl*',
            'autoimage',
            'rmsd !(:WAT,Cl*,CIO,Cs+,IB,K*,Li+,MG*,Na+,Rb+,CS,RB,NA,F,CL) mass first',
            f'trajout {self.trajectories[0]} crd {frame_control}',
            f'trajout {self.pdbs[0]} pdb onlyframes 1',
            'run',
            'clear all',
            f'parm {self.topologies[0]}',
            f'trajin {self.trajectories[0]}',
            f'strip {self.selections[0]}',
            f'trajout {self.trajectories[1]} crd',
            f'trajout {self.pdbs[1]} pdb onlyframes 1',
            'run',
            'clear all',
            f'parm {self.topologies[0]}',
            f'trajin {self.trajectories[0]}',
            f'strip {self.selections[1]}',
            f'trajout {self.trajectories[2]} crd',
            f'trajout {self.pdbs[2]} pdb onlyframes 1',
            'run',
            'quit',
        ]

        name = self.path / 'mdcrd.in'
        self.write_file('\n'.join(cpptraj_in), name)
        subprocess.call(
            f'{self.cpptraj} -i {name}',
            shell=True,
            cwd=str(self.path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        name.unlink()

    def _count_frames(self) -> None:
        """Count total frames in the processed trajectory."""
        count_script = self.path / 'count_frames.in'

        script_content = [
            f'parm {self.topologies[0]}',
            f'trajin {self.trajectories[0]}',
            f'trajinfo {self.trajectories[0]} name myinfo',
            'run',
            'quit',
        ]
        self.write_file('\n'.join(script_content), count_script)

        result = subprocess.run(
            f'{self.cpptraj} -i {count_script}',
            shell=True,
            cwd=str(self.path),
            capture_output=True,
            text=True,
        )

        for line in result.stdout.split('\n'):
            if 'frames' in line.lower() and 'total' in line.lower():
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        self.total_frames = int(part)
                        break

        if not hasattr(self, 'total_frames'):
            self.total_frames = self._estimate_frames()

        count_script.unlink(missing_ok=True)

    def _estimate_frames(self) -> int:
        """Estimate frame count by running cpptraj analysis."""
        script = self.path / 'estimate_frames.in'
        script_content = [
            f'parm {self.topologies[0]}',
            f'trajin {self.trajectories[0]}',
            'run',
            'quit',
        ]
        self.write_file('\n'.join(script_content), script)

        result = subprocess.run(
            f'{self.cpptraj} -i {script}',
            shell=True,
            cwd=str(self.path),
            capture_output=True,
            text=True,
        )

        script.unlink(missing_ok=True)

        for line in result.stdout.split('\n'):
            if 'frames' in line.lower():
                for word in line.split():
                    if word.isdigit():
                        return int(word)

        return 100

    def _split_trajectories(self) -> None:
        """Split trajectories into chunks for parallel processing."""
        actual_chunks = min(self.n_chunks, self.total_frames)

        frames_per_chunk = max(1, self.total_frames // actual_chunks)
        self.trajectory_chunks = {
            system: [] for system in ['complex', 'receptor', 'ligand']
        }

        for (top, traj, system) in zip(self.topologies,
                                       self.trajectories,
                                       ['complex', 'receptor', 'ligand'],
                                       strict=True):
            for chunk_idx in range(actual_chunks):
                start_frame = chunk_idx * frames_per_chunk + 1

                if chunk_idx == actual_chunks - 1:
                    end_frame = self.total_frames
                else:
                    end_frame = (chunk_idx + 1) * frames_per_chunk

                chunk_traj = self.path / f'{system}_chunk{chunk_idx}.crd'

                split_script = self.path / f'split_{system}_{chunk_idx}.in'
                script_content = [
                    f'parm {top}',
                    f'trajin {traj} {start_frame} {end_frame}',
                    f'trajout {chunk_traj} crd',
                    'run',
                    'quit',
                ]
                self.write_file('\n'.join(script_content), split_script)

                subprocess.run(
                    f'{self.cpptraj} -i {split_script}',
                    shell=True,
                    cwd=str(self.path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                split_script.unlink()
                self.trajectory_chunks[system].append(chunk_traj)

    @property
    def files(self) -> zip[tuple[Path, Path, Path, Path]]:
        """Get file paths for each system.

        Returns:
            Zip of (prefix, topology, trajectory, pdb) for each system.
        """
        _order = [self.path / prefix for prefix in ['complex', 'receptor', 'ligand']]
        return zip(_order, self.topologies, self.trajectories, self.pdbs, strict=True)

    @property
    def files_chunked(self) -> list[tuple]:
        """Get file paths with chunked trajectories.

        Returns:
            List of (prefix, topology, trajectory_chunks, pdb) tuples.
        """
        result = []
        for system, top, pdb in zip(
            ['complex', 'receptor', 'ligand'], self.topologies, self.pdbs, strict=True
        ):
            prefix = self.path / system
            traj_chunks = self.trajectory_chunks[system]
            result.append((prefix, top, traj_chunks, pdb))

        return result

    @staticmethod
    def write_file(lines: list[str] | str, filepath: PathLike) -> None:
        """Write lines to a file.

        Args:
            lines: Input lines (list or string).
            filepath: Output file path.
        """
        content = '\n'.join(lines) if isinstance(lines, list) else lines
        with open(str(filepath), 'w') as f:
            f.write(content)
