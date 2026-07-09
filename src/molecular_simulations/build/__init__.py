import contextlib
from pathlib import Path

from .build_amber import ExplicitSolvent, ImplicitSolvent
from .build_evb import EVBBuilder, EVBBuildError

with contextlib.suppress(ImportError):
    from .build_ligand import ComplexBuilder, LigandBuilder

PathLike = Path | str


def convert_cif_with_biopython(cif: PathLike) -> PathLike:
    """Convert a CIF file to a PDB file using biopython.

    Args:
        cif: Path to the input CIF file.

    Returns:
        Path to the written PDB file (same stem, '.pdb' suffix).
    """
    from Bio.PDB import PDBIO, MMCIFParser  # ty: ignore[unresolved-import]

    if not isinstance(cif, Path):
        cif = Path(cif)
    pdb = cif.with_suffix('.pdb')

    parser = MMCIFParser()
    structure = parser.get_structure('protein', str(cif))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb))

    return pdb


def convert_cif_with_gemmi(cif: PathLike) -> PathLike:
    """Convert a CIF file to a PDB file using gemmi.

    Args:
        cif: Path to the input CIF file.

    Returns:
        Path to the written PDB file (same stem, '.pdb' suffix).
    """
    import gemmi  # ty: ignore[unresolved-import]

    if not isinstance(cif, Path):
        cif = Path(cif)
    structure = gemmi.read_structure(str(cif))
    structure.write_pdb(str(cif.with_suffix('.pdb')))
    return cif.with_suffix('.pdb')


def add_chains(pdb: PathLike, first_res: int = 1, last_res: int = -1) -> PathLike:
    """Add chain IDs to a model.

    Residues from ``first_res`` to ``last_res`` are assigned chain 'A';
    any remaining residues are assigned chain 'B'.

    Args:
        pdb: Path to the input PDB file.
        first_res: First residue (1-indexed) of chain A. Defaults to 1.
        last_res: Last residue of chain A. If -1, uses the final residue
            (no chain B is created). Defaults to -1.

    Returns:
        Path to the written PDB file (input stem plus '_withchains.pdb').
    """
    import MDAnalysis as mda

    u = mda.Universe(pdb)
    u.add_TopologyAttr('chainID')

    split = last_res != -1
    if last_res == -1:
        assert u.residues is not None
        last_res = u.residues.n_residues

    chain_A = u.select_atoms(f'resid {first_res} to {last_res}')
    chain_A.atoms.chainIDs = 'A'

    if split:
        assert u.residues is not None
        final_res = u.residues.n_residues

        chain_B = u.select_atoms(f'resid {last_res} to {final_res}')
        chain_B.atoms.chainIDs = 'B'

    output_path = Path(pdb).parent / (Path(pdb).stem + '_withchains.pdb')
    with mda.Writer(output_path) as W:
        W.write(u.atoms)

    return output_path
