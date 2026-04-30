import contextlib
from pathlib import Path

from .build_amber import ExplicitSolvent, ImplicitSolvent
from .build_interface import InterfaceBuilder

with contextlib.suppress(ImportError):
    from .build_ligand import ComplexBuilder, LigandBuilder, PLINDERBuilder

PathLike = Path | str

def convert_cif_with_biopython(cif: PathLike) -> PathLike:
    """
    Helper function to convert a cif file to a pdb file using biopython.
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
    import gemmi  # ty: ignore[unresolved-import]

    if not isinstance(cif, Path):
        cif = Path(cif)
    structure = gemmi.read_structure(str(cif))
    structure.write_pdb(str(cif.with_suffix('.pdb')))
    return cif.with_suffix('.pdb')


def add_chains(pdb: PathLike, first_res: int = 1, last_res: int = -1) -> PathLike:
    """
    Helper function to add chain IDs to a model.
    """
    import MDAnalysis as mda

    u = mda.Universe(pdb)
    u.add_TopologyAttr('chainID')

    if last_res == -1:
        assert u.residues is not None
        last_res = u.residues.n_residues

    chain_A = u.select_atoms(f'resid {first_res} to {last_res}')
    chain_A.atoms.chainIDs = 'A'

    if last_res != -1:
        assert u.residues is not None
        final_res = u.residues.n_residues

        chain_B = u.select_atoms(f'resid {last_res} to {final_res}')
        chain_B.atoms.chainIDs = 'B'

    output_path = Path(pdb).parent / (Path(pdb).stem + '_withchains.pdb')
    with mda.Writer(output_path) as W:
        W.write(u.atoms)

    return output_path
