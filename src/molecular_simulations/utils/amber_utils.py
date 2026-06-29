import MDAnalysis as mda
import string

def assign_chainids(
    u: mda.Universe, terminus_selection: str = 'name OXT'
) -> mda.Universe:
    """Assign sequential chain IDs to residues based on chain termini.

    Adds a chainIDs topology attribute if missing, then walks the residues in
    order assigning a chain label. The chain index is incremented after each
    residue that contains a terminus atom (by default the C-terminal OXT), so
    every span ending in a terminus receives a distinct label. Labels run A-Z,
    then AA, AB, ... for systems with more than 26 chains.

    Args:
        u: MDAnalysis Universe to annotate (modified in place).
        terminus_selection: MDAnalysis selection string identifying chain
            terminus atoms used to delimit chains. Defaults to 'name OXT'.

    Returns:
        The same Universe with chainIDs assigned to every atom.
    """
    if not hasattr(u.atoms, 'chainIDs'):
        u.add_TopologyAttr('chainIDs')

    termini_atoms = u.select_atoms(terminus_selection)
    termini_resindices = set(termini_atoms.resindices)

    def get_chain_label(index):
        if index < 26:
            return string.ascii_uppercase[index]
        first = string.ascii_uppercase[(index // 26) - 1]
        second = string.ascii_uppercase[index % 26]
        return first + second

    chain_index = 0
    assert u.residues is not None
    for residue in u.residues:
        residue.atoms.chainIDs = get_chain_label(chain_index)

        if residue.resindex in termini_resindices:
            chain_index += 1

    return u
