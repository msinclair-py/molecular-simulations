"""
Reference energies for constant pH simulations.

These reference energies are calibrated to give the correct pKa values
for titratable residues. The convention is:

  ref_energies[residue_type] = [E_deprotonated, E_protonated, ...]

Where E_deprotonated = 0 and E_protonated combines the implicit-solvent
(GB) free-energy difference between states with a pKa adjustment term,
i.e. E_protonated = (E_prot - E_deprot)_GB + kT * ln(10) * pKa.

This ensures that at pH = pKa, both protonation states have equal probability.
"""


def get_ref_energies(ff: str = 'amber19'):
    """Get reference energies for constant pH simulations.

    Args:
        ff: Force field name. Currently only 'amber19' is supported.
            Defaults to 'amber19'.

    Returns:
        Maps residue type names to lists of reference energies (kJ/mol).
        Index 0 is the deprotonated state, index 1+ are protonated states.

    Raises:
        ValueError: If reference energies for the given force field have
            not been computed.
    """
    match ff.lower():
        case 'amber19':
            # Reference energies based on experimental pKa values at 300K
            # pKa adjustment term = kT * ln(10) * pKa = 2.494 * 2.303 * pKa
            # ref = (E_prot - E_deprot)_GB + pKa adjustment
            # GB term accounts for implicit solvent contribution
            # This gives the correct equilibrium at each residue's pKa
            ref_energies = {
                'CYS': [0.0, -275.17],  # pKa = 8.3
                'ASP': [0.0, -107.17],  # pKa = 3.9
                'GLU': [0.0, -96.32],  # pKa = 4.3
                'LYS': [0.0, -26.72],  # pKa = 10.5
                'HIS': [0.0, -60.43],  # pKa = 6.5 (2-state HID/HIP)
            }
        case _:
            raise ValueError(f'Forcefield {ff} not yet computed!')

    return ref_energies
