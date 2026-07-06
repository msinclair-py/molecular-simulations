"""
Reference energies for constant pH simulations.

These reference energies are calibrated to give the correct pKa values
for titratable residues. The convention is:

  ref_energies[residue_type] = [E_deprotonated, E_protonated, ...]

Where E_deprotonated = 0 and E_protonated combines the implicit-solvent
(GB) free-energy difference between states with a pKa adjustment term,
i.e. E_protonated = (E_prot - E_deprot)_GB + kT * ln(10) * pKa.

This ensures that at pH = pKa, both protonation states have equal probability.

Reference energies are specific to the implicit-solvent energy function. The
values below were regenerated with ``scripts/calibrate_reference_energies.py``
after the CustomGBForce fix (GB now contributes to every Monte Carlo move),
using capped Ace-X-Nme model compounds built by ``ConstantPHSolvent``
(ff19SB / OPC / mbondi3, GBn2 implicit). Each was calibrated iteratively so an
independent titration reproduces the experimental pKa; the validation pKa and
Hill coefficient from that titration are noted alongside each value. Regenerate
them (and re-validate) whenever the implicit-solvent model or force field
changes.
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
            # Regenerated 2026-07-06 via scripts/calibrate_reference_energies.py
            # for the CustomGBForce-corrected implicit energy (ff19SB/GBn2,
            # mbondi3). Comments give: target pKa; validation pKa and Hill n from
            # an independent titration of the model compound.
            ref_energies = {
                'CYS': [0.0, 324.03],  # pKa 8.3; validation 8.36, n 1.11 (CYX<->CYS)
                'ASP': [0.0, 91.50],  # pKa 3.9; validation 3.85, n 0.95
                'GLU': [0.0, 100.81],  # pKa 4.3; validation 4.22, n 1.03
                'LYS': [0.0, 24.11],  # pKa 10.5; validation 10.55, n 1.04
                'HIS': [0.0, 75.64],  # pKa 6.5; validation 6.44, n 0.98 (HID<->HIP)
            }
        case _:
            raise ValueError(f'Forcefield {ff} not yet computed!')

    return ref_energies
