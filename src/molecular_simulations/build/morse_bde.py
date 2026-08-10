"""Bond-dissociation-energy → Morse well-depth (``D_e``) helper for EVB.

The Morse reactive bond in the EVB diabats (see
:func:`molecular_simulations.simulate.evb_mapping.build_mapping_system`) needs a
well depth ``D_e``. Unlike the width ``alpha`` and equilibrium length ``r0`` --
which are derived from the force field's harmonic bond -- ``D_e`` is the bond
*dissociation* energy and has no force-field counterpart, so it comes from QM or
an ML predictor.

This module wraps **ALFABET** (a Machine Learning derived, Fast, Accurate Bond
dissociation Enthalpy Tool; NREL, https://github.com/NatLabRockies/alfabet) to
predict a bond's BDE from a SMILES string and convert it to ``D_e`` in the kJ/mol
OpenMM needs. ALFABET's public web server (``bde.ml.nrel.gov``) is defunct, but
the model runs offline from the ``alfabet`` pip package -- this helper is that
offline path, so ``D_e`` stays reproducible for the WT and future mutants.

Typical use::

    from molecular_simulations.build.morse_bde import predict_reactive_de

    D_e = predict_reactive_de(
        'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', bond_type='C-H'
    )  # kJ/mol
    # feed to the sampler:  run_wt_barrier.py --D-e <D_e>  (alpha/r0 auto-derive)
"""

from __future__ import annotations

#: kcal/mol -> kJ/mol (ALFABET reports BDE in kcal/mol; OpenMM wants kJ/mol).
KCAL_TO_KJ = 4.184


def bde_to_de(bde: float, units: str = 'kcal/mol') -> float:
    """Convert a bond dissociation energy to a Morse well depth ``D_e`` in kJ/mol.

    The Morse well depth is taken as the bond dissociation energy (the standard
    EVB choice; the small D_e vs D_0/BDE distinctions are within the method's
    accuracy). ALFABET reports kcal/mol; pass ``units='kJ/mol'`` for a value
    already in kJ/mol.

    Args:
        bde: Bond dissociation energy.
        units: ``'kcal/mol'`` (default, ALFABET's unit) or ``'kJ/mol'``.

    Returns:
        ``D_e`` in kJ/mol.

    Raises:
        ValueError: for an unknown ``units`` or a non-positive BDE.
    """
    if bde <= 0:
        raise ValueError(f'BDE must be positive, got {bde}')
    if units == 'kJ/mol':
        return float(bde)
    if units == 'kcal/mol':
        return float(bde) * KCAL_TO_KJ
    raise ValueError(f"units must be 'kcal/mol' or 'kJ/mol', got {units!r}")


def predict_bdes(smiles: str):
    """Predict every bond's BDE for a molecule via ALFABET (lazy import).

    Args:
        smiles: A single molecule SMILES.

    Returns:
        The ALFABET prediction as a pandas DataFrame -- one row per broken bond,
        with at least ``bond_index``, ``bond_type`` and ``bde`` (kcal/mol).

    Raises:
        ImportError: if the ``alfabet`` package is not installed
            (``pip install alfabet``; needs TensorFlow).
    """
    try:
        from alfabet import model
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            'ALFABET is required for BDE prediction: `pip install alfabet` '
            '(needs TensorFlow). The old web server bde.ml.nrel.gov is defunct; '
            'the pip package runs the model offline.'
        ) from exc
    return model.predict([smiles])


def select_bde(
    predictions, bond_index: int | None = None, bond_type: str | None = None
) -> float:
    """Pick one bond's BDE (kcal/mol) from an ALFABET prediction table.

    Exactly one selector must be given. ``bond_index`` selects that row directly;
    ``bond_type`` (e.g. ``'C-H'``) selects by bond type and requires the type to be
    unique in the molecule (raises otherwise, so the choice is never silent).

    Args:
        predictions: DataFrame from :func:`predict_bdes`.
        bond_index: Row's ``bond_index`` to select.
        bond_type: Bond type string to select (must be unique).

    Returns:
        The selected bond's BDE in kcal/mol.

    Raises:
        ValueError: if not exactly one selector is given, the selector matches no
            bond, or ``bond_type`` is ambiguous.
    """
    if (bond_index is None) == (bond_type is None):
        raise ValueError('pass exactly one of bond_index or bond_type')

    if bond_index is not None:
        rows = predictions[predictions['bond_index'] == bond_index]
        if len(rows) == 0:
            raise ValueError(f'no bond with bond_index={bond_index}')
        return float(rows['bde'].iloc[0])

    rows = predictions[predictions['bond_type'] == bond_type]
    if len(rows) == 0:
        types = sorted(set(predictions['bond_type']))
        raise ValueError(f'no {bond_type!r} bond; available types: {types}')
    if len(rows) > 1:
        idx = list(rows['bond_index'])
        raise ValueError(
            f'{bond_type!r} is ambiguous (bond_index {idx}); pass bond_index instead'
        )
    return float(rows['bde'].iloc[0])


def predict_reactive_de(
    smiles: str, bond_index: int | None = None, bond_type: str | None = None
) -> float:
    """Predict the reactive bond's Morse ``D_e`` (kJ/mol) end-to-end via ALFABET.

    Convenience over :func:`predict_bdes` + :func:`select_bde` + :func:`bde_to_de`.
    Feed the result to ``build_mapping_system(D_e=...)`` /
    ``run_wt_barrier.py --D-e ...``; ``alpha``/``r0`` then auto-derive from the
    force field.

    Args:
        smiles: The reactive molecule's SMILES (the H-donor for a hydride/proton
            transfer -- e.g. the substrate).
        bond_index: ALFABET bond index of the reactive bond, or
        bond_type: reactive bond type (e.g. ``'C-H'``, if unique).

    Returns:
        ``D_e`` in kJ/mol.
    """
    preds = predict_bdes(smiles)
    bde = select_bde(preds, bond_index=bond_index, bond_type=bond_type)
    return bde_to_de(bde, units='kcal/mol')
