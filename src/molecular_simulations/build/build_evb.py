"""Two-state EVB system builder.

Empirical valence bond (EVB) describes a bond-making/breaking reaction as a
mixture of two *diabatic* valence-bond states, each a complete classical force
field for one bonding topology:

  * State 1 (reactant): the transferring atom is bonded to the donor.
  * State 2 (product):  the same atom is bonded to the acceptor.

The two states share the *same atoms and the same coordinates* -- they differ
only in bonded connectivity around the reactive atom and in the partial charges
(the reactive region is oxidised/reduced/(de)protonated differently in each
state). Free-energy analysis then evaluates *both* diabatic Hamiltonians on the
sampled configurations, so the two topologies must be built on a common
coordinate frame with a common atom order.

:class:`EVBBuilder` takes user-supplied coordinates *and charges* for the
reactive species in each state (as mol2 files -- the natural AMBER format
carrying coordinates, partial charges and atom types together) plus an optional
protein/environment PDB, and emits two AMBER topologies -- ``reactant.prmtop``
and ``product.prmtop`` -- that share one set of coordinates (``system.inpcrd``),
along with an ``evb_meta.json`` describing the donor/acceptor/reactive atom
indices and the box. GAFF2 bonded/LJ parameters are filled in with parmchk2;
the charges are taken verbatim from the input mol2s (no AM1-BCC step), so the
whole build is free of any quantum single-point calculation.

The reactive region is built as a *single* tleap unit so the transferring atom
keeps a fixed residue identity across the two states (only its bond and charge
change). If the reactants/products are supplied as separate species (e.g. a
hydride donor D-H and an acceptor A), they are merged into that one unit.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

PathLike = str | Path


class EVBBuildError(Exception):
    """Raised when an EVB system cannot be assembled (tleap/antechamber, or a
    reactant/product mismatch)."""


@dataclass
class _Mol2:
    """Minimal parsed mol2: atom records and 1-based bond pairs.

    Only the fields EVBBuilder needs (name, element/type, coordinates, charge)
    are retained; the mol2 is otherwise treated opaquely.
    """

    names: list[str]
    types: list[str]
    coords: list[tuple[float, float, float]]
    charges: list[float]
    bonds: list[tuple[int, int, str]]  # (atom1, atom2, order), 1-based

    @property
    def n_atoms(self) -> int:
        return len(self.names)


def _parse_mol2(path: PathLike) -> _Mol2:
    """Parse the ATOM and BOND records of a Tripos mol2 file.

    Args:
        path: Path to the mol2 file.

    Returns:
        A :class:`_Mol2` with per-atom name/type/coord/charge and 1-based bonds.
    """
    names: list[str] = []
    types: list[str] = []
    coords: list[tuple[float, float, float]] = []
    charges: list[float] = []
    bonds: list[tuple[int, int, str]] = []

    section = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line.startswith('@<TRIPOS>'):
            section = line[len('@<TRIPOS>') :]
            continue
        if not line:
            continue
        if section == 'ATOM':
            f = line.split()
            # id name x y z type subst_id subst_name charge
            names.append(f[1])
            coords.append((float(f[2]), float(f[3]), float(f[4])))
            types.append(f[5])
            charges.append(float(f[8]) if len(f) > 8 else 0.0)
        elif section == 'BOND':
            f = line.split()
            bonds.append((int(f[1]), int(f[2]), f[3]))

    if not names:
        raise EVBBuildError(f'no ATOM records parsed from {path}')
    return _Mol2(names, types, coords, charges, bonds)


def _write_mol2(mol: _Mol2, path: PathLike, resname: str) -> None:
    """Write a :class:`_Mol2` as a single-residue Tripos mol2.

    Args:
        mol: The molecule to write.
        path: Destination mol2 path.
        resname: Residue (substructure) name for every atom.
    """
    lines = [
        '@<TRIPOS>MOLECULE',
        resname,
        f'{mol.n_atoms} {len(mol.bonds)} 1 0 0',
        'SMALL',
        'USER_CHARGES',
        '',
        '@<TRIPOS>ATOM',
    ]
    for i, (name, typ, (x, y, z), q) in enumerate(
        zip(mol.names, mol.types, mol.coords, mol.charges, strict=True), start=1
    ):
        lines.append(
            f'{i:>7} {name:<8} {x:>10.4f} {y:>10.4f} {z:>10.4f} '
            f'{typ:<6} 1 {resname} {q:>10.6f}'
        )
    lines.append('@<TRIPOS>BOND')
    for i, (a, b, order) in enumerate(mol.bonds, start=1):
        lines.append(f'{i:>6} {a:>6} {b:>6} {order}')
    lines.append('')
    Path(path).write_text('\n'.join(lines))


def _merge_mol2(paths: list[Path], out: Path, resname: str) -> _Mol2:
    """Merge several mol2 species into one single-residue mol2 unit.

    Atom records are concatenated in order and bond indices are offset so the
    combined unit preserves every intra-species bond. The result is the reactive
    region as a single tleap unit (needed so the transferring atom keeps a fixed
    residue across the two diabatic states).

    Args:
        paths: mol2 files to merge, in order.
        out: Destination mol2 path.
        resname: Residue name for the merged unit.

    Returns:
        The merged :class:`_Mol2`.
    """
    names: list[str] = []
    types: list[str] = []
    coords: list[tuple[float, float, float]] = []
    charges: list[float] = []
    bonds: list[tuple[int, int, str]] = []

    offset = 0
    for p in paths:
        m = _parse_mol2(p)
        names.extend(m.names)
        types.extend(m.types)
        coords.extend(m.coords)
        charges.extend(m.charges)
        bonds.extend((a + offset, b + offset, o) for a, b, o in m.bonds)
        offset += m.n_atoms

    merged = _Mol2(names, types, coords, charges, bonds)
    _write_mol2(merged, out, resname)
    return merged


class EVBBuilder:
    """Build the two diabatic AMBER topologies for an EVB reaction.

    Produces ``reactant.prmtop`` and ``product.prmtop`` sharing one coordinate
    set (``system.inpcrd``) plus ``evb_meta.json``. Charges come from the input
    mol2s verbatim; GAFF2 fills in bonded/LJ parameters. The reactive region is
    assembled as a single tleap unit (species are merged if supplied as a list).

    Args:
        out_path: Output directory (created if absent).
        reactant: mol2 path, or list of mol2 paths, for the reactive region in
            the reactant state (transferring atom bonded to the donor). Carries
            coordinates and the reactant partial charges.
        product: mol2 path, or list, for the product state (same atoms, same
            order; transferring atom bonded to the acceptor; product charges).
        protein: Optional protein/environment PDB (ff14SB). If None, only the
            reactive region is solvated (useful for gas-phase/validation runs).
        donor: Atom name of the donor heavy atom (in the reactive-region mol2).
        acceptor: Atom name of the acceptor heavy atom.
        reactive: Atom name of the transferring atom (bonded to donor in the
            reactant, acceptor in the product).
        resname: Residue name for the reactive-region unit. Defaults to 'LIG'.
        padding: Solvent box padding in Angstroms. Defaults to 12.0.
        neutralize: Whether to add counter-ions to neutralize. Defaults to True.
        water: Water model / box ('opc' or 'tip3p'). Defaults to 'opc'.
        amberhome: Path to AMBER install. If None, uses $AMBERHOME.

    Example:
        >>> EVBBuilder(
        ...     out_path='evb_system',
        ...     reactant=['DH.mol2', 'A.mol2'],
        ...     product=['D.mol2', 'AH.mol2'],
        ...     protein='enzyme.pdb',
        ...     donor='C4',
        ...     acceptor='C1',
        ...     reactive='H4',
        ... ).build()
    """

    _WATER_BOX: ClassVar[dict[str, tuple[str, str]]] = {
        'opc': ('leaprc.water.opc', 'OPCBOX'),
        'tip3p': ('leaprc.water.tip3p', 'TIP3PBOX'),
    }

    def __init__(
        self,
        out_path: PathLike,
        reactant: PathLike | list[PathLike],
        product: PathLike | list[PathLike],
        protein: PathLike | None = None,
        donor: str | None = None,
        acceptor: str | None = None,
        reactive: str | None = None,
        resname: str = 'LIG',
        padding: float = 12.0,
        neutralize: bool = True,
        water: str = 'opc',
        amberhome: str | None = None,
    ):
        """Initialize the EVBBuilder."""
        self.out = Path(out_path).resolve()
        self.build_dir = self.out / 'build'
        self.reactant_inputs = [Path(p).resolve() for p in _as_list(reactant)]
        self.product_inputs = [Path(p).resolve() for p in _as_list(product)]
        self.protein = Path(protein).resolve() if protein is not None else None
        self.donor = donor
        self.acceptor = acceptor
        self.reactive = reactive
        self.resname = resname
        self.padding = padding
        self.neutralize = neutralize

        if water not in self._WATER_BOX:
            raise EVBBuildError(f'unknown water model {water!r}')
        self.water_leaprc, self.water_box = self._WATER_BOX[water]

        if amberhome is None:
            amberhome = os.environ.get('AMBERHOME')
        if not amberhome:
            raise EVBBuildError('AMBERHOME is not set; cannot run antechamber/tleap.')
        self.amberhome = Path(amberhome)
        self.antechamber = str(self.amberhome / 'bin' / 'antechamber')
        self.parmchk2 = str(self.amberhome / 'bin' / 'parmchk2')
        self.tleap = str(self.amberhome / 'bin' / 'tleap')
        self.pdb4amber = str(self.amberhome / 'bin' / 'pdb4amber')

    # -- orchestration ------------------------------------------------------

    def build(self) -> dict:
        """Build both diabatic topologies on a shared coordinate frame.

        Returns:
            The metadata dict (also written to ``out/evb_meta.json``).

        Raises:
            EVBBuildError: On atom mismatch between states or a tleap failure.
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)

        if self.protein is not None:
            self.protein = self._prep_protein()

        reactant_mol = self._prepare_state(self.reactant_inputs, 'reactant')
        product_mol = self._prepare_state(self.product_inputs, 'product')
        self._check_states_consistent(reactant_mol, product_mol)

        # Reactant defines the shared coordinates + box; the product topology is
        # then rebuilt on exactly those coordinates with the product parameters.
        box = self._build_reactant_system()
        self._build_product_system(box)
        self._verify_shared_coordinates()

        meta = self._metadata(box)
        (self.out / 'evb_meta.json').write_text(json.dumps(meta, indent=2))
        return meta

    # -- per-state parameterization ----------------------------------------

    def _prepare_state(self, inputs: list[Path], state: str) -> _Mol2:
        """Merge (if needed), GAFF2-type and parameterize one diabatic state.

        Charges from the input mol2(s) are preserved verbatim (antechamber is
        run without a charge method); parmchk2 supplies missing bonded/LJ terms.
        Writes ``{state}.mol2``/``.frcmod``/``.lib`` (unit ``resname``).

        Args:
            inputs: mol2 file(s) for this state's reactive region.
            state: 'reactant' or 'product' (output file stem).

        Returns:
            The parsed reactive-region :class:`_Mol2` (pre-typing atom set).
        """
        merged = self.build_dir / f'{state}_input.mol2'
        mol = _merge_mol2(inputs, merged, self.resname)

        gaff = self.build_dir / f'{state}.mol2'
        frcmod = self.build_dir / f'{state}.frcmod'
        lib = self.build_dir / f'{state}.lib'

        # -c omitted -> antechamber keeps the input mol2 charges (no AM1-BCC).
        self._run(
            [
                self.antechamber,
                '-i',
                merged.name,
                '-fi',
                'mol2',
                '-o',
                gaff.name,
                '-fo',
                'mol2',
                '-at',
                'gaff2',
                '-pf',
                'y',
                '-rn',
                self.resname,
                '-dr',
                'no',
            ]
        )
        if not gaff.exists():
            raise EVBBuildError(
                f'antechamber failed to type the {state} state (see build dir)'
            )
        self._run([self.parmchk2, '-i', gaff.name, '-f', 'mol2', '-o', frcmod.name])

        leap = (
            'source leaprc.gaff2\n'
            f'{self.resname} = loadmol2 {gaff.name}\n'
            f'loadamberparams {frcmod.name}\n'
            f'saveoff {self.resname} {lib.name}\n'
            'quit\n'
        )
        self._run_tleap(leap, f'{state}_param.leap')
        if not lib.exists():
            raise EVBBuildError(f'tleap failed to write the {state} .lib')
        return mol

    # -- system assembly ----------------------------------------------------

    def _prep_protein(self) -> Path:
        """Normalize the protein PDB for tleap with pdb4amber.

        tleap's ff14SB templates need AMBER atom/residue naming; raw PDBs (e.g.
        ACE/NME caps with HH3*/CH3 names) otherwise fail with "does not have a
        type". pdb4amber renames atoms and strips hydrogens (ff14SB rebuilds
        them).

        Returns:
            Path to the prepared PDB in the build directory.
        """
        prepped = self.build_dir / 'protein_prepped.pdb'
        self._run([self.pdb4amber, '-i', str(self.protein), '-o', str(prepped), '-y'])
        if not prepped.exists():
            raise EVBBuildError('pdb4amber failed to prepare the protein PDB')
        return prepped

    def _protein_ff(self) -> list[str]:
        ffs = ['leaprc.protein.ff14SB', 'leaprc.gaff2', self.water_leaprc]
        return ffs

    def _build_reactant_system(self) -> list[float]:
        """Assemble protein + reactant reactive-region, solvate, save topology.

        Returns:
            The periodic box lengths [x, y, z] in Angstroms.
        """
        ffs = '\n'.join(f'source {ff}' for ff in self._protein_ff())
        lines = [ffs]
        units = []
        if self.protein is not None:
            lines.append(f'PROT = loadpdb {self.protein}')
            units.append('PROT')
        lines += [
            f'loadamberparams {self.build_dir}/reactant.frcmod',
            f'loadoff {self.build_dir}/reactant.lib',
            f'{self.resname} = loadmol2 {self.build_dir}/reactant.mol2',
        ]
        units.append(self.resname)

        system = 'SYS'
        lines.append(f'{system} = combine {{{" ".join(units)}}}')
        lines += [
            f'solvateBox {system} {self.water_box} {self.padding:.1f}',
        ]
        if self.neutralize:
            lines += [
                f'addIons {system} Na+ 0',
                f'addIons {system} Cl- 0',
            ]
        lines += [
            f'savepdb {system} {self.out}/reactant_solvated.pdb',
            f'saveAmberParm {system} {self.out}/reactant.prmtop {self.out}/system.inpcrd',
            'quit',
        ]
        self._run_tleap('\n'.join(lines), 'reactant_assemble.leap')
        if not (self.out / 'reactant.prmtop').exists():
            raise EVBBuildError('tleap failed to build reactant.prmtop')
        return self._read_box(self.out / 'system.inpcrd')

    def _build_product_system(self, box: list[float]) -> None:
        """Rebuild the topology on the reactant coordinates with product params.

        Loads the solvated reactant PDB (which carries the shared coordinates)
        but assigns the product reactive-region template, so the transferring
        atom is bonded to the acceptor while every coordinate is unchanged.

        Args:
            box: Periodic box lengths from the reactant build.
        """
        ffs = '\n'.join(f'source {ff}' for ff in self._protein_ff())
        bx, by, bz = box
        lines = [
            ffs,
            f'loadamberparams {self.build_dir}/product.frcmod',
            f'loadoff {self.build_dir}/product.lib',
            f'SYS = loadpdb {self.out}/reactant_solvated.pdb',
            f'set SYS box {{{bx:.3f} {by:.3f} {bz:.3f}}}',
            f'saveAmberParm SYS {self.out}/product.prmtop {self.out}/product.inpcrd',
            'quit',
        ]
        self._run_tleap('\n'.join(lines), 'product_assemble.leap')
        if not (self.out / 'product.prmtop').exists():
            raise EVBBuildError('tleap failed to build product.prmtop')

    # -- validation + metadata ---------------------------------------------

    def _check_states_consistent(self, reactant: _Mol2, product: _Mol2) -> None:
        """Ensure the two states describe the same atoms in the same order."""
        if reactant.n_atoms != product.n_atoms:
            raise EVBBuildError(
                f'reactant/product atom count mismatch: '
                f'{reactant.n_atoms} vs {product.n_atoms}'
            )
        if reactant.names != product.names:
            raise EVBBuildError(
                'reactant/product atom NAMES/order differ; the two diabatic '
                'states must list the same atoms in the same order.'
            )

    def _verify_shared_coordinates(self) -> None:
        """Confirm the two prmtops have identical atom counts (shared frame)."""
        import parmed  # ty: ignore[unresolved-import]

        r = parmed.load_file(str(self.out / 'reactant.prmtop'))
        p = parmed.load_file(str(self.out / 'product.prmtop'))
        if len(r.atoms) != len(p.atoms):
            raise EVBBuildError(
                f'reactant/product topologies differ in atom count '
                f'({len(r.atoms)} vs {len(p.atoms)}); shared-coordinate EVB '
                'requires identical atom sets.'
            )

    def _metadata(self, box: list[float]) -> dict:
        """Assemble the evb_meta.json contents (indices, box, charges)."""
        idx = self._transfer_indices()
        return {
            'system': 'evb',
            'protein': str(self.protein) if self.protein else None,
            'resname': self.resname,
            'reactant_prmtop': str(self.out / 'reactant.prmtop'),
            'product_prmtop': str(self.out / 'product.prmtop'),
            'coordinates': str(self.out / 'system.inpcrd'),
            'box': box,
            'donor_atom': idx['donor'],
            'acceptor_atom': idx['acceptor'],
            'reactive_atom': idx['reactive'],
            'donor_name': self.donor,
            'acceptor_name': self.acceptor,
            'reactive_name': self.reactive,
        }

    def _transfer_indices(self) -> dict:
        """Resolve donor/acceptor/reactive to 0-based indices in the topology.

        Uses the atom names (unique within the reactive-region residue) against
        the built reactant topology so the indices are valid for the shared
        coordinate frame.

        Returns:
            Dict of 0-based ``donor``/``acceptor``/``reactive`` indices, each
            None if the corresponding name was not supplied.
        """
        if not any([self.donor, self.acceptor, self.reactive]):
            return {'donor': None, 'acceptor': None, 'reactive': None}

        import parmed  # ty: ignore[unresolved-import]

        top = parmed.load_file(str(self.out / 'reactant.prmtop'))
        by_name = {}
        for i, atom in enumerate(top.atoms):
            if atom.residue.name == self.resname:
                by_name.setdefault(atom.name, i)

        def lookup(name: str | None) -> int | None:
            if name is None:
                return None
            if name not in by_name:
                raise EVBBuildError(
                    f'atom {name!r} not found in residue {self.resname!r} of the '
                    f'built topology; available: {sorted(by_name)}'
                )
            return by_name[name]

        return {
            'donor': lookup(self.donor),
            'acceptor': lookup(self.acceptor),
            'reactive': lookup(self.reactive),
        }

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _read_box(inpcrd: Path) -> list[float]:
        """Read the periodic box lengths from an AMBER .inpcrd last line."""
        lines = [ln for ln in Path(inpcrd).read_text().splitlines() if ln.strip()]
        vals = [float(x) for x in lines[-1].split()]
        if len(vals) < 3:
            raise EVBBuildError(f'no box found in {inpcrd}')
        return vals[:3]

    def _run(self, cmd: list[str]) -> None:
        """Run a build command in the build dir (antechamber writes to cwd)."""
        log = self.build_dir / 'build.log'
        with open(log, 'a') as fh:
            fh.write(f'\n$ {" ".join(cmd)}\n')
            subprocess.run(
                cmd,
                cwd=str(self.build_dir),
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def _run_tleap(self, script: str, name: str) -> None:
        """Write and run a tleap script from the build dir."""
        leap = self.build_dir / name
        leap.write_text(script)
        log = self.build_dir / f'{name}.log'
        with open(log, 'w') as fh:
            subprocess.run(
                [self.tleap, '-f', leap.name],
                cwd=str(self.build_dir),
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )


def _as_list(x: PathLike | list[PathLike]) -> list[PathLike]:
    """Wrap a single path in a list; pass a list through unchanged."""
    if isinstance(x, (str, Path)):
        return [x]
    return list(x)
