import MDAnalysis as mda
import numpy as np
from pathlib import Path
from rust_simulation_tools import kabsch_align  # ty: ignore[unresolved-import]

def trim_trajectory(
    u: mda.Universe,
    out: Path,
    stride: int = 1,
    align: bool = False,
    rewrap: bool = False,
    sel: str | None = None,
    align_sel: str | None = None,
) -> None:
    """Write a strided, optionally aligned subset of a trajectory to disk.

    Iterates the trajectory with the given stride, collecting positions for the
    selected atoms, optionally aligns every frame to the first frame via a
    Kabsch fit, and writes the result to a new trajectory file.

    Args:
        u: MDAnalysis Universe providing the source trajectory.
        out: Output trajectory file path.
        stride: Keep every nth frame. Defaults to 1.
        align: If True, Kabsch-align all frames to the first frame using
            align_sel (or the backbone if align_sel is None). Defaults to False.
        rewrap: Reserved for periodic re-wrapping; currently a no-op.
            Defaults to False.
        sel: Optional MDAnalysis selection string for atoms to keep. If None,
            all atoms are written. Defaults to None.
        align_sel: Optional MDAnalysis selection string (relative to the kept
            selection) defining atoms used for alignment. If None, falls back to
            'backbone or nucleicbackbone'. Defaults to None.

    Returns:
        None. Writes the trimmed trajectory to out.
    """
    selection = u.select_atoms(sel) if sel is not None else u.atoms
    assert selection is not None

    positions = np.zeros(
        (u.trajectory.n_frames // stride, selection.n_atoms, 3), dtype=np.float32
    )

    for i, _ in enumerate(u.trajectory[::stride]):
        positions[i, ...] = selection.positions.copy().astype(np.float32)

    if align:
        if align_sel is None:
            align_idx = selection.select_atoms('backbone or nucleicbackbone').ix
        else:
            align_idx = selection.select_atoms(align_sel).ix

        positions = kabsch_align(positions, positions[0], align_idx)

    if rewrap:
        pass

    with mda.Writer(str(out), n_atoms=selection.n_atoms) as w:
        for pos in positions:
            w.write(pos)
