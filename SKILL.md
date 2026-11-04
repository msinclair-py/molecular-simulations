---
name: molecular-simulations
description: Entry point for the molecular-simulations Python package (AMBER/OpenMM system building, GPU molecular dynamics, MM-PBSA, trajectory analysis, ipSAE2 interface scoring of predicted complexes, and Parsl HPC deployment). Use when a task involves setting up, running or analyzing MD simulations or scoring predicted complexes with this package, or when asked to install its agent skills; it routes to the molsim-* stage skills.
---

# molecular-simulations

`molecular-simulations` (`pip install molecular-simulations`, imported as
`molecular_simulations`) covers the path from a structure to an interpreted
result: build an AMBER system, run OpenMM MD, analyze it, and fan any stage
out across HPC nodes with Parsl. Each stage has its own skill; load the one
that matches the task.

| Stage | Skill | Use it for |
|-------|-------|------------|
| Build | `molsim-build-systems` | AMBER systems from PDB/CIF: explicit/implicit solvent, protein-ligand complexes, GAFF2 ligands |
| Simulate | `molsim-run-simulations` | OpenMM MD: explicit NPT, implicit GB, minimization, MM-PBSA, restarts, platforms |
| Analyze | `molsim-analyze-sasa` | SASA and relative per-residue exposure over a trajectory |
| Analyze | `molsim-analyze-interactions` | Interaction-energy fingerprints, linear interaction energy, clustering |
| Analyze | `molsim-analyze-ipsae` | ipSAE2 (and legacy ipSAE, ipTM, pDockQ, LIS) scoring of predicted complexes |
| Scale | `molsim-parsl-hpc` | Running any stage across nodes/GPUs: local, Polaris, Aurora settings |

Typical pipeline: `molsim-build-systems` → `molsim-run-simulations` →
`molsim-analyze-*`, with `molsim-parsl-hpc` wrapping any stage.

## Installing the package

System building calls AmberTools (`tleap`), which is not on PyPI. Either use
an existing AmberTools (set `AMBERHOME`, or pass `amberhome=` to the
builders) or install it from conda-forge alongside the package:

```bash
conda create -n molsim -c conda-forge "python>=3.10,<3.14" ambertools openmm
conda activate molsim
pip install molecular-simulations
```

Use conda-forge's `ambertools`: the package requires `numpy>=2.0`, and an
AmberTools build that pins `numpy<2` makes pip silently install the old
0.4.x release instead. If `pip show molecular-simulations` reports 0.4.x,
that is the cause. A pip warning that `ndfes`, `edgembar` or `proprep` need
`numpy<2` is harmless; the package does not use them.

Scoring predicted complexes (`molsim-analyze-ipsae`) does not need AmberTools.

## Installing these skills

If the stage skills above are not available to you, install them from the
package (they are symlinked, so they follow package upgrades):

```bash
molsim skills install --agent claude                  # ~/.claude/skills
molsim skills install --agent codex                   # ~/.agents/skills
molsim skills install --agent claude --scope project  # ./.claude/skills
molsim skills install --dest <agent skills directory> # any other agent
```

Claude Code users can instead install them from GitHub as a plugin:

```
/plugin marketplace add msinclair-py/molecular-simulations
/plugin install molecular-simulations@molecular-simulations
```

## Ground rules

- Use the package's builders, simulators and analysis classes rather than
  hand-writing `tleap` inputs, OpenMM scripts or scheduler scripts.
- Run MD on a GPU (`CUDA` or `OpenCL` platform); CPU is only for smoke
  tests.
- Documentation: https://molecular-simulations.readthedocs.io/en/latest/
