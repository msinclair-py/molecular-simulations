# Skills

Claude Code / agent skills for working with the `molecular-simulations` package.
Each subdirectory holds a `SKILL.md` (name + description frontmatter) that the
agent loads on demand when a task matches.

| Skill | Use it for |
|-------|------------|
| [build-systems](build-systems/SKILL.md) | Building AMBER systems from PDB/CIF — explicit/implicit solvent, protein-ligand complexes, GAFF2 ligand parameterization |
| [run-simulations](run-simulations/SKILL.md) | Running OpenMM MD — explicit NPT, implicit GB, minimization, MM-PBSA, restarts, platform selection |
| [analyze-sasa](analyze-sasa/SKILL.md) | Solvent-accessible surface area — absolute SASA and RelativeSASA fractional exposure, per residue over a trajectory |
| [analyze-ipsae](analyze-ipsae/SKILL.md) | ipSAE interface scoring of predicted complexes — ipSAE, ipTM, pDockQ, pDockQ2, LIS from pLDDT/PAE |
| [analyze-interactions](analyze-interactions/SKILL.md) | Interaction-energy fingerprinting, linear interaction energy (static/dynamic), and KMeans clustering of per-frame features |
| [parsl-hpc](parsl-hpc/SKILL.md) | Deploying builds/runs/analyses across HPC with Parsl — Local, Heterogeneous, Polaris, Aurora settings |

Typical pipeline: **build-systems** → **run-simulations** → **analyze-sasa** /
**analyze-ipsae** / **analyze-interactions**, with **parsl-hpc** wrapping any
stage for parallel fan-out across nodes/GPUs.
