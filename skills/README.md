# Skills

Agent skills for working with the `molecular-simulations` package, in the
open [Agent Skills](https://agentskills.io/specification) format read by
Claude Code, Codex and other coding agents. Each subdirectory holds a
`SKILL.md` (name + description frontmatter) that the agent loads on demand
when a task matches.

| Skill | Use it for |
|-------|------------|
| [molecular-simulations](../SKILL.md) | Entry point: what the package does, which stage skill to load, how to install |
| [molsim-build-systems](molsim-build-systems/SKILL.md) | Building AMBER systems from PDB/CIF — explicit/implicit solvent, protein-ligand complexes, GAFF2 ligand parameterization |
| [molsim-run-simulations](molsim-run-simulations/SKILL.md) | Running OpenMM MD — explicit NPT, implicit GB, minimization, MM-PBSA, restarts, platform selection |
| [molsim-analyze-sasa](molsim-analyze-sasa/SKILL.md) | Solvent-accessible surface area — absolute SASA and RelativeSASA fractional exposure, per residue over a trajectory |
| [molsim-analyze-ipsae](molsim-analyze-ipsae/SKILL.md) | Interface scoring of predicted complexes — ipSAE2 (proteins, nucleic acids, ligands, glycans, ions) plus legacy ipSAE, ipTM, pDockQ, pDockQ2, LIS |
| [molsim-analyze-interactions](molsim-analyze-interactions/SKILL.md) | Interaction-energy fingerprinting, linear interaction energy (static/dynamic), and KMeans clustering of per-frame features |
| [molsim-parsl-hpc](molsim-parsl-hpc/SKILL.md) | Deploying builds/runs/analyses across HPC with Parsl — Local, Heterogeneous, Polaris, Aurora settings |

Typical pipeline: **molsim-build-systems** → **molsim-run-simulations** →
**molsim-analyze-sasa** / **molsim-analyze-ipsae** /
**molsim-analyze-interactions**, with **molsim-parsl-hpc** wrapping any stage
for parallel fan-out across nodes/GPUs.

## Installing

The skills ship inside the Python package. After `pip install
molecular-simulations`, link them into your agent's skills directory:

```bash
molsim skills install --agent claude                  # ~/.claude/skills
molsim skills install --agent codex                   # ~/.agents/skills
molsim skills install --agent claude --scope project  # ./.claude/skills
molsim skills install --dest <skills directory>       # any other agent
molsim skills uninstall --agent claude                # remove them again
```

Links point into the installed package, so the skills always match its
version. Add `--copy` for independent copies.

Claude Code can also install them straight from GitHub as a plugin:

```
/plugin marketplace add msinclair-py/molecular-simulations
/plugin install molecular-simulations@molecular-simulations
```

The entry skill `molecular-simulations/SKILL.md` is a link to the top-level
[`SKILL.md`](../SKILL.md).
