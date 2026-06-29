---
name: run-simulations
description: Run OpenMM molecular dynamics with molecular-simulations — explicit-solvent NPT, implicit-solvent GB, energy minimization, and MM-PBSA binding free energy. Use when launching, configuring, or restarting an MD production run from AMBER/CHARMM inputs, choosing equilibration/production step counts, or selecting GPU/CPU platforms. For running many replicas across HPC nodes, combine with the parsl-hpc skill.
---

# Running simulations

`molecular_simulations.simulate` drives OpenMM. The main entry point is
`Simulator`, which runs a full minimize → heat → restrained equilibration →
NPT production pipeline and handles checkpoint/restart automatically.

## Input convention

`Simulator` reads `system.prmtop` and `system.inpcrd` from `path` by default
(override with `top_name` / `coor_name`). These are produced by the
`build-systems` workflow. All outputs (`eq.*`, `prod.dcd`, `prod.log`,
`prod.chk`, `prod.rst.chk`, `prod.state`) are written to `path` (or `out_path`).

## Explicit solvent (NPT)

```python
from pathlib import Path
from molecular_simulations.simulate import Simulator

# Pick step counts from physical time and timestep.
# Production uses HMR with a 4 fs timestep (0.004 ps) by default.
ns       = 100                       # production length
prod_dt  = 0.004                     # ps  (4 fs)
prod_steps = int(ns / prod_dt * 1000)  # ns -> steps

sim = Simulator(
    path=Path('/path/to/system_dir'),
    equil_steps=1_250_000,    # 2.5 ns @ 2 fs (default)
    prod_steps=prod_steps,
    temperature=300.0,
    platform='CUDA',          # 'CUDA' | 'CPU' | 'OpenCL'
    device_ids=[0],
)
sim.run()
```

What `run()` does:
1. Skips equilibration if `eq.state`/`eq.chk`/`eq.log` already exist.
2. Otherwise: minimize, slow-heat 5 K → `temperature` over `heat_steps`,
   relax backbone restraints in 5 stages, then NPT for `n_equil_cycles`.
3. Production from the equilibration checkpoint. If `prod.rst.chk` exists it
   **resumes** automatically, reading `prod.log` to compute remaining steps.

So re-invoking `sim.run()` after a crash continues where it left off — this is
the intended restart mechanism (important for preemptible HPC queues).

### Key constructor args
- `heat_steps` (default 100k), `equil_steps` (1.25M), `prod_steps` (250M = ~1 µs).
- `prod_dt_in_ps=0.004`, `hydrogen_mass_repartitioning=1.5` (amu).
- `eq_reporter_frequency=1000`, `prod_reporter_frequency=10000` (steps between
  DCD frames / log lines; checkpoint is written every `prod_freq * 10`).
- `temperature=300.0`, `force_constant=10.0` (kcal/mol·Å² backbone restraint).
- `membrane=True` — switch to `MonteCarloMembraneBarostat` (anisotropic).
- `ff='charmm'` with `params=[...psf params...]` — CHARMM inputs (1.2 nm cutoff);
  default `ff='amber'` uses PME, 1 nm cutoff, HBond constraints.

### GPU selection
- On `CUDA`, if `CUDA_VISIBLE_DEVICES` is set (e.g. by Parsl/the scheduler), the
  in-process device index is forced to `0` because the scheduler already masks
  and re-indexes GPUs. So don't fight it — let Parsl pin GPUs and leave
  `device_ids=[0]`. Without that env var, `device_ids` selects physical GPUs.

## Implicit solvent (GB)

```python
from molecular_simulations.simulate import ImplicitSimulator
from openmm.app import GBn2

sim = ImplicitSimulator(
    path='/path/to/system_dir',
    implicit_solvent=GBn2,         # default
    prod_steps=100_000_000,
    solvent_dielectric=78.5,       # 150 mM ionic screening applied via kappa
)
sim.run()
```

Same interface as `Simulator` but no barostat and a shorter equilibration. Build
the inputs with `ImplicitSolvent` (mbondi3 radii), not `ExplicitSolvent`.

## Energy minimization only

```python
from molecular_simulations.simulate import Minimizer

Minimizer(
    topology='system.prmtop',      # or a .pdb (uses amber14-all.xml)
    coordinates='system.inpcrd',
    out='min.pdb',
    platform='OpenCL',
    device_ids=[0],                # None -> CPU
).minimize()
```

## MM-PBSA binding free energy

```python
from molecular_simulations.simulate import MMPBSA
```

`MMPBSA` estimates binding free energy in parallel from a complex trajectory.
Read its docstring/signature for the required selections and decomposition
options before wiring it up — it expects already-equilibrated trajectories.

## Custom forces

`CustomForcesSimulator(path, custom_force_objects=[...])` (import from
`molecular_simulations.simulate.omm_simulator`) injects arbitrary OpenMM `Force`
objects into the system for enhanced sampling — same run pipeline otherwise.

## Tips
- One `Simulator` == one replica/one system directory. Run many in parallel by
  globbing replica dirs and submitting each through Parsl (see `parsl-hpc`).
- Monitor progress via `prod.log` (tab-separated: step, energy, temp, %
  progress, remaining time, speed, volume).
- `prod.dcd` frames are written every `prod_reporter_frequency` steps; on restart
  the reporter appends and any duplicate frames are recorded in
  `duplicate_frames.log`.
