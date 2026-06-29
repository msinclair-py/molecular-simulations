---
name: parsl-hpc
description: Configure Parsl to deploy molecular-simulations builds, MD runs, and analyses across HPC. Use when running many systems/replicas in parallel, picking a compute platform (Local single-node GPU, Heterogeneous GPU+CPU, Polaris, or Aurora via PBSPro), pinning workers to GPUs, or writing/loading a YAML settings file for a cluster job.
---

# Deploying with Parsl

`molecular_simulations.utils.parsl_settings` provides pydantic `*Settings`
classes that each emit a ready-made Parsl `Config` via `config_factory(run_dir)`.
You pick the class for your platform, fill in a few fields (or load from YAML),
build the config, `parsl.load()` it, then submit work as `@parsl.python_app`s.

## Import path (important)

```python
from molecular_simulations.utils.parsl_settings import (
    LocalSettings, LocalCPUSettings, HeterogeneousSettings,
    PolarisSettings, AuroraSettings,
)
```

Only `LocalSettings`, `PolarisSettings`, `AuroraSettings` are re-exported from
`molecular_simulations.utils`. `HeterogeneousSettings` and `LocalCPUSettings`
must be imported from the full `...utils.parsl_settings` path. (Note: the
`examples/run_omm_parsl.py` import `from molecular_simulations.simulate import
LocalSettings` is wrong — use the `utils` path.)

## The pattern

```python
import parsl
from molecular_simulations.utils.parsl_settings import LocalSettings

run_dir = '/path/to/deploy'                    # where Parsl run files go
settings = LocalSettings(worker_init='source ~/setup_env.sh')
config = settings.config_factory(run_dir)
parsl.load(config)

@parsl.python_app
def run_md(path, eq_steps=500_000, steps=250_000_000):
    # IMPORTANT: import inside the app so workers can pickle it
    from molecular_simulations.simulate.omm_simulator import Simulator
    Simulator(path, equil_steps=eq_steps, prod_steps=steps).run()

from pathlib import Path
futures = [run_md(str(p)) for p in Path('/path/to/sims').glob('replica_*')]
outputs = [f.result() for f in futures]         # block until all finish
```

`worker_init` is shell run on each worker before tasks (activate conda/venv,
`module load`, set `AMBERHOME`, etc.) — set it on whichever settings class you use.

## Platforms

### LocalSettings — single node, GPU-pinned
One `HighThroughputExecutor`, workers pinned to local GPUs via
`available_accelerators` (default 4). Use on a workstation or a single
interactive compute node. `LocalProvider` + `MpiExecLauncher`.

```python
LocalSettings(available_accelerators=4, worker_init='...', label='gpu')
```

### LocalCPUSettings — single node, CPU-only
For CPU work (e.g. analyses, MM-PBSA). `available_accelerators=[]`,
`max_workers_per_node` / `cores_per_worker` control concurrency.

### HeterogeneousSettings — single node, GPU **and** CPU executors
Emits two executors in one config: label `'gpu'` (one worker per accelerator)
and label `'cpu'`. Use when GPU MD apps and CPU analysis apps run in the same
workflow — route each `@python_app` to an executor with the `executors=[...]`
decorator argument.

### PolarisSettings — ALCF Polaris (PBSPro)
PBSPro provider, 4 GPUs/node (`select_options='ngpus=4'`), `cpu_affinity=
'alternating'`. Requires scheduler fields:

```python
PolarisSettings(
    account='MY_ALLOCATION',
    queue='debug',                 # or 'prod', etc.
    walltime='01:00:00',
    num_nodes=1,
    worker_init='module load conda; conda activate md',
    available_accelerators=4,
    cpus_per_node=64,
)
```

### AuroraSettings — ALCF Aurora (PBSPro, Intel GPUs)
12 accelerator tiles/node (`available_accelerators=[str(i) for i in range(12)]`),
`max_workers_per_node=12`, `cores_per_worker=16`, `cpu_affinity='block'`. Pair
with `platform='OpenCL'` in the `Simulator` (Aurora uses Intel GPUs, not CUDA).

```python
AuroraSettings(
    account='MY_ALLOCATION',
    queue='debug',
    walltime='01:00:00',
    num_nodes=2,
)
```

## YAML config (recommended for cluster jobs)

All settings inherit `from_yaml` / `dump_yaml`, so keep platform config in a file
and out of code:

```python
settings = PolarisSettings.from_yaml('config.yaml')
config = settings.config_factory(run_dir)
```

```yaml
# config.yaml
account: MY_ALLOCATION
queue: debug
walltime: "01:00:00"
num_nodes: 1
worker_init: "module load conda; conda activate md; export AMBERHOME=..."
available_accelerators: 4
```

Generate a starter file with `PolarisSettings(account=..., queue=..., walltime=...
).dump_yaml('config.yaml')`.

## Notes
- `PolarisSettings`/`AuroraSettings` require `account`, `queue`, `walltime` (no
  defaults) — construction fails without them.
- `Simulator` auto-detects scheduler-masked GPUs (`CUDA_VISIBLE_DEVICES`), so
  leave `device_ids=[0]` in the app and let Parsl pin accelerators.
- `max_blocks=1` in the providers by default — one PBS job/block. Edit the
  settings class if you need multiple concurrent blocks.
- Run files land under `<run_dir>/runinfo` (Local/Heterogeneous) or `<run_dir>`
  (Polaris/Aurora).
