import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import yaml
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher, SimpleLauncher
from parsl.providers import LocalProvider, PBSProProvider
from pydantic import BaseModel, Field

PathLike = str | Path
_T = TypeVar('_T')


def get_node_count():
    """Infer the node count for a job from the scheduler environment.

    For all LocalProvider-using protocols, checks for evidence of SLURM or PBS
    to infer the number of nodes allocated to a job.

    Returns:
        Number of nodes for the job, defaulting to 1 if neither SLURM nor PBS
        environment variables are found.
    """
    import os

    if num_nodes := os.environ.get('SLURM_JOB_NUM_NODES'):
        return int(num_nodes)

    if nodefile := os.environ.get('PBS_NODEFILE'):
        with open(nodefile) as f:
            return len(f.readlines())

    return 1


class BaseSettings(BaseModel):
    """Base pydantic model with YAML serialization helpers."""

    def dump_yaml(self, filename: PathLike) -> None:
        """Serialize the settings to a YAML file.

        Args:
            filename: Destination path for the YAML file.
        """
        with open(filename, mode='w') as fp:
            yaml.dump(json.loads(self.model_dump_json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: type[_T], filename: PathLike) -> _T:
        """Construct a settings instance from a YAML file.

        Args:
            filename: Path to the YAML file to load.

        Returns:
            A settings instance populated from the file contents.
        """
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class BaseComputeSettings(ABC, BaseSettings):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    @abstractmethod
    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a new Parsl configuration.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config for this compute platform.
        """


class BaseLocalSettings(BaseComputeSettings):
    """Shared settings for LocalProvider-based compute platforms."""

    available_accelerators: int | Sequence[str] = 4
    worker_init: str = ''
    nodes: int = Field(default_factory=get_node_count)
    retries: int = 1
    label: str = 'executor_label'
    worker_port_range: tuple[int, int] = (10000, 20000)


class LocalSettings(BaseLocalSettings):
    """Settings for a single-node, GPU-pinned local executor."""

    label: str = 'gpu'
    mpi_launcher: bool = True

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a Parsl configuration pinning workers to local GPUs.

        By default this uses ``MpiExecLauncher`` with ``--cpu-bind``, which is
        what ALCF-style nodes (e.g. Polaris) expect. On a plain single GPU box
        the stock ``mpiexec`` rejects ``--cpu-bind`` and workers fail to
        register; set ``mpi_launcher=False`` to fall back to ``SimpleLauncher``,
        which launches workers directly with no MPI wrapper.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config with one GPU-affinity HighThroughputExecutor.
        """
        launcher = (
            MpiExecLauncher(bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1')
            if self.mpi_launcher
            else SimpleLauncher()
        )
        return Config(
            run_dir=str(Path(run_dir) / 'runinfo'),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    provider=LocalProvider(
                        nodes_per_block=self.nodes,
                        init_blocks=1,
                        max_blocks=1,
                        launcher=launcher,
                        worker_init=self.worker_init,
                    ),
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                ),
            ],
        )


class LocalCPUSettings(BaseLocalSettings):
    """Settings for a single-node, CPU-only local executor."""

    max_workers_per_node: int = 1
    cores_per_worker: float = 1.0
    label: str = 'cpu'
    available_accelerators: int | Sequence[str] = []

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a Parsl configuration for CPU-only local execution.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config with one CPU HighThroughputExecutor.
        """
        return Config(
            run_dir=str(Path(run_dir) / 'runinfo'),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    provider=LocalProvider(
                        nodes_per_block=self.nodes,
                        init_blocks=1,
                        max_blocks=1,
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1'
                        ),
                        worker_init=self.worker_init,
                    ),
                    label=self.label,
                    max_workers_per_node=self.max_workers_per_node,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                ),
            ],
        )


class HeterogeneousSettings(BaseLocalSettings):
    """Settings for a single node exposing both GPU and CPU executors."""

    max_workers_per_node: int = 1
    cores_per_worker: float = 1.0
    available_accelerators: int = 4

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a Parsl configuration with separate GPU and CPU executors.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config with a GPU HighThroughputExecutor and a CPU
            HighThroughputExecutor.
        """
        return Config(
            run_dir=str(Path(run_dir) / 'runinfo'),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    provider=LocalProvider(
                        nodes_per_block=self.nodes,
                        init_blocks=1,
                        max_blocks=1,
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1'
                        ),
                        worker_init=self.worker_init,
                    ),
                    label='gpu',
                    cpu_affinity='block',
                    max_workers_per_node=self.available_accelerators,
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                ),
                HighThroughputExecutor(
                    provider=LocalProvider(
                        nodes_per_block=self.nodes,
                        init_blocks=1,
                        max_blocks=1,
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind', overrides='--depth=1 --ppn 1'
                        ),
                        worker_init=self.worker_init,
                    ),
                    label='cpu',
                    max_workers_per_node=self.max_workers_per_node,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                ),
            ],
        )


class PolarisSettings(BaseComputeSettings):
    """Settings for running on the Polaris HPC system via PBSPro."""

    label: str = 'htex'
    num_nodes: int = 1
    worker_init: str = ''
    scheduler_options: str = ''
    account: str
    queue: str
    walltime: str
    cpus_per_node: int = 64
    strategy: str = 'simple'
    available_accelerators: int | Sequence[str] = 4

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a configuration for running tasks on single nodes of Polaris.

        Launches 4 workers per node, each pinned to a different GPU.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config with a PBSPro-provisioned HighThroughputExecutor.
        """
        return Config(
            retries=1,  # Allows restarts if jobs are killed by the end of a job
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    available_accelerators=self.available_accelerators,
                    address=address_by_hostname(),
                    cpu_affinity='alternating',
                    prefetch_capacity=0,  # Increase if you have many more tasks than workers
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind', overrides='--depth=64 --ppn 1'
                        ),  # Updates to the mpiexec command
                        account=self.account,
                        queue=self.queue,
                        select_options='ngpus=4',
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options=self.scheduler_options,
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            strategy=self.strategy,
        )


class AuroraSettings(BaseComputeSettings):
    """Settings for running on the Aurora HPC system via PBSPro."""

    label: str = 'htex'
    worker_init: str = ''
    num_nodes: int = 1
    scheduler_options: str = ''
    account: str
    queue: str
    walltime: str
    retries: int = 0
    cpus_per_node: int = 48  # only 4 cpus per OpenMM job
    strategy: str = 'simple'
    available_accelerators: list[str] = [str(i) for i in range(12)]

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a Parsl configuration for running on Aurora.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config with a PBSPro-provisioned HighThroughputExecutor.
        """
        return Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    available_accelerators=self.available_accelerators,
                    cpu_affinity='block',  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    # One worker per addressable accelerator; deriving this from
                    # the list (rather than a hard 12) lets a caller dial down
                    # concurrency to bound per-node thread oversubscription --
                    # unthrottled, each Intel oneAPI GPU context spawns a
                    # full-node thread pool and 12 at once exhausts the process
                    # thread limit (std::system_error: Resource temporarily
                    # unavailable). Pair with OMP_NUM_THREADS in worker_init.
                    max_workers_per_node=len(self.available_accelerators),
                    cores_per_worker=16,
                    heartbeat_period=30,
                    heartbeat_threshold=300,
                    worker_debug=False,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind', overrides='--depth=208 --ppn 1'
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        account=self.account,
                        queue=self.queue,
                        # Aurora PBS rejects jobs that do not declare the
                        # filesystems they need; pass it (and any other #PBS
                        # header lines) via scheduler_options.
                        scheduler_options=self.scheduler_options,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            retries=self.retries,
        )


class AuroraLocalSettings(BaseLocalSettings):
    """Aurora executor that runs INSIDE an existing PBS allocation.

    The same HighThroughputExecutor as :class:`AuroraSettings` (one worker per
    accelerator, ``--depth=208`` mpiexec so a single manager per node divides all
    208 threads), but provisioned with :class:`LocalProvider` -- the workers run
    on the nodes the job ALREADY holds rather than qsub-ing a fresh block. Use
    this when the driver itself runs inside a qsub'd job (e.g. on the allocation's
    head node): it keeps the slow OpenMM cold-import and parsl orchestration off
    the (often contended) login node. Defaults to 6 tiles/node -- the concurrency
    that stays under the per-node thread limit (see AuroraSettings).
    """

    label: str = 'htex'
    available_accelerators: list[str] = [str(i) for i in range(6)]

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create an in-allocation Parsl configuration for Aurora.

        Args:
            run_dir: Directory in which to store Parsl run files.

        Returns:
            A Parsl Config whose HighThroughputExecutor launches workers on the
            current allocation's nodes via LocalProvider + MpiExecLauncher.
        """
        return Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    available_accelerators=self.available_accelerators,
                    cpu_affinity='block',
                    prefetch_capacity=0,
                    max_workers_per_node=len(self.available_accelerators),
                    cores_per_worker=16,
                    heartbeat_period=30,
                    heartbeat_threshold=300,
                    worker_debug=False,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind', overrides='--depth=208 --ppn 1'
                        ),
                        worker_init=self.worker_init,
                        nodes_per_block=self.nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,
                    ),
                ),
            ],
            run_dir=str(Path(run_dir) / 'runinfo'),
            retries=self.retries,
        )
