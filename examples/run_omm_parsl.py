#!/usr/bin/env python
from pathlib import Path

import parsl

from molecular_simulations.utils.parsl_settings import LocalSettings

# Housekeeping stuff
run_dir = '/path/to/deploy/parsl'  # likely same as root simulation dir
sim_root_dir = Path('/path/to/deploy/parsl')  # could be different
paths = sim_root_dir.glob('replica_*')  # get all sim replica dirs

settings = LocalSettings.from_yaml('config.yaml')
config = settings.config_factory(run_dir)

sim_length = 10  # ns
timestep = 4  # fs
n_steps = int(sim_length / timestep * 1_000_000)


@parsl.python_app
def run_md(path: str, eq_steps=500_000, steps=250_000_000):
    from molecular_simulations.simulate.omm_simulator import Simulator

    simulator = Simulator(path, equil_steps=eq_steps, prod_steps=steps)
    simulator.run()


# load parsl
parsl.load(config)

# collect futures
futures = []
for path in paths:
    futures.append(run_md(str(path), steps=n_steps))

# run workers in parallel
outputs = [x.result() for x in futures]
