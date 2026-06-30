#!/usr/bin/env python
from pathlib import Path

from molecular_simulations.simulate import Simulator

path = Path('/path/to/simulation/inputs')
sim_length = 10  # ns
timestep = 4  # fs

n_steps = int(sim_length / timestep * 1_000_000)  # production steps
eq_steps = 500_000  # 1ns; 2 fs timestep

simulator = Simulator(path, equil_steps=eq_steps, prod_steps=n_steps)
simulator.run()
