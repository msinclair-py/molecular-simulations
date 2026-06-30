from pathlib import Path

from molecular_simulations.build import ExplicitSolvent

pdb = Path('/path/to/input.pdb')
out_path = Path('/path/to/simulation/inputs')

builder = ExplicitSolvent(out_path, pdb)
builder.build()
