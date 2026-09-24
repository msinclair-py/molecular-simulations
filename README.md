# molecular-simulations

![CI/CD](https://github.com/msinclair-py/molecular-simulations/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/msinclair-py/molecular-simulations/branch/master/graph/badge.svg)](https://codecov.io/gh/msinclair-py/molecular-simulations)
[![PyPI version](https://img.shields.io/pypi/v/molecular-simulations)](https://pypi.org/project/molecular-simulations/)
[![Documentation Status](https://readthedocs.org/projects/molecular-simulations/badge/?version=latest)](https://molecular-simulations.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python toolkit for building, running, and analyzing molecular dynamics simulations using the AMBER force field ecosystem and OpenMM.

📖 **[Full Documentation](https://molecular-simulations.readthedocs.io/en/latest/index.html)**

## Features

### 🔨 System Building
- **Explicit solvent systems** with OPC water model
- **Implicit solvent** support for faster calculations
- **Small molecule parameterization** via GAFF2

### ⚡ Simulation Engine
- **OpenMM integration** (v8.0+) with GPU acceleration
- **Advanced simulations** Constant-pH and Empirical Valence Bond
- **HPC deployment** via [Parsl](https://parsl.readthedocs.io/) for PBS schedulers
- **MM-PBSA calculations** for binding free energy estimation in parallel
- Flexible configuration for various cluster environments

### 📊 Analysis Tools
- **Automatic clustering** with KMeans++ and dimensionality reduction (PCA)
- **Protein-protein interaction analysis** using covariance matrix approach
- **Interaction energy fingerprinting** (electrostatic + Lennard-Jones)
- **Linear interaction energy** calculations (static and dynamic)
- **Interface scoring** with ipTM, ipSAE, pDockQ, and pDockQ2
- **SASA calculations** (absolute and relative) via MDAnalysis
- **Residue energy footprinting** for binding site characterization

## Installation

> **Note:** `molecular-simulations` requires [**AmberTools**](https://ambermd.org/AmberTools.php) (for system building with `tleap`), which is **not distributed on PyPI**. AmberTools must be installed separately via a conda-forge-based package manager (Anaconda/Miniconda, Mamba, or Pixi) before installing this package with `pip`.

### Using conda / mamba

```bash
conda create -n molsim python=3.12
conda activate molsim
conda install -c dacase -c conda-forge ambertools-dac openmm
pip install molecular-simulations
```

`dacase::ambertools-dac` is the official channel maintained by the AmberTools developers and tracks the current AmberTools release; the community-maintained `conda-forge::ambertools` package also works if you prefer to stay on a single channel. `mamba` is a drop-in replacement for `conda` (`mamba create ...`, `mamba install ...`) and resolves environments much faster.

### Using pixi

```bash
pixi init molsim
cd molsim
pixi workspace channel add dacase
pixi add python=3.12 ambertools-dac openmm
pixi add --pypi molecular-simulations
```

### Installing the package alone from PyPI

If AmberTools is already available in your environment (or you only need OpenMM-based simulation/analysis features without system building), you can install just the Python package:

```bash
pip install molecular-simulations
```

For small molecule support (requires RDKit and OpenBabel):

```bash
pip install molecular-simulations[ligand]
```

For development:

```bash
pip install molecular-simulations[dev]
```

## Quick Start

### Building a Solvated System

```python
from molecular_simulations.build import ExplicitSolvent
from pathlib import Path

# Build an explicitly solvated system
pdb_file = Path('/path/to/protein.pdb')
output_dir = Path('/path/to/outputs')

builder = ExplicitSolvent(output_dir, pdb_file)
builder.build()

# Outputs: topology (.prmtop) and coordinates (.inpcrd)
```

### Running a Simulation

```python
from molecular_simulations.simulate import Simulator

# Initialize and run simulation
sim = Simulator(
    path=builder.out.parent, # Directory containing simulation inputs
)
sim.run()
```

### Analyzing Trajectories

#### Interaction Energy Fingerprinting

```python
from molecular_simulations.analysis import Fingerprinter

fp = Fingerprinter(
    topology='/path/to/system.prmtop',
    trajectory='/path/to/trajectory.dcd',
    target_selection='segid A',
    binder_selection='segid B'
)
fp.run()
fp.save()  # Saves to fingerprint.npz
```

#### Automatic Clustering

```python
from molecular_simulations.analysis import AutoKMeans

clusterer = AutoKMeans(
    data_directory='/path/to/features/', # should be populated with .npy files
    max_clusters=10,
    reduction_algorithm='PCA',
    reduction_kws={'n_components': 2}
)
clusterer.run()
clusterer.save_labels()   # Saves cluster assignments
clusterer.save_centers()  # Saves cluster centroids
```

#### MM-PBSA implementation

```python
from molecular_simulations.simulate.mmpbsa import MMPBSA

mmpbsa = MMPBSA(
    top='/path/to/file.prmtop',
    dcd='/path/to/traj.dcd',
    selections=[':1-100', ':101-200'], # cpptraj-style selections
    n_cpus=1,                          # CPU-based parallelism supported
    amberhome='/path/to/dir',          # should be above /bin/cpptraj
    parallel_mode='frame'              # frame or serial
)
```

#### Protein-Protein Interaction Analysis

```python
from molecular_simulations.analysis import PPInteractions

ppi = PPInteractions(
    top='/path/to/topology.prmtop',
    traj='/path/to/trajectory.dcd',
    out='/path/to/outputs/',
    sel1='chainID A',
    sel2='chainID B'
)
ppi.run()  # Analyzes H-bonds, salt bridges, hydrophobic contacts
```

#### Interface Scoring (ipSAE/pDockQ)

```python
from molecular_simulations.analysis import ipSAE

scorer = ipSAE(
    structure_file='/path/to/complex.pdb',
    pae_file='/path/to/pae.json',  # AlphaFold PAE matrix
    plddt_file='/path/to/plddt.json'
)
scores = scorer.run()
# Returns: ipTM, ipSAE, pDockQ, pDockQ2 scores
```

#### SASA Calculations

```python
from molecular_simulations.analysis import SASA, RelativeSASA

# Absolute SASA
sasa = SASA(universe, selection='protein')
sasa.run()
results = sasa.measure_sasa()

# Relative SASA (normalized by max accessible area)
rsasa = RelativeSASA(universe, selection='protein')
rsasa.run()
```

## Supported Force Fields

|==============AMBER==============|
| Component | Force Field | Notes |
|-----------|-------------|-------|
| Proteins | ff19SB | Fixed-charge, recommended |
| Proteins | ff15ipq | Polarizable |
| DNA | OL21 | Latest AMBER DNA parameters |
| RNA | OL3 | Standard RNA parameters |
| Small molecules | GAFF2 | General AMBER Force Field 2 |
| Water (explicit) | OPC | 4-point model |
| Water (polarizable) | SPC/Eb | For ff15ipq systems |

|=============CHARMM=============|
See OpenMM documentation on providing parameter
sets. CHARMM does not translate very well to the
pre-build XML files for unusual lipids, small molecules,
etc.

## Requirements

- Python ≥ 3.10
- OpenMM ≥ 8.0
- MDAnalysis ≥ 2.7
- Parsl ≥ 2024.1.29
- NumPy, SciPy, scikit-learn, Polars
- AmberTools (not available on PyPI — install via conda/mamba/pixi, see [Installation](#installation))
- Optional: RDKit, OpenBabel (for ligand parameterization)

## Known Issues

- OpenMM versions 8.0-8.1 may exhibit slower integration times for larger systems due to a known bug
- Advanced features are tested but not all edge cases have been encountered, please open an Issue if you find one

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this software in your research, please cite:

```bibtex
@software{molecular_simulations,
  author = {Sinclair, Matt},
  title = {molecular-simulations: A Python toolkit for MD simulation and analysis},
  url = {https://github.com/msinclair-py/molecular-simulations},
  year = {2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenMM](https://openmm.org/) for molecular dynamics engine
- [MDAnalysis](https://www.mdanalysis.org/) for trajectory analysis
- [Parsl](https://parsl-project.org/) for parallel workflow execution
- [AMBER](https://ambermd.org/) force field developers
