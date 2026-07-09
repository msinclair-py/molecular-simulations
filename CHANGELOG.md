# Changelog

All notable changes to molecular-simulations are documented in this file.

Releases from 0.4.4 onward are generated automatically by
[release-please](https://github.com/googleapis/release-please) from
[Conventional Commits](https://www.conventionalcommits.org/). This project
adheres to [Semantic Versioning](https://semver.org/).

## [0.5.1](https://github.com/msinclair-py/molecular-simulations/compare/v0.5.0...v0.5.1) (2026-07-09)


### Bug Fixes

* **build:** form disulfides instead of leaving cysteines reduced ([bd56cb1](https://github.com/msinclair-py/molecular-simulations/commit/bd56cb1a1e41647a738cc5a7fa8bddd7e1c34321))
* **build:** form disulfides instead of leaving cysteines reduced ([63f63e4](https://github.com/msinclair-py/molecular-simulations/commit/63f63e4877c541207c4699982c3c6214a2127228))

## [0.5.0](https://github.com/msinclair-py/molecular-simulations/compare/v0.4.4...v0.5.0) (2026-07-08)


### ⚠ BREAKING CHANGES

* `Simulator(ff='charmm', params=...)` is no longer supported; only the AMBER force field remains.

### Features

* drop CHARMM force field support ([dd4751e](https://github.com/msinclair-py/molecular-simulations/commit/dd4751e7bb342e60a123f3ec7f310aa9d9b96503))


### Bug Fixes

* **ci:** ruff-format tests; skip-gate the real kabsch align tests on numpy 2.x ([4f70e68](https://github.com/msinclair-py/molecular-simulations/commit/4f70e686cde55330bd293d300082c7cb29700b12))
* **cph:** correct GB titration, build/validate protonated systems, recalibrate reference energies ([#26](https://github.com/msinclair-py/molecular-simulations/issues/26)) ([f40deeb](https://github.com/msinclair-py/molecular-simulations/commit/f40deebf1a36d84466b70104b262927d2c829355))
* feed kabsch_align float64/int64; require rust-simulation-tools&gt;=0.2.5 ([abb660f](https://github.com/msinclair-py/molecular-simulations/commit/abb660f74c2e805d9ddd04b8e83dc8e410ac58bd))
* repair get_node_count and parameterize_ligand; de-brittle assemble test ([38ab0fd](https://github.com/msinclair-py/molecular-simulations/commit/38ab0fdcd34d74e7aaf20777fc6f46f50f62a632))
* repair SASA/autocluster/Minimizer source bugs + de-mock tests ([b3ae9f5](https://github.com/msinclair-py/molecular-simulations/commit/b3ae9f59e4cc99b201693e94f459da16a40ad90b))
* repair the SASA, autocluster and Minimizer source bugs; de-mock their tests ([8eec391](https://github.com/msinclair-py/molecular-simulations/commit/8eec391b61b5439fd589a81e2045db092ad1c00c))


### Documentation

* add missing sphinx-autodoc-typehints to docs extras ([57c60d5](https://github.com/msinclair-py/molecular-simulations/commit/57c60d57d88baddc4f4eed3e838f11719df10e0a))
* add usage skills and fix broken examples ([2a5aad3](https://github.com/msinclair-py/molecular-simulations/commit/2a5aad38ce820acbf30de6951d20020ab11c8c19))
* fix the strict (-W) build (missing _static, pydantic JsonValue) ([51e9050](https://github.com/msinclair-py/molecular-simulations/commit/51e905029f012100d426f813eabd2b2ad24988d5))
* make the strict (-W) build pass in CI ([53fdc81](https://github.com/msinclair-py/molecular-simulations/commit/53fdc81ac8248cf725f39a497b646a730146998a))
* make the strict (-W) Sphinx build pass ([d35af8c](https://github.com/msinclair-py/molecular-simulations/commit/d35af8ca3c0ebf4084222f39012601e4e552f18f))
* single-source the changelog via release-please CHANGELOG.md ([18f79e5](https://github.com/msinclair-py/molecular-simulations/commit/18f79e5b219d02d03448e408356c7a87ca2f87f8))

## 0.4.4

Baseline release for automated changelog generation.
