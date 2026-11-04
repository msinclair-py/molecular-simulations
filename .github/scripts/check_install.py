"""Check an environment built from a documented install recipe.

Run with the environment's Python. Fails if molecular-simulations is not the
newest release on PyPI (a silent fallback such as 0.4.x means a dependency
conflict), if NumPy is older than 2, or if tleap cannot load the force fields
the builders use.
"""

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

LEAPRC = [
    'leaprc.protein.ff19SB',
    'leaprc.water.opc',
    'leaprc.gaff2',
    'leaprc.DNA.OL21',
    'leaprc.GLYCAM_06j-1',
]


def main() -> int:
    """Return 0 if every check passes."""
    failures = []

    installed = importlib.metadata.version('molecular-simulations')
    with urllib.request.urlopen(
        'https://pypi.org/pypi/molecular-simulations/json'
    ) as r:
        latest = json.load(r)['info']['version']
    print(f'molecular-simulations {installed} (latest on PyPI: {latest})')
    if installed != latest:
        failures.append(f'installed {installed}, not the latest release {latest}')

    import numpy

    print(f'numpy {numpy.__version__}')
    if int(numpy.__version__.split('.')[0]) < 2:
        failures.append(f'numpy {numpy.__version__} < 2')

    tleap = Path(sys.prefix) / 'bin' / 'tleap'
    script = ''.join(f'source {name}\n' for name in LEAPRC) + 'quit\n'
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'leap.in').write_text(script)
        result = subprocess.run(
            [str(tleap), '-f', 'leap.in'],
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**os.environ, 'AMBERHOME': sys.prefix},
        )
    summary = [line for line in result.stdout.splitlines() if 'Exiting LEaP' in line]
    print(f'tleap: {summary[-1] if summary else result.stderr.strip()[:200]}')
    if result.returncode != 0 or not summary or 'Errors = 0' not in summary[-1]:
        failures.append('tleap could not load the force fields')

    for failure in failures:
        print(f'FAIL: {failure}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
