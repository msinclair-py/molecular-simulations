#!/usr/bin/env bash
# Run a README install recipe verbatim, then check the environment it built.
#
# Usage: test_install_recipe.sh conda|pixi
#   conda: needs `conda` (with `conda activate` available in this shell).
#   pixi:  needs `pixi`.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$REPO/.github/scripts"
case "$1" in
    conda) heading='### Using conda / mamba' ;;
    pixi) heading='### Using pixi' ;;
    *) echo "usage: $0 conda|pixi" >&2; exit 2 ;;
esac

block="$(python3 "$SCRIPTS/readme_block.py" "$REPO/README.md" "$heading")"
echo "--- README recipe ($1):"
echo "$block"
echo '---'
cd "$(mktemp -d)"
# A non-interactive shell lacks the `conda activate` function; load it.
if [ "$1" = conda ] && [ "$(type -t conda)" != function ]; then
    eval "$(conda shell.bash hook)"
fi
eval "$block"

if [ "$1" = conda ]; then
    python "$SCRIPTS/check_install.py"
    echo '--- this checkout installs into the recipe environment'
    python -m pip install -q "$REPO"
    python -c 'import molecular_simulations, importlib.metadata as m; print("installed checkout", m.version("molecular-simulations"))'
else
    pixi run python "$SCRIPTS/check_install.py"
fi
