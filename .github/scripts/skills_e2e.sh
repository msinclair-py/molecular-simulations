#!/usr/bin/env bash
# End-to-end check that every skill install route is discovered by Claude Code.
#
# Requires: `molsim` from an installed wheel on PATH, the `claude` CLI, and
# ANTHROPIC_API_KEY set to a dummy value. Each session fails authentication
# after skill discovery, which is expected; the init event decides.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK="python $REPO/.github/scripts/check_agent_skills.py"
WORK="$(mktemp -d)"
read -r -a SKILLS <<< "$(molsim skills list | tr '\n' ' ')"
echo "Bundled skills: ${SKILLS[*]}"

# Start a session in $1 and check its skills; remaining args go to the checker.
session() {
    local dir="$1"
    shift
    (cd "$dir" && claude -p x --output-format stream-json --verbose \
        ${CLAUDE_ARGS[@]+"${CLAUDE_ARGS[@]}"} > "$WORK/events.jsonl" 2> "$WORK/stderr.txt") || true
    $CHECK "$WORK/events.jsonl" --expect "${SKILLS[@]}" "$@" \
        || { cat "$WORK/stderr.txt"; return 1; }
}

mkdir -p "$WORK/empty" "$WORK/project"
git -C "$WORK/project" init -q
CLAUDE_ARGS=()

echo '--- control: nothing installed'
session "$WORK/empty" --none

echo '--- molsim skills install --agent claude (user, symlinks)'
molsim skills install --agent claude
session "$WORK/empty"
molsim skills uninstall --agent claude

echo '--- molsim skills install --agent claude --copy (user, copies)'
molsim skills install --agent claude --copy
session "$WORK/empty"
molsim skills uninstall --agent claude

echo '--- molsim skills install --agent claude --scope project'
(cd "$WORK/project" && molsim skills install --agent claude --scope project)
session "$WORK/project"

echo '--- claude --plugin-dir <repo>'
CLAUDE_ARGS=(--plugin-dir "$REPO")
session "$WORK/empty" --namespace molecular-simulations
CLAUDE_ARGS=()

echo '--- plugin installed from the repo marketplace'
claude plugin marketplace add "$REPO"
claude plugin install molecular-simulations@molecular-simulations --scope user
session "$WORK/empty" --namespace molecular-simulations

echo 'All skill install routes discovered.'
