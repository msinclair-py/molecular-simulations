"""Check which bundled skills an agent session discovered.

Claude Code: reads the ``--output-format stream-json`` events of
``claude -p`` and looks at the ``init`` event, which lists the session's
skills before any model call. The run itself is expected to fail
authentication (CI uses a dummy API key), so only the init event matters.

Codex: reads the JSON of ``codex debug prompt-input``, which renders the
model prompt without sending it; each discovered skill is listed there as a
line ``- <name>: <description> (file: ...)``.

Usage:
    check_agent_skills.py OUTPUT --expect NAME [NAME ...] [--namespace NS]
    check_agent_skills.py OUTPUT --expect NAME [NAME ...] --none
    check_agent_skills.py OUTPUT --agent codex --expect NAME [NAME ...]
"""

import argparse
import json
import re
import sys
from collections.abc import Iterator

CODEX_SKILL = re.compile(r'^- ([a-z0-9-]+): .*\(file: ', re.MULTILINE)


def claude_skills(path: str) -> list[str] | None:
    """Return the skills in a Claude Code init event, or None without one."""
    with open(path) as f:
        for line in f:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('subtype') == 'init':
                return event.get('skills', [])
    return None


def strings(value: object) -> Iterator[str]:
    """Yield every string nested in parsed JSON."""
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def codex_skills(path: str) -> list[str] | None:
    """Return the skills listed in a Codex prompt dump, or None if unreadable."""
    try:
        with open(path) as f:
            prompt = json.load(f)
    except ValueError:
        return None
    return [name for text in strings(prompt) for name in CODEX_SKILL.findall(text)]


def main() -> int:
    """Return 0 if the session loaded exactly the expected skills."""
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='claude -p events or codex prompt dump')
    parser.add_argument('--agent', choices=('claude', 'codex'), default='claude')
    parser.add_argument('--expect', nargs='+', required=True, help='skill names')
    parser.add_argument(
        '--namespace', help='plugin namespace, e.g. molecular-simulations'
    )
    parser.add_argument('--none', action='store_true', help='expect none loaded')
    args = parser.parse_args()

    reader = claude_skills if args.agent == 'claude' else codex_skills
    skills = reader(args.output)
    if skills is None:
        print(f'FAIL: no {args.agent} session output to read', file=sys.stderr)
        return 1

    prefix = f'{args.namespace}:' if args.namespace else ''
    wanted = {prefix + name for name in args.expect}
    ours = {skill for skill in skills if skill.split(':')[-1] in args.expect}
    if args.none:
        ok, expected = not ours, set()
    else:
        ok, expected = ours == wanted, wanted
    status = 'ok' if ok else 'FAIL'
    print(f'{status}: loaded {sorted(ours)}, expected {sorted(expected)}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
