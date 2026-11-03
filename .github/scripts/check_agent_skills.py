"""Check which bundled skills a Claude Code session discovered.

Reads the ``--output-format stream-json`` events of ``claude -p`` and looks
at the ``init`` event, which lists the session's skills before any model
call. The run itself is expected to fail authentication (CI uses a dummy
API key), so only the init event matters.

Usage:
    check_agent_skills.py EVENTS --expect NAME [NAME ...] [--namespace NS]
    check_agent_skills.py EVENTS --expect NAME [NAME ...] --none
"""

import argparse
import json
import sys


def main() -> int:
    """Return 0 if the session loaded exactly the expected skills."""
    parser = argparse.ArgumentParser()
    parser.add_argument('events', help='stream-json output of claude -p')
    parser.add_argument('--expect', nargs='+', required=True, help='skill names')
    parser.add_argument(
        '--namespace', help='plugin namespace, e.g. molecular-simulations'
    )
    parser.add_argument('--none', action='store_true', help='expect none loaded')
    args = parser.parse_args()

    init = None
    with open(args.events) as f:
        for line in f:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('subtype') == 'init':
                init = event
                break
    if init is None:
        print(
            'FAIL: no init event; Claude Code did not start a session', file=sys.stderr
        )
        return 1

    prefix = f'{args.namespace}:' if args.namespace else ''
    wanted = {prefix + name for name in args.expect}
    ours = {
        skill for skill in init.get('skills', []) if skill.split(':')[-1] in args.expect
    }
    if args.none:
        ok, expected = not ours, set()
    else:
        ok, expected = ours == wanted, wanted
    status = 'ok' if ok else 'FAIL'
    print(f'{status}: loaded {sorted(ours)}, expected {sorted(expected)}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
