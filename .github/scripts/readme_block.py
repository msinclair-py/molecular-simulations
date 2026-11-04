"""Print the first ```bash block under a README heading.

Lets CI run the documented install commands verbatim.

Usage:
    readme_block.py README.md "### Using pixi"
"""

import sys


def main() -> int:
    """Print the block, or exit 1 if the heading or block is missing."""
    path, heading = sys.argv[1], sys.argv[2]
    with open(path) as f:
        lines = f.read().splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        print(f'Heading not found: {heading}', file=sys.stderr)
        return 1
    block = None
    for line in lines[start + 1 :]:
        if block is None:
            if line.startswith('#'):
                break
            if line.strip() == '```bash':
                block = []
        elif line.strip() == '```':
            print('\n'.join(block))
            return 0
        else:
            block.append(line)
    print(f'No bash block under: {heading}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
