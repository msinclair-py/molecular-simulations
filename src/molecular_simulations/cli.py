"""Command-line interface for molecular-simulations.

Currently provides ``molsim skills``, which installs the agent skills bundled
with the package into the directories that coding agents scan:

    molsim skills install --agent claude    # ~/.claude/skills
    molsim skills install --agent codex     # ~/.agents/skills
    molsim skills install --agent claude --scope project   # ./.claude/skills
    molsim skills install --dest .github/skills            # any other agent
    molsim skills list | path | uninstall

Skills are symlinked by default, so they track the installed package version;
``--copy`` writes independent copies instead.
"""

import argparse
import shutil
import sys
from pathlib import Path

AGENT_DIRS = {'claude': '.claude/skills', 'codex': '.agents/skills'}
# Written into copied skills so uninstall can recognize them.
SKILL_MARKER = '.molsim-skill'


def bundled_skills_dir() -> Path:
    """Locate the skills shipped with the package.

    Wheels carry them in ``molecular_simulations/skills``; editable installs
    read the repository's top-level ``skills`` directory.

    Returns:
        Directory holding one ``<name>/SKILL.md`` folder per skill.

    Raises:
        FileNotFoundError: If no skills directory is found.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / 'skills', here.parents[1] / 'skills'):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError('No bundled skills found in molecular_simulations')


def bundled_skills() -> list[Path]:
    """Return the bundled skill directories, sorted by name."""
    return sorted(p.parent for p in bundled_skills_dir().glob('*/SKILL.md'))


def target_dir(agent: str | None, scope: str, dest: str | None) -> Path:
    """Resolve the skills directory to install into.

    Args:
        agent: Agent name, a key of ``AGENT_DIRS``.
        scope: ``user`` (home directory) or ``project`` (current directory).
        dest: Explicit directory; overrides ``agent`` and ``scope``.

    Returns:
        Target directory.
    """
    if dest is not None:
        return Path(dest).expanduser()
    base = Path.home() if scope == 'user' else Path.cwd()
    return base / AGENT_DIRS[agent]


def is_ours(path: Path, skill: Path) -> bool:
    """Check whether ``path`` is an install of ``skill`` by this command."""
    if path.is_symlink():
        return path.resolve() == skill.resolve()
    return (path / SKILL_MARKER).is_file()


def install(target: Path, copy: bool = False, force: bool = False) -> list[str]:
    """Install every bundled skill into ``target``.

    Args:
        target: Agent skills directory.
        copy: Copy the skills instead of symlinking them.
        force: Replace existing entries that this command did not create.

    Returns:
        Messages, one per skill.

    Raises:
        FileExistsError: If an unrelated entry exists and ``force`` is False.
    """
    target.mkdir(parents=True, exist_ok=True)
    skills = bundled_skills()
    clashes = [
        target / s.name
        for s in skills
        if (target / s.name).exists() or (target / s.name).is_symlink()
        if not is_ours(target / s.name, s)
    ]
    if clashes and not force:
        raise FileExistsError(
            'Not replacing existing skills (use --force): '
            + ', '.join(str(p) for p in clashes)
        )

    messages = []
    for skill in skills:
        path = target / skill.name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        if copy:
            shutil.copytree(skill, path)
            (path / SKILL_MARKER).touch()
            messages.append(f'copied {skill.name} -> {path}')
        else:
            try:
                path.symlink_to(skill.resolve(), target_is_directory=True)
                messages.append(f'linked {path} -> {skill.resolve()}')
            except OSError:
                # e.g. Windows without symlink privileges.
                shutil.copytree(skill, path)
                (path / SKILL_MARKER).touch()
                messages.append(f'copied {skill.name} -> {path} (symlink failed)')
    return messages


def uninstall(target: Path) -> list[str]:
    """Remove skills this command installed into ``target``.

    Args:
        target: Agent skills directory.

    Returns:
        Messages, one per removed skill.
    """
    messages = []
    for skill in bundled_skills():
        path = target / skill.name
        if not (path.exists() or path.is_symlink()) or not is_ours(path, skill):
            continue
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
        messages.append(f'removed {path}')
    return messages


def main(argv: list[str] | None = None) -> int:
    """Run the ``molsim`` command.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog='molsim', description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    skills = commands.add_parser('skills', help='Manage bundled agent skills')
    actions = skills.add_subparsers(dest='action', required=True)
    actions.add_parser('list', help='List bundled skills')
    actions.add_parser('path', help='Print the bundled skills directory')
    for name, text in (('install', 'Install skills'), ('uninstall', 'Remove skills')):
        sub = actions.add_parser(name, help=f'{text} for an agent')
        sub.add_argument('--agent', choices=sorted(AGENT_DIRS), default='claude')
        sub.add_argument('--scope', choices=('user', 'project'), default='user')
        sub.add_argument('--dest', help='Skills directory for any other agent')
        if name == 'install':
            sub.add_argument('--copy', action='store_true', help='Copy, not link')
            sub.add_argument('--force', action='store_true', help='Replace clashes')
    args = parser.parse_args(argv)

    if args.action == 'list':
        for skill in bundled_skills():
            print(skill.name)
        return 0
    if args.action == 'path':
        print(bundled_skills_dir())
        return 0

    target = target_dir(args.agent, args.scope, args.dest)
    try:
        if args.action == 'install':
            messages = install(target, copy=args.copy, force=args.force)
        else:
            messages = uninstall(target)
    except FileExistsError as err:
        print(err, file=sys.stderr)
        return 1
    print('\n'.join(messages) or f'No molecular-simulations skills in {target}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
