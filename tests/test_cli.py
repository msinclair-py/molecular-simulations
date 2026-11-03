"""Tests for the molsim CLI and the bundled agent skills."""

import re

import pytest

from molecular_simulations import cli

pytestmark = pytest.mark.fast


def frontmatter(path):
    """Parse the name/description frontmatter of a SKILL.md."""
    text = path.read_text()
    assert text.startswith('---\n'), f'{path} must start with frontmatter'
    block = text.split('---\n')[1]
    return dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)


class TestBundledSkills:
    """Every bundled skill follows the Agent Skills specification."""

    def test_expected_skills(self):
        names = [skill.name for skill in cli.bundled_skills()]
        assert names[0] == 'molecular-simulations'
        assert all(name.startswith('molsim-') for name in names[1:])
        assert len(names) == 7

    @pytest.mark.parametrize('skill', cli.bundled_skills(), ids=lambda p: p.name)
    def test_spec(self, skill):
        meta = frontmatter(skill / 'SKILL.md')
        name = meta['name']
        assert name == skill.name
        assert re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', name) and len(name) <= 64
        assert 0 < len(meta['description']) <= 1024
        assert len((skill / 'SKILL.md').read_text().splitlines()) < 500


class TestInstall:
    """molsim skills install/uninstall into a skills directory."""

    def test_symlink_install_and_uninstall(self, tmp_path):
        assert cli.main(['skills', 'install', '--dest', str(tmp_path)]) == 0
        for skill in cli.bundled_skills():
            link = tmp_path / skill.name
            assert link.is_symlink()
            assert link.resolve() == skill.resolve()
            assert (link / 'SKILL.md').is_file()

        # Reinstalling over our own links is fine.
        assert cli.main(['skills', 'install', '--dest', str(tmp_path)]) == 0

        assert cli.main(['skills', 'uninstall', '--dest', str(tmp_path)]) == 0
        assert list(tmp_path.iterdir()) == []

    def test_copy_install(self, tmp_path):
        cli.install(tmp_path, copy=True)
        skill = cli.bundled_skills()[1]
        copied = tmp_path / skill.name
        assert copied.is_dir() and not copied.is_symlink()
        assert (copied / 'SKILL.md').read_text() == (skill / 'SKILL.md').read_text()

        cli.uninstall(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_existing_skill_is_not_replaced_without_force(self, tmp_path, capsys):
        other = tmp_path / 'molsim-build-systems'
        other.mkdir()
        (other / 'SKILL.md').write_text('someone else')

        assert cli.main(['skills', 'install', '--dest', str(tmp_path)]) == 1
        assert 'use --force' in capsys.readouterr().err
        assert (other / 'SKILL.md').read_text() == 'someone else'
        # Uninstall leaves skills it did not install alone.
        cli.uninstall(tmp_path)
        assert other.is_dir()

        cli.install(tmp_path, force=True)
        assert other.is_symlink()

    @pytest.mark.parametrize(
        'agent, scope, expected',
        [
            ('claude', 'user', 'home/.claude/skills'),
            ('codex', 'user', 'home/.agents/skills'),
            ('claude', 'project', 'project/.claude/skills'),
        ],
    )
    def test_target_dir(self, tmp_path, monkeypatch, agent, scope, expected):
        monkeypatch.setattr(cli.Path, 'home', lambda: tmp_path / 'home')
        project = tmp_path / 'project'
        project.mkdir()
        monkeypatch.chdir(project)
        assert cli.target_dir(agent, scope, None) == tmp_path / expected
        assert cli.target_dir(agent, scope, '/x/skills') == cli.Path('/x/skills')

    def test_list(self, capsys):
        assert cli.main(['skills', 'list']) == 0
        assert capsys.readouterr().out.split() == [
            skill.name for skill in cli.bundled_skills()
        ]
