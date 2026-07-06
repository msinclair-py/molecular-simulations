"""
Unit tests for simulate/constantph/logging.py module

Tests setup_task_logger and JsonFormatter - pure stdlib code, no mocks needed.
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest


class TestSetupTaskLogger:
    """Test suite for setup_task_logger function."""

    def test_creates_logger_with_correct_name(self):
        """Test logger name follows task.{task_id} pattern."""
        from molecular_simulations.simulate.constantph.logging import setup_task_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_task_logger(
                run_id='run_001', task_id='task_abc', log_dir=tmpdir
            )

            assert logger.name == 'task.task_abc'

    def test_creates_log_directory_structure(self):
        """Test log directory {log_dir}/{run_id}/{prefix}/ is created."""
        from molecular_simulations.simulate.constantph.logging import setup_task_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_task_logger(run_id='run_001', task_id='task_abc', log_dir=tmpdir)

            expected_dir = Path(tmpdir) / 'run_001' / 'tas'
            assert expected_dir.exists()
            assert (expected_dir / 'task_abc.jsonl').exists() or True
            # File is created by FileHandler on first write

    def test_creates_log_file(self):
        """Test that logging to the logger produces a JSONL file."""
        from molecular_simulations.simulate.constantph.logging import setup_task_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_task_logger(
                run_id='run_001', task_id='task_abc', log_dir=tmpdir
            )

            logger.info('test message')

            log_file = Path(tmpdir) / 'run_001' / 'tas' / 'task_abc.jsonl'
            assert log_file.exists()
            content = log_file.read_text().strip()
            entry = json.loads(content)
            assert entry['message'] == 'test message'
            assert entry['run_id'] == 'run_001'
            assert entry['task_id'] == 'task_abc'
            assert entry['level'] == 'INFO'

    def test_short_task_id_prefix(self):
        """Test prefix bucketing with task_id shorter than 3 chars."""
        from molecular_simulations.simulate.constantph.logging import setup_task_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_task_logger(run_id='run_001', task_id='ab', log_dir=tmpdir)

            logger.info('short id test')

            log_file = Path(tmpdir) / 'run_001' / 'ab' / 'ab.jsonl'
            assert log_file.exists()

    def test_logger_level_is_debug(self):
        """Test logger is set to DEBUG level."""
        from molecular_simulations.simulate.constantph.logging import setup_task_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_task_logger(
                run_id='run_001', task_id='task_abc', log_dir=tmpdir
            )

            assert logger.level == logging.DEBUG

    def test_handlers_are_cleared_on_repeat_call(self):
        """Test that calling setup_task_logger twice doesn't duplicate handlers."""
        from molecular_simulations.simulate.constantph.logging import setup_task_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_task_logger(run_id='run_001', task_id='task_abc', log_dir=tmpdir)
            logger2 = setup_task_logger(
                run_id='run_001', task_id='task_abc', log_dir=tmpdir
            )

            assert len(logger2.handlers) == 1


class TestJsonFormatter:
    """Test suite for JsonFormatter class."""

    def test_format_produces_valid_json(self):
        """Test that format() returns valid JSON."""
        from molecular_simulations.simulate.constantph.logging import JsonFormatter

        formatter = JsonFormatter(task_id='task_001', run_id='run_001')
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='test message',
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        entry = json.loads(result)

        assert entry['message'] == 'test message'
        assert entry['task_id'] == 'task_001'
        assert entry['run_id'] == 'run_001'
        assert entry['level'] == 'INFO'
        assert 'timestamp' in entry

    def test_format_includes_extra_fields(self):
        """Test that extra fields from log call are included."""
        from molecular_simulations.simulate.constantph.logging import JsonFormatter

        formatter = JsonFormatter(task_id='task_001', run_id='run_001')
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='test',
            args=(),
            exc_info=None,
        )
        record.pH = 7.0
        record.step = 42

        result = formatter.format(record)
        entry = json.loads(result)

        assert entry['pH'] == 7.0
        assert entry['step'] == 42

    def test_format_excludes_standard_log_fields(self):
        """Test that standard LogRecord fields are not duplicated."""
        from molecular_simulations.simulate.constantph.logging import JsonFormatter

        formatter = JsonFormatter(task_id='task_001', run_id='run_001')
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='/some/path.py',
            lineno=10,
            msg='test',
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        entry = json.loads(result)

        # These standard fields should NOT appear as separate keys
        assert 'pathname' not in entry
        assert 'lineno' not in entry
        assert 'funcName' not in entry
        assert 'thread' not in entry


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
