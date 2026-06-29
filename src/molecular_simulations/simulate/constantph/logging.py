import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_task_logger(run_id: str, task_id: str, log_dir: str) -> logging.Logger:
    """Create a logger for a specific task with a deterministic path.

    Args:
        run_id: Identifier for the overall run; used as a top-level directory.
        task_id: Identifier for the task; used (with a prefix bucket) to form
            the log file path.
        log_dir: Base directory under which log files are written.

    Returns:
        Configured logging.Logger that writes JSONL records for the task.
    """
    # Hierarchical structure: logs/{run_id}/{task_id_prefix}/{task_id}.jsonl
    # The prefix bucketing prevents directory explosion
    prefix = task_id[:3] if len(task_id) >= 3 else task_id
    task_log_dir = Path(log_dir) / run_id / prefix
    task_log_dir.mkdir(parents=True, exist_ok=True)

    log_path = task_log_dir / f'{task_id}.jsonl'

    logger = logging.getLogger(f'task.{task_id}')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path)
    handler.setFormatter(JsonFormatter(task_id=task_id, run_id=run_id))
    logger.addHandler(handler)

    return logger


class JsonFormatter(logging.Formatter):
    """Structured JSON logging formatter for easy aggregation.

    Formats log records as JSON objects, injecting the run and task identifiers
    along with any extra fields supplied via ``extra={}``.

    Attributes:
        task_id: Identifier for the task, emitted with each record.
        run_id: Identifier for the run, emitted with each record.
    """

    def __init__(self, task_id: str, run_id: str):
        super().__init__()
        self.task_id = task_id
        self.run_id = run_id

    def format(self, record) -> str:
        """Render a log record as a JSON string.

        Args:
            record: The logging.LogRecord to serialize.

        Returns:
            JSON-encoded string containing the timestamp, run/task identifiers,
            level, message, and any extra fields attached to the record.
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'run_id': self.run_id,
            'task_id': self.task_id,
            'level': record.levelname,
            'message': record.getMessage(),
        }

        # Merge in any extra fields passed via extra={}
        for key, val in record.__dict__.items():
            if key not in (
                'msg',
                'args',
                'levelname',
                'levelno',
                'pathname',
                'filename',
                'module',
                'lineno',
                'funcName',
                'created',
                'msecs',
                'relativeCreated',
                'thread',
                'threadName',
                'processName',
                'process',
                'message',
                'exc_info',
                'exc_text',
                'stack_info',
                'name',
            ):
                entry[key] = val

        return json.dumps(entry)
