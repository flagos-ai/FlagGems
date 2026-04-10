"""Logging configuration for combination tests.

Provides a structured JSON Lines (JSONL) logging system for combination tests,
following the same patterns as ``flag_gems.logging_utils``:

* Named logger under the ``flag_gems`` namespace.
* ``FileHandler`` with a ``_flaggems_owned`` marker for clean teardown.
* ``JsonFormatter`` that writes one JSON object per line.

Usage::

    # Automatic (via conftest.py pytest hooks):
    pytest tests/combination/ --combo-log-dir=./my_logs

    # Manual:
    from tests.combination.logging_config import TestLogger
    tl = TestLogger(log_dir="./my_logs")
    tl.log_test_start("test_foo", {"batch": 4})
    tl.log_test_result("test_foo", "passed", duration=1.23)
    tl.log_session_summary(total=10, passed=9, failed=1, skipped=0, duration=5.0)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGGER_NAME = "flag_gems.combination_test"
DEFAULT_LOG_DIR = Path("combination_test_logs")


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Format each log record as a single-line JSON object (JSONL)."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge extra fields injected via ``extra={"test_data": {...}}``
        if hasattr(record, "test_data") and isinstance(record.test_data, dict):
            log_data.update(record.test_data)
        return json.dumps(log_data, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Setup / Teardown  (mirrors logging_utils.py patterns)
# ---------------------------------------------------------------------------


def _remove_file_handlers(logger):
    """Remove and close FileHandlers owned by this module."""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(
            h, "_flaggems_owned", False
        ):
            h.close()
            logger.removeHandler(h)


def setup_combination_logging(log_dir=None, session_id=None):
    """Initialise the combination-test logger with a JSONL FileHandler.

    Args:
        log_dir: Directory for log files.  Created automatically.
        session_id: Optional session identifier embedded in the filename.

    Returns:
        (logger, log_file_path)
    """
    logger = logging.getLogger(LOGGER_NAME)
    # Avoid duplicate handlers on repeated calls.
    _remove_file_handlers(logger)

    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{session_id}" if session_id else ""
    filename = log_dir / f"combo_test_{ts}{suffix}.jsonl"

    handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
    handler._flaggems_owned = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter())

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger, filename


def teardown_combination_logging():
    """Remove file handlers for the combination-test logger."""
    logger = logging.getLogger(LOGGER_NAME)
    _remove_file_handlers(logger)


# ---------------------------------------------------------------------------
# TestLogger — convenience wrapper
# ---------------------------------------------------------------------------


class TestLogger:
    """High-level interface for logging combination-test events.

    Each ``log_*`` method writes a structured JSON record via the standard
    ``logging`` module so that the output is a valid JSONL file.
    """

    def __init__(self, log_dir=None, session_id=None):
        self.logger, self.log_file = setup_combination_logging(
            log_dir=log_dir, session_id=session_id
        )
        self._session_start = time.monotonic()
        self._counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0}

    # -- test lifecycle -----------------------------------------------------

    def log_test_start(self, test_name, params=None):
        """Record the start of a test."""
        self.logger.info(
            "test_start",
            extra={
                "test_data": {
                    "event": "test_start",
                    "test_name": test_name,
                    "params": params or {},
                }
            },
        )

    def log_test_result(self, test_name, outcome, duration=None, error_msg=None):
        """Record the outcome of a test (passed / failed / skipped / xfailed)."""
        outcome_key = outcome if outcome in self._counts else "failed"
        self._counts[outcome_key] = self._counts.get(outcome_key, 0) + 1

        data = {
            "event": "test_result",
            "test_name": test_name,
            "outcome": outcome,
            "duration": round(duration, 4) if duration is not None else None,
        }
        if error_msg:
            # Truncate long tracebacks for the JSON record.
            data["error_msg"] = error_msg[:2000]

        level = logging.WARNING if outcome == "failed" else logging.INFO
        self.logger.log(level, "test_result", extra={"test_data": data})

    # -- accuracy checks ----------------------------------------------------

    def log_accuracy_check(
        self,
        name,
        check_type,
        dtype=None,
        num_ops=None,
        expected_atol=None,
        expected_rtol=None,
        actual_max_error=None,
        actual_mean_error=None,
        passed=True,
        **extra,
    ):
        """Record the result of an accuracy comparison."""
        data = {
            "event": "accuracy_check",
            "name": name,
            "check_type": check_type,
            "dtype": str(dtype) if dtype is not None else None,
            "num_ops": num_ops,
            "expected_atol": expected_atol,
            "expected_rtol": expected_rtol,
            "actual_max_error": actual_max_error,
            "actual_mean_error": actual_mean_error,
            "passed": passed,
        }
        data.update(extra)
        level = logging.WARNING if not passed else logging.INFO
        self.logger.log(level, "accuracy_check", extra={"test_data": data})

    # -- numerical issues ---------------------------------------------------

    def log_numerical_issue(self, test_name, issue_type, details=None):
        """Record a numerical problem (NaN / Inf / gradient anomaly)."""
        self.logger.warning(
            "numerical_issue",
            extra={
                "test_data": {
                    "event": "numerical_issue",
                    "test_name": test_name,
                    "issue_type": issue_type,
                    "details": details or "",
                }
            },
        )

    # -- session summary ----------------------------------------------------

    def log_session_summary(self, total, passed, failed, skipped, duration=None):
        """Write an end-of-session summary record."""
        if duration is None:
            duration = round(time.monotonic() - self._session_start, 2)
        self.logger.info(
            "session_summary",
            extra={
                "test_data": {
                    "event": "session_summary",
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "pass_rate": round(passed / total * 100, 1) if total else 0,
                    "duration": duration,
                }
            },
        )

    def close(self):
        """Flush and remove handlers."""
        teardown_combination_logging()
