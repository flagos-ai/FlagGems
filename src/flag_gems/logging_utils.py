"""Logging helpers for flag_gems.

Notes
-----
1) When you enter through the public APIs `enable`, `only_enable`, or the
    context manager `use_gems`, the `record` flag controls whether op-level
    logging is enabled and where it is written.
2) If you import `flag_gems` and call operators directly (e.g., `flag_gems.mm`)
    without those helpers, call `setup_flaggems_logging()` yourself to initialize
    the logging mode and file handler.
"""

import logging
from pathlib import Path


class LogOncePerLocationFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.logged_locations = set()

    def filter(self, record):
        key = (record.pathname, record.lineno)
        if key in self.logged_locations:
            return False
        self.logged_locations.add(key)
        return True


def _remove_file_handlers(logger: logging.Logger):
    # Remove and close only the FileHandlers created by setup_flaggems_logging.
    # This avoids touching unrelated FileHandlers attached by other modules.
    removed = False
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "_flaggems_owned", False):
            h.close()
            logger.removeHandler(h)
            removed = True
    return removed


def setup_flaggems_logging(path=None, record=True, once=False):
    logger = logging.getLogger("flag_gems")

    # If caller asks for recording, refresh file handler (new path overwrites old).
    if record:
        _remove_file_handlers(logger)
    else:
        return

    filename = Path(path or Path.home() / ".flaggems/oplist.log")
    handler = logging.FileHandler(filename, mode="w")
    handler._flaggems_owned = True

    if once:
        handler.addFilter(LogOncePerLocationFilter())

    formatter = logging.Formatter("[%(levelname)s] %(name)s.%(funcName)s: %(message)s")
    handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False


def teardown_flaggems_logging(logger: logging.Logger | None = None):
    """Remove file handlers for the flag_gems logger (used on context exit)."""

    logger = logger or logging.getLogger("flag_gems")
    _remove_file_handlers(logger)
