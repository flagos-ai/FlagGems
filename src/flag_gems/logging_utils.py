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
import sys
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


_ORIGINAL_LOGGER_DEBUG = logging.Logger.debug
_COMPILE_SAFE_DEBUG_INSTALLED = False


def install_compile_safe_debug_for_flaggems():
    """Skip flag_gems debug logging while torch.compile is tracing.

    Dynamo can graph-break on python logging internals when debug calls are
    executed inside traced paths. We keep normal debug behavior outside compile.
    """

    global _COMPILE_SAFE_DEBUG_INSTALLED
    if _COMPILE_SAFE_DEBUG_INSTALLED:
        return

    def _compile_safe_debug(self, msg, *args, **kwargs):
        if self.name.startswith("flag_gems"):
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None:
                compiler = getattr(torch_mod, "compiler", None)
                if compiler is not None and compiler.is_compiling():
                    return
        return _ORIGINAL_LOGGER_DEBUG(self, msg, *args, **kwargs)

    logging.Logger.debug = _compile_safe_debug
    _COMPILE_SAFE_DEBUG_INSTALLED = True


def install_dynamo_ignore_logger_methods():
    """Tell torch._dynamo to treat logging logger methods as no-ops."""

    try:
        import torch._dynamo.config as dynamo_config
    except Exception:
        return

    dynamo_config.ignore_logger_methods.update(
        {
            logging.Logger.debug,
            logging.Logger.info,
            logging.Logger.warning,
            logging.Logger.error,
            logging.Logger.critical,
            logging.Logger.exception,
            logging.Logger.log,
        }
    )


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
