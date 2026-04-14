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
import traceback
from pathlib import Path

import torch


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


# prevision check logger

_precision_logger = None
precision_config = {
    "enabled": False,
    "rtol": 1e-4,
    "atol": 1e-5,
    "log_once": True,
    "logged_ops": set(),
}


def setup_precision_logging(path=None):
    global _precision_logger
    _precision_logger = logging.getLogger("flag_gems.precision")
    _remove_file_handlers(_precision_logger)

    filename = Path(path or Path.home() / ".flaggems/precision.log")
    filename.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(filename, mode="w")
    handler._flaggems_owned = True

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    _precision_logger.setLevel(logging.DEBUG)
    _precision_logger.addHandler(handler)
    _precision_logger.propagate = False


def get_precision_logger():
    return _precision_logger


def get_tensor_info(t):
    if isinstance(t, torch.Tensor):
        return f"{tuple(t.shape)}:{t.dtype}"
    elif isinstance(t, (list, tuple)):
        infos = [get_tensor_info(x) for x in t]
        return [i for i in infos if i]
    return None


def get_call_location():
    for frame in traceback.extract_stack():
        if "flag_gems" not in frame.filename and "torch" not in frame.filename:
            return f"{frame.filename}:{frame.lineno}"
    return "unknown"


def compare_outputs(fg_out, pt_out, rtol, atol):
    if isinstance(fg_out, torch.Tensor) and isinstance(pt_out, torch.Tensor):
        if fg_out.shape != pt_out.shape:
            return False, {
                "error": "shape_mismatch",
                "fg": tuple(fg_out.shape),
                "pt": tuple(pt_out.shape),
            }
        try:
            fg = fg_out.detach().float()
            pt = pt_out.detach().float()
            abs_diff = torch.abs(fg - pt)
            max_abs = abs_diff.max().item()
            max_rel = (abs_diff / (torch.abs(pt) + 1e-12)).max().item()
            is_close = torch.allclose(fg, pt, rtol=rtol, atol=atol)
            return is_close, {"max_abs": max_abs, "max_rel": max_rel}
        except:
            return True, {}
    elif isinstance(fg_out, (tuple, list)) and isinstance(pt_out, (tuple, list)):
        for i, (fg, pt) in enumerate(zip(fg_out, pt_out)):
            ok, info = compare_outputs(fg, pt, rtol, atol)
            if not ok:
                info["index"] = i
                return False, info
    return True, {}


def enable_precision_check(
    rtol=1e-4, atol=1e-5, log_once=True, max_checks=10, path=None
):
    setup_precision_logging(path)
    precision_config.update(
        {
            "enabled": True,
            "rtol": rtol,
            "atol": atol,
            "log_once": log_once,
            "max_checks": max_checks,
            "logged_ops": set(),
        }
    )


def disable_precision_check():
    """close precision log file and disable precision check."""
    precision_config["enabled"] = False
