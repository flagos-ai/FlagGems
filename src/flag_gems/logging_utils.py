# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


# Precision check data file writer
# We intentionally use a plain file rather than the logging framework here
# because precision results are structured data, not runtime diagnostics.

_precision_file = None
precision_config = {
    "enabled": False,
    "rtol": 1e-4,
    "atol": 1e-5,
    "log_once": True,
    "logged_ops": set(),
}


def setup_precision_logging(path=None):
    """Open (or reopen) the precision results data file for writing."""
    global _precision_file
    _close_precision_file()

    filename = Path(path or Path.home() / ".flaggems/precision.log")
    filename.parent.mkdir(parents=True, exist_ok=True)
    _precision_file = open(filename, mode="w")  # noqa: SIM115


def _close_precision_file():
    """Close the precision data file if open."""
    global _precision_file
    if _precision_file is not None and not _precision_file.closed:
        _precision_file.close()
    _precision_file = None


def write_precision_result(record: dict):
    """Write a precision check result as a JSON line to the data file.

    Each call appends one JSON object (JSONL format) so the output can be
    post-processed with standard tools such as ``jq`` or Python's ``json``
    module.
    """
    import json
    from datetime import datetime, timezone

    if _precision_file is not None and not _precision_file.closed:
        record["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        _precision_file.write(json.dumps(record, default=str) + "\n")
        _precision_file.flush()


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

            # Mask out positions where both are NaN or both are the same Inf
            # These are not precision errors — they are semantically identical.
            both_nan = torch.isnan(fg) & torch.isnan(pt)
            both_same_inf = torch.isinf(fg) & torch.isinf(pt) & (fg == pt)
            ignore_mask = both_nan | both_same_inf

            # If all elements are in the ignore set, they match perfectly
            if ignore_mask.all():
                return True, {"max_abs": 0.0, "max_rel": 0.0}

            # Check for mismatched NaN/Inf (one side has it, the other doesn't)
            fg_special = torch.isnan(fg) | torch.isinf(fg)
            pt_special = torch.isnan(pt) | torch.isinf(pt)
            mismatch_special = (fg_special != pt_special) & ~ignore_mask
            if mismatch_special.any():
                # Find first mismatch for reporting
                idx = mismatch_special.nonzero(as_tuple=False)[0]
                return False, {
                    "error": "special_value_mismatch",
                    "fg": fg[tuple(idx)].item(),
                    "pt": pt[tuple(idx)].item(),
                }

            # Compare only finite, non-ignored elements
            valid = ~ignore_mask & ~fg_special
            if valid.any():
                abs_diff = torch.abs(fg[valid] - pt[valid])
                max_abs = abs_diff.max().item()
                denom = torch.abs(pt[valid]) + 1e-12
                max_rel = (abs_diff / denom).max().item()
            else:
                max_abs = 0.0
                max_rel = 0.0

            is_close = (
                torch.allclose(fg[valid], pt[valid], rtol=rtol, atol=atol)
                if valid.any()
                else True
            )
            return is_close, {"max_abs": max_abs, "max_rel": max_rel}
        except Exception as e:
            return True, {"error": "exception", "message": str(e)}
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
    """Close precision data file and disable precision check."""
    _close_precision_file()
    precision_config["enabled"] = False
