"""Compatibility re-export for Hopper FA3 kernels.

The FA3 implementations now live under ``ops.fa3_ws``.  This module preserves
the historical import path used by ``flash_api_v3.py`` and external callers.
"""

from .fa3_ws import kernels as _fa3_ws_kernels
from .fa3_ws.kernels import *  # noqa: F401,F403

__all__ = _fa3_ws_kernels.__all__


def __getattr__(name):
    return getattr(_fa3_ws_kernels, name)


def __dir__():
    return sorted(set(__all__) | set(globals()) | set(dir(_fa3_ws_kernels)))
