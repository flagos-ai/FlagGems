import functools
import os
import shutil
from pathlib import Path


@functools.lru_cache(maxsize=None)  # this is the same as functools.cache in Python 3.9+
def cache_dir_path() -> Path:
    """Return the cache directory for generated files in flaggems."""
    _cache_dir = os.environ.get("FLAGGEMS_CACHE_DIR")
    if _cache_dir is None:
        _cache_dir = Path.home() / ".flaggems"
    else:
        _cache_dir = Path(_cache_dir)
    return _cache_dir


def cache_dir() -> Path:
    """Return cache directory for generated files in flaggems. Create it if it does not exist."""
    _cache_dir = cache_dir_path()
    os.makedirs(_cache_dir, exist_ok=True)
    return _cache_dir


def code_cache_dir() -> Path:
    _code_cache_dir = cache_dir() / "code_cache"
    os.makedirs(_code_cache_dir, exist_ok=True)
    return _code_cache_dir


def config_cache_dir() -> Path:
    _config_cache_dir = cache_dir() / "config_cache"
    os.makedirs(_config_cache_dir, exist_ok=True)
    return _config_cache_dir


def clear_cache():
    """Clear the cache directory for code cache."""
    _cache_dir = cache_dir_path()
    shutil.rmtree(_cache_dir)
