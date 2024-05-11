import functools
import os
from pathlib import Path
import shutil


@functools.cache
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


def clear_cache():
    """Clear the cache directory for code cache."""
    _cache_dir = cache_dir_path()
    shutil.rmtree(_cache_dir)
