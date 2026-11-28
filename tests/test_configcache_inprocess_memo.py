"""What this file does:
It checks that `ConfigCache` serves repeated lookups from an in-process memo instead of
hitting the persistent backend on every call, while still observing configs written by
another process.

Why this matters:
`LibEntry.run` queries the config cache on every kernel launch. With the SQLite backend
each query opens a session and issues a SELECT, which dominates wall time for short
kernels. The memo removes that per-launch round trip.

How it is tested:
A fake `PersistantModel` counts backend round trips, so memo hits/misses are asserted
directly without a real database.
"""

import pytest
import triton

from flag_gems.utils.libentry import ConfigCache
from flag_gems.utils.models import PersistantModel


class CountingModel(PersistantModel):
    """Minimal in-memory `PersistantModel` that records how often it is queried."""

    def __init__(self):
        super().__init__()
        self.store = {}
        self.get_config_calls = 0
        self.put_config_calls = 0

    def get_config(self, name, key):
        self.get_config_calls += 1
        return self.store.get((name, tuple(key)))

    def put_config(self, name, key, config):
        self.put_config_calls += 1
        if isinstance(config, dict):
            config = triton.Config(config)
        self.store[(name, tuple(key))] = config

    def get_benchmark(self, name, keys, config):  # pragma: no cover - unused here
        return None

    def put_benchmark(self, name, keys, config, benchmark):  # pragma: no cover
        return None


def _config(**kwargs):
    return triton.Config(kwargs)


def test_repeated_hits_use_the_memo():
    model = CountingModel()
    cache = ConfigCache("table", model)
    key = (16, 32, 64)
    cfg = _config(BLOCK_SIZE=64)
    model.store[("table", key)] = cfg

    assert cache.get(key) is cfg
    assert model.get_config_calls == 1

    for _ in range(50):
        assert cache.get(key) is cfg
    # Still exactly one round trip for 51 lookups.
    assert model.get_config_calls == 1


def test_misses_are_not_memoized():
    """A config tuned by another process must still become visible."""
    model = CountingModel()
    cache = ConfigCache("table", model)
    key = (1, 2, 3)

    assert cache.get(key) is None
    assert cache.get(key) is None
    # Misses keep consulting the backend rather than caching "absent".
    assert model.get_config_calls == 2

    cfg = _config(BLOCK_SIZE=32)
    model.store[("table", key)] = cfg
    assert cache.get(key) is cfg
    assert model.get_config_calls == 3


def test_set_populates_memo_and_backend():
    model = CountingModel()
    cache = ConfigCache("table", model)
    key = (7, 8)
    cfg = _config(BLOCK_SIZE=128)

    cache.set(key, cfg)

    assert model.put_config_calls == 1
    # Served from the memo without a further backend read.
    assert cache.get(key) is cfg
    assert model.get_config_calls == 0


def test_contains_operator_uses_memo():
    model = CountingModel()
    cache = ConfigCache("table", model)
    key = (4, 5)
    cfg = _config(BLOCK_SIZE=16)
    cache.set(key, cfg)
    model.get_config_calls = 0

    assert key in cache
    assert (9, 9) not in cache
    # The hit was served from the memo; the miss went to the backend.
    assert model.get_config_calls == 1


def test_item_access_and_errors():
    model = CountingModel()
    cache = ConfigCache("table", model)
    key = (11, 12)
    cfg = _config(BLOCK_SIZE=64)

    cache[key] = cfg
    assert cache[key] is cfg

    with pytest.raises(KeyError):
        _ = cache[(99, 99)]
