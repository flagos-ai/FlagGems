# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cache lifecycle correctness across lookup hits, misses and lock rechecks."""

import importlib

import pytest
import torch
import triton
import triton.language as tl

runtime = importlib.import_module("flag_gems.utils.libentry")


@triton.jit
def cache_kernel(X, BLOCK: tl.constexpr):
    pass


class Compiled:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args: self.launches.append((grid, args))


def setup(monkeypatch, device):
    entry = runtime.LibEntry(cache_kernel)
    # Logical device isolation only: the resource lease exposes one real GPU.
    monkeypatch.setattr(entry, "kernel_cache", ({}, {}))
    monkeypatch.setattr(runtime.torch_device_fn, "current_device", lambda: device)
    cache = entry._cpu_cache if device == "cpu" else entry.kernel_cache[device]
    return entry, cache, torch.ones(16, device="cpu")


@pytest.mark.parametrize("device", ["cpu", 0, 1])
@pytest.mark.parametrize("style", ["positional", "keywords"])
def test_lookup_hit_miss_and_invalidation(monkeypatch, device, style):
    entry, cache, x = setup(monkeypatch, device)
    cold = []

    def first(*args, **kwargs):
        compiled = Compiled()
        cold.append(compiled)
        return compiled

    monkeypatch.setattr(entry, "_first_run", first)

    def call(block):
        if style == "keywords":
            return entry.run(X=x, BLOCK=block, grid=(1,))
        return entry.run(x, block, grid=(1,))

    a, _ = call(64)
    b, _ = call(64)
    c, _ = call(128)
    assert a is b and c is not a
    assert len(cold) == 2 and len(cache) == 2
    assert a.launches[0][0] == (1, 1, 1)
    assert a.launches[0][1][0] is x
    cache.clear()
    d, _ = call(64)
    assert d is not a and len(cold) == 3


@pytest.mark.parametrize("device", ["cpu", 0])
@pytest.mark.parametrize("callable_grid", [False, True])
def test_lookup_rechecks_cache_after_lock_acquisition(
    monkeypatch, device, callable_grid
):
    entry, cache, x = setup(monkeypatch, device)
    key = entry.key([x], [], [64])
    compiled = Compiled()
    hooks, locks = [], []

    class FilledByOtherCaller:
        def __enter__(self):
            locks.append("entered")
            cache[key] = (
                compiled,
                {"BLOCK": 64},
                {},
                {},
                ((lambda meta: hooks.append(meta), {"extra": 7}),),
            )

        def __exit__(self, *args):
            locks.append("exited")

    monkeypatch.setattr(entry, "lock", FilledByOtherCaller())
    monkeypatch.setattr(
        entry,
        "_first_run",
        lambda *a, **k: pytest.fail("must reuse cache filled while waiting"),
    )
    grid = (lambda meta: (meta["BLOCK"] // 32,)) if callable_grid else (2,)
    result, const = entry.run(x, 64, grid=grid)
    assert result is compiled and const == {"BLOCK": 64}
    assert locks == ["entered", "exited"]
    assert compiled.launches[0][0] == (2, 1, 1)
    assert compiled.launches[0][1][0] is x
    assert hooks[0]["X"] is x and hooks[0]["extra"] == 7


@pytest.mark.parametrize("device", ["cpu", 0])
def test_lookup_failed_compile_does_not_poison_cache(monkeypatch, device):
    entry, cache, x = setup(monkeypatch, device)
    compiled = Compiled()
    attempts = []

    def first(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("compile failed")
        return compiled

    monkeypatch.setattr(entry, "_first_run", first)
    with pytest.raises(RuntimeError, match="compile failed"):
        entry.run(x, 64, grid=(1,))
    assert not cache
    assert entry.run(x, 64, grid=(1,))[0] is compiled
    assert entry.run(x, 64, grid=(1,))[0] is compiled
    assert len(attempts) == 2 and len(compiled.launches) == 1


def test_lookup_device_caches_remain_separate(monkeypatch):
    entry = runtime.LibEntry(cache_kernel)
    # Logical device isolation only: the resource lease exposes one real GPU.
    monkeypatch.setattr(entry, "kernel_cache", ({}, {}))
    x = torch.ones(16, device="cpu")
    device, cold = [0], []
    monkeypatch.setattr(runtime.torch_device_fn, "current_device", lambda: device[0])

    def first(*args, **kwargs):
        compiled = Compiled()
        cold.append((device[0], compiled))
        return compiled

    monkeypatch.setattr(entry, "_first_run", first)
    results = []
    for selected in [0, 1, 0, "cpu", "cpu", 1]:
        device[0] = selected
        results.append(entry.run(x, 64, grid=(1,))[0])
    assert [d for d, _ in cold] == [0, 1, "cpu"]
    assert results[0] is results[2]
    assert results[1] is results[5]
    assert results[3] is results[4]
    assert len({id(k) for k in results}) == 3
