# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import pytest
import torch
import triton

import flag_gems
from flag_gems.ops.add import add
from flag_gems.ops.arange import arange_start
from flag_gems.ops.bitwise_and import bitwise_and_tensor
from flag_gems.ops.ge import ge_scalar
from flag_gems.ops.lt import lt_scalar
from flag_gems.ops.mul import mul, mul_
from flag_gems.ops.sub import sub
from flag_gems.ops.where import where_self
from flag_gems.utils import aten_plan_cache as plan_cache
from flag_gems.utils.aten_plan_cache import (
    cache_stats,
    clear_caches,
    disable,
    enable,
    install,
    reset_stats,
    uninstall,
)
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def cached_add(x, y):
    return x + y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def cached_add_scalar(x, scalar):
    return x + scalar


@pytest.fixture(autouse=True)
def _enable_plan_cache():
    assert install(max_size=128, include=None, exclude=("*arange_func*",))
    clear_caches()
    yield
    uninstall()


def _assert_no_tensor_metadata(value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    assert not isinstance(value, torch.Tensor)
    if is_dataclass(value):
        for field in fields(value):
            if field.name not in ("kernel", "libentry"):
                _assert_no_tensor_metadata(getattr(value, field.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_tensor_metadata(key, seen)
            _assert_no_tensor_metadata(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_tensor_metadata(item, seen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_launch_plan_functional_out_inplace_and_dtype(dtype):
    cached_add.clear_plan_cache()
    x = torch.randn((17, 31), device=flag_gems.device, dtype=dtype)
    y = torch.randn_like(x)

    actual = cached_add(x, y)
    torch.testing.assert_close(actual, x + y)
    actual_second = cached_add(x.clone(), y.clone())
    torch.testing.assert_close(actual_second, x + y)

    out = torch.empty_like(x)
    returned = cached_add(x, y, out0=out)
    assert returned is out
    torch.testing.assert_close(out, x + y)

    inplace = x.clone()
    expected = inplace + y
    returned = cached_add(inplace, y, out0=inplace)
    assert returned is inplace
    torch.testing.assert_close(inplace, expected)

    info = cached_add.cache_info()
    assert info.last_hits >= 1
    assert info.misses == 3
    for vendor_cache in cached_add._aten_routing_plan_cache.lru.values():
        for signature, plan in vendor_cache.items():
            _assert_no_tensor_metadata(signature)
            _assert_no_tensor_metadata(plan)


def test_launch_plan_dynamic_shape_lru_and_scalar_constexpr():
    cached_add_scalar.clear_plan_cache()
    for shape, scalar in (((64,), 1), ((127,), 1), ((64,), 1), ((64,), 2)):
        x = torch.randn(shape, device=flag_gems.device)
        torch.testing.assert_close(cached_add_scalar(x, scalar), x + scalar)
    info = cached_add_scalar.cache_info()
    assert info.lru_hits == 1
    assert info.misses == 3


def test_launch_plan_noncontiguous_stride_and_broadcast():
    cached_add.clear_plan_cache()
    x = torch.randn((23, 11), device=flag_gems.device).T
    y = torch.randn((1, 23), device=flag_gems.device)
    expected = x + y
    actual = cached_add(x, y)
    torch.testing.assert_close(actual, expected)

    x2 = torch.randn((23, 11), device=flag_gems.device).T
    y2 = torch.randn((1, 23), device=flag_gems.device)
    actual2 = cached_add(x2, y2)
    torch.testing.assert_close(actual2, x2 + y2)
    assert cached_add.cache_info().last_hits == 1


def test_launch_plan_rank_zero_tensor():
    cached_add.clear_plan_cache()
    x = torch.tensor(2.0, device=flag_gems.device)
    y = torch.tensor(3.0, device=flag_gems.device)
    torch.testing.assert_close(cached_add(x, y), x + y)
    torch.testing.assert_close(cached_add(x.clone(), y.clone()), x + y)
    assert cached_add.cache_info().last_hits == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph")
def test_launch_plan_cuda_graph_capture_replay():
    cached_add.clear_plan_cache()
    x = torch.randn((4096,), device="cuda")
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    cached_add(x, y, out0=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cached_add(x, y, out0=out)

    for _ in range(3):
        x.copy_(torch.randn_like(x))
        y.copy_(torch.randn_like(y))
        expected = x + y
        graph.replay()
        torch.testing.assert_close(out, expected)
    assert cached_add.cache_info().last_hits >= 1


def test_direct_arange_and_mul_launch_plan_semantics():
    clear_caches()

    expected_range = torch.arange(0, 257, 3, device=flag_gems.device)
    actual_range = arange_start(0, 257, 3, device=flag_gems.device)
    actual_range_second = arange_start(0, 257, 3, device=flag_gems.device)
    torch.testing.assert_close(actual_range, expected_range)
    torch.testing.assert_close(actual_range_second, expected_range)
    arange_stats = [
        stats
        for name, stats in cache_stats()["per_operator"].items()
        if "arange_func" in name
    ]
    assert arange_stats and arange_stats[0]["bypasses"] >= 2

    x = torch.randn((31, 17), device=flag_gems.device)
    y = torch.randn_like(x)
    torch.testing.assert_close(mul(x, y), x * y)
    torch.testing.assert_close(mul(x.clone(), y.clone()), x * y)

    out = torch.empty_like(x)
    assert mul(x, y, out=out) is out
    torch.testing.assert_close(out, x * y)

    inplace = x.clone()
    expected = inplace * 2.5
    assert mul_(inplace, 2.5) is inplace
    torch.testing.assert_close(inplace, expected)

    a = torch.randn((23, 1), device=flag_gems.device)
    b = torch.randn((1, 19), device=flag_gems.device)
    torch.testing.assert_close(mul(a, b), a * b)
    torch.testing.assert_close(mul(a.clone(), b.clone()), a * b)
    assert cache_stats()["hits"] >= 2


def _qsa_metadata_chain(size):
    indexes = arange_start(0, size, 1, device=flag_gems.device)
    shifted = sub(indexes, 3)
    scaled = mul(shifted, 4)
    added = add(scaled, 1)
    lower = ge_scalar(added, 0)
    upper = lt_scalar(added, size * 4)
    valid = bitwise_and_tensor(lower, upper)
    return where_self(valid, added, indexes)


def test_qsa_metadata_chain_warm_hit_rate():
    expected = _qsa_metadata_chain(2048)
    for _ in range(9):
        torch.testing.assert_close(_qsa_metadata_chain(2048), expected)
    reset_stats()
    for _ in range(100):
        torch.testing.assert_close(_qsa_metadata_chain(2048), expected)
    stats = cache_stats()
    assert stats["misses"] == 0
    assert stats["hit_rate"] > 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph")
def test_qsa_metadata_chain_cuda_graph_replay():
    expected = _qsa_metadata_chain(2048)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _qsa_metadata_chain(2048)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, expected)


def test_runtime_disable_and_reversible_uninstall():
    from flag_gems.utils.libentry import LibEntry
    from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction

    assert LibEntry.run is plan_cache._cached_libentry_run
    assert PointwiseDynamicFunction.__call__ is plan_cache._cached_pointwise_call
    disable()
    assert not cache_stats()["enabled"]
    assert (
        cached_add(
            torch.ones(4, device=flag_gems.device),
            torch.ones(4, device=flag_gems.device),
        ).sum()
        == 8
    )
    uninstall()
    assert LibEntry.run is not plan_cache._cached_libentry_run
    assert PointwiseDynamicFunction.__call__ is not plan_cache._cached_pointwise_call
    assert enable()


def test_install_is_idempotent_and_keeps_warm_plans():
    x = torch.randn(64, device=flag_gems.device)
    y = torch.randn_like(x)
    cached_add.clear_plan_cache()
    torch.testing.assert_close(cached_add(x, y), x + y)
    before = cached_add.cache_info()
    assert install()
    torch.testing.assert_close(cached_add(x, y), x + y)
    after = cached_add.cache_info()
    assert after.misses == before.misses
    assert after.hits == before.hits + 1


def test_device_context_and_pid_are_part_of_cache_identity(monkeypatch):
    cache = plan_cache._TwoLevelCache(("unit", "operator"))
    contexts = iter((("nvidia", "cuda", 0, (9, 0)), ("nvidia", "cuda", 1, (9, 0))))
    monkeypatch.setattr(plan_cache, "_current_device_context", lambda: next(contexts))

    signature0, value = cache.lookup(("shape", (64,)))
    assert value is None
    cache.insert(signature0, "plan0")
    signature1, value = cache.lookup(("shape", (64,)))
    assert value is None
    assert signature1 != signature0

    cache.pid -= 1
    monkeypatch.setattr(
        plan_cache,
        "_current_device_context",
        lambda: ("nvidia", "cuda", 0, (9, 0)),
    )
    _, value = cache.lookup(("shape", (64,)))
    assert value is None
    assert cache.info().size == 0


def test_tensor_device_context_is_reused():
    plan_cache._tensor_device_context.cache_clear()
    device = torch.device(flag_gems.device)

    first = plan_cache._tensor_device_context(device)
    second = plan_cache._tensor_device_context(device)

    assert first is second
    assert plan_cache._tensor_device_context.cache_info().hits == 1


def test_validated_launch_plan_reuses_device_context(monkeypatch):
    context = ("nvidia", "cuda", "cuda", 0, (9, 0))
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            return lambda *args: launches.append((grid, args))

    class FakeEntry:
        _has_flagtune_tuner = False

    plan = plan_cache.LibEntryLaunchPlan(
        kernel=FakeKernel(),
        grid=(1, 1, 1),
        argument_sources=(),
        device_context=context,
        tuning_epoch=0,
        constexprs=(),
        num_warps=None,
        num_stages=None,
        num_ctas=None,
    )

    def unexpected_context_scan(*args, **kwargs):
        raise AssertionError("validated cache hit must reuse its device context")

    monkeypatch.setattr(plan_cache, "_argument_device_context", unexpected_context_scan)
    plan_cache._run_libentry_plan(FakeEntry(), plan, _aten_plan_device_context=context)

    assert launches == [((1, 1, 1), ())]


def test_codegen_config_mutation_invalidates_pointwise_plan():
    x = torch.randn(64, device=flag_gems.device)
    y = torch.randn_like(x)
    cached_add.clear_plan_cache()
    torch.testing.assert_close(cached_add(x, y), x + y)
    original = cached_add.config.max_grid_size
    try:
        cached_add.config.max_grid_size = (original[0] - 1, *original[1:])
        torch.testing.assert_close(cached_add(x, y), x + y)
    finally:
        cached_add.config.max_grid_size = original
    assert cached_add.cache_info().misses == 2


def test_flagtune_change_invalidates_launch_epoch():
    class FakeEntry:
        _has_flagtune_tuner = True

        def __init__(self):
            self.changed = True

        def _apply_flagtune(self):
            changed, self.changed = self.changed, False
            return changed

    entry = FakeEntry()
    cache = plan_cache._TwoLevelCache(("unit", "flagtune"))
    entry._aten_launch_plan_cache = cache
    signature, _ = cache.lookup(("shape", (1,)))
    cache.insert(signature, "old-plan")

    plan_cache._refresh_flagtune(entry)
    assert entry._aten_plan_tuning_epoch == 1
    assert cache.info().size == 0


def test_optional_plan_builder_fails_open(monkeypatch):
    x = torch.randn(64, device=flag_gems.device)
    y = torch.randn_like(x)
    cached_add.clear_plan_cache()

    def incompatible_vendor_abi(*args, **kwargs):
        raise RuntimeError("unsupported wrapper ABI")

    monkeypatch.setattr(plan_cache, "_build_pointwise_plan", incompatible_vendor_abi)
    torch.testing.assert_close(cached_add(x, y), x + y)
    assert cached_add.cache_info().size == 0


@pytest.mark.parametrize("shape", [(0,), (3, 0, 7)])
def test_empty_outputs_reuse_structural_plans(shape):
    x = torch.empty(shape, device=flag_gems.device)
    y = torch.empty_like(x)
    cached_add.clear_plan_cache()
    torch.testing.assert_close(cached_add(x, y), x + y)
    torch.testing.assert_close(cached_add(x, y), x + y)
    assert cached_add.cache_info().hits == 1


def test_bounded_lru_eviction_rebuilds_dynamic_shapes():
    assert enable(max_size=2)
    cached_add.clear_plan_cache()
    for size in (17, 31, 47, 17):
        x = torch.randn(size, device=flag_gems.device)
        y = torch.randn_like(x)
        torch.testing.assert_close(cached_add(x, y), x + y)
    info = cached_add.cache_info()
    assert info.size == 2
    assert info.evictions == 2
    assert info.misses == 4


def test_zero_capacity_and_operator_filters_bypass_cache():
    x = torch.randn(64, device=flag_gems.device)
    y = torch.randn_like(x)
    for policy in (
        {"max_size": 0},
        {"max_size": 128, "include": ("does_not_match.*",)},
        {"max_size": 128, "include": None, "exclude": ("*cached_add*",)},
    ):
        assert enable(**policy)
        cached_add.clear_plan_cache()
        for _ in range(2):
            torch.testing.assert_close(cached_add(x, y), x + y)
        assert cached_add.cache_info().size == 0
        assert cached_add.cache_info().hits == 0


@pytest.mark.parametrize("prefer_1d", [False, True])
def test_balanced_grid_change_invalidates_routing_and_launch_plans(prefer_1d):
    config = replace(
        cached_add.config,
        balance_grid=False,
        prefer_1d_tile=prefer_1d,
        max_grid_size=(3, 1, 1),
    )
    operator = pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )(cached_add._scalar_fn)
    x = torch.randn((129, 127), device=flag_gems.device)
    y = torch.randn_like(x)
    for balanced in (False, True):
        operator.config.balance_grid = balanced
        torch.testing.assert_close(operator(x, y), x + y)
        torch.testing.assert_close(operator(x, y), x + y)
    info = operator.cache_info()
    assert info.misses == 2
    assert info.hits == 2
    for vendor_cache in operator._aten_routing_plan_cache.lru.values():
        for signature, plan in vendor_cache.items():
            assert plan.launch_plan is not None
            assert signature[2][-1][-1] in (False, True)


def test_stateful_callable_grids_bypass_launch_cache():
    class MutableGrid:
        def __init__(self):
            self.blocks = 1

        def __call__(self, metadata):
            return (self.blocks,)

    grid = MutableGrid()
    assert plan_cache._grid_signature(grid) is None
    grid.blocks = 2
    assert plan_cache._grid_signature(grid) is None


def test_launch_constants_reject_nested_tensors_and_opaque_owners():
    tensor = torch.ones(1, device=flag_gems.device)
    for value in (tensor, (tensor,), {"tensor": tensor}, SimpleNamespace(base=tensor)):
        with pytest.raises(TypeError):
            plan_cache._launch_constant(value)
    assert plan_cache._launch_constant((True, 16, torch.float32)) == (
        True,
        16,
        torch.float32,
    )


def test_original_flagtune_path_invalidates_existing_launch_plans():
    from flag_gems.utils.libentry import LibEntry

    class Tuner:
        fn = cached_add._scalar_fn

        def __init__(self):
            self.changed = True

        def apply_flagtune(self):
            changed, self.changed = self.changed, False
            return changed

    cache = plan_cache._TwoLevelCache(("unit", "flagtune-original"))
    signature, _ = cache.lookup(("shape", (1,)))
    cache.insert(signature, "old-plan")
    entry = SimpleNamespace(
        fn=Tuner(),
        kernel_cache=[{"old": "kernel"}],
        _cpu_cache={"old": "kernel"},
        _has_flagtune_tuner=True,
        _aten_launch_plan_cache=cache,
    )
    entry._apply_flagtune = lambda: LibEntry._apply_flagtune(entry)
    assert entry._apply_flagtune()
    assert entry._aten_plan_tuning_epoch == 1
    assert cache.info().size == 0
    assert not entry.kernel_cache[0]
    assert not entry._cpu_cache
    plan_cache._refresh_flagtune(entry)
    assert entry._aten_plan_tuning_epoch == 1


@pytest.mark.parametrize(
    "switch, expected", [(None, False), ("0", False), ("  true  ", True)]
)
def test_import_time_public_api_opt_in(switch, expected):
    environment = os.environ.copy()
    environment.pop("FLAGGEMS_ATEN_PLAN_CACHE", None)
    if switch is not None:
        environment["FLAGGEMS_ATEN_PLAN_CACHE"] = switch
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, flag_gems; "
            "print(json.dumps(flag_gems.aten_plan_cache_stats()))",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    stats = json.loads(result.stdout.splitlines()[-1])
    assert stats["installed"] is expected
    assert stats["enabled"] is expected
