"""Contracts for the optional tuned global Split-K family."""

from contextlib import contextmanager, nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="requires the Hygon backend"
)


@pytest.fixture
def ops(monkeypatch):
    mm = import_module("flag_gems.runtime.backend._hygon.ops.mm")
    import triton

    monkeypatch.setattr(mm.torch_device_fn, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        triton.runtime.driver.active,
        "get_current_target",
        lambda: SimpleNamespace(backend="hip", arch="gfx936"),
    )
    return mm


def operand(shape, strides, pointer):
    return SimpleNamespace(
        ndim=2,
        shape=shape,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        stride=lambda i=None: strides if i is None else strides[i],
        is_contiguous=lambda: strides == (shape[1], 1),
        t=lambda: SimpleNamespace(is_contiguous=lambda: strides == (1, shape[0])),
        untyped_storage=lambda: SimpleNamespace(
            data_ptr=lambda: pointer, nbytes=lambda: 2 * shape[0] * shape[1]
        ),
        data_ptr=lambda: pointer,
        requires_grad=False,
        is_inference=lambda: False,
    )


@pytest.mark.parametrize("nt", [False, True])
@pytest.mark.parametrize(
    "m,n",
    [
        (31, 2048),
        (32, 2048),
        (33, 2112),
        (127, 2112),
        (128, 2112),
        (129, 2176),
        (1024, 2048),
        (1024, 2049),
    ],
)
def test_expanded_domain_is_area_bounded_and_layout_symmetric(ops, m, n, nt):
    k = 7168
    a = operand((m, k), (k, 1), 0x100000)
    b = operand((k, n), (1, k) if nt else (n, 1), 0x100000000)
    c = operand((m, n), (n, 1), 0x200000000)
    assert ops._splitk_eligible(a, b, c, expanded_pool=True) == (
        m * n <= ops._SPLITK_WORKSPACE_BYTES // 8
    )
    assert ops._splitk_eligible(a, b, c) == (((m + 31) // 32) * ((n + 31) // 32) <= 64)
    c.untyped_storage = lambda: SimpleNamespace(
        data_ptr=lambda: a.data_ptr(), nbytes=lambda: 2 * m * n
    )
    assert not ops._splitk_eligible(a, b, c, expanded_pool=True)


@pytest.mark.parametrize("planes", [0, 1, 2, 4, 8])
def test_workspace_pruning_keeps_local_and_direct_stores(ops, planes):
    m, n = 127, 2112
    configs = ops.runtime.ops_get_configs("mm_hygon_splitk")
    w = SimpleNamespace(numel=lambda: planes * m * n)
    pruned = ops._splitk_prune(
        configs,
        dict(M=m, N=n, K=7168, W=w),
        ASYNC_LOAD_MASK=3,
        WAVE_PRIVATE_SYNC_SUPPORTED=True,
    )
    expected = [
        c
        for c in configs
        if c.kwargs["KIND"] == 1
        or c.kwargs["SPLITS"] == 1
        or c.kwargs["SPLITS"] <= planes
    ]
    assert [c.all_kwargs() for c in pruned] == [c.all_kwargs() for c in expected]
    assert any(c.kwargs["KIND"] == 1 for c in pruned)


def test_optional_family_dispatch_uses_symmetric_load_contract(ops, monkeypatch):
    sentinel = object()
    seen = []
    monkeypatch.setattr(
        ops,
        "_launch_mm_splitk",
        lambda *args, **kw: seen.append((args, kw)) or sentinel,
    )
    a, b, c = object(), object(), object()
    assert (
        ops._dispatch_hygon_mm_candidate(a, b, c, ops._TUNED_SPLITK_CANDIDATE)
        is sentinel
    )
    assert seen == [((a, b, c), dict(symmetric_async=True))]


@pytest.mark.parametrize("route", ["mm", "mv", "splitk"])
def test_profile_resolution_uses_operand_device(ops, monkeypatch, route):
    # Simulate a different ambient device without requiring heterogeneous GPUs.
    # Target-dependent profile defaults must resolve inside the operand scope.
    active = []
    calls = []
    device = object()

    @contextmanager
    def use_device(value):
        active.append(value)
        try:
            yield
        finally:
            active.pop()

    def apply():
        assert active == [device]
        calls.append(route)

    tuner = SimpleNamespace(
        _run_mode="normal", _flagtune_mode=ops.runtime.TuningMode.EXPANDED
    )
    entry = SimpleNamespace(fn=tuner, _apply_flagtune=apply)
    monkeypatch.setattr(ops.torch_device_fn, "device", use_device)
    monkeypatch.setattr(ops, "gemm_tuning_key", lambda _: ("test",))
    monkeypatch.setattr(ops, "try_mm_family", lambda *a, **kw: True)
    a = SimpleNamespace(device=device)
    if route == "splitk":
        monkeypatch.setattr(ops, "_mm_splitk_partial", entry)
        monkeypatch.setattr(ops, "_splitk_eligible", lambda *a, **kw: True)
        assert ops.try_mm_splitk(a, None, None, compare_local_split=True)
    elif route == "mm":
        monkeypatch.setattr(ops, "mm_kernel", entry)
        monkeypatch.setattr(ops, "try_mm_splitk", lambda *a, **kw: False)
        ops._dispatch_hygon_mm(a, None, None)
    else:
        monkeypatch.setattr(ops, "mv_kernel", entry)
        monkeypatch.setattr(ops, "try_mv_mmac", lambda *a, **kw: True)
        ops._dispatch_hygon_mv(a, None, None)
    assert not active and calls == [route]
