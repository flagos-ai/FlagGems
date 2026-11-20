"""Contracts for the optional tuned global Split-K family."""

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


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
