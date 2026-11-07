"""Storage and address contracts of the cooperative MM family."""

import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def cooperative():
    return importlib.import_module(
        "flag_gems.runtime.backend._hygon.ops.mm_cooperative"
    )


def operands(m, n, k, dtype=torch.bfloat16):
    def tensor(shape, stride, pointer):
        return SimpleNamespace(
            ndim=2,
            shape=shape,
            stride=lambda: stride,
            dtype=dtype,
            device=torch.device("cuda:0"),
            data_ptr=lambda: pointer,
            requires_grad=False,
            is_inference=lambda: False,
        )

    return (
        tensor((m, k), (k, 1), 0x100000),
        tensor((k, n), (1, k), 0x1000000000),
        tensor((m, n), (n, 1), 0x2000000000),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (35, 80, 64),
        (37, 81, 192),
        (1035, 12288, 2048),
        (131073, 8192, 64),
        (17, 262145, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pipeline_tail_and_global_large_storage_contract(cooperative, shape, dtype):
    # Large global objects are legal when the per-CTA local offsets fit.
    args = operands(*shape, dtype)
    for group in [0, 3, 4]:
        assert cooperative.CooperativeMMConfig(group).supports(*args)


@pytest.mark.parametrize("k", [0, 32, 65, 8192])
def test_invalid_structured_pitch_or_reduction_extent(cooperative, k):
    assert not cooperative.CooperativeMMConfig().supports(*operands(35, 80, k))


def test_output_alias_alignment_and_storage_layout(cooperative):
    config = cooperative.CooperativeMMConfig()
    a, b, c = operands(35, 80, 192)
    c.data_ptr = a.data_ptr
    assert not config.supports(a, b, c)
    # An adjacent view of the same allocation has no byte overlap.
    c.data_ptr = lambda: a.data_ptr() + 2 * 35 * 192
    assert config.supports(a, b, c)
    c.data_ptr = lambda: a.data_ptr() + 2 * 35 * 192 + 2
    assert not config.supports(a, b, c)
    a, b, c = operands(35, 80, 192)
    b.stride = lambda: (80, 1)
    assert not config.supports(a, b, c)


def test_output_local_offset_and_grouped_grid_limits(cooperative):
    config = cooperative.CooperativeMMConfig()
    # 512*N, not the total output tensor size, bounds scalar local stores.
    assert not config.supports(*operands(35, 2**22, 192))
    large = operands(2**31 - 1, 2**22 - 1, 64)
    # Move the conceptual arrays far enough apart for this metadata-only case.
    for i, t in enumerate(large):
        t.data_ptr = lambda i=i: (i + 1) * 2**60
    assert config.supports(*large)
    assert not replace(config, group_m=4).supports(*large)
