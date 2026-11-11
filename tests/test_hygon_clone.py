"""Numerical and layout regressions for the Hygon small-kernel paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)

@pytest.mark.clone
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
@pytest.mark.parametrize("kind", ["transpose", "slice", "broadcast", "empty"])
def test_hygon_clone_layout(memory_format, kind):
    base = torch.randn((7, 13, 4), dtype=torch.float64)
    if kind == "transpose":
        a = base.transpose(1, 2)
    elif kind == "slice":
        a = base[:, ::2, 1:]
    elif kind == "broadcast":
        a = base[:1].expand(5, 13, 4)
    else:
        a = base[:0]
    expected = torch.clone(a, memory_format=memory_format)
    # Rebuild the original view on device, including its holes and zero strides.
    gpu_base = base.to(flag_gems.device)
    gpu = torch.as_strided(gpu_base, a.shape, a.stride(), a.storage_offset())
    result = flag_gems.clone(gpu, memory_format=memory_format)
    assert result.shape == expected.shape
    assert result.stride() == expected.stride()
    assert torch.equal(result.cpu(), expected)

"""Regression coverage for expanded Hygon layouts and precise reductions."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)
