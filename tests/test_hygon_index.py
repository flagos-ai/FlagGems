"""Numerical and layout regressions for the Hygon small-kernel paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)

@pytest.mark.index
@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("suffix", [None, 1, 3, 0])
@pytest.mark.parametrize("broadcast", [False, True])
def test_hygon_mask_compaction(density, suffix, broadcast):
    torch.manual_seed(7301)
    shape = (1, 5, 17, 19)
    if broadcast:
        a = torch.arange(5, dtype=torch.int64).reshape(1, 5, 1, 1).expand(shape)
    else:
        a = torch.randn(shape, dtype=torch.float64)
    gpu = a.to(flag_gems.device)
    if broadcast:
        gpu = (
            torch.arange(5, dtype=torch.int64, device=flag_gems.device)
            .reshape(1, 5, 1, 1)
            .expand(shape)
        )
    if suffix is not None:
        a = a.unsqueeze(-1).expand(*shape, suffix)
        gpu = gpu.unsqueeze(-1).expand(*shape, suffix)
    mask = torch.rand(shape) < density
    expected = a[mask]
    result = flag_gems.index(gpu, [mask.to(flag_gems.device)]).cpu()
    assert torch.equal(result, expected)

@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("suffix", [4, 48, 128])
def test_hygon_gather_negative_indices(dtype, suffix):
    base = torch.arange(7 * (suffix * 2 + 1), dtype=torch.float64).reshape(
        7, suffix * 2 + 1
    )
    a = base[:, 1::2]
    i = torch.tensor([-1, -7, 0, 5, 5, 2], dtype=dtype)
    device_base = base.to(flag_gems.device)
    device_a = torch.as_strided(device_base, a.shape, a.stride(), a.storage_offset())
    result = flag_gems.index(device_a, [i.to(flag_gems.device)])
    assert torch.equal(result.cpu(), a[i])

"""Regression coverage for expanded Hygon layouts and precise reductions."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)

@pytest.mark.index
@pytest.mark.parametrize("column", [False, True])
def test_expanded_index_negative_strided_indices(column):
    x = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    storage = torch.tensor([9, -1, 9, 0, 9, 2, 9])
    ix = storage[1::2]
    indices = [None, ix] if column else [ix]
    device_ix = storage.to(flag_gems.device)[1::2]
    actual = flag_gems.index(
        x.to(flag_gems.device), [None, device_ix] if column else [device_ix]
    )
    assert torch.equal(actual.cpu(), torch.ops.aten.index.Tensor(x, indices))

@pytest.mark.index
@pytest.mark.parametrize("suffix", [4, 33])
@pytest.mark.parametrize("empty", [False, True])
def test_expanded_index_broadcast_mask(suffix, empty):
    x = torch.arange(3 * 5 * suffix, dtype=torch.float32).reshape(3, 5, suffix)
    base = (
        torch.zeros(5, dtype=torch.bool)
        if empty
        else torch.tensor([True, False, True, False, True])
    )
    mask = base.expand(3, 5)
    actual = flag_gems.index(
        x.to(flag_gems.device), [base.to(flag_gems.device).expand(3, 5)]
    )
    assert torch.equal(actual.cpu(), torch.ops.aten.index.Tensor(x, [mask]))

@pytest.mark.index
def test_expanded_index_empty_integer():
    x = torch.arange(4, device=flag_gems.device)
    actual = flag_gems.index(
        x, [torch.empty(0, dtype=torch.int64, device=flag_gems.device)]
    )
    assert actual.shape == (0,)
