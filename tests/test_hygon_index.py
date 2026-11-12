"""Numerical and layout regressions for the Hygon small-kernel paths."""

import math

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


@pytest.mark.index
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_hygon_index_offsets_above_int32(index_dtype):
    import ctypes

    from flag_gems.ops.index import index as generic_index

    # Torch-FL's flagos device API does not expose mem_get_info. Query the
    # Hygon runtime directly before allocating this boundary regression input.
    torch.empty(1, device=flag_gems.device)
    hip = ctypes.CDLL("libamdhip64.so")
    free_bytes, total_bytes = ctypes.c_size_t(), ctypes.c_size_t()
    assert hip.hipMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)) == 0
    if free_bytes.value < 3 * 2**30:
        pytest.skip("not enough free device memory for the int32-offset boundary")
    rows = 2**28 + 1
    inp = torch.zeros((rows, 8), dtype=torch.int8, device=flag_gems.device)
    sentinel = torch.arange(1, 9, dtype=torch.int8)
    inp[-1] = sentinel.to(flag_gems.device)
    inp[0] = -sentinel.to(flag_gems.device)
    idx = torch.tensor([-1, rows - 1, 0], dtype=index_dtype, device=flag_gems.device)
    expected = torch.stack((sentinel, sentinel, -sentinel))
    for implementation in (flag_gems.index, generic_index):
        actual = implementation(inp, [idx])
        assert torch.equal(actual.cpu(), expected)
        transposed = implementation(inp.t(), [None, idx])
        assert torch.equal(transposed.cpu(), expected.t())


@pytest.mark.index
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "shape,count,dtype",
    [
        ((1, 3, 3), 56, torch.float32),
        ((25, 3), 57, torch.float32),
        ((91, 2, 2), 208, torch.float32),
        ((4819, 4, 32), 126636, torch.float32),
        ((854, 256), 40992, torch.float32),
        ((4096,), 80599, torch.int64),
        ((65536,), 1289579, torch.int64),
    ],
)
def test_hygon_dense_gather_repeated_negative_indices(shape, count, dtype, index_dtype):
    # Full-output validation for dense gather regimes, including repeated
    # positive and negative boundary rows.
    x = torch.arange(math.prod(shape), dtype=dtype).reshape(shape)
    rows = torch.tensor([-shape[0], -1, 0, shape[0] - 1], dtype=index_dtype)
    ix = rows.repeat((count + 3) // 4)[:count]
    actual = flag_gems.index(x.to(flag_gems.device), [ix.to(flag_gems.device)])
    assert torch.equal(actual.cpu(), x[ix])
