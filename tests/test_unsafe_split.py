import math

import pytest
import torch

import flag_gems
from flag_gems.config import has_c_extension
from tests import accuracy_utils as utils


def _torch_device_available(device_name):
    if device_name == "cpu":
        return False
    if device_name == "npu" and not hasattr(torch, "npu"):
        try:
            __import__("torch_npu")
        except ImportError:
            return False
    device_mod = getattr(torch, device_name, None)
    if device_mod is None or not hasattr(device_mod, "is_available"):
        return False
    try:
        return bool(device_mod.is_available())
    except Exception:
        return False


def _test_device():
    if _torch_device_available(flag_gems.device):
        return torch.device(flag_gems.device)
    for device_name in ("npu", "cuda"):
        if _torch_device_available(device_name):
            return torch.device(device_name)
    return None


TEST_DEVICE = _test_device()


pytestmark = pytest.mark.skipif(
    not (
        has_c_extension
        and hasattr(torch.ops.flag_gems, "unsafe_split")
        and hasattr(torch.ops.flag_gems, "unsafe_split_with_sizes")
        and TEST_DEVICE is not None
    ),
    reason="unsafe_split C++ wrapper requires an available FlagGems backend device",
)


# unsafe_split only creates views, so dtype does not change the implementation.
# Keep one floating dtype and one integer dtype for coverage.
DTYPES = [torch.float32, torch.int32]  # One float dtype and one int dtype are enough.

SPLIT_CASES = [
    ((8,), 2, 0),
    ((4, 6), 4, -1),
    ((2, 3, 5), 2, 1),
    ((5,), 1, 0),
    ((5,), 5, 0),
    ((5,), 8, 0),
    ((0, 3), 2, 0),
    ((3, 0), 0, 1),
]

SPLIT_WITH_SIZES_CASES = [
    ((8,), [3, 0, 5], 0),
    ((4, 6), [1, 2, 3], -1),
    ((2, 3, 5), [0, 1, 2], 1),
    ((5,), [2, 0, 3], 0),
    ((0, 3), [0], 0),
    ((0, 3), [0, 0], 0),
    ((0, 3), [], 0),
    ((3, 0), [0], 1),
]

DIM_CASES = [
    ((2, 3, 4), 1, -3),
    ((2, 3, 4), 2, -1),
    ((2, 3, 4), 2, 2),
]

INVALID_DIMS = [
    ((2, 3, 4), 3),
    ((2, 3, 4), -4),
]


def _make_input(shape, dtype, device=None):
    if device is None:
        device = TEST_DEVICE
    numel = math.prod(shape)
    if numel == 0:
        out = torch.empty(shape, dtype=dtype, device=device)
    else:
        out = torch.arange(numel, dtype=dtype, device=device).reshape(shape)
    assert out.device.type != "cpu", (
        "unsafe_split C++ wrapper tests require backend input, "
        f"but got {out.device}; flag_gems.device={flag_gems.device}"
    )
    return out


def _make_noncontiguous_input(dtype, device=None):
    return _make_input((5, 4, 6), dtype, device).transpose(0, 1)[:, 1:, :]


def _wrap_dim(dim, ndim):
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim
    return dim


def _assert_matches_torch_reference(result, expected, base):
    assert len(result) == len(expected)
    base_ptr = base.untyped_storage().data_ptr()
    for res, ref in zip(result, expected):
        assert tuple(res.shape) == tuple(ref.shape)
        assert res.stride() == ref.stride()
        assert res.storage_offset() == ref.storage_offset()
        assert res.untyped_storage().data_ptr() == base_ptr
        assert ref.untyped_storage().data_ptr() == base_ptr
        assert res.data_ptr() == ref.data_ptr()
        utils.gems_assert_equal(res, utils.to_reference(ref))


@pytest.mark.unsafe_split
@pytest.mark.parametrize("shape, split_size, dim", SPLIT_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split(shape, split_size, dim, dtype):
    inp = _make_input(shape, dtype)
    expected = torch.ops.aten.unsafe_split.Tensor(inp, split_size, dim)

    result = torch.ops.flag_gems.unsafe_split(inp, split_size, dim)

    _assert_matches_torch_reference(result, expected, inp)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("shape, split_sizes, dim", SPLIT_WITH_SIZES_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_with_sizes(shape, split_sizes, dim, dtype):
    inp = _make_input(shape, dtype)
    expected = torch.ops.aten.unsafe_split_with_sizes.default(inp, split_sizes, dim)

    result = torch.ops.flag_gems.unsafe_split_with_sizes(inp, split_sizes, dim)

    _assert_matches_torch_reference(result, expected, inp)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("split_size", [-1, 0])
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_invalid_split_size(split_size, dtype):
    inp = _make_input((5,), dtype)

    with pytest.raises(RuntimeError):
        torch.ops.aten.unsafe_split.Tensor(inp, split_size, 0)

    with pytest.raises(RuntimeError):
        torch.ops.flag_gems.unsafe_split(inp, split_size, 0)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("shape, split_size, dim", DIM_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_dim_boundaries(shape, split_size, dim, dtype):
    inp = _make_input(shape, dtype)
    expected = torch.ops.aten.unsafe_split.Tensor(inp, split_size, dim)

    result = torch.ops.flag_gems.unsafe_split(inp, split_size, dim)

    _assert_matches_torch_reference(result, expected, inp)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("shape, split_size, dim", DIM_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_with_sizes_dim_boundaries(shape, split_size, dim, dtype):
    inp = _make_input(shape, dtype)
    wrapped_dim = _wrap_dim(dim, inp.dim())
    split_sizes = [inp.size(wrapped_dim)]
    expected = torch.ops.aten.unsafe_split_with_sizes.default(inp, split_sizes, dim)

    result = torch.ops.flag_gems.unsafe_split_with_sizes(inp, split_sizes, dim)

    _assert_matches_torch_reference(result, expected, inp)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("shape, dim", INVALID_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_invalid_dim(shape, dim, dtype):
    inp = _make_input(shape, dtype)

    with pytest.raises((IndexError, RuntimeError)):
        torch.ops.aten.unsafe_split.Tensor(inp, 1, dim)

    with pytest.raises((IndexError, RuntimeError)):
        torch.ops.flag_gems.unsafe_split(inp, 1, dim)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("shape, dim", INVALID_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_with_sizes_invalid_dim(shape, dim, dtype):
    inp = _make_input(shape, dtype)

    with pytest.raises((IndexError, RuntimeError)):
        torch.ops.aten.unsafe_split_with_sizes.default(inp, [inp.numel()], dim)

    with pytest.raises((IndexError, RuntimeError)):
        torch.ops.flag_gems.unsafe_split_with_sizes(inp, [inp.numel()], dim)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_scalar_tensor_errors(dtype):
    inp = _make_input((), dtype)

    with pytest.raises(RuntimeError):
        torch.ops.aten.unsafe_split.Tensor(inp, 1, 0)

    with pytest.raises(RuntimeError):
        torch.ops.flag_gems.unsafe_split(inp, 1, 0)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_with_sizes_scalar_tensor_errors(dtype):
    inp = _make_input((), dtype)

    with pytest.raises(RuntimeError):
        torch.ops.aten.unsafe_split_with_sizes.default(inp, [1], 0)

    with pytest.raises(RuntimeError):
        torch.ops.flag_gems.unsafe_split_with_sizes(inp, [1], 0)


@pytest.mark.unsafe_split
@pytest.mark.parametrize(
    "split_sizes",
    [
        [2, 2],
        [2, 4],
        [2, -1, 4],
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_with_sizes_invalid_split_sizes(split_sizes, dtype):
    inp = _make_input((5,), dtype)

    with pytest.raises(RuntimeError):
        torch.ops.aten.unsafe_split_with_sizes.default(inp, split_sizes, 0)

    with pytest.raises(RuntimeError):
        torch.ops.flag_gems.unsafe_split_with_sizes(inp, split_sizes, 0)


@pytest.mark.unsafe_split
@pytest.mark.parametrize("dtype", DTYPES)
def test_unsafe_split_noncontiguous(dtype):
    inp = _make_noncontiguous_input(dtype)
    assert not inp.is_contiguous()

    for dim in [0, 1, -1]:
        split_result = torch.ops.flag_gems.unsafe_split(inp, 2, dim)
        split_expected = torch.ops.aten.unsafe_split.Tensor(inp, 2, dim)
        _assert_matches_torch_reference(split_result, split_expected, inp)

        wrapped_dim = _wrap_dim(dim, inp.dim())
        split_sizes = [1, 0, inp.size(wrapped_dim) - 1]
        sizes_result = torch.ops.flag_gems.unsafe_split_with_sizes(
            inp, split_sizes, dim
        )
        sizes_expected = torch.ops.aten.unsafe_split_with_sizes.default(
            inp, split_sizes, dim
        )
        _assert_matches_torch_reference(sizes_result, sizes_expected, inp)
