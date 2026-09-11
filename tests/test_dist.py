import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    P_LIST = [2]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    P_LIST = [0, 1, 1.5, 2, 3, 0.5]

# torch.dist accepts any real p plus inf / -inf:
# inf -> max|x - y|, -inf -> min|x - y|, 0 -> count of non-zero differences,
# general p -> (sum |x - y| ** p) ** (1 / p).
P_LIST_ALL = P_LIST + [float("inf"), float("-inf")]


def composed_dist(x, y, p=2.0):
    # torch-native dist via basic torch ops (sub+abs+pow+sum).
    diff = torch.abs(x - y)
    if p == float("inf"):
        return torch.amax(diff)
    elif p == float("-inf"):
        return torch.amin(diff)
    elif p == 0:
        return torch.sum(diff != 0, dtype=torch.float32).to(x.dtype)
    else:
        return torch.pow(torch.sum(torch.pow(diff, p)), 1.0 / p).to(x.dtype)


# torch_npu's native dist only supports p in {0, 1, 2} -- other norms return
# inaccurate results on NPU. When the reference runs on NPU (not CPU), use
# the composed version for any p outside {0, 1, 2}.
_ASCEND_SUPPORTED_P = (0.0, 1.0, 2.0)


def _ref_dist(x, y, p=2.0):
    if (
        flag_gems.vendor_name == "ascend"
        and not cfg.TO_CPU
        and p not in _ASCEND_SUPPORTED_P
    ):
        return composed_dist(x, y, p=p)
    return torch.dist(x, y, p)


SHAPES = [
    (1,),  # tiny, single-kernel path with BLOCK_SIZE = 1
    (16,),
    (64,),
    (512,),
    (1024,),  # single-kernel path
    (4096,),
    (32, 1024),  # two-stage reduction path (numel > 16384)
    (65537,),
    (128, 4096),
    (1, 1000000),  # large two-stage reduction
]


@pytest.mark.dist
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("p", P_LIST_ALL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dist_accuracy(shape, p, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    y = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = _ref_dist(ref_x, ref_y, p)

    out = flag_gems.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, dtype)


# (input_shape, other_shape) pairs exercising broadcasting: torch.dist
# broadcasts the two operands against each other before reducing.
BROADCAST_SHAPES = [
    ((4,), (1,)),
    ((3, 4), (4,)),
    ((3, 4), (1, 4)),
    ((2, 3, 4), (3, 4)),
]


@pytest.mark.dist
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dist_broadcast(input_shape, other_shape, p, dtype):
    torch.manual_seed(0)
    x = torch.randn(input_shape, device=flag_gems.device, dtype=dtype)
    y = torch.randn(other_shape, device=flag_gems.device, dtype=dtype)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = _ref_dist(ref_x, ref_y, p)

    out = flag_gems.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, dtype)


# Non-contiguous inputs (e.g. produced by transpose / slicing) must be
# handled by making them contiguous internally, not read with wrong strides.
@pytest.mark.dist
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dist_non_contiguous(p, dtype):
    torch.manual_seed(0)
    x = torch.randn((64, 128), device=flag_gems.device, dtype=dtype).t()
    y = torch.randn((64, 128), device=flag_gems.device, dtype=dtype).t()
    assert not x.is_contiguous() and not y.is_contiguous()

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = _ref_dist(ref_x, ref_y, p)

    out = flag_gems.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, dtype)


# Negative p: torch.dist computes (sum |x - y| ** p) ** (1 / p) for any real
# p, including negative values (vector_norm semantics).
@pytest.mark.dist
@pytest.mark.parametrize("p", [-1, -0.5])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dist_negative_p(p, dtype):
    torch.manual_seed(0)
    x = torch.randn((32, 1024), device=flag_gems.device, dtype=dtype)
    y = torch.randn((32, 1024), device=flag_gems.device, dtype=dtype)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = _ref_dist(ref_x, ref_y, p)

    out = flag_gems.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, dtype)


# float64 accuracy: fp64 inputs must be accumulated in fp64, not silently
# downcast to fp32. Covers the fast paths (p = 2 / 1 / inf / -inf) and a
# general p, on both the single-kernel and the two-stage reduction paths.
@pytest.mark.dist
@pytest.mark.parametrize("shape", [(64, 257), (128, 4096)])
@pytest.mark.parametrize("p", [2, 1, 1.5, float("inf"), float("-inf")])
def test_dist_fp64(shape, p):
    if not utils.fp64_is_supported:
        pytest.skip("fp64 not supported on this device")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    y = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = _ref_dist(ref_x, ref_y, p)

    out = flag_gems.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, torch.float64)


# Empty tensors: torch.dist returns 0 for finite non-negative p and raises
# for inf / -inf / negative p (no identity element over an empty reduction).
@pytest.mark.dist
@pytest.mark.parametrize("p", [0, 1, 2, 1.5])
def test_dist_empty(p):
    x = torch.empty(0, device=flag_gems.device)
    y = torch.empty(0, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = _ref_dist(ref_x, ref_y, p)

    out = flag_gems.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, x.dtype)


@pytest.mark.dist
@pytest.mark.parametrize("p", [float("inf"), float("-inf"), -1])
def test_dist_empty_raises(p):
    x = torch.empty(0, device=flag_gems.device)
    y = torch.empty(0, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.dist(x, y, p)
