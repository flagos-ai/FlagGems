import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def composed_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    # torch-native pairwise_distance via basic torch ops (sub+abs+pow+sum).
    diff = torch.abs(x1 - x2 + eps)
    if p == float("inf"):
        return torch.amax(diff, dim=-1, keepdim=keepdim)
    elif p == float("-inf"):
        return torch.amin(diff, dim=-1, keepdim=keepdim)
    elif p == 0.0:
        return torch.sum(diff != 0, dim=-1, keepdim=keepdim, dtype=torch.float32).to(
            x1.dtype
        )
    else:
        return torch.pow(
            torch.sum(torch.pow(diff, p), dim=-1, keepdim=keepdim), 1.0 / p
        ).to(x1.dtype)


# torch_npu's native pairwise_distance only supports p in {0, 1, 2} -- inf,
# -inf, and arbitrary real p all crash (core dump). When the reference runs on
# NPU (not CPU), use the composed version for any p outside {0, 1, 2}.
_ASCEND_SUPPORTED_P = (0.0, 1.0, 2.0)


def _ref_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    # On ascend backend, native pairwise_distance only supports p in {0, 1, 2}.
    # Fall back to composed op for unsupported p values.
    if (
        flag_gems.vendor_name == "ascend"
        and not cfg.TO_CPU
        and p not in _ASCEND_SUPPORTED_P
    ):
        return composed_pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
    # On iluvatar, CPU-mode torch.pairwise_distance precision does not match
    # the device-side implementation. Use composed op when TO_CPU is enabled.
    if flag_gems.vendor_name == "iluvatar" and cfg.TO_CPU:
        return composed_pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
    return torch.nn.functional.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)


# torch.nn.functional.pairwise_distance computes ||x1 - x2 + eps||_p and accepts
# any real p, including inf / -inf / 0. The gems kernel is expected to match torch
# for all of these (inf -> max|diff|, -inf -> min|diff|, 0 -> nonzero count).
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    P_LIST = [2.0]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    P_LIST = [0, 1.0, 1.5, 2.0]

SHAPES = [
    (7,),  # 1-D: a single pair of D-dim vectors -> scalar output
    (64, 64),
    (1024, 257),
    (64, 65536),  # split-K (small N, large D)
    (1, 10000000),
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("p", P_LIST + [float("inf"), float("-inf")])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_accuracy(shape, p, keepdim, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(
            x1, x2, p=p, eps=1e-6, keepdim=keepdim
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


# (x1_shape, x2_shape) pairs exercising broadcasting: torch broadcasts x2 against
# x1 before reducing over the last dim. Requires the op to broadcast internally.
BROADCAST_SHAPES = [
    ((4,), (1,)),  # 1-D vs single-element vector
    ((3, 4), (4,)),  # 2-D vs trailing 1-D
    ((3, 4), (1, 4)),  # 2-D vs row-broadcast
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("x1_shape, x2_shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_broadcast(x1_shape, x2_shape, p, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(x1_shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(x2_shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(x1, x2, p=p, eps=1e-6)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ndim >= 3: torch reduces over the LAST dim and returns shape[:-1]
# (shape[:-1] + (1,) with keepdim). The kernel must treat every leading dim as
# batch rows (N = numel // D), not collapse the whole tensor to a single pair.
NDIM_GE3_SHAPES = [
    (2, 3, 4),  # 3-D -> out (2, 3)
    (2, 3, 4, 5),  # 4-D -> out (2, 3, 4)
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("shape", NDIM_GE3_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_ndim3plus(shape, p, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(x1, x2, p=p, eps=1e-6)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Broadcasting where the broadcast result is ndim >= 3: x2 is broadcast against
# x1 to a 3-D/4-D shape, then reduced over the last dim. Exercises the broadcast
# path and the multi-row (numel // D) path together.
BROADCAST_NDIM3_SHAPES = [
    ((2, 3, 4), (3, 4)),  # 3-D broadcast -> out (2, 3)
    ((2, 3, 4, 5), (5,)),  # 4-D broadcast -> out (2, 3, 4)
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("x1_shape, x2_shape", BROADCAST_NDIM3_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_broadcast_ndim3plus(x1_shape, x2_shape, p, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(x1_shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(x2_shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(x1, x2, p=p, eps=1e-6)

    utils.gems_assert_close(res_out, ref_out, dtype)


# float64 accuracy: verify fp64 inputs are accumulated in fp64 (not silently
# downcast to fp32, which would lose ~9 digits of precision). Covers fast paths
# (p=2/1/inf/-inf) + a general p (1.5), on a per-row and a split-K shape.
FP64_P_LIST = [2.0, 1.0, 1.5, float("inf"), float("-inf")]
FP64_SHAPES = [(64, 257), (8, 65536)]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("shape", FP64_SHAPES)
@pytest.mark.parametrize("p", FP64_P_LIST)
def test_pairwise_distance_fp64(shape, p):
    if not utils.fp64_is_supported:
        pytest.skip("fp64 not supported on this device")
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    x2 = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(x1, x2, p=p, eps=1e-6)

    utils.gems_assert_close(res_out, ref_out, torch.float64)
