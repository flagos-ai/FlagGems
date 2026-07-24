import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def composed_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """NPU-native pairwise_distance via basic torch ops (sub+abs+pow+sum).
    Supports arbitrary p, unlike torch_npu's LpNormV2 which only accepts {0,1,2}.
    Used as the reference when aten would crash on p not in {0,1,2,inf,-inf}."""
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
_ATEN_SUPPORTED_P = (0.0, 1.0, 2.0)


def _ref_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    if not cfg.TO_CPU and p not in _ATEN_SUPPORTED_P:
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
    P_LIST = [-1.1, 0, 1.0, 1.5, 2.0, 4.3]

SHAPES = [
    (7,),  # 1-D: a single pair of D-dim vectors -> scalar output
    (64, 64),
    (1024, 257),
    (1, 10000000),
    (8, 8192),
    (64, 65536),
    (100, 5000),
    (128, 4097),
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
    ((4,), (1,)),  # 1-D vs single-element vector (e.g. [1,2,4,100] vs [3])
    ((4,), (4,)),  # no broadcast (sanity)
    ((3, 4), (4,)),  # 2-D vs trailing 1-D
    ((3, 4), (1, 4)),  # 2-D vs row-broadcast
    ((2, 8), (1,)),  # 2-D vs scalar-vector
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("x1_shape, x2_shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_broadcast(x1_shape, x2_shape, p, keepdim, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(x1_shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(x2_shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(
            x1, x2, p=p, eps=1e-6, keepdim=keepdim
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


# ndim >= 3: torch reduces over the LAST dim and returns shape[:-1]
# (shape[:-1] + (1,) with keepdim). The kernel must treat every leading dim as
# batch rows (N = numel // D), not collapse the whole tensor to a single pair.
NDIM_GE3_SHAPES = [
    (2, 3, 4),  # 3-D -> out (2, 3)
    (4, 8, 16),  # 3-D, larger
    (2, 3, 4, 5),  # 4-D -> out (2, 3, 4)
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("shape", NDIM_GE3_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_ndim3plus(shape, p, keepdim, dtype):
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


# Broadcasting where the broadcast result is ndim >= 3: x2 is broadcast against
# x1 to a 3-D/4-D shape, then reduced over the last dim. Exercises the broadcast
# path and the multi-row (numel // D) path together.
BROADCAST_NDIM3_SHAPES = [
    ((2, 3, 4), (3, 4)),  # -> out (2, 3)
    ((2, 3, 4), (4,)),  # -> out (2, 3)
    ((2, 3, 4), (1,)),  # -> out (2, 3)
    ((2, 3, 4, 5), (4, 5)),  # -> out (2, 3, 4)
    ((2, 3, 4, 5), (5,)),  # -> out (2, 3, 4)
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("x1_shape, x2_shape", BROADCAST_NDIM3_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pairwise_distance_broadcast_ndim3plus(x1_shape, x2_shape, p, keepdim, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(x1_shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(x2_shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = _ref_pairwise_distance(ref_x1, ref_x2, p=p, eps=1e-6, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(
            x1, x2, p=p, eps=1e-6, keepdim=keepdim
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
