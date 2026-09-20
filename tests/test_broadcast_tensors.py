import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# Test for broadcast_tensors
BROADCAST_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (2, 3), (1, 3), (3, 1), (2, 3, 4), (1, 3, 4), (3, 1, 4), (3, 4, 1)]
)


@pytest.mark.broadcast_tensors
@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_tensors(shape, dtype):
    # Test broadcasting two tensors
    # Create two tensors with different shapes that can be broadcast
    if len(shape) == 1:
        # For 1D: broadcast (n,) and (1,)
        inp1 = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(1, dtype=dtype, device=flag_gems.device)
    elif len(shape) == 2:
        # For 2D: broadcast (n, m) and (1, m) or (n, 1)
        if shape[0] == 1:
            inp1 = torch.randn(1, shape[1], dtype=dtype, device=flag_gems.device)
            inp2 = torch.randn(shape[0], shape[1], dtype=dtype, device=flag_gems.device)
        else:
            inp1 = torch.randn(shape[0], 1, dtype=dtype, device=flag_gems.device)
            inp2 = torch.randn(shape[0], shape[1], dtype=dtype, device=flag_gems.device)
    else:
        # For 3D: broadcast (1, m, n) and (k, m, n)
        inp1 = torch.randn(1, shape[1], shape[2], dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(
            shape[0], shape[1], shape[2], dtype=dtype, device=flag_gems.device
        )

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.broadcast_tensors(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.broadcast_tensors(inp1, inp2)

    # Compare each tensor in the list
    assert len(res_out) == len(ref_out)
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_equal(res, ref)


@pytest.mark.broadcast_tensors
@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_tensors_three_inputs(shape, dtype):
    # Test broadcasting three tensors
    inp1 = torch.randn(1, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape[-1], dtype=dtype, device=flag_gems.device)
    inp3 = torch.randn(*shape, dtype=dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)
    ref_inp3 = utils.to_reference(inp3)

    ref_out = torch.broadcast_tensors(ref_inp1, ref_inp2, ref_inp3)
    with flag_gems.use_gems():
        res_out = torch.broadcast_tensors(inp1, inp2, inp3)

    # Compare each tensor in the list
    assert len(res_out) == len(ref_out)
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_equal(res, ref)
