import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Covers representative (N, M) pairs exercising different grid sizes and BLOCK_M widths.
PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


@pytest.mark.pdist_forward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# _pdist_forward only supports float32 in the reference implementation
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_forward(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 2.0
    ref_out = torch.ops.aten._pdist_forward(ref_inp, p)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_forward(inp, p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_forward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# _pdist_forward only supports float32 in the reference implementation
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_forward_p1(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 1.0
    ref_out = torch.ops.aten._pdist_forward(ref_inp, p)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_forward(inp, p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_forward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# _pdist_forward only supports float32 in the reference implementation
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_forward_pinf(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = float("inf")
    ref_out = torch.ops.aten._pdist_forward(ref_inp, p)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_forward(inp, p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_forward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# _pdist_forward only supports float32 in the reference implementation
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_forward_general_p(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 3.0
    ref_out = torch.ops.aten._pdist_forward(ref_inp, p)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_forward(inp, p)

    utils.gems_assert_close(res_out, ref_out, dtype)
