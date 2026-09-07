import pytest
import torch

import flag_gems
from flag_gems.utils import get_device_properties

from . import accuracy_utils as utils

# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
PDIST_SHAPES = utils.PDIST_SHAPES
# Wider than the kernels' 1024-lane feature tile, so the tile loop is exercised
PDIST_WIDE_SHAPES = [(2, 1025), (8, 2048), (16, 4100)]


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES + PDIST_WIDE_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 2.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES + PDIST_WIDE_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p1(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 1.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES + PDIST_WIDE_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_pinf(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = float("inf")
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES + PDIST_WIDE_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p_general(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 3.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES + PDIST_WIDE_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p0(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 0.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES + PDIST_WIDE_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p_large(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 100.0
    ref_out = torch.pdist(ref_inp, p=p)
    with flag_gems.use_gems():
        res_out = torch.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.skipif(
    get_device_properties(0).total_memory < (16 * 1024**3),
    reason="the pdist output alone is 4.3 GB at this row count",
)
def test_pdist_pair_count_beyond_int32():
    # N * (N - 1) exceeds int32 here; with 32-bit pair arithmetic the kernels
    # returned without writing a single element. Spot-check pairs against
    # directly computed distances rather than materializing a second 4.3 GB
    # reference; the last pair only decodes correctly with 64-bit indexing.
    N = 46342
    x = torch.randn(N, 1, dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        res = torch.pdist(x)

    assert res.numel() == N * (N - 1) // 2
    xf = x.flatten()

    def flat_index(i, j):
        return i * N - (i * (i + 1)) // 2 + (j - i - 1)

    for i, j in [(0, 1), (23170, 23171), (0, N - 1), (N - 2, N - 1)]:
        torch.testing.assert_close(res[flat_index(i, j)], (xf[i] - xf[j]).abs())
