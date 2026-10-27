import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Large reduce dimension to exercise the scan path.
CUMTRAPZ_LARGE_SHAPES = [(1, 8192), (32, 4097)]


# cumulative_trapezoid reduces along a dim, so the scalar shape () is invalid
# (matches torch, which raises for 0-dim input). Use shapes with rank >= 1.
CUMTRAPZ_SHAPES = [
    (1,),
    (8,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


@pytest.mark.cumulative_trapezoid
@pytest.mark.parametrize("shape", CUMTRAPZ_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cumulative_trapezoid(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.cumulative_trapezoid(ref_inp)
    res_out = flag_gems.cumulative_trapezoid(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cumulative_trapezoid
@pytest.mark.parametrize("shape", CUMTRAPZ_LARGE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cumulative_trapezoid_large(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.cumulative_trapezoid(ref_inp)
    res_out = flag_gems.cumulative_trapezoid(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cumulative_trapezoid
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cumulative_trapezoid_dim(dim, dtype):
    # Fixed 3-D shape with distinct, non-power-of-2 extents so integrating along
    # each candidate dim (0/1/2/-1) exercises a different reduce length.
    shape = (8, 33, 17)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.cumulative_trapezoid(ref_inp, dim=dim)
    res_out = flag_gems.cumulative_trapezoid(res_inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cumulative_trapezoid
@pytest.mark.parametrize("dx", [0.5, 2.0, 3.5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cumulative_trapezoid_dx(dx, dtype):
    # Fixed shape matching one of the benchmarked workloads; the dx sweep here
    # checks uniform-spacing scaling independent of tensor size.
    shape = (20, 320, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.cumulative_trapezoid(ref_inp, dx=dx)
    res_out = flag_gems.cumulative_trapezoid(res_inp, dx=dx)

    utils.gems_assert_close(res_out, ref_out, dtype)
