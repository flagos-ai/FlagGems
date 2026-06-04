import pytest
import torch

from flag_gems.testing import consts


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", consts.POINTWISE_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_deg2rad_benchmark(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    benchmark(torch.ops.aten.deg2rad, inp)
