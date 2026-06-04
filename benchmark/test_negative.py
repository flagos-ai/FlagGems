import pytest
import torch

from flag_gems.testing import benchmark, consts


@pytest.mark.negative
@pytest.mark.parametrize("shape", consts.POINTWISE_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_negative_benchmark(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    benchmark(torch.ops.aten.negative, inp)
