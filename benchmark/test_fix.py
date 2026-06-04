import pytest
import torch

from flag_gems.testing import consts


@pytest.mark.fix
@pytest.mark.parametrize("shape", consts.POINTWISE_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_fix_benchmark(shape, dtype, benchmark):
    inp = torch.randn(shape, dtype=dtype, device="cuda") * 10
    benchmark(torch.ops.aten.fix, inp)
