import pytest
import torch

from . import base, consts


@pytest.mark.not_equal
def test_not_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="not_equal",
        torch_op=torch.not_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def not_equal_scalar_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp, 0.5


@pytest.mark.not_equal_scalar
def test_not_equal_scalar():
    bench = base.GenericBenchmark(
        op_name="not_equal_scalar",
        input_fn=not_equal_scalar_input_fn,
        torch_op=torch.not_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
