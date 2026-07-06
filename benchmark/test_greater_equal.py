import pytest
import torch

from . import base, consts, utils


def _greater_equal_scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.greater_equal
def test_greater_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater_equal",
        torch_op=torch.greater_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_equal_
def test_greater_equal_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater_equal_",
        torch_op=lambda a, b: a.greater_equal_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.greater_equal_scalar
def test_greater_equal_scalar():
    bench = base.GenericBenchmark(
        input_fn=_greater_equal_scalar_input_fn,
        op_name="greater_equal_scalar",
        torch_op=torch.greater_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
