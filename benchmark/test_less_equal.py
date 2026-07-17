import pytest
import torch

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0


@pytest.mark.less_equal
def test_less_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="less_equal",
        torch_op=torch.less_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.less_equal_scalar
def test_less_equal_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="less_equal_scalar",
        torch_op=torch.less_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
