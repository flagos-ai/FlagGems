import pytest
import torch

from . import base, consts, utils

SCALAR_VALUES = (0, 1.0, -1.0, 0.5)


def _input_fn_scalar(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    for scalar in SCALAR_VALUES:
        yield inp1, scalar


@pytest.mark.less
def test_less():
    bench = base.BinaryPointwiseBenchmark(
        op_name="less",
        torch_op=torch.less,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.less_scalar
def test_less_scalar():
    bench = base.GenericBenchmark(
        op_name="less_scalar",
        input_fn=_input_fn_scalar,
        torch_op=torch.less,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
