import pytest
import torch

from . import base, consts, utils


def _input_fn_scalar(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = 0
    yield inp1, inp2


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
