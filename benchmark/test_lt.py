import pytest
import torch

from . import base, consts, utils


def _input_fn_scalar(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = 0
    yield inp1, inp2


@pytest.mark.lt
def test_lt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lt_scalar
def test_lt_scalar():
    bench = base.GenericBenchmark(
        op_name="lt_scalar",
        input_fn=_input_fn_scalar,
        torch_op=torch.lt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
