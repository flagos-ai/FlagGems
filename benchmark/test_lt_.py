import pytest
import torch

from . import base, consts, utils


def _input_fn_scalar(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


@pytest.mark.lt_
def test_lt_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt_",
        torch_op=lambda a, b: torch.ops.aten.lt_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.lt_scalar_
def test_lt_scalar_():
    bench = base.GenericBenchmark(
        op_name="lt_scalar_",
        input_fn=_input_fn_scalar,
        torch_op=lambda a, b: torch.ops.aten.lt_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
