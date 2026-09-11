import pytest
import torch

from . import base, consts, utils


def _input_fn_scalar(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


@pytest.mark.less_equal_
def test_less_equal_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="less_equal_",
        torch_op=lambda a, b: torch.ops.aten.less_equal_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.less_equal_scalar_
def test_less_equal_scalar_():
    bench = base.GenericBenchmark(
        op_name="less_equal_scalar_",
        input_fn=_input_fn_scalar,
        torch_op=lambda a, b: torch.ops.aten.less_equal_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
