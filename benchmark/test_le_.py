import pytest
import torch

from . import base, consts, utils


def _input_fn_scalar(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


@pytest.mark.le_
def test_le_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="le_",
        torch_op=lambda a, b: torch.ops.aten.le_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.le_scalar_
def test_le_scalar_():
    bench = base.GenericBenchmark(
        op_name="le_scalar_",
        input_fn=_input_fn_scalar,
        torch_op=lambda a, b: torch.ops.aten.le_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
