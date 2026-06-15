import pytest
import torch

from . import base, consts, utils


@pytest.mark.ixor__
def test_ixor__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ixor__",
        torch_op=torch.ops.aten.__ixor__.Tensor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _scalar_input_fn_ixor(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    scalar = True if dtype == torch.bool else 0x00FF
    yield inp, scalar


@pytest.mark.ixor___scalar
def test_ixor___scalar():
    bench = base.GenericBenchmark(
        op_name="ixor___scalar",
        torch_op=torch.ops.aten.__ixor__.Scalar,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        input_fn=_scalar_input_fn_ixor,
        is_inplace=True,
    )
    bench.run()
