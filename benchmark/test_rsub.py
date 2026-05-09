import pytest
import torch

from . import base, consts, utils


def _tensor_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, inp2


def _scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.rsub_tensor
def test_rsub_tensor():
    bench = base.GenericBenchmark(
        input_fn=_tensor_input_fn,
        op_name="rsub.Tensor",
        torch_op=torch.rsub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rsub_scalar
def test_rsub_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="rsub.Scalar",
        torch_op=torch.rsub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
