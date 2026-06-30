import pytest
import torch

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.001


@pytest.mark.eq
def test_eq():
    bench = base.BinaryPointwiseBenchmark(
        op_name="eq",
        torch_op=torch.eq,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.eq_scalar
def test_eq_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="eq_scalar",
        torch_op=torch.eq,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
