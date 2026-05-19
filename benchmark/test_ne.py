import pytest
import torch

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0


@pytest.mark.ne
def test_ne():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ne",
        torch_op=torch.ne,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.ne_scalar
def test_ne_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="ne_scalar",
        torch_op=torch.ne,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
