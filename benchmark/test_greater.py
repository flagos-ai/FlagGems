import pytest
import torch

from . import base, consts, utils


@pytest.mark.greater
def test_greater():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def greater_scalar_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=torch.bool, device=device)
    yield inp, 0, {"out": out}


@pytest.mark.greater_scalar_out
def test_greater_scalar_out():
    bench = base.GenericBenchmark(
        op_name="greater_scalar_out",
        input_fn=greater_scalar_out_input_fn,
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
