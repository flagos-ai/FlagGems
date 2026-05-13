import pytest
import torch

from . import base, consts


@pytest.mark.greater
def test_greater():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def greater_out_input_fn(shape, dtype, device):
    inp1 = base.generate_tensor_input(shape, dtype, device)
    inp2 = base.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=torch.bool, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.greater_out
def test_greater_out():
    bench = base.GenericBenchmark(
        op_name="greater_out",
        torch_op=torch.greater,
        input_fn=greater_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
