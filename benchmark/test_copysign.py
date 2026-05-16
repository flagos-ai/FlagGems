import pytest
import torch

from . import base, consts, utils


@pytest.mark.copysign
def test_copysign():
    bench = base.BinaryPointwiseBenchmark(
        op_name="copysign",
        torch_op=torch.copysign,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.copysign_out
def test_copysign_out():
    def input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        out = torch.empty(shape, dtype=dtype, device=device)
        yield inp1, inp2, {"out": out}

    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="copysign_out",
        torch_op=torch.copysign,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
