import pytest
import torch

from . import base, utils


@pytest.mark.igammac
def test_igammac():
    bench = base.BinaryPointwiseBenchmark(
        op_name="igammac",
        torch_op=torch.special.gammaincc,
        dtypes=[torch.float32],
    )
    bench.run()


def _input_fn_out(shape, dtype, device):
    x = utils.generate_tensor_input(shape, dtype, device)
    y = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(x)
    yield x, y, {"out": out}


@pytest.mark.igammac_out
def test_igammac_out():
    bench = base.GenericBenchmark(
        op_name="igammac_out",
        input_fn=_input_fn_out,
        torch_op=torch.ops.aten.special_gammaincc.out,
        dtypes=[torch.float32],
    )
    bench.run()
