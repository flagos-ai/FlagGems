import pytest
import torch

from . import base, consts, utils


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES (Issue #2693).
@pytest.mark.div_tensor
def test_div():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_tensor_
def test_div_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor_",
        torch_op=lambda a, b: a.div_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.div_out
def test_div_out():
    def input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        out = torch.empty_like(inp1)
        yield inp1, inp2, {"out": out}

    bench = base.GenericBenchmark(
        op_name="div_out",
        input_fn=input_fn,
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
