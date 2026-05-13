import pytest
import torch

from . import base, consts


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


@pytest.mark.div_scalar_mode
def test_div_scalar_mode():
    def input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        yield inp, 0.5, {"rounding_mode": "floor"}

    bench = base.GenericBenchmark(
        op_name="div_scalar_mode",
        torch_op=torch.div,
        input_fn=input_fn,
        dtypes=[torch.float32],
    )
    bench.run()
