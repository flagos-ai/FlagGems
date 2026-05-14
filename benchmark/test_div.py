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


def div_scalar_mode_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, 0.001, {"rounding_mode": None}


@pytest.mark.div_scalar_mode_
def test_div_scalar_mode_():
    bench = base.GenericBenchmark(
        op_name="div_scalar_mode_",
        torch_op=lambda a, scalar, rounding_mode=None: a.div_(
            scalar, rounding_mode=rounding_mode
        ),
        dtypes=consts.FLOAT_DTYPES,
        input_fn=div_scalar_mode_input_fn,
        is_inplace=True,
    )
    bench.run()
