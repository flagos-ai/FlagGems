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


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.div_scalar
def test_div_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="div_scalar",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
