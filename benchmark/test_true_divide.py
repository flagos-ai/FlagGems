import pytest
import torch

from . import base, consts, utils


def _true_divide_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.div_tensor
def test_true_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.true_divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_scalar
def test_true_divide_scalar():
    bench = base.GenericBenchmark(
        input_fn=_true_divide_scalar_input_fn,
        op_name="div_scalar",
        torch_op=torch.true_divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_scalar_
def test_true_divide_inplace_scalar():
    bench = base.GenericBenchmark(
        input_fn=_true_divide_scalar_input_fn,
        op_name="div_scalar_",
        torch_op=lambda a, b: a.true_divide_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
