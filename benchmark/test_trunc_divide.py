import pytest
import torch

from . import base, consts, utils


def _binary_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, inp2


# Note: tl.math.div_rz only supports float32, so we only benchmark float32
@pytest.mark.trunc_divide
def test_trunc_divide():
    bench = base.GenericBenchmark(
        op_name="trunc_divide",
        input_fn=_binary_input_fn,
        torch_op=lambda a, b: torch.div(a, b, rounding_mode="trunc"),
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.trunc_divide_
def test_trunc_divide_inplace():
    bench = base.GenericBenchmark(
        op_name="trunc_divide_",
        input_fn=_binary_input_fn,
        torch_op=lambda a, b: a.div_(b, rounding_mode="trunc"),
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
