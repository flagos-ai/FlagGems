import pytest
import torch

from . import base, consts, utils


def _binary_input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2


@pytest.mark.special_xlogy
def test_special_xlogy():
    bench = base.GenericBenchmark(
        op_name="special_xlogy",
        input_fn=_binary_input_fn,
        torch_op=torch.special.xlogy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_xlogy_
def test_special_xlogy_():
    bench = base.GenericBenchmark(
        op_name="special_xlogy_",
        input_fn=_binary_input_fn,
        torch_op=lambda x, y: torch.ops.aten.xlogy_(x, y),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
