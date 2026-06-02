import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 3.14


@pytest.mark.clamp_max
def test_clamp_max():
    bench = base.GenericBenchmark(
        op_name="clamp_max",
        input_fn=_input_fn,
        torch_op=torch.clamp_max,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_max_
def test_clamp_max_inplace():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clamp_max_",
        torch_op=torch.clamp_max_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
