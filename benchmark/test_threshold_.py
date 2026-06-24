import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp,


@pytest.mark.threshold_
def test_threshold_():
    bench = base.GenericBenchmark(
        op_name="threshold_",
        input_fn=_input_fn,
        torch_op=lambda x: torch.threshold_(x, 0.0, -1.0),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
