import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    inp.view(-1)[0] = float("nan")
    if inp.numel() > 1:
        inp.view(-1)[1] = float("inf")
    if inp.numel() > 2:
        inp.view(-1)[2] = float("-inf")

    yield inp,


@pytest.mark.nan_to_num
def test_nan_to_num():
    bench = base.GenericBenchmark(
        op_name="nan_to_num",
        input_fn=_input_fn,
        torch_op=torch.nan_to_num,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
