import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, cur_dtype, device):
    dep_token = utils.generate_tensor_input(shape, cur_dtype, device)
    yield 5, 1, 10, dep_token


@pytest.mark.functional_sym_constrain_range_for_size
def test_functional_sym_constrain_range_for_size():
    bench = base.GenericBenchmark(
        op_name="functional_sym_constrain_range_for_size",
        torch_op=torch.ops.aten._functional_sym_constrain_range_for_size,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=_input_fn,
    )
    bench.run()
