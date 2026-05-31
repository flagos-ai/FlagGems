import pytest
import torch

from . import base, consts


def log_normal__input_fn(shape, cur_dtype, device):
    self = torch.empty(shape, dtype=cur_dtype, device=device)
    mean = 1.0
    std = 2.0
    yield self, mean, std


@pytest.mark.log_normal_
def test_log_normal_inplace():
    bench = base.GenericBenchmark(
        op_name="log_normal_",
        input_fn=log_normal__input_fn,
        torch_op=torch.Tensor.log_normal_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
