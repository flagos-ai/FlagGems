import random

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts


class IndexSelectBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Shape for grad tensor (index_len, feature_dims...)
        INDEX_SELECT_BACKWARD_SHAPES = [
            (16, 64),
            (32, 128),
            (64, 256),
            (16, 16, 64),
            (32, 32, 128),
        ]
        return INDEX_SELECT_BACKWARD_SHAPES

    def set_more_metrics(self):
        return ["gbps"]


def _input_fn(shape, dtype, device):
    # Randomly choose dim
    dim = random.randint(0, len(shape) - 1)
    index_len = shape[dim]
    dim_size_out = index_len + 8  # Make output larger
    self_sizes = list(shape)
    self_sizes[dim] = dim_size_out
    self_sizes = tuple(self_sizes)

    grad = torch.randn(shape, dtype=dtype, device=device)
    index = torch.randint(0, dim_size_out, (index_len,), device=device)

    yield (grad, self_sizes, dim, index)


def _get_gbps(args, latency):
    grad, self_sizes, dim, index = args
    # Input: grad tensor + index tensor; Output: self_sizes tensor
    out = torch.empty(self_sizes, dtype=grad.dtype, device=grad.device)
    io_amount = shape_utils.size_in_bytes(grad) + shape_utils.size_in_bytes(out)
    io_amount += shape_utils.size_in_bytes(index)
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.index_select_backward
def test_index_select_backward():
    bench = IndexSelectBackwardBenchmark(
        op_name="index_select_backward",
        torch_op=torch.ops.aten.index_select_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.set_gems(flag_gems.index_select_backward)
    bench.run()
