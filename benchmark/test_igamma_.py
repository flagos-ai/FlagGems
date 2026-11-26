import pytest
import torch

from . import base


@pytest.mark.igamma_
def test_igamma_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="igamma_",
        torch_op=lambda a, b: a.igamma_(b),
        # torch.igamma_ CUDA reference does not support fp16/bfloat16.
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
