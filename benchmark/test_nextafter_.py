import pytest
import torch

from . import base


@pytest.mark.nextafter_
def test_nextafter_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="nextafter_",
        torch_op=lambda a, b: a.nextafter_(b),
        # Kernel uses int32 bitcast which only supports float32; fp16/bf16 produce incorrect results.
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
