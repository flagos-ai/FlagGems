import pytest
import torch

from . import base


@pytest.mark.nextafter_
def test_nextafter_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="nextafter_",
        torch_op=lambda a, b: a.nextafter_(b),
        # Kernel uses generic int bitcast based on dtype bitwidth; all float dtypes supported.
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
