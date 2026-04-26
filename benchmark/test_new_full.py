import pytest
import torch

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, shape, 3.1415926  # self, size, fill_value


@pytest.mark.new_full
def test_new_full():
    bench = base.GenericBenchmark(
        op_name="new_full", input_fn=_input_fn, torch_op=torch.Tensor.new_full
    )
    bench.run()
