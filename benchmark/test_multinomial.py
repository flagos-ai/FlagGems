import pytest
import torch

from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    dist = torch.rand(shape, dtype=dtype, device=device)
    n_samples = 10000
    yield dist, n_samples, True,


@pytest.mark.multinomial
def test_multinomial_with_replacement():
    bench = utils.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="multinomial",
        torch_op=torch.multinomial,
        dtypes=(torch.float16, torch.float32),
    )
    bench.run()
