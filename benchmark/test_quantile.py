import pytest
import torch

from . import performance_utils as base
from . import utils


class QuantileBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        more_shapes_1d = [(4,), (1024,), (65535)]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 15, 3)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 3)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


def quantile_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    q = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=cur_dtype, device=device)
    yield inp, q, 0


# @pytest.mark.skipif(True, reason="Skipping Triton version due to poor performance")
@pytest.mark.quantile
def test_quantile():
    bench = QuantileBenchmark(
        input_fn=quantile_input_fn,
        op_name="quantile",
        torch_op=torch.quantile,
        dtypes=[torch.float32],
    )
    bench.run()
