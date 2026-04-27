import pytest
import torch

from benchmark.performance_utils import GenericBenchmark


def svd_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp,


SVD_SHAPES = [
    (3, 3),
    (8, 8),
    (16, 16),
    (16, 8),
    (8, 16),
    (10, 3, 3),
    (100, 8, 8),
    (1000, 8, 8),
    (50, 16, 16),
    (200, 16, 16),
    (32, 32),
    (64, 32),
    (32, 64),
    (10, 32, 32),
    (50, 32, 32),
    (10, 64, 32),
]


class SVDBenchmark(GenericBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = SVD_SHAPES

    def set_more_shapes(self):
        return None


@pytest.mark.svd
def test_perf_svd():
    bench = SVDBenchmark(
        op_name="svd",
        torch_op=torch.svd,
        input_fn=svd_input_fn,
        dtypes=[torch.float32],
    )
    bench.run()
