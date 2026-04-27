import pytest
import torch

import flag_gems
from benchmark.performance_utils import Config, GenericBenchmark


def svd_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp,


SVD_SHAPES = [
    (3, 3),
    (8, 8),
    (12, 12),
    (14, 6),
    (6, 14),
    (11, 7),
    (15, 9),
    (9, 15),
    (16, 3, 3),
    (64, 8, 8),
    (128, 12, 12),
    (24, 24),
    (48, 16),
    (16, 48),
    (10, 24, 24),
    (16, 48, 16),
]


class SVDBenchmark(GenericBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = SVD_SHAPES
        self.shape_desc = "(..., M, N)"

    def set_more_shapes(self):
        return None

    def init_user_config(self):
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.shapes = SVD_SHAPES
        self.shape_desc = "(..., M, N)"


@pytest.mark.svd
def test_perf_svd():
    bench = SVDBenchmark(
        op_name="svd",
        torch_op=torch.svd,
        input_fn=svd_input_fn,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.svd)
    bench.run()
