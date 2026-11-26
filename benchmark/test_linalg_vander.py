import pytest
import torch

from . import base

# linalg_vander benchmark
VANDER_SHAPES = [
    (4,),
    (8,),
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (512,),
    (1024,),
    (2, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
]


class LinalgVanderBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = VANDER_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, {}


@pytest.mark.linalg_vander
def test_linalg_vander():
    bench = LinalgVanderBenchmark(
        op_name="linalg_vander",
        torch_op=torch.linalg.vander,
        # torch.linalg.vander only supports float32 on GPU;
        # float16/bf16 are not accelerated and produce low-precision results
        dtypes=[torch.float32],
    )
    bench.run()
