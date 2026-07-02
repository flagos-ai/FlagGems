import pytest
import torch

from flag_gems.ops._cdist_forward import _cdist_forward

from . import base

# Shapes for cdist benchmark: (P, M), (R, M) -> (P, R)
# torch.cdist doesn't support float16 on CUDA
CDIST_FORWARD_SHAPES = [
    ((4, 8), (6, 8)),
    ((8, 16), (8, 16)),
    ((16, 32), (16, 32)),
    ((32, 64), (32, 64)),
    ((64, 128), (64, 128)),
]


class CdistForwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CDIST_FORWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape1, shape2 in self.shapes:
            x1 = torch.randn(*shape1, dtype=cur_dtype, device=self.device)
            x2 = torch.randn(*shape2, dtype=cur_dtype, device=self.device)
            yield x1, x2, 2.0

    def get_tflops(self, op, *args, **kwargs):
        x1, x2, _ = args
        # FLOPs = 2 * P * R * M (for L2 distance computation)
        return 2 * x1.shape[-2] * x2.shape[-2] * x1.shape[-1]


@pytest.mark.cdist_forward
def test_cdist_forward():
    bench = CdistForwardBenchmark(
        op_name="cdist_forward",
        torch_op=torch.cdist,
        # torch.cdist doesn't support float16 on CUDA; only float32 is numerically stable
        dtypes=[torch.float32],
    )
    bench.set_gems(_cdist_forward)
    bench.run()
