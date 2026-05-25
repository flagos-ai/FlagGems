import pytest
import torch

from . import base

# Square matrix shapes for linalg_slogdet benchmark
SLOGDET_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


class SlogdetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet():
    from flag_gems.ops.linalg_slogdet import linalg_slogdet

    bench = SlogdetBenchmark(
        op_name="linalg_slogdet",
        torch_op=torch.linalg.slogdet,
        # slogdet involves log and division, float16/bf16 lack precision
        dtypes=[torch.float32],
    )
    bench.set_gems(linalg_slogdet)
    bench.run()
