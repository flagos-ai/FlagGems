import pytest
import torch

import flag_gems

from . import base, consts

# Sparse semi-structured MM shapes
SPARSE_SEMI_STRUCTURED_MM_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 128),
    (512, 512),
]


class SparseSemiStructuredMMBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_SEMI_STRUCTURED_MM_SHAPES

    def get_input_iter(self, cur_dtype):
        K4 = 32  # K = 4 * K4
        for shape in self.shapes:
            M, N = shape
            mat1 = torch.randn(M, 4 * K4, dtype=cur_dtype, device=self.device)
            mat1_meta = torch.randint(
                0, 2, (M, K4), dtype=torch.bool, device=self.device
            )
            mat2 = torch.randn(4 * K4, N, dtype=cur_dtype, device=self.device)
            yield mat1, mat1_meta, mat2


@pytest.mark.sparse_semi_structured_mm
def test_sparse_semi_structured_mm():
    bench = SparseSemiStructuredMMBenchmark(
        op_name="sparse_semi_structured_mm",
        torch_op=flag_gems._sparse_semi_structured_mm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
