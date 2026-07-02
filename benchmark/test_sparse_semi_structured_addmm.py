import pytest
import torch

import flag_gems

from . import base, consts

# Sparse semi-structured addmm shapes
SPARSE_SEMI_STRUCTURED_ADDMM_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 128),
    (512, 512),
]


class SparseSemiStructuredAddmmBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_SEMI_STRUCTURED_ADDMM_SHAPES

    def get_input_iter(self, cur_dtype):
        K4 = 32  # K = 4 * K4
        for shape in self.shapes:
            M, N = shape
            input_tensor = torch.randn(M, N, dtype=cur_dtype, device=self.device)
            mat1 = torch.randn(M, 4 * K4, dtype=cur_dtype, device=self.device)
            mat1_meta = torch.randint(
                0, 2, (M, K4), dtype=torch.bool, device=self.device
            )
            mat2 = torch.randn(4 * K4, N, dtype=cur_dtype, device=self.device)
            yield input_tensor, mat1, mat1_meta, mat2


@pytest.mark.sparse_semi_structured_addmm
def test_sparse_semi_structured_addmm():
    bench = SparseSemiStructuredAddmmBenchmark(
        op_name="sparse_semi_structured_addmm",
        torch_op=flag_gems._sparse_semi_structured_addmm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
