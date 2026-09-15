import pytest
import torch

import flag_gems

from . import base

VENDOR = flag_gems.vendor_name


_LINALG_DET_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (4, 32, 32),
    (8, 64, 64),
]

_LINALG_DET_DTYPES = [torch.float32]


class LinalgDetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _LINALG_DET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.underscore_linalg_det
def test_underscore_linalg_det():
    def _torch_linalg_det(A):
        return torch.ops.aten._linalg_det(A)

    def _gems_linalg_det(A):
        return flag_gems.ops._linalg_det(A)

    bench = LinalgDetBenchmark(
        op_name="_linalg_det",
        torch_op=_torch_linalg_det,
        dtypes=_LINALG_DET_DTYPES,
    )
    bench.set_gems(_gems_linalg_det)
    bench.run()
