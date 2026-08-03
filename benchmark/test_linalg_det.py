import pytest
import torch

import flag_gems
from flag_gems import linalg_det

from . import base


def _torch_det(A):
    if A.device.type == "npu":
        out = torch.linalg.qr(A, mode="r")
        r = out[1] if isinstance(out, tuple) else out
        return r.diagonal(dim1=-2, dim2=-1).prod(-1)
    return torch.linalg.det(A)


def _torch_det_out(A, *, out):
    if A.device.type == "npu":
        out.copy_(_torch_det(A))
        return out
    return torch.linalg.det(A, out=out)


DET_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (4096, 4, 4),
    (1024, 8, 8),
    (1024, 16, 16),
    (128, 16, 16),
    (4, 32, 32),
    (512, 32, 32),
    (256, 64, 64),
    (32, 128, 128),
    (8, 256, 256),
]

DET_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class DetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.linalg_det
def test_linalg_det():
    bench = DetBenchmark(
        op_name="linalg_det",
        torch_op=_torch_det,
        dtypes=DET_DTYPES,
    )
    bench.set_gems(linalg_det)
    bench.run()


class DetOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out = torch.empty(shape[:-2], dtype=cur_dtype, device=self.device)
            yield (A, {"out": out})


@pytest.mark.linalg_det_out
def test_linalg_det_out():
    bench = DetOutBenchmark(
        op_name="linalg_det_out",
        torch_op=_torch_det_out,
        dtypes=DET_DTYPES,
    )
    bench.set_gems(linalg_det)
    bench.run()
