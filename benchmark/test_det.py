import math

import pytest
import torch

import flag_gems
from flag_gems import det

from . import base, consts
from .conftest import Config

VENDOR = flag_gems.vendor_name


if VENDOR == "ascend":
    Config.mode = consts.BenchMode.OPERATOR


def _small_ops_det(A):
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    B = math.prod(batch_shape) if batch_shape else 1
    LU = A.clone().reshape(B, n, n)
    sign = torch.ones(B, dtype=A.dtype, device=A.device)
    bidx = torch.arange(B, device=A.device)
    for k in range(n):
        p = LU[:, k:, k].abs().argmax(dim=-1) + k
        swap = p != k
        sign = torch.where(swap, -sign, sign)
        row_k = LU[:, k, :].clone()
        row_p = LU[bidx, p, :].clone()
        LU[:, k, :] = row_p
        LU[bidx[swap], p[swap], :] = row_k[swap]
        pivot = LU[:, k, k]
        safe_pivot = torch.where(pivot == 0, torch.ones_like(pivot), pivot)
        col = LU[:, k + 1 :, k]
        mult = torch.where(
            (pivot == 0).unsqueeze(-1),
            torch.zeros_like(col),
            col / safe_pivot.unsqueeze(-1),
        )
        LU[:, k + 1 :, k] = mult
        LU[:, k + 1 :, k + 1 :] -= mult.unsqueeze(-1) * LU[:, k : k + 1, k + 1 :]
    det = LU.diagonal(dim1=-2, dim2=-1).prod(dim=-1) * sign
    return det.reshape(batch_shape)


def _torch_det(A):
    if A.device.type == "npu":
        return _small_ops_det(A)
    return torch.det(A)


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


@pytest.mark.det
def test_det():
    bench = DetBenchmark(
        op_name="det",
        torch_op=_torch_det,
        dtypes=DET_DTYPES,
    )
    bench.set_gems(det)
    bench.run()
