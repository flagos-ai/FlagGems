import math

import pytest
import torch

import flag_gems

from . import base


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


def _small_ops_det(matrix):
    n = matrix.shape[-1]
    batch_shape = matrix.shape[:-2]
    batch = math.prod(batch_shape) if batch_shape else 1
    lu = matrix.clone().reshape(batch, n, n)
    sign = torch.ones(batch, dtype=matrix.dtype, device=matrix.device)
    batch_index = torch.arange(batch, device=matrix.device)
    for k in range(n):
        pivot_row = lu[:, k:, k].abs().argmax(dim=-1) + k
        swap = pivot_row != k
        sign = torch.where(swap, -sign, sign)
        row_k = lu[:, k, :].clone()
        row_p = lu[batch_index, pivot_row, :].clone()
        lu[:, k, :] = row_p
        lu[batch_index[swap], pivot_row[swap], :] = row_k[swap]
        pivot = lu[:, k, k]
        safe_pivot = torch.where(pivot == 0, torch.ones_like(pivot), pivot)
        column = lu[:, k + 1 :, k]
        multiplier = torch.where(
            (pivot == 0).unsqueeze(-1),
            torch.zeros_like(column),
            column / safe_pivot.unsqueeze(-1),
        )
        lu[:, k + 1 :, k] = multiplier
        lu[:, k + 1 :, k + 1 :] -= (
            multiplier.unsqueeze(-1) * lu[:, k : k + 1, k + 1 :]
        )
    result = lu.diagonal(dim1=-2, dim2=-1).prod(dim=-1) * sign
    return result.reshape(batch_shape)


def _torch_det(matrix):
    if matrix.device.type == "npu":
        return _small_ops_det(matrix)
    return torch.det(matrix)


class DetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DET_SHAPES
        self.shape_desc = "batch..., N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        return (torch.randn(shape, dtype=case.dtype, device=self.device),)


@pytest.mark.det
def test_det():
    bench = DetBenchmark(
        op_name="det",
        torch_op=_torch_det,
        dtypes=DET_DTYPES,
    )
    bench.set_gems(lambda matrix: flag_gems.det(matrix))
    bench.run()
