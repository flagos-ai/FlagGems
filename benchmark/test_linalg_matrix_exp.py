import math

import pytest
import torch

import flag_gems
from flag_gems import linalg_matrix_exp, linalg_matrix_exp_out

from . import base


def _small_ops_matrix_exp(A):
    n = A.shape[-1]
    if n == 0:
        return A.clone()
    batch_shape = A.shape[:-2]
    B = math.prod(batch_shape) if batch_shape else 1
    A_flat = A.reshape(B, n, n)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    out = torch.empty_like(A_flat)
    for i in range(B):
        m = A_flat[i]
        norm = m.abs().sum(-2).max().item()
        s = math.ceil(math.log2(norm)) if norm > 1.0 else 0
        ms = m * (2.0**-s)
        r = eye.clone()
        term = eye.clone()
        for k in range(1, 31):
            term = (term @ ms) / k
            r = r + term
        for _ in range(s):
            r = r @ r
        out[i] = r
    return out.reshape(batch_shape + (n, n) if batch_shape else (n, n))


def _torch_matrix_exp(A):
    if A.device.type == "npu":
        return _small_ops_matrix_exp(A)
    return torch.linalg.matrix_exp(A)


def _torch_matrix_exp_out(A, *, out):
    out.copy_(_torch_matrix_exp(A))
    return out


MATRIX_EXP_SHAPES = [
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

MATRIX_EXP_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class MatrixExpBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MATRIX_EXP_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.linalg_matrix_exp
def test_linalg_matrix_exp():
    bench = MatrixExpBenchmark(
        op_name="linalg_matrix_exp",
        torch_op=_torch_matrix_exp,
        dtypes=MATRIX_EXP_DTYPES,
    )
    bench.set_gems(linalg_matrix_exp)
    bench.run()


class MatrixExpOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MATRIX_EXP_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out = torch.empty(shape, dtype=cur_dtype, device=self.device)
            yield (A, {"out": out})


@pytest.mark.linalg_matrix_exp_out
def test_linalg_matrix_exp_out():
    bench = MatrixExpOutBenchmark(
        op_name="linalg_matrix_exp_out",
        torch_op=_torch_matrix_exp_out,
        dtypes=MATRIX_EXP_DTYPES,
    )
    bench.set_gems(linalg_matrix_exp_out)
    bench.run()
