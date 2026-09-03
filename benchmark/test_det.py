# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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


# Single unbatched matrices (16x16..256x256) plus batched cases that cover both
# large batch / small matrix and small batch / large matrix regimes, exercising
# the register, blocked, and panel kernel paths across matrix sizes.
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

# det uses LU decomposition and only supports float32/float64 (no fp16/bf16),
# matching torch.det; float64 is added only when the device supports it.
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
