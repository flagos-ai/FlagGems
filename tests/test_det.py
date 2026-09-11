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

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

DET_SHAPES = (
    [(2, 2), (16, 16), (128, 128)]
    if QUICK_MODE
    else [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ]
)
DET_BATCH_SHAPES = (
    [(2, 3, 16, 16)]
    if QUICK_MODE
    else [
        (2, 4, 4),
        (3, 8, 8),
        (2, 3, 16, 16),
        (4, 2, 32, 32),
        (3, 64, 64),
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
)
# det uses LU decomposition and only supports float32/float64 (no fp16/bf16),
# matching torch.det; float64 is added only when the device supports it.
DET_DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])


def _scaled_randn(shape, dtype):
    n = shape[-1]
    return torch.randn(shape, dtype=dtype, device=flag_gems.device) / math.sqrt(n)


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


def _ref_det(A):
    if A.device.type == "npu":
        return _small_ops_det(A)
    prev = torch.get_num_threads()
    torch.set_num_threads(min(prev, 64))
    try:
        return torch.det(A)
    finally:
        torch.set_num_threads(prev)


@pytest.mark.det
@pytest.mark.parametrize("shape", DET_SHAPES)
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_random(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.det
@pytest.mark.parametrize("shape", DET_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_random_batch(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.det
@pytest.mark.parametrize("shape", [(4, 4), (16, 16), (2, 32, 32)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_positive_definite(shape, dtype):
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A = (B @ B.transpose(-2, -1) + n * eye) / (2 * n)

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.det
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (2, 3, 3)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_negative_determinant(shape, dtype):
    n = shape[-1]
    A = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A[[0, 1]] = A[[1, 0]]
    if len(shape) > 2:
        A = A.unsqueeze(0).expand(shape).contiguous()

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.det
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_diagonal(shape, dtype):
    n = shape[-1]
    diag = torch.arange(1, n + 1, dtype=dtype, device=flag_gems.device)
    diag[0] = -diag[0]
    A = torch.diag(diag)
    if len(shape) > 2:
        A = A.unsqueeze(0).expand(shape).contiguous()

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.det
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (2, 3, 3)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_singular(shape, dtype):
    A = _scaled_randn(shape, dtype)
    A[..., 0, :] = A[..., 1, :]

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.det
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_empty(dtype):
    for shape in [(0, 0), (3, 0, 0)]:
        A = torch.empty(shape, dtype=dtype, device=flag_gems.device)

        ref_A = utils.to_reference(A)
        ref_out = _ref_det(ref_A)

        res_out = flag_gems.det(A)

        utils.gems_assert_close(res_out, ref_out, dtype)

    A = torch.empty((0, 3, 3), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.det(A)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype


@pytest.mark.det
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_det_non_contiguous(dtype):
    n = 16
    base = _scaled_randn((n, n), dtype)
    A = base.transpose(-2, -1)
    assert not A.is_contiguous()

    ref_A = utils.to_reference(A)
    ref_out = _ref_det(ref_A)

    res_out = flag_gems.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.det
def test_det_errors():
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.det(torch.randn(3, 4, device=flag_gems.device))

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.det(torch.randn(3, device=flag_gems.device))
