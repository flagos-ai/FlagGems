# Copyright 2026 FlagOS Contributors
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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm import bmm
from flag_gems.ops.mm import mm
from flag_gems.ops.svd import svd
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _scale_rows_kernel(value, scale):
    return value * scale


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _symmetrize_kernel(value, transposed):
    return 0.5 * (value + transposed)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(value):
    return value


@libentry()
@triton.jit
def _polar_postprocess_kernel(
    Up,
    S,
    V,
    U,
    H,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    Up += batch * M * N
    S += batch * N
    V += batch * N * N
    U += batch * M * N
    H += batch * N * N

    up = tl.load(
        Up + rows[:, None] * N + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        other=0.0,
    )
    v = tl.load(
        V + rows[:, None] * N + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < N),
        other=0.0,
    )
    singular_values = tl.load(S + cols, cols < N, other=0.0)

    polar_u = tl.dot(up, tl.trans(v), allow_tf32=False)
    polar_h = tl.dot(v * singular_values[None, :], tl.trans(v), allow_tf32=False)

    tl.store(
        U + rows[:, None] * N + cols[None, :],
        polar_u,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        H + rows[:, None] * N + cols[None, :],
        polar_h,
        (rows[:, None] < N) & (cols[None, :] < N),
    )


def _batched_matmul(left, right):
    if left.ndim == 2:
        return mm(left, right)

    batch_shape = left.shape[:-2]
    batch = math.prod(batch_shape)
    if batch == 0:
        return torch.empty(
            (*batch_shape, left.shape[-2], right.shape[-1]),
            dtype=left.dtype,
            device=left.device,
        )

    left_3d = left.reshape(batch, left.shape[-2], left.shape[-1])
    right_3d = right.reshape(batch, right.shape[-2], right.shape[-1])
    result = bmm(left_3d, right_3d)
    return result.reshape(*batch_shape, left.shape[-2], right.shape[-1])


def _validate_input(A):
    if A.ndim < 2:
        raise RuntimeError(
            "linalg.polar: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-2] < A.shape[-1]:
        raise RuntimeError(
            "linalg.polar: input must have at least as many rows as columns, "
            f"but got {A.shape[-2]} by {A.shape[-1]} matrices"
        )
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_polar only supports float32 input, got {A.dtype}")


def _check_out(A, out, name):
    if out.dtype != A.dtype:
        raise RuntimeError(
            f"Expected out tensor {name} to have dtype {A.dtype}, got {out.dtype}"
        )
    if out.device != A.device:
        raise RuntimeError(
            f"Expected out tensor {name} to be on {A.device}, got {out.device}"
        )


def _linalg_polar_impl(A, out_U=None, out_H=None):
    _validate_input(A)
    m = A.shape[-2]
    n = A.shape[-1]
    U_shape = A.shape
    H_shape = (*A.shape[:-2], n, n)

    if out_U is not None:
        _check_out(A, out_U, "U")
        _check_out(A, out_H, "H")
        if out_U.shape != U_shape:
            out_U.resize_(U_shape)
        if out_H.shape != H_shape:
            out_H.resize_(H_shape)

    if A.numel() == 0:
        U = out_U
        H = out_H
        if U is None:
            U = torch.empty_like(A, memory_format=torch.contiguous_format)
            H = torch.empty(H_shape, dtype=A.dtype, device=A.device)
        return U, H

    # A = Up @ diag(S) @ Vh, hence the right polar decomposition is
    # U = Up @ Vh and H = Vh^H @ diag(S) @ Vh.
    Up, S, V = svd(A, some=True, compute_uv=True)
    if m <= 16 and n <= 16:
        direct_out = (
            out_U is not None and out_U.is_contiguous() and out_H.is_contiguous()
        )
        U = (
            out_U
            if direct_out
            else torch.empty(U_shape, dtype=A.dtype, device=A.device)
        )
        H = (
            out_H
            if direct_out
            else torch.empty(H_shape, dtype=A.dtype, device=A.device)
        )
        batch = math.prod(A.shape[:-2]) if A.ndim > 2 else 1
        with torch_device_fn.device(A.device):
            _polar_postprocess_kernel[(batch,)](
                Up.contiguous(),
                S.contiguous(),
                V.contiguous(),
                U,
                H,
                m,
                n,
                BLOCK_M=max(16, triton.next_power_of_2(m)),
                BLOCK_N=16,
            )
        if out_U is not None and not direct_out:
            _copy_kernel(U, out0=out_U)
            _copy_kernel(H, out0=out_H)
            return out_U, out_H
        return U, H

    Vh = V.mH
    U = _batched_matmul(Up, Vh)
    scaled_Vh = _scale_rows_kernel(Vh, S.unsqueeze(-1))
    H = _batched_matmul(V, scaled_Vh)
    H = _symmetrize_kernel(H, H.mT)
    if out_U is not None:
        _copy_kernel(U, out0=out_U)
        _copy_kernel(H, out0=out_H)
        return out_U, out_H
    return U.contiguous(), H.contiguous()


def linalg_polar(A):
    logger.debug("GEMS LINALG_POLAR")
    return _linalg_polar_impl(A)


def linalg_polar_out(A, *, U, H):
    logger.debug("GEMS LINALG_POLAR_OUT")
    return _linalg_polar_impl(A, U, H)
