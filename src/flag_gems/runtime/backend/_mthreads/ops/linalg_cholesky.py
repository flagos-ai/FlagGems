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

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_cholesky import linalg_cholesky as default_linalg_cholesky
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# PyTorch's linalg_cholesky only accepts fp32/fp64, and Moore Threads hardware
# does not support fp64 compute (support_fp64 is False). The specialized kernel
# therefore targets fp32 only; fp64 (and any other dtype) falls back to the
# generic implementation. Accumulation happens in the native element type.
_SUPPORTED_DTYPES = {torch.float32}


@libentry()
@triton.jit
def cholesky_kernel(
    A,
    L,
    N,
    batch_stride,
    stride_a,
    stride_l,
    BLOCK_ROW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Cholesky decomposition kernel (column-by-column, right-looking).

    Each program computes the lower-triangular factor L of one matrix in the
    batch such that A = L @ L^T. Columns are processed sequentially (they are
    data-dependent), but within a column all rows i >= j are computed in
    parallel and the dot-product reduction over previous columns is vectorized.
    Accumulation uses the input element type to preserve precision.
    """
    pid = tl.program_id(0)
    base = pid * batch_stride

    rows = tl.arange(0, BLOCK_ROW)
    ks = tl.arange(0, BLOCK_K)

    for j in range(N):
        k_mask = ks < j
        # L[j, k] for k < j
        ljk = tl.load(L + base + j * stride_l + ks, mask=k_mask, other=0.0)

        # Diagonal: L[j, j] = sqrt(A[j, j] - sum_{k<j} L[j, k]^2)
        diag_sum = tl.sum(ljk * ljk, axis=0)
        a_diag = tl.load(A + base + j * stride_a + j)
        l_diag = tl.sqrt(a_diag - diag_sum)
        tl.store(L + base + j * stride_l + j, l_diag)

        # Off-diagonal rows i > j, all computed in parallel:
        #   L[i, j] = (A[i, j] - sum_{k<j} L[i, k] * L[j, k]) / L[j, j]
        row_mask = (rows > j) & (rows < N)
        lik = tl.load(
            L + base + rows[:, None] * stride_l + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        dot = tl.sum(lik * ljk[None, :], axis=1)
        a_col = tl.load(A + base + rows * stride_a + j, mask=row_mask, other=0.0)
        l_col = (a_col - dot) / l_diag
        tl.store(L + base + rows * stride_l + j, l_col, mask=row_mask)

        # Column j+1 reads column j, which was just written cooperatively by
        # multiple threads. Synchronize so those stores are visible before the
        # next iteration loads them.
        tl.debug_barrier()


def _use_triton_kernel(x):
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.numel() == 0 or x.dim() < 2:
        return False
    if x.is_complex():
        return False
    if x.shape[-1] != x.shape[-2]:
        return False
    return True


def linalg_cholesky(A, upper=False):
    logger.debug("GEMS_MTHREADS LINALG_CHOLESKY")

    if not _use_triton_kernel(A):
        return default_linalg_cholesky(A, upper=upper)

    shape = A.shape
    n = shape[-1]

    if len(shape) == 2:
        batch_size = 1
    else:
        batch_size = 1
        for dim in shape[:-2]:
            batch_size *= dim

    # Symmetrize to guard against non-symmetric inputs and improve stability.
    if len(shape) == 2:
        A_sym = (A + A.transpose(-2, -1)) / 2
    else:
        A_view = A.reshape(-1, n, n)
        A_sym = ((A_view + A_view.transpose(1, 2)) / 2).reshape(shape)

    A_sym = A_sym.contiguous()
    L = torch.empty_like(A_sym)

    if len(shape) > 2:
        A_kernel = A_sym.reshape(-1, n, n)
        L_kernel = L.reshape(-1, n, n)
        stride_a = A_kernel.stride(1)
        stride_l = L_kernel.stride(1)
        batch_stride = A_kernel.stride(0)
    else:
        A_kernel = A_sym
        L_kernel = L
        stride_a = A_sym.stride(0)
        stride_l = L.stride(0)
        batch_stride = stride_a * n

    grid = (batch_size,)
    block = triton.next_power_of_2(n)
    # More rows per column benefit from more warps; keep it bounded.
    num_warps = max(1, min(16, block // 32))

    with torch_device_fn.device(A.device):
        with torch.no_grad():
            cholesky_kernel[grid](
                A_kernel,
                L_kernel,
                n,
                batch_stride,
                stride_a,
                stride_l,
                BLOCK_ROW=block,
                BLOCK_K=block,
                num_warps=num_warps,
            )

    # Keep only the lower triangle (kernel does not touch the upper part).
    if len(shape) > 2:
        L = torch.tril(L.reshape(-1, n, n)).reshape(shape)
    else:
        L = torch.tril(L)

    if upper:
        L = L.transpose(-2, -1).conj()

    return L
