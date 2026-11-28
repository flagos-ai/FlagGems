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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Largest matrix side handled by the register-resident kernel, per dtype. The
# kernel keeps two BLOCK x BLOCK tiles live per program (working matrix +
# inverse accumulator): 32 KiB of registers at BLOCK = 64 in fp32, double that
# in fp64, where the 64-wide tile already spills and loses to cuSOLVER on
# single matrices. Beyond the per-dtype bound the call falls through to the
# vendor (cuSOLVER) `aten::inverse.out`.
_INVERSE_KERNEL_MAX_N = {torch.float32: 64, torch.float64: 32}


@libentry()
@triton.jit
def _inverse_kernel(
    A_ptr,
    Out_ptr,
    n,
    BLOCK: tl.constexpr,
    USE_GATHER: tl.constexpr = True,
):
    """Invert one matrix per program via pivoted Gauss-Jordan elimination.

    The working matrix ``W`` and the inverse accumulator ``X`` are both held in
    registers and receive every row swap, so partial pivoting is tracked exactly
    (the augmented ``[A | I]`` scheme with both blocks kept on chip). Rows and
    columns are extracted either with ``tl.gather`` or with a masked block-wide
    reduction, see ``USE_GATHER``; column ``k`` is pinned to ``e_k`` after each
    elimination step to stop rounding error from accumulating in the
    already-eliminated column.

    ``USE_GATHER`` picks the row/column extraction primitive. ``tl.gather`` only
    lowers for a single warp on gcu400, while the masked reduction scales to
    ``num_warps > 1`` and is the cheaper of the two for BLOCK <= 32; for
    BLOCK = 64 the 64-deep reduction tree costs more than the (single-warp)
    gather, so the gather path stays the default there.
    """
    pid = tl.program_id(0)
    base = pid * n * n
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    idx2d = rows[:, None] * n + cols[None, :]
    mask = (rows[:, None] < n) & (cols[None, :] < n)
    r2 = rows[:, None]
    c2 = cols[None, :]
    diag2d = r2 == c2

    # Row-index vectors for tl.gather: ``row_at`` selects one row of a block
    # (axis 0), ``col_at`` one column (axis 1).
    row_at = tl.full((1, BLOCK), 0, tl.int32)
    col_at = tl.full((BLOCK, 1), 0, tl.int32)

    # Non-power-of-two sizes are padded to BLOCK with the identity, making the
    # padded system block diagonal; the leading n x n block is then the inverse.
    W = tl.load(A_ptr + base + idx2d, mask=mask, other=0.0)
    W = tl.where(mask, W, tl.where(diag2d, 1.0, 0.0))
    X = tl.where(diag2d, 1.0, 0.0).to(A_ptr.dtype.element_ty)

    for k in range(n):
        is_k = r2 == k
        is_c = c2 == k

        # Partial pivoting: p = argmax_{i >= k} |W[i, k]|.
        if USE_GATHER:
            col_k = tl.sum(tl.gather(W, col_at + k, axis=1), axis=1)
        else:
            col_k = tl.sum(tl.where(is_c, W, 0.0), axis=1)
        pick = tl.where(rows >= k, tl.abs(col_k), -1.0)
        p = tl.argmax(pick, axis=0)
        is_p = r2 == p

        # Swap rows k and p in both blocks (p == k is a no-op).
        if USE_GATHER:
            row_k_w = tl.sum(tl.gather(W, row_at + k, axis=0), axis=0)
            row_p_w = tl.sum(tl.gather(W, row_at + p, axis=0), axis=0)
            row_k_x = tl.sum(tl.gather(X, row_at + k, axis=0), axis=0)
            row_p_x = tl.sum(tl.gather(X, row_at + p, axis=0), axis=0)
        else:
            row_k_w = tl.sum(tl.where(is_k, W, 0.0), axis=0)
            row_p_w = tl.sum(tl.where(is_p, W, 0.0), axis=0)
            row_k_x = tl.sum(tl.where(is_k, X, 0.0), axis=0)
            row_p_x = tl.sum(tl.where(is_p, X, 0.0), axis=0)
        W = tl.where(is_p, row_k_w[None, :], tl.where(is_k, row_p_w[None, :], W))
        X = tl.where(is_p, row_k_x[None, :], tl.where(is_k, row_p_x[None, :], X))

        # Normalize the pivot row by its diagonal entry.
        pivot = tl.sum(tl.where(is_k & is_c, W, 0.0))
        inv_pivot = 1.0 / pivot
        W = tl.where(is_k, W * inv_pivot, W)
        X = tl.where(is_k, X * inv_pivot, X)

        # Eliminate column k on every other row: row_i -= W[i, k] * row_k.
        if USE_GATHER:
            factor = tl.sum(tl.gather(W, col_at + k, axis=1), axis=1)
            pivot_row_w = tl.sum(tl.gather(W, row_at + k, axis=0), axis=0)
            pivot_row_x = tl.sum(tl.gather(X, row_at + k, axis=0), axis=0)
        else:
            factor = tl.sum(tl.where(is_c, W, 0.0), axis=1)
            pivot_row_w = tl.sum(tl.where(is_k, W, 0.0), axis=0)
            pivot_row_x = tl.sum(tl.where(is_k, X, 0.0), axis=0)
        W = tl.where(is_k, W, W - factor[:, None] * pivot_row_w[None, :])
        X = tl.where(is_k, X, X - factor[:, None] * pivot_row_x[None, :])

        # Column k is exactly e_k now; pin it to keep the dead column exact.
        W = tl.where(is_c, tl.where(is_k, 1.0, 0.0), W)

    tl.store(Out_ptr + base + idx2d, X, mask=mask)


def _inverse_kernel_impl(A):
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    batch = 1
    for dim in batch_shape:
        batch *= dim

    if batch == 0:
        return torch.empty_like(A)

    # Both buffers must be freshly materialized: `empty_like(A)` keeps the
    # input's strides, so for a non-contiguous operand the 2-D reshape below
    # would be a strided view while the kernel addresses flat `n*n` tiles.
    A_work = A.reshape(batch, n, n).contiguous()
    out_work = torch.empty((batch, n, n), dtype=A.dtype, device=A.device)

    block = max(triton.next_power_of_2(n), 16)
    # gcu400: the triton-gpu -> triton-gcu lowering only implements ``tl.gather``
    # for a single warp.  With num_warps > 1 it marks the op illegal and the
    # compiler dies with a segfault instead of raising ("failed to legalize
    # operation 'tt.gather' that was explicitly marked illegal").  Measured on
    # this kernel: num_warps=1 compiles for BLOCK 16/32/64, num_warps 2/4/8
    # all crash when ``tl.gather`` is kept in the IR.
    #
    # The masked-reduction extraction has no such restriction, and measured on
    # gcu400 (fp32, 50 reps after warmup) it wins for BLOCK <= 32: BLOCK=8/16
    # prefer num_warps=1, BLOCK=32 prefers num_warps=2 (a single warp leaves the
    # 32x32 tile too starved to hide the reduction latency).  BLOCK=64 keeps the
    # single-warp gather path, where the 64-deep reduction tree is the slower
    # option.
    # A direct-kernel sweep suggested gather was 3.2-3.3x faster for BLOCK=32, but
    # that did not reproduce through the benchmark harness (32x32: masked 512 us vs
    # gather 703 us; 128x32x32: 3.38 ms vs 5.06 ms), so the masked path stays.
    if block <= 16:
        use_gather, num_warps = False, 1
    elif block == 32:
        use_gather, num_warps = False, 2
    else:
        use_gather, num_warps = True, 1
    if A.dtype is torch.float64:
        # fp64 was profiled with the gather extraction only; keep it unchanged.
        use_gather, num_warps = True, 1

    with torch_device_fn.device(A.device):
        _inverse_kernel[(batch,)](
            A_work,
            out_work,
            n,
            BLOCK=block,
            USE_GATHER=use_gather,
            num_warps=num_warps,
        )
    return out_work.reshape(A.shape)


def inverse(A):
    """Invert a square matrix (batched), equivalent to ``torch.inverse``.

    Args:
        A: tensor of shape (*, n, n) with floating point or complex dtype.

    Returns:
        A tensor of the same shape holding the inverse of every matrix.
    """
    logger.debug("GEMS_ENFLAME INVERSE")

    assert A.dim() >= 2, "inverse: input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "inverse: input must be a square matrix"
    # Low precision (fp16/bf16) and integer inputs are rejected by torch.inverse
    # itself; the generated kernel additionally only implements the real dtypes.
    assert A.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), f"inverse: unsupported dtype {A.dtype}"

    n = A.shape[-1]
    use_kernel = (
        A.dtype in _INVERSE_KERNEL_MAX_N and n <= _INVERSE_KERNEL_MAX_N[A.dtype]
    )

    if use_kernel:
        return _inverse_kernel_impl(A)

    # Larger real matrices spill the register tile, and complex inputs are not
    # implemented in the kernel, so both are routed through the vendor
    # (cuSOLVER) implementation. The `.out` overload is used so the call does
    # not re-enter the FlagGems `aten::inverse` registration.
    out = torch.empty_like(A)
    return torch.ops.aten.inverse.out(A, out=out)
