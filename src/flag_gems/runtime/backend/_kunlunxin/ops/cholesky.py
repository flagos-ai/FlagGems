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

logger = logging.getLogger(__name__)

# Lanes per row chunk.  The XPU3 backend cannot lower 1-D tiles of 128/256
# lanes for this loop shape (``TritonXPUUnrollControl`` / ``uni_sram``); 64 is
# the widest tile that compiles, so rows longer than 64 columns are swept in
# chunks of 64.  See harness/KernelOpt_Knowledge/wiki/techniques/.
_ROW_CHUNK = 64


@triton.jit
def _cholesky_row_kernel(
    A_ptr,
    U_ptr,
    Info_ptr,
    N,
    batch_stride,
    row_stride,
    col_stride,
    i,
    CBLK: tl.constexpr,
):
    """Upper-factor Cholesky of row ``i`` for every matrix; one launch per row.

    Lanes span a chunk of columns; the k-loop reads only rows produced by
    *previous launches*, so every write -> read dependency crosses a kernel
    boundary.  A single-program whole-matrix version is not usable here: the
    static-unrolled form of the total-update Cholesky blows up the XPU3
    legaliser, and reading back rows written inside one program is unordered on
    this backend (observed max errors 1e-1..1e0).  One launch per row keeps the
    dependency ordered and only costs launch overhead.

    ``Info_ptr[b]`` receives ``i + 1`` for the first row whose pivot is not
    strictly positive (0 while the factorisation is still valid), matching the
    ``aten::cholesky`` error contract without any device-side NaN compare.
    """
    b = tl.program_id(0)
    base = b * batch_stride
    lanes = tl.arange(0, CBLK)
    dt = A_ptr.dtype.element_ty

    # Accumulate the squared norm of the already-computed column i.  Kept out
    # of the per-chunk loop below so it is summed exactly once.
    sq = tl.zeros((), dtype=dt)
    for k in range(i):
        fk = tl.load(U_ptr + base + k * row_stride + i * col_stride)
        sq += fk * fk

    arg = tl.load(A_ptr + base + i * row_stride + i * col_stride) - sq
    ok = arg > 0.0
    # Guard the square root so a non-positive pivot cannot emit NaN/inf into the
    # buffer; the caller discards the result once Info says the factor failed.
    d = tl.sqrt(tl.where(ok, arg, 1.0))
    inv = 1.0 / d

    for c in range(tl.cdiv(N, CBLK)):
        js = c * CBLK + lanes
        jmask = js < N
        acc = tl.zeros((CBLK,), dtype=dt)
        for k in range(i):
            fk = tl.load(U_ptr + base + k * row_stride + i * col_stride)
            pk = tl.load(
                U_ptr + base + k * row_stride + js * col_stride,
                mask=jmask,
                other=0.0,
            )
            acc += fk * pk
        a_row = tl.load(
            A_ptr + base + i * row_stride + js * col_stride,
            mask=jmask,
            other=0.0,
        )
        val = tl.where(js == i, d, tl.where(js > i, (a_row - acc) * inv, 0.0))
        tl.store(U_ptr + base + i * row_stride + js * col_stride, val, mask=jmask)

    prev = tl.load(Info_ptr + b)
    tl.store(Info_ptr + b, tl.where(ok, prev, tl.where(prev == 0, i + 1, prev)))


def cholesky(A, upper=False):
    """Cholesky decomposition of a symmetric positive-definite matrix."""
    logger.debug("GEMS_KUNLUNXIN CHOLESKY")
    if A.dim() < 2:
        raise RuntimeError(
            "cholesky: The input tensor A must have at least 2 dimensions."
        )
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("cholesky: only float32 and float64 are supported")
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "cholesky: A must be batches of square matrices, but they are "
            "%d by %d matrices" % (A.shape[-2], A.shape[-1])
        )

    n = int(A.shape[-1])
    # A zero-sized batch is vacuously factorised; ATen hands back a fresh empty
    # tensor rather than aliasing the input.
    if n == 0 or A.numel() == 0:
        return torch.empty_like(A)

    num = int(A.numel()) // (n * n)
    a3 = A.reshape(num, n, n)
    if not a3.is_contiguous():
        # Data staging only (vendor copy kernel): the factorisation kernel
        # indexes the matrices with real strides, so it needs a dense input.
        a3 = a3.contiguous()

    # Output is fully written by the kernel (zeros above the diagonal are stored
    # explicitly), so a plain allocation is enough.
    out = torch.empty((num, n, n), dtype=a3.dtype, device=a3.device)
    info = torch.zeros(num, dtype=torch.int32, device=a3.device)

    cblk = min(triton.next_power_of_2(n), _ROW_CHUNK)
    for i in range(n):
        _cholesky_row_kernel[(num,)](
            a3,
            out,
            info,
            n,
            a3.stride(0),
            a3.stride(1),
            a3.stride(2),
            i,
            CBLK=cblk,
            num_warps=1,
            num_stages=1,
        )

    # Tiny D2H read of the failure flags (no numeric work on the host).
    first = 0
    for flag in info.tolist():
        if flag != 0:
            first = int(flag)
            break
    if first != 0:
        raise torch.linalg.LinAlgError(
            "cholesky: The factorization could not be completed because the "
            "input is not positive-definite (the leading minor of order %d is "
            "not positive-definite)." % first
        )

    # The kernel produces the upper factor U (same layout that ATen uses for
    # ``upper=True``); the default lower factor is its transpose, returned as a
    # view exactly like ATen does for the upper case.
    if upper:
        return out.reshape(A.shape)
    return out.mT.reshape(A.shape)


__all__ = ["cholesky"]
