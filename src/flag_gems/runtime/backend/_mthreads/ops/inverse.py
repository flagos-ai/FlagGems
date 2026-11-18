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

"""Triton implementation of ``aten::inverse`` (batched, square matrices).

Two device paths, selected only by shape/dtype metadata on the host:

* ``n <= 32``: one program per matrix performs pivoted Gauss-Jordan on the
  augmented ``[A | I]`` system held in one register tile.  One launch per call.
* ``n > 32``: a block Schur complement that splits the side in half and recurses,
  built from the same register block inverse plus a Triton block matmul kernel
  (no framework math).  The split wins because the direct path's serial chain
  grows faster than linearly in ``n``.

Every call recomputes everything from the current tensor contents; no state is
carried between calls and the input is never mutated or aliased.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.inverse import inverse as default_inverse
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized;
# every other dtype (including complex) falls back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float32}

_SMALL_MAX = 32

# The register inverse is latency-bound on its per-step reduction/barrier chain,
# so its best warp count is not predictable from the tile size.  Small sides keep
# a fixed choice (a one- or two-warp block is best both alone and when hundreds
# of programs run); wider leaves are searched, keyed on the side, whose next
# power of two already selects the tile width (supplied through heuristics).
_LEAF_CONFIGS = [triton.Config({}, num_warps=w) for w in (1, 2, 4, 8, 16)]


def _leaf_prune(configs, named_args, **kwargs):
    n = named_args["n"]
    if n <= 8:
        want = 1
    elif n <= 16:
        want = 2
    else:
        return configs
    return [c for c in configs if c.num_warps == want]


@triton.autotune(
    configs=_LEAF_CONFIGS,
    key=["n"],
    prune_configs_by={"early_config_prune": _leaf_prune},
)
@triton.heuristics({"BLOCK": lambda args: max(triton.next_power_of_2(args["n"]), 4)})
@triton.jit
def _inv_block_kernel(
    A_ptr,
    O_ptr,
    n,
    a_off,
    o_off,
    sa,
    sb,
    osa,
    osb,
    AS0,
    AS1,
    AS2,
    B1,
    B2,
    OBS,
    BLOCK: tl.constexpr,
):
    """Invert one ``n x n`` block per program (``n <= BLOCK``).

    The augmented system ``[A | I]`` is one ``BLOCK x 2*BLOCK`` register tile, so
    a Gauss-Jordan step is one column extract, one pivot-row extract and three
    tile passes.  The pivot row is *not* moved to position ``k``: each step
    records which row was used, and the epilogue scatters row ``r`` of the
    accumulator to output row ``perm[r]``, which is the same arithmetic with one
    bookkeeping vector instead of a per-step row swap.  Rows/cols beyond ``n``
    are padded with the identity, making the padded system block diagonal.
    """
    p0 = tl.program_id(0)
    p1 = tl.program_id(1)
    p2 = tl.program_id(2)

    a_base = a_off + p0 * AS0 + p1 * AS1 + p2 * AS2
    o_base = o_off + ((p0 * B1 + p1) * B2 + p2) * OBS

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 2 * BLOCK)
    r2 = rows[:, None]
    c2 = cols[None, :]

    in_a = (r2 < n) & (c2 < n)
    # Only the padding region gets an identity on the left half: inside the
    # loaded n x n block the memory load already supplies the value.
    eye_l = (c2 < BLOCK) & (r2 == c2) & (r2 >= n)
    eye_r = (c2 >= BLOCK) & (r2 == c2 - BLOCK)

    T = tl.load(A_ptr + a_base + r2 * sa + c2 * sb, mask=in_a, other=0.0)
    T = T + tl.where(eye_l | eye_r, 1.0, 0.0)

    used = tl.zeros((BLOCK,), dtype=tl.int32)
    perm = tl.zeros((BLOCK,), dtype=tl.int32) + BLOCK

    for k in range(0, n):
        colk = tl.sum(tl.where(c2 == k, T, 0.0), axis=1)
        cand = tl.where((used == 0) & (rows < n), tl.abs(colk), -1.0)
        mx = tl.max(cand, axis=0)
        p = tl.min(tl.where(cand == mx, rows, BLOCK), axis=0)

        piv = tl.sum(tl.where(rows == p, colk, 0.0), axis=0)
        ip = 1.0 / piv
        prow = tl.sum(tl.where((rows == p)[:, None], T, 0.0), axis=0) * ip
        # column k after the row scale: only row p changes, so the factors are
        # the pre-scale column with the pivot row excluded.
        f = tl.where(rows == p, 0.0, colk)

        T = T * tl.where(rows == p, ip, 1.0)[:, None]
        T = T - f[:, None] * prow[None, :]
        T = tl.where(c2 == k, tl.where(r2 == p, 1.0, 0.0), T)

        used = tl.where(rows == p, 1, used)
        perm = tl.where(rows == k, p, perm)

    # dest[r] = the output row that accumulator row r belongs to.
    kk = tl.arange(0, BLOCK)
    dest = tl.sum(tl.where(perm[None, :] == rows[:, None], kk[None, :], 0), axis=1)

    dcol = c2 - BLOCK
    keep = (c2 >= BLOCK) & (dcol < n) & (r2 < n)
    tl.store(
        O_ptr + o_base + dest[:, None] * osa + dcol * osb,
        T,
        mask=keep,
    )


@triton.jit
def _mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    mr,
    nc,
    klen,
    a_off,
    b_off,
    c_off,
    d_off,
    asa,
    asb,
    bsa,
    bsb,
    csa,
    csb,
    dsa,
    dsb,
    ab,
    bb,
    cb,
    db,
    alpha,
    beta,
    HAS_D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """``C = alpha * (A @ B) + beta * D`` for one ``mr x nc`` product per batch.

    A serial-K broadcast accumulation is used instead of ``tl.dot``: it keeps the
    arithmetic in the operand dtype, so it also serves float64 blocks.
    """
    pb = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)

    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    rok = om < mr
    cok = on < nc

    dt = A_ptr.dtype.element_ty
    acc = tl.zeros((BM, BN), dtype=dt)
    for t in range(0, klen):
        a = tl.load(A_ptr + a_off + pb * ab + om * asa + t * asb, mask=rok, other=0.0)
        b = tl.load(B_ptr + b_off + pb * bb + t * bsa + on * bsb, mask=cok, other=0.0)
        acc += a[:, None] * b[None, :]

    out = acc * alpha
    if HAS_D:
        d = tl.load(
            D_ptr + d_off + pb * db + om[:, None] * dsa + on[None, :] * dsb,
            mask=rok[:, None] & cok[None, :],
            other=0.0,
        )
        out = out + beta * d

    tl.store(
        C_ptr + c_off + pb * cb + om[:, None] * csa + on[None, :] * csb,
        out,
        mask=rok[:, None] & cok[None, :],
    )


@triton.jit
def _mm_dot_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    mr,
    nc,
    klen,
    a_off,
    b_off,
    c_off,
    d_off,
    asa,
    asb,
    bsa,
    bsb,
    csa,
    csb,
    dsa,
    dsb,
    ab,
    bb,
    cb,
    db,
    alpha,
    beta,
    HAS_D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """``C = alpha * (A @ B) + beta * D``, K-blocked with ``tl.dot``.

    Used only for fp32 blocks whose operand bases and row strides are 16-element
    aligned (see ``_dot_aligned``); both operands are then read as 2-D tiles so
    the K dimension is contiguous.  Other layouts use ``_mm_kernel``.
    """
    pb = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)

    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    rok = om < mr
    cok = on < nc

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, klen, BK):
        oki = k0 + tl.arange(0, BK)
        kok = oki < klen
        a = tl.load(
            A_ptr + a_off + pb * ab + om[:, None] * asa + oki[None, :] * asb,
            mask=rok[:, None] & kok[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr + b_off + pb * bb + oki[:, None] * bsa + on[None, :] * bsb,
            mask=kok[:, None] & cok[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")

    out = acc * alpha
    if HAS_D:
        d = tl.load(
            D_ptr + d_off + pb * db + om[:, None] * dsa + on[None, :] * dsb,
            mask=rok[:, None] & cok[None, :],
            other=0.0,
        )
        out = out + beta * d

    tl.store(
        C_ptr + c_off + pb * cb + om[:, None] * csa + on[None, :] * csb,
        out,
        mask=rok[:, None] & cok[None, :],
    )


def _dot_aligned(descs):
    """Whether every operand block is safe for the ``tl.dot`` product kernel.

    On this backend the K-blocked dot returns wrong values when a block base is
    not on a 16-element boundary or its row stride is not a multiple of 16, so
    those layouts fall back to the scalar accumulator.
    """
    for d in descs:
        if d[3] != 1 or d[1] % 16 or d[2] % 16 or d[4] % 16:
            return False
    return True


def _launch_block_inv(src, dst, m, batch):
    """Launch the register inverse for one ``(batch, m, m)`` block."""
    _inv_block_kernel[(batch, 1, 1)](
        src[0],
        dst[0],
        m,
        src[1],
        dst[1],
        src[2],
        src[3],
        dst[2],
        dst[3],
        src[4],
        0,
        0,
        1,
        1,
        dst[4],
    )


def _launch_mm(a, b, c, d, mr, nc, kl, batch, alpha, beta):
    """Launch ``c = alpha * (a @ b) + beta * d`` for an ``mr x nc`` product."""

    def _pow2(x):
        return max(triton.next_power_of_2(x), 16)

    bm = 64 if mr > 64 else _pow2(mr)
    bn = 64 if nc > 64 else _pow2(nc)
    has_d = d is not None
    dsc = d if has_d else c
    grid = (batch, (mr + bm - 1) // bm, (nc + bn - 1) // bn)
    if a[0].dtype == torch.float32 and _dot_aligned((a, b, c, dsc)):
        _mm_dot_kernel[grid](
            a[0],
            b[0],
            c[0],
            dsc[0],
            mr,
            nc,
            kl,
            a[1],
            b[1],
            c[1],
            dsc[1],
            a[2],
            a[3],
            b[2],
            b[3],
            c[2],
            c[3],
            dsc[2],
            dsc[3],
            a[4],
            b[4],
            c[4],
            dsc[4],
            alpha,
            beta,
            HAS_D=has_d,
            BM=bm,
            BN=bn,
            BK=32,
            num_warps=4,
        )
        return
    _mm_kernel[grid](
        a[0],
        b[0],
        c[0],
        dsc[0],
        mr,
        nc,
        kl,
        a[1],
        b[1],
        c[1],
        dsc[1],
        a[2],
        a[3],
        b[2],
        b[3],
        c[2],
        c[3],
        dsc[2],
        dsc[3],
        a[4],
        b[4],
        c[4],
        dsc[4],
        alpha,
        beta,
        HAS_D=has_d,
        BM=bm,
        BN=bn,
        num_warps=4,
    )


def _block(desc, r0, c0):
    """Zero-copy sub-block descriptor ``(tensor, off, sa, sb, batch_stride)``."""
    return (desc[0], desc[1] + r0 * desc[2] + c0 * desc[3], desc[2], desc[3], desc[4])


def _flat(src, off, rows, cols):
    """Scratch descriptor for a ``rows x cols`` row-major block at ``off``."""
    return (src[0], off, cols, 1, src[4])


def _inv_into(src, dst, m, batch):
    """Write the inverse of the ``m x m`` block ``src`` into ``dst``."""
    if m <= _SMALL_MAX:
        _launch_block_inv(src, dst, m, batch)
        return

    h = m // 2  # first diagonal block
    k = m - h  # second diagonal block (>= h)

    # One flat scratch per recursion level: X1 (h x h), E (h x k), S (k x k),
    # G (k x h).
    n1 = h * h
    n2 = h * k
    n3 = k * k
    n4 = k * h
    total = n1 + n2 + n3 + n4
    buf = torch.empty((batch, total), dtype=src[0].dtype, device=src[0].device)
    flat = (buf, 0, 0, 0, total)
    x1 = _flat(flat, 0, h, h)
    e = _flat(flat, n1, h, k)
    s = _flat(flat, n1 + n2, k, k)
    g = _flat(flat, n1 + n2 + n3, k, h)

    a11 = _block(src, 0, 0)
    a12 = _block(src, 0, h)
    a21 = _block(src, h, 0)
    a22 = _block(src, h, h)
    d11 = _block(dst, 0, 0)
    d12 = _block(dst, 0, h)
    d21 = _block(dst, h, 0)
    d22 = _block(dst, h, h)

    _inv_into(a11, x1, h, batch)  # X1 = A11^-1
    _launch_mm(x1, a12, e, None, h, k, h, batch, 1.0, 0.0)  # E = X1 A12
    _launch_mm(a21, e, s, a22, k, k, h, batch, -1.0, 1.0)  # S = A22 - A21 E
    _inv_into(s, d22, k, batch)  # Si = S^-1
    _launch_mm(e, d22, d12, None, h, k, k, batch, -1.0, 0.0)  # TR = -(E Si)
    _launch_mm(d22, a21, g, None, k, h, k, batch, 1.0, 0.0)  # G = Si A21
    _launch_mm(g, x1, d21, None, k, h, h, batch, -1.0, 0.0)  # BL = -(G X1)
    _launch_mm(e, d21, d11, x1, h, h, k, batch, -1.0, 1.0)  # TL = X1 - E BL


def _use_triton_kernel(A):
    """Whether ``A`` can be handled by the MUSA register-tile kernels."""
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if A.is_complex() or A.requires_grad:
        return False
    if A.dim() < 2 or A.shape[-1] != A.shape[-2] or A.shape[-1] == 0:
        return False
    if A.is_contiguous():
        return True
    # Non-contiguous input is only understood up to three batch axes; deeper
    # layouts rely on the generic implementation.
    return A.dim() - 2 <= 3


def inverse(self):
    logger.debug("GEMS_MTHREADS INVERSE")

    A = self
    if not _use_triton_kernel(A):
        return default_inverse(A)

    n = A.shape[-1]
    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)

    batch_shape = A.shape[:-2]
    batch = 1
    for d in batch_shape:
        batch *= d
    if batch == 0 or n == 0:
        return out

    with torch_device_fn.device(out.device):
        if n <= _SMALL_MAX:
            if A.is_contiguous():
                grid = (batch, 1, 1)
                as0, as1, as2 = n * n, 0, 0
            elif batch == 1:
                grid = (1, 1, 1)
                as0, as1, as2 = 0, 0, 0
            elif len(batch_shape) == 1:
                grid = (batch_shape[0], 1, 1)
                as0, as1, as2 = A.stride(-3), 0, 0
            elif len(batch_shape) == 2:
                grid = (batch_shape[0], batch_shape[1], 1)
                as0, as1, as2 = A.stride(-4), A.stride(-3), 0
            else:
                grid = (batch_shape[0], batch_shape[1], batch_shape[2])
                as0, as1, as2 = A.stride(-5), A.stride(-4), A.stride(-3)
            _inv_block_kernel[grid](
                A,
                out,
                n,
                0,
                0,
                A.stride(-2),
                A.stride(-1),
                n,
                1,
                as0,
                as1,
                as2,
                grid[1],
                grid[2],
                n * n,
            )
            return out

        # Large side: block Schur complement over contiguous scratch.  The batch
        # slots are addressed with one flat stride, which requires a collapsible
        # batch layout (guaranteed for a contiguous input).
        if A.is_contiguous():
            ab = n * n
        else:
            assert batch == 1, "inverse: unsupported batch layout"
            ab = 0
        src = (A, 0, A.stride(-2), A.stride(-1), ab)
        dst = (out, 0, n, 1, n * n)
        _inv_into(src, dst, n, batch)

    return out
