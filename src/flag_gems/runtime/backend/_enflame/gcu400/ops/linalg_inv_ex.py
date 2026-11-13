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

"""gcu400 linalg_inv_ex: register-tiled Gauss-Jordan inverse with LAPACK info.

Why this override exists (measured 2026-11-12, GCU400):
  * The generic kernel keeps the augmented [A | I] system in ONE
    2*BLOCK_N-wide register tile.  At BLOCK_N = 128 that 128x256 tile
    silently miscompiles for num_warps <= 2 (P10 family: reduction-plane x
    warp-count wrong code): the pivot search reads zero and the kernel
    reports a spurious singular matrix (info=2, n=80..128).  num_warps >= 4
    is correct, but the @triton.autotune wrapper selects the FASTEST
    config, which is the broken num_warps=1, so every n > 64 call returns
    garbage.  The old workaround for this range was a redispatch to the
    ATen CompositeExplicitAutograd decomposition, which is no longer
    allowed: host functions must not hand computation to aten/native ops.
  * This kernel keeps TWO (BLOCK_N, BLOCK_N) tiles instead - the evolving
    left block and the identity that becomes A^{-1} - so no tile is wider
    than 128, and num_warps is pinned per BLOCK_N to a probed-correct
    domain (autotune optimizes speed, not correctness, on this backend).

Algorithm: one program per matrix, no row swaps.  At step k the pivot row
p is picked among rows not yet used (partial pivoting over column k), the
pivot row is scaled in place, every other row receives the rank-1 update,
and `pos[p] = k` remembers where logical inverse row k physically lives;
the final store scatters through `pos`.  `info` follows LAPACK order: the
first step whose pivot is exactly zero stores k+1 and the elimination
continues with a guarded denominator (matching the generic kernel).

Resource envelope: DSM hard limit on gcu400 is 896 KB per program.  Two
128x128 fp32 tiles plus working vectors fit; BLOCK_N = 256 requires
1.8-3.2 MB and fails to launch, so matrices wider than 128 raise
NotImplementedError instead of silently degrading (the removed aten
fallback used to cover them; see the module changelog note in the repo
docs).

Contract: one Triton launch per call (plus the tiny info-flag reduction
kernel only when check_errors=True, mirroring the generic
implementation); the host only allocates outputs (torch.empty) and never
calls aten compute ops.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_inv_ex import LinalgInvExResult

__all__ = ["linalg_inv_ex"]

# Two (BLOCK_N, BLOCK_N) fp32 tiles + pivot/working vectors stay under the
# gcu400 DSM budget of 896 KB per program up to BLOCK_N = 128; BLOCK_N = 256
# needs 1.8 MB+ and the launch is rejected.
_MAX_TRITON_N = 128


@triton.jit
def _gcu_inv_ex_kernel(
    A,
    OUT,
    INFO,
    n,
    a_row,
    a_col,
    d0,
    d1,
    d2,
    d3,
    e0,
    e1,
    e2,
    e3,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # Batch offset of this matrix in A (levels innermost-first).
    rem = pid.to(tl.int64)
    a_off = rem * 0
    if R >= 1:
        a_off += (rem % d0) * e0
        rem = rem // d0
    if R >= 2:
        a_off += (rem % d1) * e1
        rem = rem // d1
    if R >= 3:
        a_off += (rem % d2) * e2
        rem = rem // d2
    if R >= 4:
        a_off += (rem % d3) * e3

    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    row_ok = rows < n
    col_ok = cols < n
    m2 = row_ok[:, None] & col_ok[None, :]
    diag = rows[:, None] == cols[None, :]

    a_ptrs = A + a_off + rows[:, None] * a_row + cols[None, :] * a_col
    a = tl.load(a_ptrs, mask=m2, other=0.0).to(tl.float32)
    x = tl.where(m2 & diag, 1.0, 0.0)

    used = tl.zeros((BLOCK_N,), dtype=tl.int32)
    pos = tl.zeros((BLOCK_N,), dtype=tl.int32)
    z_a = a * 0.0
    z_x = x * 0.0
    singular = 0

    for k in range(0, n):
        # column k of the working block
        colk = tl.sum(tl.where(cols[None, :] == k, a, z_a), axis=1)
        av = tl.abs(colk)
        av = tl.where(row_ok & (used == 0), av, -1.0)
        pval = tl.max(av, axis=0)
        prow = tl.min(tl.where(av == pval, rows, BLOCK_N), axis=0)
        piv = tl.sum(tl.where(rows == prow, colk, 0.0), axis=0)

        singular = tl.where((piv == 0.0) & (singular == 0), k + 1, singular)
        denom = tl.where(piv == 0.0, 1.0, piv)
        inv = 1.0 / denom
        factor = colk * inv

        sel = (rows == prow)[:, None]
        ap = tl.sum(tl.where(sel, a, z_a), axis=0)
        xp = tl.sum(tl.where(sel, x, z_x), axis=0)
        aps = ap * inv
        xps = xp * inv
        a = tl.where(sel, aps[None, :], a - factor[:, None] * ap[None, :])
        x = tl.where(sel, xps[None, :], x - factor[:, None] * xp[None, :])
        used = used | tl.where(rows == prow, 1, 0)
        pos = tl.where(rows == prow, k, pos)

    o_ptrs = OUT + pid.to(tl.int64) * (n * n) + pos[:, None] * n + cols[None, :]
    tl.store(o_ptrs, x.to(OUT.dtype.element_ty), mask=m2)
    tl.store(INFO + pid, singular.to(tl.int32))


@triton.jit
def _info_flag_kernel(INFO, FLAG, B, BLOCK_B: tl.constexpr):
    """Reduce the per-matrix info values to a single 0/1 flag on device."""
    offs = tl.arange(0, BLOCK_B)
    acc = 0
    for base in range(0, B, BLOCK_B):
        vals = tl.load(INFO + base + offs, mask=(base + offs) < B, other=0)
        acc = tl.maximum(acc, tl.max(tl.abs(vals), axis=0))
    tl.store(FLAG, (acc != 0).to(tl.int32))


def _batch_levels(batch_shape, batch_strides):
    """Innermost-first (dim, stride) levels, merging contiguous neighbours."""
    dims = []
    sts = []
    for d, s in zip(reversed(batch_shape), reversed(batch_strides)):
        if dims and sts[-1] * dims[-1] == s:
            dims[-1] *= d
        else:
            dims.append(d)
            sts.append(s)
    if len(dims) > 4:
        raise NotImplementedError("linalg_inv_ex: unsupported batch stride layout")
    while len(dims) < 4:
        dims.append(1)
        sts.append(0)
    return dims, sts


def _num_warps(block_n):
    # Probed-correct (and fastest-correct) warp domains on gcu400 for the
    # (BLOCK_N, BLOCK_N) tile pair of this kernel: nw=1 is wrong at
    # BLOCK_N=128 and slow at BLOCK_N=64; nw>=4 is correct everywhere but
    # slow at BLOCK_N=128; nw=2 is the fast-correct pick for BLOCK_N=128.
    if block_n <= 32:
        return 1
    if block_n <= 64:
        return 4
    return 2


def linalg_inv_ex(A, *, check_errors=False):
    """Compute the inverse of a square matrix.

    Args:
        A: Input tensor of shape (*, n, n), n <= 128 on this backend.
        check_errors: If True, raises torch.linalg.LinAlgError when any
            matrix in the batch is singular.

    Returns:
        A namedtuple (inverse, info): `inverse` has the shape/dtype of A;
        `info` is an int32 tensor of shape A.shape[:-2] holding 0 on
        success or k+1 (LAPACK order) when the k-th pivot is exactly zero.
    """
    if A.dim() < 2:
        raise RuntimeError(
            "linalg_inv_ex: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "linalg_inv_ex: A must be batches of square matrices, but they are "
            "%d by %d matrices" % (A.shape[-2], A.shape[-1])
        )

    n = A.shape[-1]
    if n > _MAX_TRITON_N:
        raise NotImplementedError(
            "linalg_inv_ex: gcu400 supports matrices up to %d wide in Triton "
            "(on-chip tile budget); got n=%d" % (_MAX_TRITON_N, n)
        )

    batch_shape = A.shape[:-2]
    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    nn = n * n
    B = A.numel() // nn if nn else 0
    if n == 0 or B == 0:
        return LinalgInvExResult(out, info)

    if A.is_contiguous():
        a_row, a_col = n, 1
        dims, sts = ([B], [nn]) if batch_shape else ([], [])
    else:
        st = A.stride()
        a_row, a_col = st[-2], st[-1]
        dims, sts = _batch_levels(batch_shape, st[:-2])
    R = len(dims)
    padded_d = list(dims) + [1] * (4 - R)
    padded_e = list(sts) + [0] * (4 - R)

    block_n = max(4, triton.next_power_of_2(n))
    _gcu_inv_ex_kernel[(B,)](
        A,
        out,
        info,
        n,
        a_row,
        a_col,
        padded_d[0],
        padded_d[1],
        padded_d[2],
        padded_d[3],
        padded_e[0],
        padded_e[1],
        padded_e[2],
        padded_e[3],
        R=R,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
        num_stages=1,
    )

    if check_errors:
        flag = torch.empty(1, dtype=torch.int32, device=A.device)
        bb = max(1, triton.next_power_of_2(min(B, 1024)))
        _info_flag_kernel[(1,)](info, flag, B, bb, num_warps=1)
        if flag.item() != 0:
            raise torch.linalg.LinAlgError(
                "torch.linalg.inv_ex: The diagonal element of the LU "
                "decomposition is zero."
            )

    return LinalgInvExResult(out, info)
