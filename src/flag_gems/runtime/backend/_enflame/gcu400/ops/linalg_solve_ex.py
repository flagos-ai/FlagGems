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

"""gcu400 linalg_solve_ex: shape-dispatched fast paths over the ATen fallback.

Background: the generic Triton kernel keeps the LU/X workspaces in global
memory and interleaves vector stores with scalar loads inside one program; on
gcu400 the scalar load can observe the pre-store value even with explicit
barriers (fp32 residual 1.5e-2 for cond~4 systems at n=16/32/64), so the
public op was switched to a redispatch of ``aten::linalg_solve_ex``
(CompositeImplicitAutograd).  That decomposition lands in the FlagGems
``_linalg_solve_ex`` fused Triton kernel, whose per-batch elimination walks a
long serial chain of dependent global loads (~200us for n=16, ~0.45x of
native).  A register-tile rewrite (modeled on the lu_factor_ex register
kernel) was measured at ~148us: gcu400 charges ~0.4-0.6us per cross-lane
extraction/broadcast, so in-register elimination cannot beat the vendor
factorization kernels.

Measured dispatch on gcu400 (do_bench kernel mode, median, fp32):
- 1 <= n <= 4: the FlagGems fused Triton kernel (reached through the ATen
  composite redispatch) wins (19us vs 24-29us for native and the vendor
  composition below) thanks to its single tiny launch, so the redispatch is
  kept there.
- n >= 5: the vendor getrf wins the factorization (its fixed cost of ~17.5us
  at n=8 / ~36.5us at n=16 is far below any in-register Triton elimination
  on this backend), so ``aten::linalg_lu_factor_ex.out`` is captured before
  ``use_gems`` installs the FlagGems registrations and called directly with
  pre-allocated outputs; ``LU``, ``pivots`` and ``info`` then match
  ``torch.linalg.solve_ex`` exactly.

  The triangular-solve half is where this copy diverges from the plain
  vendor composition: ``aten::linalg_lu_solve.out`` costs 25-26us (n=8) to
  52-53us (n=16) of mostly fixed overhead on the benchmark shapes, because
  its interchange replay plus forward/backward substitution runs as long
  serial chains.  For 5 <= n <= 32 and k <= 128 the solve is instead done by
  the fused Triton kernel below in a single launch:

    X = U^-1 L^-1 P B

  computed through exact nilpotent algebra, all in registers:
  - the LAPACK interchange sequence is replayed on a "current position"
    vector with element-wise selects only (one independent scalar pivot
    load per step, no cross-lane extraction), and P is built as a one-hot
    tile consumed by a single dot;
  - L^-1 = (I-N)(I+N^2)...(I+N^(2^(STAGES-1))) and (I+M)^-1 for the scaled
    strict upper triangle of U are the exact finite Neumann factorizations
    of the nilpotent strict triangles (N^n = 0), applied to the right-hand
    side as acc-fused dots (v += N^(2^s) v), highest power first.  The
    serial O(n^2) substitution chain collapses to 4*STAGES-1 [16x16]-class
    dots under num_warps=4 (~1us/dot of exposed chain latency), with fp32
    rounding staying in the same cond(A)*eps band as the vendor getrs
    (validated element-wise against an fp64 CPU reference across
    n/k/batch/pivot/random/near-singular shapes; tested shapes hold
    1e-6..5e-5 worst-case relative error vs 3e-7..1e-4 for the vendor
    solve).  num_warps=1 would cut the chain ~3x (kernel 24-29us ->
    9-10us) but is data-dependently miscompiled on this backend (P07
    family: 43/174-case silent misvalue, deterministic per config) and is
    therefore forbidden for this kernel; any structural change must re-run
    the per-config accuracy gate.
  - k == 1 with a contiguous right-hand side gets a dedicated code path: a
    2D load of a [*, n, 1] B makes the row stride 1, which the JIT
    specializes into a K-continuous (transposed) load layout - a known
    slow domain on this backend (+7-12us).  The K1 path loads B as one 1D
    vector, broadcasts it across the BLOCK_K lanes (matrix ops never mix
    right-hand-side columns, so every column of the accumulator stays
    identical) and stores the solution with a 1D write.

  Larger n or k, and right-hand sides the kernel cannot address, keep the
  captured vendor ``linalg_lu_solve.out`` exactly as before.

  Measured effect vs the plain vendor composition (do_bench kernel mode,
  median): n=16 shapes 87.3 -> 61.7-64.0us against native 91.3-98.4us
  (1.12 -> 1.46-1.58x), n=8 shapes 41.4 -> 38.4-40.5us against native
  45.2-52.1us (1.12-1.25 -> 1.14-1.34x), (32,8,1) 70.6 -> 57.0us
  (1.08 -> 1.35x); benchmark geomean 1.21 -> 1.41x.  The remaining gap to
  native is dominated by the vendor getrf fixed cost (17.4us at n=8 /
  36.5us at n=16, 44-58% of the wall) plus the num_warps=4 dot-chain
  latency forced by the P07 miscompile domain.

Everything else (left=False, vector right-hand sides, broadcast batches,
other dtypes, empty inputs, CPU tensors) keeps the exact redispatch behavior
of the previous implementation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["linalg_solve_ex"]

_COMPOSITE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)

# Captured at import time, before ``use_gems`` installs the FlagGems kernels,
# so the vendor leaf kernels stay reachable (same approach as the complex path
# of _linalg_solve_ex / linalg_matrix_sqrth).  PyTorch builds without
# get_kernel keep importing FlagGems normally and simply lose this fast path.
_CAN_CAPTURE_NATIVE_KERNELS = hasattr(torch.library, "get_kernel")

if _CAN_CAPTURE_NATIVE_KERNELS:
    _DEVICE_KEYSET = torch._C.DispatchKeySet(
        getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
    )
    try:
        _NATIVE_LU_FACTOR_EX_OUT = torch.library.get_kernel(
            torch.ops.aten.linalg_lu_factor_ex.out, runtime.device.dispatch_key
        )
        _NATIVE_LINALG_LU_SOLVE_OUT = torch.library.get_kernel(
            torch.ops.aten.linalg_lu_solve.out, runtime.device.dispatch_key
        )
    except Exception:  # pragma: no cover - vendor kernels always exist today
        _CAN_CAPTURE_NATIVE_KERNELS = False

# Below this size the FlagGems fused Triton kernel (reached through the ATen
# composite redispatch) is faster than the two vendor kernel launches, so the
# redispatch is kept there.
_VENDOR_SOLVE_MIN_N = 5

# Register-tile ceiling for the fused Neumann solve kernel: BLOCK_N stays at
# 16/32 so every dot is a full-width MMA fragment (partial-N dot shapes are
# a known silent-miscompile domain on this backend).
_NEUMANN_MAX_N = 32
# BLOCK_K ceiling (fp32 tile [32, 128] = 16KB, well inside the DSM budget).
_NEUMANN_MAX_K = 128


@libentry()
@triton.jit
def _linalg_lu_solve_neumann_kernel(
    LU,
    PIV,
    B,
    X,
    k,
    b_batch_stride,
    b_row_stride,
    b_col_stride,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    K1: tl.constexpr,
):
    """X = U^-1 L^-1 P B for one batch element per program, all in registers.

    ``LU``/``PIV``/``X`` are contiguous with ``N`` known at compile time, so
    their addressing derives from ``N``; only ``B`` carries runtime strides.
    ``STAGES`` is the number of Neumann factors: the strict triangles are
    nilpotent with index N, so (I-N)(I+N^2)...(I+N^(2^(STAGES-1))) inverts
    I+N exactly for N <= 2^STAGES.
    """
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_N)
    colsn = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)

    lu = tl.load(
        LU + pid * N * N + rows[:, None] * N + colsn[None, :],
        mask=(rows[:, None] < N) & (colsn[None, :] < N),
        other=0.0,
    )
    if K1:
        # One right-hand-side column: a 2D load of [*, n, 1] would carry a row
        # stride of 1 and hit the transposed-load slow path, so load the
        # vector in 1D and let every BLOCK_K lane carry the same column.
        b_vec = tl.load(
            B + pid * b_batch_stride + rows * b_row_stride,
            mask=rows < N,
            other=0.0,
        )
        bt = tl.broadcast_to(b_vec[:, None], (BLOCK_N, BLOCK_K))
    else:
        bt = tl.load(
            B
            + pid * b_batch_stride
            + rows[:, None] * b_row_stride
            + cols[None, :] * b_col_stride,
            mask=(rows[:, None] < N) & (cols[None, :] < k),
            other=0.0,
        )

    # Replay the LAPACK interchanges on a "current position of original row j"
    # vector: every update is an element-wise select on broadcast scalars, so
    # the only per-step cost is one independent scalar load of the pivot.
    pos = rows
    for kk in tl.static_range(N):
        p = tl.load(PIV + pid * N + kk) - 1
        pos = tl.where(pos == kk, p, tl.where(pos == p, kk, pos))
    # P[i, j] = (row j sits at final position i); one dot applies it to B.
    pf = tl.where(pos[None, :] == rows[:, None], 1.0, 0.0)
    v = tl.dot(pf, bt, input_precision="ieee")

    # y = L^-1 P B with L = I + N, N strictly lower (nilpotent): the exact
    # ladder (I-N)(I+N^2)...(I+N^(2^(STAGES-1))) applied highest power first;
    # each factor acts as an acc-fused dot: v += N^(2^s) v.
    n_tri = tl.where(rows[:, None] > colsn[None, :], lu, 0.0)
    n2 = tl.dot(n_tri, n_tri, input_precision="ieee")
    if STAGES >= 3:
        n4 = tl.dot(n2, n2, input_precision="ieee")
    if STAGES >= 4:
        n8 = tl.dot(n4, n4, input_precision="ieee")
    if STAGES >= 5:
        n16 = tl.dot(n8, n8, input_precision="ieee")
        v = tl.dot(n16, v, acc=v, input_precision="ieee")
    if STAGES >= 4:
        v = tl.dot(n8, v, acc=v, input_precision="ieee")
    if STAGES >= 3:
        v = tl.dot(n4, v, acc=v, input_precision="ieee")
    v = tl.dot(n2, v, acc=v, input_precision="ieee")
    neg_n = 0.0 - n_tri
    v = tl.dot(neg_n, v, acc=v, input_precision="ieee")

    # x = U^-1 y with U = D (I + M): scale by D^-1, then the same exact
    # ladder on the scaled strict upper triangle M (also nilpotent).
    d = tl.sum(tl.where(rows[:, None] == colsn[None, :], lu, 0.0), axis=1)
    inv_d = tl.where(rows < N, 1.0 / d, 0.0)
    m_tri = tl.where(colsn[None, :] > rows[:, None], lu * inv_d[:, None], 0.0)
    v = v * inv_d[:, None]
    m2 = tl.dot(m_tri, m_tri, input_precision="ieee")
    if STAGES >= 3:
        m4 = tl.dot(m2, m2, input_precision="ieee")
    if STAGES >= 4:
        m8 = tl.dot(m4, m4, input_precision="ieee")
    if STAGES >= 5:
        m16 = tl.dot(m8, m8, input_precision="ieee")
        v = tl.dot(m16, v, acc=v, input_precision="ieee")
    if STAGES >= 4:
        v = tl.dot(m8, v, acc=v, input_precision="ieee")
    if STAGES >= 3:
        v = tl.dot(m4, v, acc=v, input_precision="ieee")
    v = tl.dot(m2, v, acc=v, input_precision="ieee")
    neg_m = 0.0 - m_tri
    v = tl.dot(neg_m, v, acc=v, input_precision="ieee")

    if K1:
        x0 = tl.sum(tl.where(cols[None, :] == 0, v, 0.0), axis=1)
        tl.store(X + pid * N + rows, x0, mask=rows < N)
    else:
        tl.store(
            X + pid * N * k + rows[:, None] * k + cols[None, :],
            v,
            mask=(rows[:, None] < N) & (cols[None, :] < k),
        )


def _plain_fp32_matrix_rhs(A, B, left):
    """Shape guard shared by both fast paths: plain fp32 ``AX = B`` with
    matrix right-hand sides and matching batch shapes (no broadcasting, no
    vector RHS, no left=False, nothing empty, on this device)."""
    if not left:
        return False
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        return False
    if A.ndim < 2 or B.ndim < 2:
        return False
    if A.shape[-1] != A.shape[-2]:
        return False
    n = A.shape[-1]
    if B.shape[-2] != n:
        return False
    if tuple(A.shape[:-2]) != tuple(B.shape[:-2]):
        return False
    if A.numel() == 0 or B.numel() == 0:
        return False
    if A.device.type != runtime.device.name:
        return False
    return True


def _kernel_batch_strides(T):
    """(batch, row, col) strides for a tensor the kernels address with a flat
    batch index, or None when the layout cannot be flattened that way
    (multi-batch non-contiguous)."""
    if T.ndim == 2:
        return 0, T.stride(-2), T.stride(-1)
    if T.is_contiguous():
        return T.stride(-3), T.stride(-2), T.stride(-1)
    return None


def _raise_singular(info):
    index = int(torch.argmax((info != 0).reshape(-1).to(torch.int32)))
    raise torch.linalg.LinAlgError(
        f"torch.linalg.solve_ex: (Batch element {index}): The solver failed "
        "because the input matrix is singular."
    )


def _vendor_solve_ex(A, B, check_errors):
    """Factor with the vendor getrf, then solve through the exact Neumann
    triangular-inverse ladder (fused Triton kernel) for the register-tile
    sizes, or with the captured vendor lu_solve leaf otherwise.

    The factorization always comes from the vendor ``.out`` leaf the native
    ATen path bottoms out in, so ``LU``, ``pivots`` and ``info`` match
    ``torch.linalg.solve_ex`` exactly; only the triangular solves differ,
    and they compute the same re-associated product.
    """
    n = A.shape[-1]
    k = B.shape[-1]
    batch_shape = A.shape[:-2]
    batch = A.numel() // (n * n)

    LU = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    pivots = torch.empty(batch_shape + (n,), dtype=torch.int32, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    _NATIVE_LU_FACTOR_EX_OUT.call_boxed(
        _DEVICE_KEYSET,
        A,
        pivot=True,
        check_errors=False,
        LU=LU,
        pivots=pivots,
        info=info,
    )

    X = torch.empty(B.shape, dtype=A.dtype, device=A.device)
    b_strides = None
    if n <= _NEUMANN_MAX_N and k <= _NEUMANN_MAX_K and _CAN_CAPTURE_NATIVE_KERNELS:
        b_strides = _kernel_batch_strides(B)
    if b_strides is not None:
        # num_warps=4 is pinned: the ladder dots run as one serial chain per
        # program (~1us/dot at nw=4), and nw=1 would cut that ~3x (0.3us/dot,
        # kernel 24-29us -> 9-10us), but nw<4 is a data-dependent silent
        # miscompile domain on this backend (P07 family: 43/174-case misvalue
        # at nw=1, deterministic per config, one failure at nw=2).  Any
        # structural kernel change must re-run the per-config accuracy gate.
        _linalg_lu_solve_neumann_kernel[(batch,)](
            LU,
            pivots,
            B,
            X,
            k,
            b_strides[0],
            b_strides[1],
            b_strides[2],
            N=n,
            BLOCK_N=max(triton.next_power_of_2(n), 16),
            BLOCK_K=max(triton.next_power_of_2(k), 16),
            STAGES=max((n - 1).bit_length(), 2),
            K1=(k == 1),
            num_warps=4,
        )
    else:
        _NATIVE_LINALG_LU_SOLVE_OUT.call_boxed(
            _DEVICE_KEYSET, LU, pivots, B, left=True, adjoint=False, out=X
        )

    if check_errors and torch.any(info != 0):
        _raise_singular(info)

    return X, info


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    """Solve AX = B, returning (result, info); see the module docstring."""
    logger.debug("GEMS_ENFLAME LINALG_SOLVE_EX")
    if (
        _plain_fp32_matrix_rhs(A, B, left)
        and A.shape[-1] >= _VENDOR_SOLVE_MIN_N
        and _CAN_CAPTURE_NATIVE_KERNELS
    ):
        return _vendor_solve_ex(A, B, check_errors)

    return torch.ops.aten.linalg_solve_ex.default.redispatch(
        _COMPOSITE_KEYSET, A, B, left=left, check_errors=check_errors
    )
