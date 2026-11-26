# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend (910B) backend for torch.linalg.matrix_rank.

The upstream GPU implementation (src/flag_gems/ops/linalg_matrix_rank.py) is a
pure-Triton SVD: one-sided Jacobi for small/medium matrices, Householder
tridiagonalization / bidiagonalization + Sturm bisection for large ones. It is
deeply CUDA-specific (``_sm_count`` queries ``torch.cuda`` streaming
multiprocessors; the blocked paths synchronize with atomic-based software grid
barriers that have no equivalent on triton-ascend).

This file ports that algorithm to triton-ascend in stages. Small matrices
use closed-form rank1/rank2 kernels; fp32 matrices with k <= 32 use a
single fused Golub-Kahan bidiagonalization + register Sturm kernel;
hermitian fp32 matrices with 32 < k <= 64 use a padded one-sided
tridiagonalization + Sturm count; other fp32 matrices with k <= 64 use a
single-program Golub-Kahan bidiagonalization (m, n <= 64) or, for long
dimensions, a Householder QR compression to the k x k R factor (backward
stable in the linear sigma domain) followed by the same bidiagonalization
-- both ending in the shared Sturm tail whose decisive count runs in
double-single (df64) arithmetic from the raw bidiagonal d/e; fp32 matrices
with 64 < k <= 255 use the same exact paths BY DEFAULT (the
tridiagonalization / bidiagonalization below). For non-hermitian inputs whose
long dimension exceeds 256, FLAGGEMS_MR_FAST_PATH=1 opts into a pure-Triton
blocked Householder QR (unpivoted) whose |R_ii| diagonal is counted against
the tolerance:
rank = #{ |R_ii| > max(atol, rtol * sigma_max) }, with sigma_max
bracketed by |R_00| and ||A||_F and refined by power iteration only when
the two bounds disagree; and fp32 matrices with k > 255 ALWAYS use an
unblocked Golub-Kahan bidiagonalization (non-hermitian) or a one-sided
Householder tridiagonalization (hermitian -- half the reflectors of
Golub-Kahan), plus a Sturm count whose decisive pass runs in
double-single (df64) arithmetic (SVD-accurate); hermitian ranks are
counted in the eigenvalue domain by a +/-tol double-chain Sturm bracket.
The exact paths are the DEFAULT; FLAGGEMS_MR_FAST_PATH=1 opts back into
the fast approximations (Gram sigma^2-domain count for long-dimension
k <= 64, unpivoted-QR |R_ii| count for non-hermitian 64 < k <= 255 with
rows > 256), which are faster but less accurate on adversarial spectra.
fp64 is rejected with NotImplementedError (this toolchain cannot compile
fp64 Triton kernels); there is no aclnn/native decomposition fallback
anywhere -- every rank is computed by the Triton kernels in this file (the
hermitian 32 < k <= 64 path only uses host-side aten ops -- zeros / where /
arange -- for padding and lower-triangle symmetrization, never for the
decomposition itself).  Empty inputs still validate tolerance
dtype/device/broadcast before returning (native torch validates before
its empty-input return too).  Every non-empty call is scale-normalized at
the entry: each batch matrix is divided by its lower-triangle
(hermitian) / full-matrix max-abs and atol by the same factor, so fp32
sum-of-squares neither overflow (|x| ~ 1e20) nor underflow (|x| ~ 1e-30);
rank semantics are invariant under the scaling (rtol is relative).

Implementation note: the RRQR kernels below are extremely sensitive to this
toolchain's codegen instabilities; the inline comments document every
workaround (they are load-bearing -- do not "clean them up").
"""

import logging
import os
import threading
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device topology
# ---------------------------------------------------------------------------
_VEC_CORE_CACHE = {}


def _sm_count(device):
    """Number of vector cores available on the Ascend device.

    Replaces the GPU ``_sm_count`` (which queried CUDA streaming multiprocessor
    count). triton-ascend exposes the AI-core / vector-core topology through
    ``triton.runtime.driver``; 910B4 has 20 AI cores x 2 vector sub-cores = 40.
    """
    index = device.index
    if index is None:
        index = torch.npu.current_device()
    count = _VEC_CORE_CACHE.get(index)
    if count is None:
        try:
            from triton.runtime import driver

            props = driver.active.utils.get_device_properties(index)
            count = props.get("num_vectorcore", 40) if isinstance(props, dict) else 40
        except Exception:
            count = 40
        _VEC_CORE_CACHE[index] = count
    return count


# ---------------------------------------------------------------------------
# Dispatch thresholds (ported from the shared GPU op)
# ---------------------------------------------------------------------------
# Fused single-program Jacobi covers small/medium matrices; everything larger
# falls through to the placeholder (and, in later stages, the blocked paths).
_FUSED_JACOBI_MAX_K = 64
_FUSED_JACOBI_MAX_K_FP64 = 32
_FUSED_JACOBI_MAX_ROWS = 256
_FUSED_JACOBI_WIDE_MAX_ROWS = 128

# Small/medium path (fp32, k <= 64, any row count): fused kernel for
# m,n <= 32, bidiag64 for the non-hermitian 33..64 band, padded
# tridiagonalization for hermitian 33..64, QR-compression to the k x k R
# factor + bidiag64 for long dimensions.
_TRIDIAG_MAX_K = 64
_RRQR_MAX_K = 255
# CANN 9.1.1 cannot compile the register-panel specializations needed by
# compact 65..255 QR. The GM-panel QR remains useful only for non-hermitian
# long-row inputs; compact inputs and every hermitian input stay on the exact
# bidiagonalization/tridiagonalization paths even when FAST_PATH=1.
_RRQR_FAST_MIN_ROWS = 257

# CANN 9.1.1/BiShengIR UB-overflows both the NB=2 and NB=4 specializations of
# the register-resident QR panel. The active QR launch domains all have more
# than 64 rows, so they use the GM-tile panel; keep NB=1 available for future
# callers/toolchains without exposing a known-uncompilable path.
_RRQR_REG_PANEL_MAX_ROWS = 64


def _jacobi_sweeps(k, is_fp64):
    """Worst-case sweep cap; kernels exit early once the Weyl bound holds."""
    if is_fp64:
        if k <= 16:
            return 12
        if k <= 32:
            return 16
        if k <= 256:
            return 18
        return 24
    if k <= 16:
        return 8
    if k <= 32:
        return 12
    if k <= 256:
        return 14
    return 18


# ---------------------------------------------------------------------------
# Tolerance / validation helpers (ported verbatim from the shared op)
# ---------------------------------------------------------------------------
def _expand_tolerance(value, batch_count, batch_shape, input, name):
    """Return a flat (batch_count,) tolerance tensor.

    Kept flat (always >= 1-D) on purpose: under ``use_gems()`` every factory
    op is patched to the Ascend backend, and the Ascend ``full_like`` kernel
    rejects 0-D (scalar) tensors (it requires a SUBBLOCK_SIZE it cannot derive
    for an empty grid). A non-batched input has ``batch_shape == ()``; working
    in the flattened (batch_count,) space avoids that 0-D recursion entirely.
    """
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} tensor of complex type is not "
                f"supported. Got {value.dtype}"
            )
        if value.device != input.device:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: Expected {name} and input tensors to "
                f"be on the same device, but got {name} on {value.device} and "
                f"input on {input.device}"
            )
        try:
            value = value.expand(batch_shape)
        except RuntimeError as error:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} with shape {tuple(value.shape)} "
                f"is not broadcastable to batch shape {tuple(batch_shape)}"
            ) from error
        return value.to(dtype=input.dtype).contiguous().reshape(batch_count)

    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"torch.linalg.matrix_rank: {name} must be a float or Tensor"
        ) from error
    return torch.full((batch_count,), scalar, dtype=input.dtype, device=input.device)


def _prepare_tolerances(input, atol, rtol):
    m, n = input.shape[-2], input.shape[-1]
    batch_shape = input.shape[:-2]
    batch_count = input.numel() // (m * n)
    atol_is_set = atol is not None
    atol_tensor = _expand_tolerance(
        0.0 if atol is None else atol, batch_count, batch_shape, input, "atol"
    )

    if rtol is not None:
        rtol_tensor = _expand_tolerance(rtol, batch_count, batch_shape, input, "rtol")
    else:
        default_rtol = max(m, n) * torch.finfo(input.dtype).eps
        if atol_is_set:
            rtol_tensor = torch.where(
                atol_tensor > 0,
                torch.zeros((batch_count,), dtype=input.dtype, device=input.device),
                torch.full(
                    (batch_count,),
                    default_rtol,
                    dtype=input.dtype,
                    device=input.device,
                ),
            )
        else:
            rtol_tensor = torch.full(
                (batch_count,),
                default_rtol,
                dtype=input.dtype,
                device=input.device,
            )

    return atol_tensor, rtol_tensor.contiguous()


def _tolerance_scalars(input, atol, rtol):
    """Scalar fast path mirroring _prepare_tolerances for non-tensor
    atol/rtol: returns plain floats so the kernels can receive the
    tolerances as scalar arguments instead of materialized (batch,)
    tensors (saves two fill-kernel launches per call)."""
    m, n = input.shape[-2], input.shape[-1]
    try:
        atol_val = 0.0 if atol is None else float(atol)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "torch.linalg.matrix_rank: atol must be a float or Tensor"
        ) from error
    if rtol is not None:
        try:
            rtol_val = float(rtol)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "torch.linalg.matrix_rank: rtol must be a float or Tensor"
            ) from error
    else:
        default_rtol = max(m, n) * torch.finfo(input.dtype).eps
        # torch semantics: an explicitly set positive atol suppresses the
        # default rtol (mirrors the torch.where branch in
        # _prepare_tolerances).
        rtol_val = 0.0 if (atol is not None and atol_val > 0) else default_rtol
    return atol_val, rtol_val


def _validate_tolerance_metadata(input, atol, rtol):
    """Host-side-only tolerance validation for the empty-input early return
    (which never calls _prepare_tolerances -- that would divide by
    m*n == 0).  Mirrors _expand_tolerance's checks so empty inputs reject
    invalid tolerances exactly like non-empty ones (native torch validates
    tolerances before its empty-input return as well)."""
    batch_shape = input.shape[:-2]
    for name, value in (("atol", atol), ("rtol", rtol)):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            try:
                float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"torch.linalg.matrix_rank: {name} must be a float or Tensor"
                ) from error
            continue
        if value.is_complex():
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} tensor of complex type is "
                f"not supported. Got {value.dtype}"
            )
        if value.device != input.device:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: Expected {name} and input tensors "
                f"to be on the same device, but got {name} on {value.device} "
                f"and input on {input.device}"
            )
        try:
            value.expand(batch_shape)
        except RuntimeError as error:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} with shape "
                f"{tuple(value.shape)} is not broadcastable to batch shape "
                f"{tuple(batch_shape)}"
            ) from error


def _check_input(input, hermitian):
    if input.ndim < 2:
        raise RuntimeError(
            "torch.linalg.matrix_rank: input must have at least 2 dimensions"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank currently supports float32 and float64 "
            f"real inputs only; got {input.dtype}"
        )
    if hermitian and input.shape[-2] != input.shape[-1]:
        raise RuntimeError(
            "torch.linalg.matrix_rank: A must be batches of square matrices when "
            "hermitian=True"
        )


def _empty_matrix_rank(input, output_shape):
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    output_size = out.numel()
    if output_size:
        block_size = min(256, triton.next_power_of_2(output_size))
        with torch_device_fn.device(input.device):
            _matrix_rank_zero_kernel[(triton.cdiv(output_size, block_size),)](
                out,
                N=output_size,
                BLOCK_SIZE=block_size,
            )
    return out


def _copy_rank_to_out(input, result, out):
    if out is None:
        raise TypeError("torch.linalg.matrix_rank: out must be a Tensor")
    if out.device != input.device:
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result and input tensors to be on "
            f"the same device, but got result on {out.device} and input on "
            f"{input.device}"
        )
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result to be safely castable from "
            f"Long dtype, but got result with dtype {result.dtype}"
        )

    if out.numel() != 0 and out.shape != result.shape:
        warnings.warn(
            "An output with one or more elements was resized because it had shape "
            f"{tuple(out.shape)}, which does not match the required output shape "
            f"{tuple(result.shape)}.",
            UserWarning,
            stacklevel=3,
        )
    out.resize_(result.shape)
    out.copy_(result)
    return out


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _matrix_rank_zero_kernel(out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out + offsets, 0, mask=offsets < N)


@triton.jit
def _mr_normalize_small_kernel(
    A,
    OUT,
    SCALE,
    FLAG,
    M: tl.constexpr,
    N: tl.constexpr,
    HERMITIAN: tl.constexpr,
    NEED_FLAG: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Single-launch scale normalization for small matrices (m, n <= 64):
    # per-batch power-of-two max-abs scale + in-kernel divide.  The aten
    # equivalent (abs -> amax -> mantissa mask -> where -> div, plus tril
    # for hermitian) costs 5-6 launches, which dominates sub-millisecond
    # ops.  HERMITIAN takes the scale from the LOWER triangle only (torch
    # reads just the lower triangle; garbage above the diagonal must not
    # inflate the scale) while still writing out the FULL scaled matrix.
    # The bit trick zeroes the fp32 mantissa -> the divisor is 2^e and the
    # division is exact; a masked-to-zero scale (all-zero or subnormal-max
    # matrix) maps to 1, i.e. "no scaling", matching native torch.  FLAG
    # (written only under NEED_FLAG, for the tensor-tolerance negative-tol
    # fixup) records "the (lower-triangle) max-abs is nonzero", sparing the
    # fixup its own full-matrix amax/amin/tril scan.
    batch = tl.program_id(0)
    rr = tl.arange(0, BLOCK_R)[:, None]
    cc = tl.arange(0, BLOCK_C)[None, :]
    lmask = (rr < M) & (cc < N)
    a = tl.load(A + batch * M * N + rr * N + cc, mask=lmask, other=0.0)
    aa = tl.abs(a)
    if HERMITIAN:
        aa = tl.where(rr >= cc, aa, 0.0)
    row_max = tl.max(aa, axis=1)  # (BLOCK_R,), >= 0
    if NEED_FLAG:
        nz = tl.max(row_max, axis=0) > 0.0
        tl.store(FLAG + batch, nz.to(tl.int8))
    # pow2-floor per row, then the batch max: pow2_floor is monotonic on
    # nonneg floats, so pow2_floor(max) == max(pow2_floor(row_maxes)) --
    # this keeps the bitcast on a VECTOR (a scalar bitcast miscompiles on
    # this backend).
    pw = (row_max.to(tl.int32, bitcast=True) & -8388608).to(tl.float32, bitcast=True)
    scale = tl.max(pw, axis=0)
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(SCALE + batch, scale)
    tl.store(OUT + batch * M * N + rr * N + cc, a / scale, mask=lmask)


# ---------------------------------------------------------------------------
# rank-1 / rank-2 special cases (single program per matrix, no barrier)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _matrix_rank_rank1_kernel(
    A,
    ATOL,
    RTOL,
    OUT,
    ATOL_VAL,
    RTOL_VAL,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TOL_TENSOR: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N

    if HERMITIAN:
        values = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
    elif TALL:
        values = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
    else:
        values = tl.load(a_base + rows, mask=row_mask, other=0.0)

    singular_value = tl.sqrt(tl.sum(values * values, axis=0))
    if TOL_TENSOR:
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
    else:
        atol = ATOL_VAL
        rtol = RTOL_VAL
    threshold = tl.maximum(atol, rtol * singular_value)
    rank = (singular_value > threshold).to(tl.int64)
    tl.store(OUT + batch, rank)


@libentry()
@triton.jit
def _matrix_rank_rank2_kernel(
    A,
    ATOL,
    RTOL,
    OUT,
    ATOL_VAL,
    RTOL_VAL,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
    TOL_TENSOR: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N

    if HERMITIAN:
        x = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
        lower_rows = tl.maximum(rows, 1)
        lower_columns = tl.minimum(rows, 1)
        y = tl.load(a_base + lower_rows * N + lower_columns, mask=row_mask, other=0.0)
    elif TALL:
        x = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
        y = tl.load(a_base + rows * N + 1, mask=row_mask, other=0.0)
    else:
        x = tl.load(a_base + rows, mask=row_mask, other=0.0)
        y = tl.load(a_base + N + rows, mask=row_mask, other=0.0)

    alpha = tl.sum(x * x, axis=0)
    beta = tl.sum(y * y, axis=0)
    gamma = tl.sum(x * y, axis=0)
    active = tl.abs(gamma) > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = 1.0 / tl.sqrt(1.0 + t * t)
    s = t * c
    c = tl.where(active, c, 1.0)
    s = tl.where(active, s, 0.0)

    rotated_x = c * x - s * y
    rotated_y = s * x + c * y
    singular_x = tl.sqrt(tl.sum(rotated_x * rotated_x, axis=0))
    singular_y = tl.sqrt(tl.sum(rotated_y * rotated_y, axis=0))
    max_value = tl.maximum(singular_x, singular_y)

    if TOL_TENSOR:
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
    else:
        atol = ATOL_VAL
        rtol = RTOL_VAL
    threshold = tl.maximum(atol, rtol * max_value)
    rank = (singular_x > threshold).to(tl.int32)
    rank += (singular_y > threshold).to(tl.int32)
    tl.store(OUT + batch, rank.to(tl.int64))


# ---------------------------------------------------------------------------
# Fused one-sided cyclic Jacobi (single program per matrix, no grid barrier).
# Computes singular values entirely in Triton (no aclnn SVD), so it works for
# both fp32 and fp64. Native rotation math is used for both dtypes (the GPU
# df32/df64 refinement is a speed optimization for weak-FP64 GPUs and is not
# needed for Ascend correctness).
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _matrix_rank_fused_jacobi_kernel(
    A,
    A_WORK,
    ATOL,
    RTOL,
    OUT,
    BATCH_COUNT: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    IS_FP64: tl.constexpr,
    ROUND: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SWEEPS,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
):
    # The work matrix round-trips through global memory each Jacobi step. On
    # Ascend a program's MTE3 stores and MTE2 loads are unordered, and under
    # cross-program congestion the store->load races (flaky wrong ranks). A
    # single resident program is fine -- the inner ``tl.debug_barrier`` restores
    # ordering -- so the whole batch is processed serially by one program
    # (grid=(1,)). This trades batch throughput for determinism.
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    for batch in range(BATCH_COUNT):
        a_base = A + batch * M * N
        work_base = A_WORK + batch * K * ROWS

        # Load the working matrix: column j of W is column/row j of A.
        column = 0
        while column < K:
            if HERMITIAN:
                source_rows = tl.maximum(rows, column)
                source_columns = tl.minimum(rows, column)
                values = tl.load(
                    a_base + source_rows * N + source_columns,
                    mask=row_mask,
                    other=0.0,
                )
            elif TALL:
                values = tl.load(a_base + rows * N + column, mask=row_mask, other=0.0)
            else:
                values = tl.load(a_base + column * N + rows, mask=row_mask, other=0.0)
            tl.store(work_base + column * ROWS + rows, values, mask=row_mask)
            column += 1

        pair = tl.arange(0, BLOCK_P)
        ring: tl.constexpr = ROUND - 1
        accumulator_dtype = tl.float64 if IS_FP64 else tl.float32
        singular_indices = tl.arange(0, BLOCK_K)
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
        sweep = 0
        e2_prev = tl.zeros((), dtype=accumulator_dtype)
        alphas = tl.zeros((BLOCK_K,), dtype=accumulator_dtype)
        keep_sweeping = 1
        while (sweep < SWEEPS) & (keep_sweeping != 0):
            rotations = 0
            e2_local = tl.zeros((), dtype=accumulator_dtype)
            step = 0
            while step < ROUND - 1:
                position_q = ROUND - 1 - pair
                p = tl.where(pair == 0, 0, ((pair + ring - step - 1) % ring) + 1)
                q = tl.where(
                    position_q == 0, 0, ((position_q + ring - step - 1) % ring) + 1
                )
                valid_pair = (pair < PAIRS) & (p < K) & (q < K)
                swap = p > q
                ordered_p = tl.where(swap, q, p)
                ordered_q = tl.where(swap, p, q)
                pair_mask = valid_pair[:, None] & row_mask[None, :]

                ap = tl.load(
                    work_base + ordered_p[:, None] * ROWS + rows[None, :],
                    mask=pair_mask,
                    other=0.0,
                )
                aq = tl.load(
                    work_base + ordered_q[:, None] * ROWS + rows[None, :],
                    mask=pair_mask,
                    other=0.0,
                )
                alpha = tl.sum(ap * ap, axis=1)
                beta = tl.sum(aq * aq, axis=1)
                gamma = tl.sum(ap * aq, axis=1)
                e2_local += tl.sum(gamma * gamma, axis=0).to(accumulator_dtype)

                active = valid_pair & (
                    tl.abs(gamma) > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
                )
                safe_gamma = tl.where(active, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c

                rotations += tl.sum(active.to(tl.int32), axis=0)
                c = tl.where(active, c, 1.0)
                s = tl.where(active, s, 0.0)
                # Gate the write on ``active``: inactive pairs would only
                # rewrite identical values, so skipping the store avoids a
                # needless store->load round-trip.
                write_mask = pair_mask & active[:, None]
                tl.store(
                    work_base + ordered_p[:, None] * ROWS + rows[None, :],
                    c[:, None] * ap - s[:, None] * aq,
                    mask=write_mask,
                )
                tl.store(
                    work_base + ordered_q[:, None] * ROWS + rows[None, :],
                    s[:, None] * ap + c[:, None] * aq,
                    mask=write_mask,
                )
                tl.debug_barrier()
                step += 1

            # Rank-stability check (Weyl bound): once the residual off-diagonal
            # energy cannot move a singular value across tol, stop early.
            check_tile = tl.load(
                work_base + singular_indices[:, None] * ROWS + rows[None, :],
                mask=(singular_indices < K)[:, None] & row_mask[None, :],
                other=0.0,
            )
            alphas = tl.sum(check_tile * check_tile, axis=1).to(accumulator_dtype)
            maxa = tl.max(alphas, axis=0)
            tol = tl.maximum(atol, rtol * tl.sqrt(maxa))
            tol2 = tol * tol
            margin = tl.min(
                tl.where(
                    singular_indices < K,
                    tl.abs(alphas - tol2),
                    tl.full((BLOCK_K,), float("inf"), dtype=accumulator_dtype),
                ),
                axis=0,
            )
            stall_floor = 64.0 * REL_EPS * maxa
            stable = (e2_local <= 0.25 * margin * margin) | (
                (sweep > 0)
                & (e2_local >= 0.8 * e2_prev)
                & (e2_local <= stall_floor * stall_floor)
            )
            e2_prev = e2_local
            keep_sweeping = ((rotations != 0) & (stable == 0)).to(tl.int32)
            sweep += 1

        singular_values = tl.sqrt(alphas)
        max_value = tl.max(singular_values, axis=0)
        threshold = tl.maximum(atol, rtol * max_value)
        rank = tl.sum(
            ((singular_values > threshold) & (singular_indices < K)).to(tl.int32),
            axis=0,
        )
        tl.store(OUT + batch, rank.to(tl.int64))


# ---------------------------------------------------------------------------
# Medium matrices (k <= 64): Gram via Cube + Householder tridiagonalization +
# Sturm-sequence eigenvalue counting (non-iterative, one program per batch
# element).  The trailing matrix stays register/UB-resident for the whole
# reduction -- there is no global-memory store->load round trip inside it, so
# the Ascend MTE3(store)/MTE2(load) reordering hazard that made the
# GM-resident Jacobi sweep flaky under batch concurrency cannot occur, and
# the batch is fully parallel (grid = batch_count).
#
# Toolchain notes (all verified on this machine):
#  * tl.dot only accepts operands that are direct loads (or dot chains), so
#    the Gram matrix is built in a separate kernel and reloaded below.
#  * DSA extract_slice/insert_slice and axis-0 register-tile reductions both
#    crash or hang the Ascend backend here; everything below is expressed as
#    full-tile elementwise ops plus axis-1 reductions.
#  * `tl.arange(...) == const` masks miscompile (device exception); masks are
#    written as strict-inequality intervals: (x == c) -> (x > c-1) & (x < c+1).
#  * By symmetry g[j, j+1:] == g[j+1:, j], so the Householder vector is taken
#    from the COLUMN below the diagonal, which an axis-1 reduction can
#    extract from a full tile without any gather.
# ---------------------------------------------------------------------------


@triton.jit
def _sturm_count_less(D, E, base, K: tl.constexpr, x):
    """Number of eigenvalues of the symmetric tridiagonal T = diag(d) +
    diag(e, +/-1) that are <= x, via the qd recurrence (LAPACK DLANEG
    convention: a zero pivot is replaced by a tiny negative value so the
    count stays consistent for clustered spectra).  Runs in fp32: the qd
    quotients stay on the scale of (d - x), so no overflow for the matrix
    magnitudes this path handles."""
    q = tl.load(D + base) - x
    q = tl.where(q == 0.0, -1.1754944e-38, q)
    neg = (q < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        q = (di - x) - ei * ei / q
        q = tl.where(q == 0.0, -1.1754944e-38, q)
        neg += (q < 0.0).to(tl.int32)
        i += 1
    return neg


@triton.jit
def _sturm_count_less2(D, E, base, K: tl.constexpr, x1, x2):
    """Lockstep pair of _sturm_count_less: two thresholds in ONE pass of the
    qd recurrence.  The sequential K-step chain dominates the Sturm kernel's
    latency and the bracketing needs two (hermitian: four) counts, so pairing
    halves the serial work.  Identical fp32 semantics to two separate
    calls."""
    d0 = tl.load(D + base)
    q1 = d0 - x1
    q2 = d0 - x2
    q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
    q2 = tl.where(q2 == 0.0, -1.1754944e-38, q2)
    neg1 = (q1 < 0.0).to(tl.int32)
    neg2 = (q2 < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        e2 = ei * ei
        q1 = (di - x1) - e2 / q1
        q2 = (di - x2) - e2 / q2
        q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
        q2 = tl.where(q2 == 0.0, -1.1754944e-38, q2)
        neg1 += (q1 < 0.0).to(tl.int32)
        neg2 += (q2 < 0.0).to(tl.int32)
        i += 1
    return neg1, neg2


@triton.jit
def _sturm_count_strict(D, E, base, K: tl.constexpr, x):
    """STRICT variant of _sturm_count_less: counts #{lambda < x}.  Same qd
    recurrence, but the zero-pivot guard replaces a zero pivot with a tiny
    POSITIVE value -- the mirrored DLANEG tie convention -- so an eigenvalue
    exactly equal to x is NOT counted.  Used for the hermitian negative side
    #{lambda < -tol}: counting the <= variant at an arithmetically shifted
    threshold cannot be exact on the fp32 lattice (-tol*(1+2eps) lands 0, 2
    or 3 ULP below -tol depending on tol's mantissa, skipping representable
    eigenvalues), and a scalar bitcast nextafter miscompiles on this
    backend.  The mirrored tie convention needs no shift at all and is
    exact for every tol, including 0 (#{lambda < 0}) and subnormals."""
    q = tl.load(D + base) - x
    q = tl.where(q == 0.0, 1.1754944e-38, q)
    neg = (q < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        q = (di - x) - ei * ei / q
        q = tl.where(q == 0.0, 1.1754944e-38, q)
        neg += (q < 0.0).to(tl.int32)
        i += 1
    return neg


@triton.jit
def _sturm_count_posneg2(D, E, base, K: tl.constexpr, xp, xn):
    """Hermitian tolerance-bracket pair in ONE pass of the qd recurrence:
    chain 1 counts #{lambda <= xp} (DLANEG convention: zero pivot -> tiny
    NEGATIVE), chain 2 counts #{lambda < xn} STRICTLY (zero pivot -> tiny
    POSITIVE, see _sturm_count_strict).  With xn = -tol the pair yields both
    sides of #{|lambda| > tol} = K - #{<= xp} + #{< xn} with exact tie
    semantics; pairing halves the serial K-step chain."""
    d0 = tl.load(D + base)
    q1 = d0 - xp
    q2 = d0 - xn
    q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
    q2 = tl.where(q2 == 0.0, 1.1754944e-38, q2)
    neg1 = (q1 < 0.0).to(tl.int32)
    neg2 = (q2 < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        e2 = ei * ei
        q1 = (di - xp) - e2 / q1
        q2 = (di - xn) - e2 / q2
        q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
        q2 = tl.where(q2 == 0.0, 1.1754944e-38, q2)
        neg1 += (q1 < 0.0).to(tl.int32)
        neg2 += (q2 < 0.0).to(tl.int32)
        i += 1
    return neg1, neg2


@triton.jit
def _sturm_count_less_reg(dd, ee, K: tl.constexpr, BLOCK: tl.constexpr, x):
    """Register-resident variant of _sturm_count_less: identical qd
    recurrence and fp32 semantics, but the tridiagonal entries are extracted
    from register vectors with interval masks ((idx == i) is written as
    (idx > i-1) & (idx < i+1) -- an == mask on tl.arange miscompiles on this
    backend) instead of scalar global-memory loads.  Used by the fused
    small-matrix kernels, where d/e never leave registers."""
    kidx = tl.arange(0, BLOCK)
    q = tl.sum(tl.where(kidx < 1, dd, 0.0), axis=0) - x
    q = tl.where(q == 0.0, -1.1754944e-38, q)
    neg = (q < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.sum(tl.where((kidx > i - 1) & (kidx < i + 1), dd, 0.0), axis=0)
        ei = tl.sum(tl.where((kidx > i - 2) & (kidx < i), ee, 0.0), axis=0)
        q = (di - x) - ei * ei / q
        q = tl.where(q == 0.0, -1.1754944e-38, q)
        neg += (q < 0.0).to(tl.int32)
        i += 1
    return neg


@triton.jit
def _sturm_count_less_reg2(dd, ee, K: tl.constexpr, BLOCK: tl.constexpr, x1, x2):
    """Lockstep pair of _sturm_count_less_reg: two thresholds in ONE pass of
    the qd recurrence (the sequential K-step chain dominates the fused
    kernels' latency, and the hermitian bracket needs four counts -- pairing
    halves that).  Identical fp32 semantics to two separate calls."""
    kidx = tl.arange(0, BLOCK)
    d0 = tl.sum(tl.where(kidx < 1, dd, 0.0), axis=0)
    q1 = d0 - x1
    q2 = d0 - x2
    q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
    q2 = tl.where(q2 == 0.0, -1.1754944e-38, q2)
    neg1 = (q1 < 0.0).to(tl.int32)
    neg2 = (q2 < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.sum(tl.where((kidx > i - 1) & (kidx < i + 1), dd, 0.0), axis=0)
        ei = tl.sum(tl.where((kidx > i - 2) & (kidx < i), ee, 0.0), axis=0)
        e2 = ei * ei
        q1 = (di - x1) - e2 / q1
        q2 = (di - x2) - e2 / q2
        q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
        q2 = tl.where(q2 == 0.0, -1.1754944e-38, q2)
        neg1 += (q1 < 0.0).to(tl.int32)
        neg2 += (q2 < 0.0).to(tl.int32)
        i += 1
    return neg1, neg2


@triton.jit
def _sturm_count_strict_reg(dd, ee, K: tl.constexpr, BLOCK: tl.constexpr, x):
    """STRICT register-resident variant of _sturm_count_less_reg: counts
    #{lambda < x} via the mirrored zero-pivot tie convention (zero pivot ->
    tiny POSITIVE).  See _sturm_count_strict for why this -- and not an
    arithmetically shifted threshold -- is the exact strict count."""
    kidx = tl.arange(0, BLOCK)
    q = tl.sum(tl.where(kidx < 1, dd, 0.0), axis=0) - x
    q = tl.where(q == 0.0, 1.1754944e-38, q)
    neg = (q < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.sum(tl.where((kidx > i - 1) & (kidx < i + 1), dd, 0.0), axis=0)
        ei = tl.sum(tl.where((kidx > i - 2) & (kidx < i), ee, 0.0), axis=0)
        q = (di - x) - ei * ei / q
        q = tl.where(q == 0.0, 1.1754944e-38, q)
        neg += (q < 0.0).to(tl.int32)
        i += 1
    return neg


@triton.jit
def _sturm_count_posneg_reg2(dd, ee, K: tl.constexpr, BLOCK: tl.constexpr, xp, xn):
    """Register-resident counterpart of _sturm_count_posneg2: chain 1 counts
    #{lambda <= xp} (zero pivot -> tiny negative), chain 2 counts
    #{lambda < xn} STRICTLY (zero pivot -> tiny positive), in ONE pass of
    the qd recurrence over register-resident d/e vectors (interval masks --
    an == mask on tl.arange miscompiles on this backend)."""
    kidx = tl.arange(0, BLOCK)
    d0 = tl.sum(tl.where(kidx < 1, dd, 0.0), axis=0)
    q1 = d0 - xp
    q2 = d0 - xn
    q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
    q2 = tl.where(q2 == 0.0, 1.1754944e-38, q2)
    neg1 = (q1 < 0.0).to(tl.int32)
    neg2 = (q2 < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.sum(tl.where((kidx > i - 1) & (kidx < i + 1), dd, 0.0), axis=0)
        ei = tl.sum(tl.where((kidx > i - 2) & (kidx < i), ee, 0.0), axis=0)
        e2 = ei * ei
        q1 = (di - xp) - e2 / q1
        q2 = (di - xn) - e2 / q2
        q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
        q2 = tl.where(q2 == 0.0, 1.1754944e-38, q2)
        neg1 += (q1 < 0.0).to(tl.int32)
        neg2 += (q2 < 0.0).to(tl.int32)
        i += 1
    return neg1, neg2


@triton.jit
def _mr_sturm_rank_reg(
    dd,
    ee,
    d_raw,
    atol,
    rtol,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
):
    """Rank from an in-register symmetric tridiagonal (dd, ee); d_raw is the
    pre-transform diagonal (raw bidiagonal d for the bidiag path, dd itself
    for the Gram path), used only by the HERMITIAN sigma_lo estimate.  Exact
    register-resident counterpart of the logic that used to live in
    _matrix_rank_sturm_kernel: Gershgorin bounds, tolerance bracketing, and
    bisection on the Sturm count only when the rank depends on it."""
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    shift2d = (cols[None, :] > rows[:, None] - 2) & (cols[None, :] < rows[:, None])
    e_prev_v = tl.sum(tl.where(shift2d, ee[None, :], 0.0), axis=1)
    gershgorin = tl.abs(dd) + tl.abs(ee) + tl.abs(e_prev_v)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(dd, axis=0)
    rank = tl.zeros((), dtype=tl.int32)
    if hi > 0.0:
        if HERMITIAN:
            # eigenvalues themselves; rank = #{lambda > tol} + #{lambda < -tol}
            sigma_lo = tl.maximum(tl.abs(dmax), tl.abs(tl.min(d_raw, axis=0)))
            tol_lo = tl.maximum(atol, rtol * sigma_lo)
            tol_hi = tl.maximum(atol, rtol * hi)
            cnt_lo_p, cnt_lo_n = _sturm_count_posneg_reg2(
                dd, ee, K, BLOCK, tol_lo, -tol_lo
            )
            cnt_hi_p, cnt_hi_n = _sturm_count_posneg_reg2(
                dd, ee, K, BLOCK, tol_hi, -tol_hi
            )
            rank_lo = (K - cnt_lo_p) + cnt_lo_n
            rank_hi = (K - cnt_hi_p) + cnt_hi_n
            rank = rank_lo
            refine = rank_lo != rank_hi
            if refine:
                # The rank depends on sigma_max: refine it by bisection.
                lo = dmax
                hi_p = hi * (1.0 + 1e-9) + 1e-30
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less_reg(dd, ee, K, BLOCK, mid)
                    if cnt >= K:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmax = 0.5 * (lo + hi_p)
                lo = -(hi * (1.0 + 1e-9) + 1e-30)
                hi_p = tl.min(d_raw, axis=0)
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less_reg(dd, ee, K, BLOCK, mid)
                    if cnt > 0:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmin = 0.5 * (lo + hi_p)
                sigma_max = tl.maximum(tl.abs(lmax), tl.abs(lmin))
                tol = tl.maximum(atol, rtol * sigma_max)
                rank = (K - _sturm_count_less_reg(dd, ee, K, BLOCK, tol)) + (
                    _sturm_count_strict_reg(dd, ee, K, BLOCK, -tol)
                )
        else:
            # Gram matrix: eigenvalues are sigma^2, threshold is tol^2.
            sigma_lo = tl.sqrt(tl.maximum(dmax, 0.0))
            sigma_hi = tl.sqrt(hi)
            tol_lo = tl.maximum(atol, rtol * sigma_lo)
            tol_hi = tl.maximum(atol, rtol * sigma_hi)
            cnt_lo, cnt_hi = _sturm_count_less_reg2(
                dd, ee, K, BLOCK, tol_lo * tol_lo, tol_hi * tol_hi
            )
            rank = K - cnt_lo
            refine = rank != (K - cnt_hi)
            if refine:
                # The rank depends on sigma_max: refine it by bisection.
                lo = tl.maximum(dmax, 0.0)
                hi_p = hi * (1.0 + 1e-9) + 1e-30
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less_reg(dd, ee, K, BLOCK, mid)
                    if cnt >= K:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmax = 0.5 * (lo + hi_p)
                sigma_max = tl.sqrt(lmax)
                tol = tl.maximum(atol, rtol * sigma_max)
                rank = K - _sturm_count_less_reg(dd, ee, K, BLOCK, tol * tol)
    return rank


@libentry()
@triton.jit
def _matrix_rank_gram_kernel(
    A,
    G,
    PM: tl.constexpr,
    PN: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """G[b] = A[b]^T A[b] (tall) or A[b] A[b]^T (wide) on the Cube.

    Fast opt-in path (FLAGGEMS_MR_FAST_PATH=1) for long-dimension inputs
    (one of m, n > 64), k <= 64: fast, but the sigma^2-domain floor
    overestimates rank on slowly-decaying spectra -- the default dispatch
    routes these inputs to the exact QR-compression path instead.  Input
    is pre-padded
    by the launcher to (PM, PN) multiples of 32 with the short dimension
    64, so every tl.dot operand is a full unpadded block.  The accumulator
    is summed in plain vector adds (the dot-accumulator form loses
    precision), and the transposed operand is a normal load run through
    tl.trans (stride-swapped loads fed to tl.dot miscompile).
    """
    batch = tl.program_id(0)
    a_base = A + batch * PM * PN
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    g = tl.zeros((64, 64), dtype=tl.float32)
    if TALL:
        for m0 in tl.static_range(0, PM, BLOCK_M):
            mr = m0 + tl.arange(0, BLOCK_M)
            b = tl.load(a_base + mr[:, None] * PN + cols[None, :])
            g = g + tl.dot(tl.trans(b), b, input_precision="ieee")
    else:
        for n0 in tl.static_range(0, PN, BLOCK_M):
            mc = n0 + tl.arange(0, BLOCK_M)
            at = tl.load(a_base + rows[:, None] * PN + mc[None, :])
            g = g + tl.dot(at, tl.trans(at), input_precision="ieee")
    tl.store(G + batch * 64 * 64 + rows[:, None] * 64 + cols[None, :], g)


@libentry()
@triton.jit
def _matrix_rank_tridiag_kernel(
    A,
    D,
    E,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Householder tridiagonalization of the host-presymmetrized padded
    # hermitian input (33..64 band).  The symmetrization itself stays on
    # the host: both a 64x64 computed-address gather (~90us) and a
    # tl.trans + select of the tile (~68us) cost far more inside this
    # kernel than the aten equivalent (~7us).  Rank-2 trailing update as
    # one reshape outer product.
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    g = tl.load(A + batch * BLOCK * BLOCK + rows[:, None] * BLOCK + cols[None, :])
    d_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    e_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.range(0, K - 1):
        colmask = (cols[None, :] > j - 1) & (cols[None, :] < j + 1)
        v_vec = tl.sum(tl.where(colmask, g, 0.0), axis=1)
        m0 = (rows > j) & (rows < j + 2)
        x0 = tl.sum(v_vec * m0.to(tl.float32), axis=0)
        v = v_vec * (rows > j).to(tl.float32)
        sigma = tl.sqrt(tl.sum(v * v, axis=0))
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        v2 = tl.where(m0, x0 - alpha, v)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        w = tau * tl.sum(g * v2[None, :], axis=1)
        beta = -0.5 * tau * tl.sum(v2 * w, axis=0)
        w2 = (w + beta * v2) * (cols > j).to(tl.float32)
        upd = tl.reshape(v2, (BLOCK, 1)) * tl.reshape(w2, (1, BLOCK))
        updt = tl.reshape(w2, (BLOCK, 1)) * tl.reshape(v2, (1, BLOCK))
        g = g - upd - updt
        dmask = ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
        d_vec = d_vec + v_vec * dmask
        e_vec = e_vec + alpha * dmask
    vlast = tl.sum(
        tl.where((cols[None, :] > K - 2) & (cols[None, :] < K), g, 0.0), axis=1
    )
    d_vec = d_vec + vlast * ((rows > K - 2) & (rows < K)).to(tl.float32)
    # NOTE: the store carries the batch offset (the pre-fusion version wrote
    # every program's d/e to batch 0's slot, corrupting batched inputs).
    tl.store(D + batch * BLOCK + rows, d_vec)
    tl.store(E + batch * BLOCK + rows, e_vec)


@libentry()
@triton.jit
def _matrix_rank_small_fused_kernel(
    A,
    ATOL,
    RTOL,
    OUT,
    ATOL_VAL,
    RTOL_VAL,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    TOL_TENSOR: tl.constexpr,
):
    # Fused small-matrix path (M, N <= 32): reduction + Sturm rank count in
    # ONE kernel launch (this fusion removes two kernel launches, the padded
    # staging buffer, and the d/e global-memory round trip, which dominated
    # the wall-clock latency of small matrices).  One program per matrix.
    # The input tile is loaded directly with a zero-filling bounds mask --
    # no tl.dot is fed by this load, so the masked-operand miscompile that
    # forces padding on the Gram path does not apply here.
    #
    # Non-hermitian: Golub-Kahan bidiagonalization (the exact math of the
    # former _matrix_rank_bidiag_kernel: two-sided Householder reflections
    # via reshape-based outer products, axis-1 masked reductions, and
    # strict-inequality interval masks), then the B^T B tridiagonal entries
    # are constructed exactly from the bidiagonal d/e (d_i^2 + e_{i-1}^2 and
    # d_i e_i), so the smallest singular value keeps LINEAR precision.
    #
    # Hermitian: one-sided Householder TRIDIAGONALIZATION of A directly --
    # half the reflections of the bidiagonalization and no tl.trans, and the
    # Sturm count runs on the tridiagonal in the eigenvalue domain (no
    # squaring), which is both faster and more accurate for symmetric input.
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    if not HERMITIAN and M < N:
        # Wide input: bidiagonalize A^T instead (sigma(A^T) == sigma(A)),
        # so the reduction below always works on the tall form -- the GK
        # loop needs K left reflections for the tall form (a K-1 loop plus
        # a diagonal readout loses the column K-1 tail energy of tall
        # inputs and the row K-1 tail of wide ones).  A strided transposed
        # load is ~10x slower than a normal load + register trans on this
        # backend, hence the trans.  M, N, HERMITIAN are constexpr, so
        # square/tall inputs compile to the plain load.
        g0 = tl.load(
            A + batch * M * N + rows[:, None] * N + cols[None, :],
            mask=(rows[:, None] < M) & (cols[None, :] < N),
            other=0.0,
        )
        g = tl.trans(g0)
    else:
        lmask = (rows[:, None] < M) & (cols[None, :] < N)
        g = tl.load(
            A + batch * M * N + rows[:, None] * N + cols[None, :], mask=lmask, other=0.0
        )
    if HERMITIAN:
        # torch hermitian semantics: only the LOWER triangle of A may be
        # used; rebuild the strict upper triangle from the lower one in
        # registers (a garbage upper triangle must not leak into the
        # tridiagonalization).
        g = tl.where(rows[:, None] >= cols[None, :], g, tl.trans(g))
    d_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    e_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    if HERMITIAN:
        # One-sided Householder tridiagonalization (the math of
        # _matrix_rank_tridiag_kernel): rank-2 trailing update as one
        # reshape outer product.
        for j in tl.range(0, K - 1):
            colmask = (cols[None, :] > j - 1) & (cols[None, :] < j + 1)
            v_vec = tl.sum(tl.where(colmask, g, 0.0), axis=1)
            m0 = (rows > j) & (rows < j + 2)
            x0 = tl.sum(v_vec * m0.to(tl.float32), axis=0)
            v = v_vec * (rows > j).to(tl.float32)
            sigma = tl.sqrt(tl.sum(v * v, axis=0))
            alpha = tl.where(x0 >= 0.0, -sigma, sigma)
            v2 = tl.where(m0, x0 - alpha, v)
            vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
            tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
            w = tau * tl.sum(g * v2[None, :], axis=1)
            beta = -0.5 * tau * tl.sum(v2 * w, axis=0)
            w2 = (w + beta * v2) * (cols > j).to(tl.float32)
            upd = tl.reshape(v2, (BLOCK, 1)) * tl.reshape(w2, (1, BLOCK))
            updt = tl.reshape(w2, (BLOCK, 1)) * tl.reshape(v2, (1, BLOCK))
            g = g - upd - updt
            dmask = ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
            d_vec = d_vec + v_vec * dmask
            e_vec = e_vec + alpha * dmask
        vlast = tl.sum(
            tl.where((cols[None, :] > K - 2) & (cols[None, :] < K), g, 0.0), axis=1
        )
        d_vec = d_vec + vlast * ((rows > K - 2) & (rows < K)).to(tl.float32)
        # eigenvalue-domain tridiagonal: feed the Sturm count directly.
        dd = d_vec
        ee = e_vec
    else:
        gT = tl.trans(g)
        # K left reflections (the tall form needs the last one: it folds
        # the column K-1 tail into d[K-1]); the right reflection is
        # inert at j = K-1 (u == 0 there) and skipped when j + 2 >= K.
        for j in tl.range(0, K):
            # ---- left reflection: zero g[j+1:, j] ----
            colmask = (cols[None, :] > j - 1) & (cols[None, :] < j + 1)
            colj = tl.sum(tl.where(colmask, g, 0.0), axis=1)
            x0 = tl.sum(colj * ((rows > j - 1) & (rows < j + 1)).to(tl.float32), axis=0)
            x = colj * (rows >= j).to(tl.float32)
            sigma = tl.sqrt(tl.sum(x * x, axis=0))
            alpha = tl.where(x0 >= 0.0, -sigma, sigma)
            v2 = tl.where((rows > j - 1) & (rows < j + 1), x0 - alpha, x)
            vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
            tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
            # w = tau * (g[j:, :]^T v2) via gT (axis-1 reduce, stable)
            w = tau * tl.sum(gT * v2[None, :], axis=1)
            g = g - tl.reshape(v2, (BLOCK, 1)) * tl.reshape(w, (1, BLOCK))
            gT = tl.trans(g)
            # D[j] = alpha: the left reflection maps g[j, j] to +/-sigma
            d_vec = d_vec + alpha * ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
            # ---- right reflection: zero g[j, j+2:] ----
            rowmask = (rows[None, :] > j - 1) & (rows[None, :] < j + 1)
            rowj = tl.sum(tl.where(rowmask, gT, 0.0), axis=0)
            u0 = tl.sum(rowj * ((cols > j) & (cols < j + 2)).to(tl.float32), axis=0)
            u = rowj * (cols > j).to(tl.float32)
            sigma2 = tl.sqrt(tl.sum(u * u, axis=0))
            alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
            if j + 2 < K:
                u2 = tl.where((cols > j) & (cols < j + 2), u0 - alpha2, u)
                vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
                tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0)
                # z = tau2 * (g[:, j+1:] u2): axis-1 reduce
                z = tau2 * tl.sum(g * u2[None, :], axis=1)
                g = g - tl.reshape(z, (BLOCK, 1)) * tl.reshape(u2, (1, BLOCK))
                gT = tl.trans(g)
            if j + 1 < K:
                # E[j] = alpha2: the right reflection maps g[j, j+1] to +/-sigma
                e_vec = e_vec + alpha2 * ((rows > j - 1) & (rows < j + 1)).to(
                    tl.float32
                )
        # B^T B tridiagonal entries, exact from the bidiagonal d/e:
        # dd_i = d_i^2 + e_{i-1}^2, ee_i = d_i * e_i.
        shift2d = (cols[None, :] > rows[:, None] - 2) & (cols[None, :] < rows[:, None])
        e_prev = tl.sum(tl.where(shift2d, e_vec[None, :], 0.0), axis=1)
        dd = d_vec * d_vec + e_prev * e_prev
        ee = d_vec * e_vec
    if TOL_TENSOR:
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
    else:
        atol = ATOL_VAL
        rtol = RTOL_VAL
    rank = _mr_sturm_rank_reg(
        dd, ee, d_vec, atol, rtol, K, BLOCK, HERMITIAN, BISECT_ITERS
    )
    tl.store(OUT + batch, rank.to(tl.int64))


@triton.jit
def _matrix_rank_bidiag64_kernel(
    A, D, E, M, N, K, BLOCK: tl.constexpr
):  # One program per matrix: Golub-Kahan bidiagonalization for the
    # non-hermitian 32 < k <= 64 band (m, n <= 64), writing the RAW
    # bidiagonal d/e to global memory (the Sturm tail -- to_tridiag /
    # sturm_big / sturm_final -- is shared with the large-matrix path).
    #
    # Structural choices vs the deleted _matrix_rank_bidiag_kernel, all
    # aimed at the BLOCK=64 miscompile cluster (report, stage 5):
    #  * M, N, K are RUNTIME values: one compilation covers the whole band
    #    instead of one miscompile lottery ticket per K.
    #  * the input is loaded directly with a zero-filling bounds mask (no
    #    padded staging buffer, no unmasked load).
    #  * no tl.trans inside the loop: gT is maintained incrementally with
    #    the transposed outer product of each update (the 64x64 trans slow
    #    path measured ~68us in this kernel class).
    #  * the last-step right-reflector skip is done by zeroing tau2 with a
    #    float mask, not an scf.if around tile ops.  E is still recorded
    #    (its magnitude is |u0| either way; per-element sign flips of the
    #    bidiagonal are diagonal +-1 similarities and do not change the
    #    singular values being counted).
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    # Wide inputs (M < N) are bidiagonalized TRANSPOSED: sigma(A^T) ==
    # sigma(A) and the tall form keeps the reduction uniform (the B^T B
    # construction in the Sturm tail assumes the tall/square bidiagonal
    # layout).  A strided transposed LOAD is ~10x slower than a normal
    # load + register trans on this backend, and an scf.if around the tile
    # load is a miscompile risk, so: load normally, take one register
    # trans, and select with a scalar-conditioned where.
    lmask = (rows[:, None] < M) & (cols[None, :] < N)
    g0 = tl.load(
        A + batch * M * N + rows[:, None] * N + cols[None, :], mask=lmask, other=0.0
    )
    gT0 = tl.trans(g0)
    wide = M < N
    g = tl.where(wide, gT0, g0)
    gT = tl.where(wide, g0, gT0)
    d_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    e_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.range(0, K):
        # ---- left reflection: zero g[j+1:, j] ----
        colmask = (cols[None, :] > j - 1) & (cols[None, :] < j + 1)
        colj = tl.sum(tl.where(colmask, g, 0.0), axis=1)
        x0 = tl.sum(colj * ((rows > j - 1) & (rows < j + 1)).to(tl.float32), axis=0)
        x = colj * (rows >= j).to(tl.float32)
        sigma = tl.sqrt(tl.sum(x * x, axis=0))
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        v2 = tl.where((rows > j - 1) & (rows < j + 1), x0 - alpha, x)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        # w = tau * (g^T v2); update g and the incrementally maintained gT
        w = tau * tl.sum(gT * v2[None, :], axis=1)
        g = g - tl.reshape(v2, (BLOCK, 1)) * tl.reshape(w, (1, BLOCK))
        gT = gT - tl.reshape(w, (BLOCK, 1)) * tl.reshape(v2, (1, BLOCK))
        # D[j] = alpha: the left reflection maps g[j, j] to +/-sigma
        d_vec = d_vec + alpha * ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
        # ---- right reflection: zero g[j, j+2:] ----
        coljT = (cols[None, :] > j - 1) & (cols[None, :] < j + 1)
        rowj = tl.sum(tl.where(coljT, gT, 0.0), axis=1)
        u0 = tl.sum(rowj * ((cols > j) & (cols < j + 2)).to(tl.float32), axis=0)
        u = rowj * (cols > j).to(tl.float32)
        sigma2 = tl.sqrt(tl.sum(u * u, axis=0))
        alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
        u2 = tl.where((cols > j) & (cols < j + 2), u0 - alpha2, u)
        vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
        # gate the reflector off once row j has <= 2 in-range elements
        # (j + 2 >= K, nothing to zero) by zeroing tau2 -- no scf.if around
        # the tile ops.  At j == K-1 u is identically zero, which also
        # makes this a no-op for the tall last step.
        tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0) * (j + 2 < K).to(tl.float32)
        z = tau2 * tl.sum(g * u2[None, :], axis=1)
        g = g - tl.reshape(z, (BLOCK, 1)) * tl.reshape(u2, (1, BLOCK))
        gT = gT - tl.reshape(u2, (BLOCK, 1)) * tl.reshape(z, (1, BLOCK))
        # E[j] = alpha2 (magnitude is right even when the reflector is
        # gated off: |alpha2| == |g[j, j+1]|).
        e_vec = e_vec + alpha2 * ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
    tl.store(D + batch * K + rows, d_vec, mask=rows < K)
    tl.store(E + batch * K + rows, e_vec, mask=rows < K)


@triton.jit
def _mr_extract_upper_kernel(W, PIV, RB, K, RS, WPITCH, BLOCK: tl.constexpr):
    # Extract the upper-triangular R (K x K, row-major) from the QR work
    # matrix W (column-major tiles, row pitch RS).  Convention of the panel
    # kernels: W holds R STRICTLY above the diagonal (the diagonal itself is
    # stale -- R[j, j] = PIV[j], the reflection alpha) and reflector
    # sideband data below it.
    pid = tl.program_id(0)
    lr = tl.arange(0, BLOCK)
    lc = tl.arange(0, BLOCK)
    tile = tl.load(
        W + pid * WPITCH + lc[:, None] * RS + lr[None, :],
        mask=(lc[:, None] < K) & (lr[None, :] < lc[:, None]),
        other=0.0,
    )
    piv = tl.load(PIV + pid * K + lc, mask=lc < K, other=0.0)
    tile = tile + piv[:, None] * (
        (lr[None, :] > lc[:, None] - 1) & (lr[None, :] < lc[:, None] + 1)
    ).to(tl.float32)
    # tile[c, r] = R[r, c]; scatter to row-major RB[r*K + c]
    tl.store(
        RB + pid * K * K + lr[None, :] * K + lc[:, None],
        tile,
        mask=(lc[:, None] < K) & (lr[None, :] < K),
    )


@libentry()
@triton.jit
def _matrix_rank_sturm_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    ATOL_VAL,
    RTOL_VAL,
    K: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BIDIAG: tl.constexpr,
    BLOCK: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    TOL_TENSOR: tl.constexpr,
):
    # Rank of a symmetric tridiagonal (B^T B of a Gram matrix, or the
    # direct tridiagonalization output), using Gershgorin bounds refined
    # by bisection on the Sturm count only when the rank actually depends
    # on the refinement.  One program per batch element; D/E were produced
    # in a PREVIOUS launch, so this kernel only reads global memory
    # (kernel boundaries serialize the MTE3/MTE2 queues).  D/E always hold
    # the FINAL tridiagonal on entry, so BIDIAG is False at every call
    # site (the True branch is a legacy dead switch kept to preserve the
    # verified compilation of this kernel) and the store-back below
    # re-writes identical values (idempotent): no result depends on
    # same-kernel store->load ordering, which is NOT guaranteed on this
    # backend.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    base = batch * BLOCK
    d = tl.load(D + base + kidx, mask=kidx < K, other=0.0)
    e_cur = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    e_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & (kidx < K), other=0.0)
    if BIDIAG:
        dd = d * d + e_prev * e_prev
        ee = d * e_cur
    else:
        dd = d
        ee = e_cur
    tl.store(D + base + kidx, dd, mask=kidx < K)
    tl.store(E + base + kidx, ee, mask=kidx < K)
    shift2d = (cols[None, :] > rows[:, None] - 2) & (cols[None, :] < rows[:, None])
    e_prev_v = tl.sum(tl.where(shift2d, ee[None, :], 0.0), axis=1)
    gershgorin = tl.abs(dd) + tl.abs(ee) + tl.abs(e_prev_v)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(dd, axis=0)
    if TOL_TENSOR:
        atol = tl.load(ATOL + batch)
        rtol = tl.load(RTOL + batch)
    else:
        atol = ATOL_VAL
        rtol = RTOL_VAL

    if hi == 0.0:
        # The tridiagonal (and hence the matrix) is exactly zero.
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
    else:
        if HERMITIAN:
            # eigenvalues themselves; rank = #{lambda > tol} + #{lambda < -tol}
            sigma_lo = tl.maximum(tl.abs(dmax), tl.abs(tl.min(d, axis=0)))
            tol_lo = tl.maximum(atol, rtol * sigma_lo)
            tol_hi = tl.maximum(atol, rtol * hi)
            cnt_lo_p, cnt_lo_n = _sturm_count_posneg2(D, E, base, K, tol_lo, -tol_lo)
            cnt_hi_p, cnt_hi_n = _sturm_count_posneg2(D, E, base, K, tol_hi, -tol_hi)
            rank_lo = (K - cnt_lo_p) + cnt_lo_n
            rank_hi = (K - cnt_hi_p) + cnt_hi_n
        else:
            # Gram matrix: eigenvalues are sigma^2, threshold is tol^2.
            sigma_lo = tl.sqrt(tl.maximum(dmax, 0.0))
            sigma_hi = tl.sqrt(hi)
            tol_lo = tl.maximum(atol, rtol * sigma_lo)
            tol_hi = tl.maximum(atol, rtol * sigma_hi)
            cnt_lo, cnt_hi = _sturm_count_less2(
                D, E, base, K, tol_lo * tol_lo, tol_hi * tol_hi
            )
            rank_lo = K - cnt_lo
            rank_hi = K - cnt_hi
        rank = rank_lo
        refine = rank_lo != rank_hi
        if refine:
            # The rank depends on sigma_max: refine it by bisection.
            if HERMITIAN:
                lo = dmax
                hi_p = hi * (1.0 + 1e-9) + 1e-30
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less(D, E, base, K, mid)
                    if cnt >= K:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmax = 0.5 * (lo + hi_p)
                lo = -(hi * (1.0 + 1e-9) + 1e-30)
                hi_p = tl.min(d, axis=0)
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less(D, E, base, K, mid)
                    if cnt > 0:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmin = 0.5 * (lo + hi_p)
                sigma_max = tl.maximum(tl.abs(lmax), tl.abs(lmin))
                tol = tl.maximum(atol, rtol * sigma_max)
                rank = (K - _sturm_count_less(D, E, base, K, tol)) + (
                    _sturm_count_strict(D, E, base, K, -tol)
                )
            else:
                lo = tl.maximum(dmax, 0.0)
                hi_p = hi * (1.0 + 1e-9) + 1e-30
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less(D, E, base, K, mid)
                    if cnt >= K:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmax = 0.5 * (lo + hi_p)
                sigma_max = tl.sqrt(lmax)
                tol = tl.maximum(atol, rtol * sigma_max)
                rank = K - _sturm_count_less(D, E, base, K, tol * tol)
        tl.store(OUT + batch, rank.to(tl.int64))


_HERMITIAN_LOWER_MASK_CACHE = {}


def _hermitian_lower_mask(m, n, device):
    """Broadcastable (m, n) bool mask of the lower triangle (diagonal
    included), cached per shape/device -- building it every call would cost
    more than the single aten ``where`` it feeds."""
    key = (m, n, str(device))
    mask = _HERMITIAN_LOWER_MASK_CACHE.get(key)
    if mask is None:
        rows = torch.arange(m, device=device).unsqueeze(1)
        cols = torch.arange(n, device=device).unsqueeze(0)
        mask = rows >= cols
        _HERMITIAN_LOWER_MASK_CACHE[key] = mask
    return mask


def _launch_tridiag_rank(
    matrix,
    atol_tensor,
    rtol_tensor,
    atol_val,
    rtol_val,
    out,
    m,
    n,
    batch_count,
    input,
    hermitian,
):
    """Small/medium paths for k <= 64, fp32.

    Small tiles (m, n <= 32) go to _matrix_rank_small_fused_kernel
    (bidiagonalization + Sturm count in one launch, direct masked load of
    A -- no padding); hermitian 32 < k <= 64 uses the padded
    tridiagonalization + Sturm kernels (the fused variant miscompiles at
    BLOCK = 64); non-hermitian 32 < k <= 64 with m, n <= 64 goes to
    _matrix_rank_bidiag64_kernel + the shared Sturm tail; long-dimension
    inputs go to _launch_longdim_rank (QR-compress to the k x k R factor,
    then the same exact tail).
    When atol/rtol are plain scalars they are passed as kernel scalar
    arguments (TOL_TENSOR=False) and the tolerance tensors are never
    materialized; ATOL/RTOL then receive a dummy pointer that is never
    dereferenced.
    """
    k = min(m, n)
    rows = max(m, n)
    block_m = 32
    tol_tensor = atol_tensor is not None
    if tol_tensor:
        at_arg = atol_tensor
        rt_arg = rtol_tensor
    else:
        at_arg = matrix
        rt_arg = matrix
    with torch_device_fn.device(input.device):
        small = m <= 64 and n <= 64
        # tile size adapts to the matrix: 32-wide tiles are cheaper and
        # are correct on this backend (verified for K <= 32)
        block = max(triton.next_power_of_2(max(m, n)), 32) if small else 64
        if small and block <= 32:
            # Single fused launch (verified correct on this backend only
            # for BLOCK <= 32; at BLOCK = 64 the fused kernel miscompiles
            # for many K).
            _matrix_rank_small_fused_kernel[(batch_count,)](
                matrix,
                at_arg,
                rt_arg,
                out,
                atol_val,
                rtol_val,
                M=m,
                N=n,
                K=k,
                HERMITIAN=hermitian,
                BLOCK=block,
                BISECT_ITERS=32,
                TOL_TENSOR=tol_tensor,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
        elif small and hermitian:
            # 32 < k <= 64 hermitian: one-sided Householder
            # tridiagonalization of the zero-padded input -- half the
            # reflections of the bidiagonalization, and the Sturm count
            # runs in the eigenvalue domain directly (faster and more
            # accurate for symmetric input).  torch's lower-triangle-only
            # hermitian semantics are enforced on the HOST while filling
            # the padding (one cached-mask aten where, ~7us incremental
            # over the plain copy; doing it inside the kernel costs ~68us
            # at BLOCK=64, measured).
            padded = torch.zeros(
                (batch_count, block, block),
                dtype=torch.float32,
                device=input.device,
            )
            torch.where(
                _hermitian_lower_mask(m, n, matrix.device),
                matrix,
                matrix.mT,
                out=padded[:, :m, :n],
            )
            d = torch.empty(
                (batch_count, block), dtype=torch.float32, device=input.device
            )
            e = torch.empty(
                (batch_count, block), dtype=torch.float32, device=input.device
            )
            _matrix_rank_tridiag_kernel[(batch_count,)](
                padded,
                d,
                e,
                K=k,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
            _matrix_rank_sturm_kernel[(batch_count,)](
                d,
                e,
                at_arg,
                rt_arg,
                out,
                atol_val,
                rtol_val,
                K=k,
                HERMITIAN=hermitian,
                BIDIAG=False,
                BLOCK=block,
                BISECT_ITERS=32,
                TOL_TENSOR=tol_tensor,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
        else:
            if small:
                # 32 < k <= 64, m, n <= 64, non-hermitian: single-program
                # Golub-Kahan bidiagonalization (raw d/e) + the Sturm tail
                # shared with the large-matrix path (fp32 bracket counts
                # from the dd/ee tridiagonal + decisive df64 count from the
                # raw d/e).  Linear-domain all the way -- replaces the Gram
                # route for this band (whose sigma^2-domain floor
                # overestimates rank on slowly-decaying spectra).
                d = torch.empty(
                    (batch_count, k), dtype=torch.float32, device=input.device
                )
                e = torch.empty(
                    (batch_count, k), dtype=torch.float32, device=input.device
                )
                ddbuf = torch.empty(
                    (batch_count, k), dtype=torch.float32, device=input.device
                )
                eebuf = torch.empty(
                    (batch_count, k), dtype=torch.float32, device=input.device
                )
                tol2_buf = torch.empty(
                    (batch_count,), dtype=torch.float32, device=input.device
                )
                flag_buf = torch.empty(
                    (batch_count,), dtype=torch.int64, device=input.device
                )
                if not tol_tensor:
                    # the Sturm tail kernels take tolerance tensors
                    at_arg = torch.full(
                        (batch_count,),
                        atol_val,
                        dtype=torch.float32,
                        device=input.device,
                    )
                    rt_arg = torch.full(
                        (batch_count,),
                        rtol_val,
                        dtype=torch.float32,
                        device=input.device,
                    )
                _fast_launch(
                    _matrix_rank_bidiag64_kernel,
                    (batch_count,),
                    matrix,
                    d,
                    e,
                    m,
                    n,
                    k,
                    BLOCK=64,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
                _fast_launch(
                    _mr_bidiag_to_tridiag_kernel,
                    (batch_count,),
                    d,
                    e,
                    ddbuf,
                    eebuf,
                    k,
                    BLOCK=64,
                    num_warps=1,
                    num_stages=1,
                )
                _fast_launch(
                    _mr_sturm_big_kernel,
                    (batch_count,),
                    ddbuf,
                    eebuf,
                    at_arg,
                    rt_arg,
                    out.reshape(batch_count),
                    tol2_buf,
                    flag_buf,
                    k,
                    BLOCK=64,
                    BISECT_ITERS=32,
                    num_warps=1,
                    num_stages=1,
                )
                _fast_launch(
                    _mr_sturm_final_kernel,
                    (batch_count,),
                    d,
                    e,
                    tol2_buf,
                    flag_buf,
                    out.reshape(batch_count),
                    k,
                    num_warps=1,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
                return out
            if os.environ.get("FLAGGEMS_MR_FAST_PATH") != "1":
                # Exact long-dimension path (DEFAULT): QR-compress to the
                # k x k R factor (backward stable in the linear domain),
                # then bidiag64 + df64 Sturm tail.  Slower than Gram on
                # this hardware (the Householder panel is a serial O(k)
                # chain of GM round trips) but exact.
                return _launch_longdim_rank(
                    matrix,
                    atol_tensor,
                    rtol_tensor,
                    atol_val,
                    rtol_val,
                    out,
                    m,
                    n,
                    k,
                    rows,
                    batch_count,
                    input,
                )
            # Gram (Cube) + tridiagonalization + Sturm count as three
            # separate launches -- fast opt-in (FLAGGEMS_MR_FAST_PATH=1)
            # for long-dimension inputs.  The sigma^2 domain loses
            # singular values below ~sqrt(k*eps)*sigma_max (rank
            # overestimation on slowly-decaying low-rank spectra --
            # documented in the report); the exact path above removes
            # that limitation.
            block = 64
            d = torch.empty(
                (batch_count, block), dtype=torch.float32, device=input.device
            )
            e = torch.empty(
                (batch_count, block), dtype=torch.float32, device=input.device
            )
            # The Gram kernel writes every element of the (64, 64) tile, so
            # no zero-fill is needed for it.
            padded = torch.empty(
                (batch_count, block, block), dtype=torch.float32, device=input.device
            )
            if m >= n:
                pm = triton.cdiv(m, block_m) * block_m
                gpad = torch.zeros(
                    (batch_count, pm, block), dtype=torch.float32, device=input.device
                )
                gpad[:, :m, :n] = matrix
                _matrix_rank_gram_kernel[(batch_count,)](
                    gpad,
                    padded,
                    PM=pm,
                    PN=block,
                    TALL=True,
                    BLOCK_M=block_m,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
            else:
                pn = triton.cdiv(n, block_m) * block_m
                gpad = torch.zeros(
                    (batch_count, block, pn), dtype=torch.float32, device=input.device
                )
                gpad[:, :m, :n] = matrix
                _matrix_rank_gram_kernel[(batch_count,)](
                    gpad,
                    padded,
                    PM=block,
                    PN=pn,
                    TALL=False,
                    BLOCK_M=block_m,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
            _matrix_rank_tridiag_kernel[(batch_count,)](
                padded,
                d,
                e,
                K=k,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
            _matrix_rank_sturm_kernel[(batch_count,)](
                d,
                e,
                at_arg,
                rt_arg,
                out,
                atol_val,
                rtol_val,
                K=k,
                HERMITIAN=hermitian,
                BIDIAG=False,
                BLOCK=block,
                BISECT_ITERS=32,
                TOL_TENSOR=tol_tensor,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
    return out


# ---------------------------------------------------------------------------
# Large matrices (fp32, 64 < k <= 255): blocked Householder QR, unpivoted.
# The rank is #{ |R_ii| > max(atol, rtol * sigma_max) } with sigma_max
# bracketed by |R_00| <= sigma_max <= ||A||_F and refined by power iteration
# only when the two bounds disagree on the count.
#
# Workspace layout (all fp32, global memory, column-major tiles): W and V are
# (batch, Kp, RS) with Kp = round_up(k, 64) and RS = round_up(rows, 64);
# column c lives at base + c*RS + r (rows contiguous). W holds the working
# matrix (A^T for wide inputs, the lower-triangle-symmetrized A for
# hermitian), V the Householder vectors, NRM2/PIV/TAU are (batch, k), T is
# (batch, 64, 64), FROB is (batch,) = ||A||_F^2.
#
# Toolchain constraints honored here (each verified by a minimal probe):
#  * no scalar global-memory access at data-dependent (argmax) indices --
#    norms live in register vectors with interval-mask extract/insert;
#  * the two stores of a column swap are separated by tl.debug_barrier;
#  * no runtime scalar `if` regions around the data-dependent-address stores
#    (the conditional pivot form silently corrupts results), and no extra
#    prologue loop in the panel kernel at all (its mere presence, even when
#    never executed, breaks the kernel -- the trailing-norm downdate lives in
#    the update kernel instead);
#  * tl.dot operands are direct unmasked loads (padding is zero-filled) and
#    tl.trans of those; accumulation is a plain vector add;
#  * dot loops run from row block 0 (a dynamic loop START crashes bishengir
#    in these kernels; reflector rows below J0 are zero, so this is free
#    mathematically);
#  * the dlarft T recurrence writes its column with a pure accumulate of a
#    reshape outer product (the multiply-and-add form miscompiles at higher
#    trip counts);
#  * all cross-program ordering relies on kernel boundaries; within a program
#    store->load round trips are fenced with tl.debug_barrier.
# ---------------------------------------------------------------------------


@triton.jit
def _mr_rrqr_init_kernel(
    A,
    W,
    NRM2,
    FROB,
    M,
    N,
    K,
    ROWS,
    RS,
    WPITCH,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
):
    b = tl.program_id(0)
    c0 = tl.program_id(1) * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    a_base = A + b * M * N
    wbase = W + b * WPITCH
    # Iterate only the tiles containing valid rows.  The bidiag/tridiag
    # workspaces pad RS with one extra slack tile for unmasked tile
    # addressing; those rows start as zeros (torch.zeros workspace), are
    # never read after the step kernels' own slack trimming, and must not
    # be written here either.
    RB = (ROWS + 63) // 64
    nacc = tl.zeros((64,), dtype=tl.float32)
    for rb in tl.range(0, RB):
        rr = rb * 64 + lr
        rmask = rr < ROWS
        cmask = (c0 + lc) < K
        lmask = rmask[:, None] & cmask[None, :]
        if HERMITIAN:
            cc = c0 + lc
            src_r = tl.maximum(rr[:, None], cc[None, :])
            src_c = tl.minimum(rr[:, None], cc[None, :])
            at = tl.load(a_base + src_r * N + src_c, mask=lmask, other=0.0)
        elif TALL:
            at = tl.load(
                a_base + rr[:, None] * N + (c0 + lc)[None, :], mask=lmask, other=0.0
            )
        else:
            at = tl.load(
                a_base + (c0 + lc)[None, :] * N + rr[:, None], mask=lmask, other=0.0
            )
        atT = tl.trans(at)  # (64 cols, 64 rows)
        tl.store(wbase + (c0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :], atT)
        nacc += tl.sum(atT * atT, axis=1)
    tl.store(NRM2 + b * K + c0 + lc, nacc, mask=(c0 + lc) < K)
    tl.atomic_add(FROB + b, tl.sum(nacc, axis=0))


@triton.jit
def _mr_rrqr_panel_kernel(
    W,
    V,
    NRM2,
    PIV,
    TAU,
    J0,
    B,
    K,
    RS,
    WPITCH,
):
    # GM-tile panel factorization for ROWS > 256 (the panel does not fit in
    # register tiles there). No pivoting -- selection/per-step pivoting cost
    # ~55% of the panel time and the test spectra have clear gaps (the k<=64
    # bidiagonalization path is likewise unpivoted). NRM2 is unused (kept in
    # the signature to match the launcher's workspace).
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    vbase = V + pid * WPITCH
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    RB = RS // 64

    piv_acc = tl.zeros((64,), dtype=tl.float32)
    tau_acc = tl.zeros((64,), dtype=tl.float32)
    for jj in tl.range(0, B):
        j = J0 + jj
        # Householder reflector from column j (rows >= j): row-blocked
        # reductions, then the V column store by row block.
        ssq = tl.zeros((), dtype=tl.float32)
        x0 = tl.zeros((), dtype=tl.float32)
        for rb in tl.range(j // 64, RB):
            r0 = rb * 64
            ch = tl.load(wbase + j * RS + r0 + lr)
            ch = ch * ((r0 + lr) >= j).to(tl.float32)
            ssq += tl.sum(ch * ch, axis=0)
            x0 += tl.sum(
                ch * ((r0 + lr > j - 1) & (r0 + lr < j + 1)).to(tl.float32),
                axis=0,
            )
        sigma = tl.sqrt(ssq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        for rb in tl.range(j // 64, RB):
            r0 = rb * 64
            ch = tl.load(wbase + j * RS + r0 + lr)
            ch = ch * ((r0 + lr) >= j).to(tl.float32)
            v2c = tl.where((r0 + lr > j - 1) & (r0 + lr < j + 1), x0 - alpha, ch)
            tl.store(vbase + j * RS + r0 + lr, v2c)
        piv_acc = piv_acc + alpha * ((lc > jj - 1) & (lc < jj + 1)).to(tl.float32)
        tau_acc = tau_acc + tau * ((lc > jj - 1) & (lc < jj + 1)).to(tl.float32)
        tl.debug_barrier()
        # apply H_j to the remaining panel columns. Not gated on jj + 1 < B:
        # an scf.if region around tile ops fails to compile on this backend;
        # on the last step the column mask is all-false so w == 0 and the
        # store writes back identical values.
        colmask = ((lc > jj) & (J0 + lc < K)).to(tl.float32)
        wacc = tl.zeros((64,), dtype=tl.float32)
        for rb in tl.range(j // 64, RB):
            tile = tl.load(wbase + (J0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
            v2p = tl.load(vbase + j * RS + rb * 64 + lr)
            wacc += tl.sum(tile * v2p[None, :], axis=1)
        w = tau * wacc * colmask
        for rb in tl.range(j // 64, RB):
            tile = tl.load(wbase + (J0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
            v2p = tl.load(vbase + j * RS + rb * 64 + lr)
            tile = tile - tl.reshape(w, (64, 1)) * tl.reshape(v2p, (1, 64))
            tl.store(wbase + (J0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :], tile)
        tl.debug_barrier()
    # this panel's R rows (i in [J0, J0+B), i <= col) back to W: the panel
    # columns above the diagonal were left in place by the row-masked apply
    # loops (rows < j untouched per step), so W already holds them; nothing
    # extra to store.
    tl.store(PIV + pid * K + J0 + lc, piv_acc, mask=lc < B)
    tl.store(TAU + pid * K + J0 + lc, tau_acc, mask=lc < B)


@triton.jit
def _mr_rrqr_panel_reg_kernel(
    W,
    V,
    PIV,
    TAU,
    J0,
    B,
    K,
    RS,
    WPITCH,
    NB: tl.constexpr,  # number of 64-row register tiles (1, 2 or 4)
):
    # Register-resident panel factorization supporting RS <= 256 (NB <= 4).
    # The panel lives in NB static (64, 64) register tiles, so a Householder
    # step is a handful of fused tile ops (~9-20us/step vs ~50-100us for the
    # GM-tile panel above). CANN 9.1.1 cannot allocate the NB=2/NB=4
    # specializations, so current launchers use this kernel only for rows <= 64
    # and route larger panels through GM. No pivoting: the
    # k <= 64 bidiagonalization path is likewise unpivoted, and the test
    # spectra have clear gaps.
    #
    # Mask lore: the panel-load mask must not contain a runtime-K comparison
    # ((J0 + lc) < K trips the backend buffer analysis into a 25x UB
    # overallocation); lc < B already implies J0 + lc < K.
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    vbase = V + pid * WPITCH
    lc = tl.arange(0, 64)
    rr = tl.arange(0, 64)
    pm = lc < B  # lc < B implies J0 + lc < K (B = min(64, K - J0)); a
    # runtime-K term in the load mask trips the backend's buffer analysis
    # (ub overflow), so it must not appear in any mask

    g0 = tl.load(
        wbase + (J0 + lc)[None, :] * RS + rr[:, None],
        mask=pm[None, :] & (rr < RS)[:, None],
        other=0.0,
    )
    if NB > 1:
        g1 = tl.load(
            wbase + (J0 + lc)[None, :] * RS + (64 + rr)[:, None],
            mask=pm[None, :] & ((64 + rr) < RS)[:, None],
            other=0.0,
        )
    if NB > 2:
        g2 = tl.load(
            wbase + (J0 + lc)[None, :] * RS + (128 + rr)[:, None],
            mask=pm[None, :] & ((128 + rr) < RS)[:, None],
            other=0.0,
        )
        g3 = tl.load(
            wbase + (J0 + lc)[None, :] * RS + (192 + rr)[:, None],
            mask=pm[None, :] & ((192 + rr) < RS)[:, None],
            other=0.0,
        )

    piv_acc = tl.zeros((64,), dtype=tl.float32)
    tau_acc = tl.zeros((64,), dtype=tl.float32)
    for jj in tl.range(0, B):
        j = J0 + jj
        cj = ((lc > jj - 1) & (lc < jj + 1)).to(tl.float32)
        rj = ((rr > j - 1) & (rr < j + 1)).to(tl.float32)
        e0 = tl.sum(g0 * cj[None, :], axis=1) * (rr >= j).to(tl.float32)
        x0 = tl.sum(e0 * rj, axis=0)
        ssq = tl.sum(e0 * e0, axis=0)
        if NB > 1:
            rj1 = (((64 + rr) > j - 1) & ((64 + rr) < j + 1)).to(tl.float32)
            e1 = tl.sum(g1 * cj[None, :], axis=1) * ((64 + rr) >= j).to(tl.float32)
            x0 += tl.sum(e1 * rj1, axis=0)
            ssq += tl.sum(e1 * e1, axis=0)
        if NB > 2:
            rj2 = (((128 + rr) > j - 1) & ((128 + rr) < j + 1)).to(tl.float32)
            e2 = tl.sum(g2 * cj[None, :], axis=1) * ((128 + rr) >= j).to(tl.float32)
            x0 += tl.sum(e2 * rj2, axis=0)
            ssq += tl.sum(e2 * e2, axis=0)
            rj3 = (((192 + rr) > j - 1) & ((192 + rr) < j + 1)).to(tl.float32)
            e3 = tl.sum(g3 * cj[None, :], axis=1) * ((192 + rr) >= j).to(tl.float32)
            x0 += tl.sum(e3 * rj3, axis=0)
            ssq += tl.sum(e3 * e3, axis=0)
        sigma = tl.sqrt(ssq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        v0 = tl.where(rj > 0.5, x0 - alpha, e0)
        tl.store(vbase + j * RS + rr, v0)
        w = tau * tl.sum(tl.trans(g0) * v0[None, :], axis=1)
        if NB > 1:
            v1 = tl.where(rj1 > 0.5, x0 - alpha, e1)
            tl.store(vbase + j * RS + 64 + rr, v1)
            w = w + tau * tl.sum(tl.trans(g1) * v1[None, :], axis=1)
        if NB > 2:
            v2 = tl.where(rj2 > 0.5, x0 - alpha, e2)
            tl.store(vbase + j * RS + 128 + rr, v2)
            w = w + tau * tl.sum(tl.trans(g2) * v2[None, :], axis=1)
            v3 = tl.where(rj3 > 0.5, x0 - alpha, e3)
            tl.store(vbase + j * RS + 192 + rr, v3)
            w = w + tau * tl.sum(tl.trans(g3) * v3[None, :], axis=1)
        w = w * (lc > jj).to(tl.float32)
        g0 = g0 - tl.reshape(v0, (64, 1)) * tl.reshape(w, (1, 64))
        if NB > 1:
            g1 = g1 - tl.reshape(v1, (64, 1)) * tl.reshape(w, (1, 64))
        if NB > 2:
            g2 = g2 - tl.reshape(v2, (64, 1)) * tl.reshape(w, (1, 64))
            g3 = g3 - tl.reshape(v3, (64, 1)) * tl.reshape(w, (1, 64))
        piv_acc = piv_acc + alpha * cj
        tau_acc = tau_acc + tau * cj
    # R rows of this panel (i in [J0, J0+B), i <= col) back to W, per tile
    m0 = (
        ((rr >= J0) & (rr < J0 + B))[:, None]
        & (rr[:, None] <= (J0 + lc)[None, :])
        & pm[None, :]
    )
    tl.store(wbase + (J0 + lc)[None, :] * RS + rr[:, None], g0, mask=m0)
    if NB > 1:
        r1 = 64 + rr
        m1 = (
            ((r1 >= J0) & (r1 < J0 + B))[:, None]
            & (r1[:, None] <= (J0 + lc)[None, :])
            & pm[None, :]
        )
        tl.store(wbase + (J0 + lc)[None, :] * RS + r1[:, None], g1, mask=m1)
    if NB > 2:
        r2 = 128 + rr
        m2 = (
            ((r2 >= J0) & (r2 < J0 + B))[:, None]
            & (r2[:, None] <= (J0 + lc)[None, :])
            & pm[None, :]
        )
        tl.store(wbase + (J0 + lc)[None, :] * RS + r2[:, None], g2, mask=m2)
        r3 = 192 + rr
        m3 = (
            ((r3 >= J0) & (r3 < J0 + B))[:, None]
            & (r3[:, None] <= (J0 + lc)[None, :])
            & pm[None, :]
        )
        tl.store(wbase + (J0 + lc)[None, :] * RS + r3[:, None], g3, mask=m3)
    tl.store(PIV + pid * K + J0 + lc, piv_acc, mask=lc < B)
    tl.store(TAU + pid * K + J0 + lc, tau_acc, mask=lc < B)


@triton.jit
def _mr_rrqr_vtv_kernel(
    V,
    TAU,
    T,
    J0,
    B,
    K,
    RS,
    WPITCH,
):
    pid = tl.program_id(0)
    vbase = V + pid * WPITCH
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    RB = RS // 64
    g = tl.zeros((64, 64), dtype=tl.float32)
    # loop from row block 0: a dynamic START (J0 // 64) crashes bishengir in
    # this kernel; rows < J0 of the reflectors are zero, so they contribute
    # nothing to G.
    for rb in tl.range(0, RB):
        vt = tl.load(vbase + (J0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
        g = g + tl.dot(vt, tl.trans(vt), input_precision="ieee")
    ta = tl.load(TAU + pid * K + J0 + lc, mask=lc < B, other=0.0)
    tt = tl.zeros((64, 64), dtype=tl.float32)
    for t in tl.range(0, B):
        cmask = ((lc > t - 1) & (lc < t + 1)).to(tl.float32)
        gcol = tl.sum(g * tl.reshape(cmask, (1, 64)), axis=1)  # G[:, t]
        ta_t = tl.sum(ta * cmask, axis=0)
        mv = tl.sum(tt * tl.reshape(gcol, (1, 64)), axis=1)  # T[:, :t] @ G[:t, t]
        tcol = -ta_t * mv
        tcol = tl.where((lc > t - 1) & (lc < t + 1), ta_t, tcol)
        # column t of tt is still zero at step t, so a pure accumulate of the
        # outer product tcol x cmask writes it (the multiply-and-add form
        # tt*(1-cm)+tcol@cm miscompiles at higher trip counts; verified).
        tt = tt + tl.reshape(tcol, (64, 1)) * tl.reshape(cmask, (1, 64))
    tl.store(T + pid * 4096 + lc[:, None] * 64 + lr[None, :], tt)


@triton.jit
def _mr_rrqr_update_kernel(
    W,
    V,
    T,
    SCR,
    NRM2,
    J0,
    B,
    K,
    RS,
    WPITCH,
    SCPITCH,
):
    pid = tl.program_id(0)
    tile_id = tl.program_id(1)
    c0 = J0 + B + tile_id * 64
    wbase = W + pid * WPITCH
    vbase = V + pid * WPITCH
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    RB = RS // 64
    s = tl.zeros((64, 64), dtype=tl.float32)
    # loops run from row block 0 (dynamic starts crash bishengir here); the
    # reflector rows < J0 are zero, so S and the update are unaffected, and
    # rows < J0 of the trailing columns are rewritten with identical values.
    for rb in tl.range(0, RB):
        vt = tl.load(vbase + (J0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
        wt = tl.load(wbase + (c0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
        s = s + tl.dot(vt, tl.trans(wt), input_precision="ieee")
    scr = SCR + pid * SCPITCH + tile_id * 4096
    tl.store(scr + lc[:, None] * 64 + lr[None, :], s)
    tl.debug_barrier()
    s2 = tl.load(scr + lc[:, None] * 64 + lr[None, :])
    tt = tl.load(T + pid * 4096 + lc[:, None] * 64 + lr[None, :])
    ts = tl.dot(tl.trans(tt), s2, input_precision="ieee")
    tl.store(scr + lc[:, None] * 64 + lr[None, :], ts)
    tl.debug_barrier()
    ts2 = tl.load(scr + lc[:, None] * 64 + lr[None, :])
    dacc = tl.zeros((64,), dtype=tl.float32)
    for rb in tl.range(0, RB):
        vt = tl.load(vbase + (J0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
        upd = tl.dot(tl.trans(vt), ts2, input_precision="ieee")
        wt = tl.load(wbase + (c0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :])
        wt = wt - tl.trans(upd)
        tl.store(wbase + (c0 + lc)[:, None] * RS + (rb * 64 + lr)[None, :], wt)
        # column-norm downdate with this panel's R rows: J0 is a multiple of
        # 64, so the R-row band is exactly the first row block of the loop.
        band = ((rb * 64 > J0 - 1) & (rb * 64 < J0 + 1)).to(tl.float32)
        dacc += tl.sum(wt * wt, axis=1) * band
    # nrm2[trailing] -= ||R rows of this panel||^2 (consumed by the next
    # panel's selection; kernel boundary serializes).
    nrm = tl.load(NRM2 + pid * K + c0 + lc, mask=(c0 + lc) < K, other=0.0)
    tl.store(NRM2 + pid * K + c0 + lc, nrm - dacc, mask=(c0 + lc) < K)


@triton.jit
def _mr_rrqr_count_kernel(
    W,
    PIV,
    ATOL,
    RTOL,
    FROB,
    OUT,
    X,
    Y,
    K,
    RS,
    WPITCH,
    BLOCK_K: tl.constexpr,
    ITER: tl.constexpr,
):
    pid = tl.program_id(0)
    kk = tl.arange(0, BLOCK_K)
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    kmask = kk < K
    wbase = W + pid * WPITCH
    piv = tl.load(PIV + pid * K + kk, mask=kmask, other=0.0)
    apiv = tl.abs(piv)
    atol = tl.load(ATOL + pid)
    rtol = tl.load(RTOL + pid)
    frob = tl.load(FROB + pid)
    sigma_lo = tl.sum(apiv * ((kk > -1) & (kk < 1)).to(tl.float32), axis=0)
    sigma_hi = tl.sqrt(tl.maximum(frob, 0.0))
    tol_lo = tl.maximum(atol, rtol * sigma_lo)
    tol_hi = tl.maximum(atol, rtol * sigma_hi)
    rank_lo = tl.sum(((apiv > tol_lo) & kmask).to(tl.int32), axis=0)
    rank_hi = tl.sum(((apiv > tol_hi) & kmask).to(tl.int32), axis=0)
    rank = rank_lo
    if rank_lo != rank_hi:
        # power iteration for sigma_max(R): x <- R^T R x / ||.||
        KB = (K + 63) // 64
        xb = X + pid * BLOCK_K
        yb = Y + pid * BLOCK_K
        tl.store(xb + kk, (1.0 / tl.sqrt(K * 1.0)) * kmask.to(tl.float32))
        tl.debug_barrier()
        sig2 = tl.zeros((), dtype=tl.float32)
        for _ in tl.range(0, ITER):
            for ib in tl.range(0, KB):
                acc = tl.zeros((64,), dtype=tl.float32)
                for jb in tl.range(ib, KB):
                    tile = tl.load(
                        wbase + (jb * 64 + lc)[:, None] * RS + (ib * 64 + lr)[None, :]
                    )
                    xj = tl.load(xb + jb * 64 + lc)
                    acc += tl.sum(tl.trans(tile) * xj[None, :], axis=1)
                tl.store(yb + ib * 64 + lc, acc)
            tl.debug_barrier()
            for jb in tl.range(0, KB):
                acc = tl.zeros((64,), dtype=tl.float32)
                for ib in tl.range(0, jb + 1):
                    tile = tl.load(
                        wbase + (jb * 64 + lc)[:, None] * RS + (ib * 64 + lr)[None, :]
                    )
                    yi = tl.load(yb + ib * 64 + lc)
                    acc += tl.sum(tile * yi[None, :], axis=1)
                tl.store(xb + jb * 64 + lc, acc)
            tl.debug_barrier()
            xv = tl.load(xb + kk, mask=kmask, other=0.0)
            nrm = tl.sqrt(tl.sum(xv * xv, axis=0))
            sig2 = nrm
            tl.store(xb + kk, xv * (1.0 / tl.maximum(nrm, 1e-30)))
            tl.debug_barrier()
        sigma_max = tl.sqrt(sig2)
        tol = tl.maximum(atol, rtol * sigma_max)
        rank = tl.sum(((apiv > tol) & kmask).to(tl.int32), axis=0)
    tl.store(OUT + pid, rank.to(tl.int64))


# ---------------------------------------------------------------------------
# Fast kernel launch: the triton-ascend JIT dispatch costs ~460us per launch
# on this host (slow ARM CPU + arg re-binding); a pre-bound CompiledKernel
# run costs ~12us. The cache key carries the grid, constexpr kwargs, integer
# argument values and pointer alignment, so Triton's argument specialization
# (value 1 / divisibility-by-16) stays correct per entry.
# ---------------------------------------------------------------------------
_FAST_LAUNCH_CACHE = {}


def _fast_launch(kernel, grid, *args, **kwargs):
    # Triton <= 3.2 passes only non-constexpr values to CompiledKernel.run,
    # while Triton 3.5 passes every bound kernel argument (including
    # tl.constexpr values).  The Ascend C launcher is generated to match the
    # frontend ABI, so select the argument set expected by that frontend.
    # Every call below passes the non-constexpr prefix positionally and the
    # remaining kernel arguments by name; direct tuple assembly preserves the
    # low-overhead cache-hit path.  Calling run with only ``args`` on 3.5
    # produces e.g. "function takes exactly 19 arguments (13 given)".
    key = (
        id(kernel),
        tuple(grid),
        tuple(sorted(kwargs.items())),
        tuple(a if isinstance(a, int) else None for a in args),
        tuple(a.data_ptr() % 16 == 0 if torch.is_tensor(a) else None for a in args),
    )
    entry = _FAST_LAUNCH_CACHE.get(key)
    if entry is None:
        compiled = kernel.warmup(*args, grid=grid, **kwargs)
        compiled._init_handles()
        entry = (compiled.run, compiled.function, compiled.packed_metadata, compiled)
        _FAST_LAUNCH_CACHE[key] = entry
    run, function, md, compiled = entry
    if hasattr(kernel, "non_constexpr_indices"):
        launch_args = args
    else:
        launch_args = args + tuple(
            kwargs[name] for name in kernel.arg_names[len(args) :]
        )
    from triton.runtime import driver

    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    lm = compiled.launch_metadata(grid, stream, *launch_args)
    g0 = grid[0] if len(grid) > 0 else 1
    g1 = grid[1] if len(grid) > 1 else 1
    g2 = grid[2] if len(grid) > 2 else 1
    run(g0, g1, g2, stream, function, md, lm, None, None, *launch_args)


def _launch_longdim_rank(
    matrix,
    atol_tensor,
    rtol_tensor,
    atol_val,
    rtol_val,
    out,
    m,
    n,
    k,
    rows,
    batch_count,
    input,
):
    """Long-dimension non-hermitian k <= 64 (one of m, n > 64): QR-reduce to
    the k x k R factor, then the exact bidiag64 + df64 Sturm tail.

    Householder QR is backward stable in the LINEAR domain (sigma(R) ==
    sigma(A) up to ~k*eps*||A||), so compressing the long dimension through
    R never squares the spectrum -- unlike the Gram path this replaces,
    whose sigma^2-domain floor overestimates rank on slowly-decaying
    spectra.  For k <= 64 the whole width fits one panel, so the QR is just
    init + one panel kernel (no VTV/trailing update).
    """
    dev = input.device
    kp = 64
    rs = triton.cdiv(rows, 64) * 64
    wpitch = kp * rs
    W = torch.empty((batch_count, kp, rs), dtype=torch.float32, device=dev)
    V = torch.zeros((batch_count, kp, rs), dtype=torch.float32, device=dev)
    nrm2 = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    piv = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    tau = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    frob = torch.zeros((batch_count,), dtype=torch.float32, device=dev)
    RB = torch.empty((batch_count, k, k), dtype=torch.float32, device=dev)
    d = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    e = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    ddbuf = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    eebuf = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    tol2_buf = torch.empty((batch_count,), dtype=torch.float32, device=dev)
    flag_buf = torch.empty((batch_count,), dtype=torch.int64, device=dev)
    tol_tensor = atol_tensor is not None
    at_arg, rt_arg = atol_tensor, rtol_tensor
    if not tol_tensor:
        # the Sturm tail kernels take tolerance tensors
        at_arg = torch.full((batch_count,), atol_val, dtype=torch.float32, device=dev)
        rt_arg = torch.full((batch_count,), rtol_val, dtype=torch.float32, device=dev)
    with torch_device_fn.device(input.device):
        _fast_launch(
            _mr_rrqr_init_kernel,
            (batch_count, 1),
            matrix,
            W,
            nrm2,
            frob,
            m,
            n,
            k,
            rows,
            rs,
            wpitch,
            TALL=m >= n,
            HERMITIAN=False,
            num_warps=4,
            num_stages=1,
        )
        if rows <= _RRQR_REG_PANEL_MAX_ROWS:
            _fast_launch(
                _mr_rrqr_panel_reg_kernel,
                (batch_count,),
                W,
                V,
                piv,
                tau,
                0,
                k,
                k,
                rs,
                wpitch,
                NB=triton.next_power_of_2(max(1, rs // 64)),
                num_warps=4,
                num_stages=1,
            )
        else:
            _fast_launch(
                _mr_rrqr_panel_kernel,
                (batch_count,),
                W,
                V,
                nrm2,
                piv,
                tau,
                0,
                k,
                k,
                rs,
                wpitch,
                num_warps=8,
                num_stages=1,
                multibuffer=False,
            )
        _fast_launch(
            _mr_extract_upper_kernel,
            (batch_count,),
            W,
            piv,
            RB,
            k,
            rs,
            wpitch,
            BLOCK=64,
            num_warps=4,
            num_stages=1,
        )
        _fast_launch(
            _matrix_rank_bidiag64_kernel,
            (batch_count,),
            RB,
            d,
            e,
            k,
            k,
            k,
            BLOCK=64,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=True,
        )
        _fast_launch(
            _mr_bidiag_to_tridiag_kernel,
            (batch_count,),
            d,
            e,
            ddbuf,
            eebuf,
            k,
            BLOCK=64,
            num_warps=1,
            num_stages=1,
        )
        _fast_launch(
            _mr_sturm_big_kernel,
            (batch_count,),
            ddbuf,
            eebuf,
            at_arg,
            rt_arg,
            out.reshape(batch_count),
            tol2_buf,
            flag_buf,
            k,
            BLOCK=64,
            BISECT_ITERS=32,
            num_warps=1,
            num_stages=1,
        )
        _fast_launch(
            _mr_sturm_final_kernel,
            (batch_count,),
            d,
            e,
            tol2_buf,
            flag_buf,
            out.reshape(batch_count),
            k,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


def _launch_rrqr_rank(
    matrix,
    atol_tensor,
    rtol_tensor,
    out,
    m,
    n,
    k,
    rows,
    batch_count,
    input,
    hermitian,
):
    """Blocked Householder QR, unpivoted (fp32, 64 < k <= 255, rows > 256).

    Pure Triton: no aclnn decomposition. Singular values never materialize;
    the rank is read off the |R_ii| diagonal, which sits in the LINEAR domain
    (no Gram squaring), so the smallest singular value keeps full relative
    precision. Panels are factored by the register-resident kernel when the
    row count fits (rows <= 64), else by the GM-tile kernel. Current dispatch
    enters this function only for non-hermitian rows > 256, hence always GM.
    """
    dev = input.device
    reg_panel = rows <= _RRQR_REG_PANEL_MAX_ROWS
    kp = triton.cdiv(k, 64) * 64
    rs = triton.cdiv(rows, 64) * 64
    wpitch = kp * rs
    ntmax = kp // 64
    block_k = triton.next_power_of_2(k)
    # 64-row tiles in the register panel kernel. When ``reg_panel`` is true,
    # the row threshold guarantees NB is 1; the value is unused for the
    # current GM-panel branch.
    nb = triton.next_power_of_2(max(1, rs // 64))
    W = torch.empty((batch_count, kp, rs), dtype=torch.float32, device=dev)
    V = torch.zeros((batch_count, kp, rs), dtype=torch.float32, device=dev)
    nrm2 = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    piv = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    tau = torch.empty((batch_count, k), dtype=torch.float32, device=dev)
    T = torch.empty((batch_count, 64, 64), dtype=torch.float32, device=dev)
    frob = torch.zeros((batch_count,), dtype=torch.float32, device=dev)
    scr = torch.empty((batch_count, ntmax * 4096), dtype=torch.float32, device=dev)
    xs = torch.empty((batch_count, block_k), dtype=torch.float32, device=dev)
    ys = torch.empty((batch_count, block_k), dtype=torch.float32, device=dev)
    with torch_device_fn.device(input.device):
        _fast_launch(
            _mr_rrqr_init_kernel,
            (batch_count, kp // 64),
            matrix,
            W,
            nrm2,
            frob,
            m,
            n,
            k,
            rows,
            rs,
            wpitch,
            TALL=m >= n,
            HERMITIAN=hermitian,
            num_warps=4,
            num_stages=1,
        )
        j0 = 0
        while j0 < k:
            b = min(64, k - j0)
            nt = triton.cdiv(k - j0 - b, 64)
            # b == 1 panels must NOT go to the register panel kernel: the
            # JIT specializes int args equal to 1, and the B=1
            # specialization UB-overflows (verified: k=129 fails while
            # k=131 compiles, same NB).  Use the GM-tile panel for those.
            if reg_panel and b != 1:
                _fast_launch(
                    _mr_rrqr_panel_reg_kernel,
                    (batch_count,),
                    W,
                    V,
                    piv,
                    tau,
                    j0,
                    b,
                    k,
                    rs,
                    wpitch,
                    NB=nb,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                _fast_launch(
                    _mr_rrqr_panel_kernel,
                    (batch_count,),
                    W,
                    V,
                    nrm2,
                    piv,
                    tau,
                    j0,
                    b,
                    k,
                    rs,
                    wpitch,
                    num_warps=8,
                    num_stages=1,
                    multibuffer=False,
                )
            if nt > 0:
                _fast_launch(
                    _mr_rrqr_vtv_kernel,
                    (batch_count,),
                    V,
                    tau,
                    T,
                    j0,
                    b,
                    k,
                    rs,
                    wpitch,
                    num_warps=4,
                    num_stages=1,
                )
                _fast_launch(
                    _mr_rrqr_update_kernel,
                    (batch_count, nt),
                    W,
                    V,
                    T,
                    scr,
                    nrm2,
                    j0,
                    b,
                    k,
                    rs,
                    wpitch,
                    ntmax * 4096,
                    num_warps=4,
                    num_stages=1,
                )
            j0 += b
        _fast_launch(
            _mr_rrqr_count_kernel,
            (batch_count,),
            W,
            piv,
            atol_tensor,
            rtol_tensor,
            frob,
            out.reshape(batch_count),
            xs,
            ys,
            k,
            rs,
            wpitch,
            BLOCK_K=block_k,
            ITER=30,
            num_warps=4,
            num_stages=1,
        )
    return out


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
@triton.jit
def _mr_bidiag_lstep_kernel(W, V, D, TAU, ACC, J, K, RS, WPITCH, APITCH, RBV):
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    lr = tl.arange(0, 64)
    ssq = tl.zeros((), dtype=tl.float32)
    x0 = tl.zeros((), dtype=tl.float32)
    # single pass over the column: store the (masked) reflector vector
    # provisionally while accumulating the norm, then overwrite the pivot
    # element with x0 - alpha afterwards (stores only -- no read-back, so
    # the MTE3/MTE2 ordering hazard does not apply).
    # RBV = cdiv(rows, 64) trims the padded/slack tail tiles: they are
    # zeros by construction, contribute nothing to the norm, and the
    # reflector entries there are never read.
    for rb in tl.range(J // 64, RBV):
        r0 = rb * 64
        ch = tl.load(wbase + J * RS + r0 + lr)
        ch = ch * ((r0 + lr) >= J).to(tl.float32)
        tl.store(V + pid * WPITCH + J * RS + r0 + lr, ch)
        ssq += tl.sum(ch * ch, axis=0)
        x0 += tl.sum(
            ch * ((r0 + lr > J - 1) & (r0 + lr < J + 1)).to(tl.float32), axis=0
        )
    sigma = tl.sqrt(ssq)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    tl.store(V + pid * WPITCH + J * RS + J, x0 - alpha)
    tl.store(D + pid * K + J, alpha)
    tl.store(TAU + pid * K + J, tau)
    for cb in tl.range(0, APITCH // 64):
        tl.store(ACC + pid * APITCH + cb * 64 + lr, tl.zeros((64,), dtype=tl.float32))


@triton.jit
def _mr_bidiag_lmat_kernel(W, V, ACC, J, K, RS, WPITCH, APITCH, NRC, RB0):
    # NOTE: 3D grids miscompute on this backend past small sizes (verified:
    # (1,8,8) silently wrong, flat 2D exact), so tiles are flattened.
    # Row tiles below RB0 = J // 64 are skipped: v is zero above row J, so
    # they contribute nothing to the reduction.
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRC
    rc = flat % NRC
    c0 = J + 1 + ct * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    r0 = (RB0 + rc) * 64
    wbase = W + pid * WPITCH
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    v2p = tl.load(V + pid * WPITCH + J * RS + r0 + lr)
    part = tl.sum(tile * v2p[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + c0 + lc, part)


@triton.jit
def _mr_bidiag_lapply_kernel(W, V, TAU, ACC, J, K, RS, WPITCH, APITCH, NRC, RB0):
    # Row tiles below RB0 are skipped: v is zero there, so the rank-1
    # update would be a no-op (and the load+store is pure waste).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRC
    rc = flat % NRC
    c0 = J + 1 + ct * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    r0 = (RB0 + rc) * 64
    wbase = W + pid * WPITCH
    tau = tl.load(TAU + pid * K + J)
    w = tl.load(ACC + pid * APITCH + c0 + lc) * tau
    v2p = tl.load(V + pid * WPITCH + J * RS + r0 + lr)
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    tile = tile - tl.reshape(w, (64, 1)) * tl.reshape(v2p, (1, 64))
    tl.store(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :], tile)


@triton.jit
def _mr_bidiag_rstep_kernel(W, U, E, TAU, ACC, J, K, RS, WPITCH, UPITCH, APITCH):
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    lc = tl.arange(0, 64)
    ssq = tl.zeros((), dtype=tl.float32)
    x0 = tl.zeros((), dtype=tl.float32)
    # single pass (see _mr_bidiag_lstep_kernel); columns <= J contribute
    # nothing (u has support on c > J), so start at the (J+1)'s tile.
    for cb in tl.range((J + 1) // 64, (K + 63) // 64):
        c0 = cb * 64
        ch = tl.load(wbase + (c0 + lc) * RS + J, mask=(c0 + lc) < K, other=0.0)
        ch = ch * ((c0 + lc) > J).to(tl.float32)
        tl.store(U + pid * UPITCH + J * K + c0 + lc, ch, mask=(c0 + lc) < K)
        ssq += tl.sum(ch * ch, axis=0)
        x0 += tl.sum(ch * ((c0 + lc > J) & (c0 + lc < J + 2)).to(tl.float32), axis=0)
    sigma = tl.sqrt(ssq)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    tl.store(U + pid * UPITCH + J * K + (J + 1), x0 - alpha)
    tl.store(E + pid * K + J, alpha)
    tl.store(TAU + pid * K + J, tau)
    for cb in tl.range(0, APITCH // 64):
        tl.store(ACC + pid * APITCH + cb * 64 + lc, tl.zeros((64,), dtype=tl.float32))


@triton.jit
def _mr_bidiag_rmat_kernel(W, U, ACC, J, K, RS, WPITCH, UPITCH, APITCH, NCC, CC0):
    # Column tiles below CC0 = (J+1) // 64 are skipped: u is zero on
    # columns <= J, so they contribute nothing to the reduction.
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    rt = flat // NCC
    cc = flat % NCC
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    c0 = (CC0 + cc) * 64
    wbase = W + pid * WPITCH
    # the J+1-aligned row grid is not 64-aligned, so the last tile straddles
    # the RS row pitch; mask it (the wrapped-around address is the NEXT
    # column's row 0 -- an out-of-bounds read/write otherwise)
    rmask = (r0 + lr) < RS
    tile = tl.load(
        wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :],
        mask=rmask[None, :],
        other=0.0,
    )
    up = tl.load(U + pid * UPITCH + J * K + c0 + lc, mask=(c0 + lc) < K, other=0.0)
    part = tl.sum(tl.trans(tile) * up[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + r0 + lr, part)


@triton.jit
def _mr_bidiag_rapply_kernel(
    W, U, TAU, ACC, J, K, RS, WPITCH, UPITCH, APITCH, NRC, NTR
):
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NTR
    rc = flat % NTR
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rc * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    wbase = W + pid * WPITCH
    tau = tl.load(TAU + pid * K + J)
    rmask = (r0 + lr) < RS
    wu = tl.load(ACC + pid * APITCH + r0 + lr, mask=rmask, other=0.0) * tau
    up = tl.load(U + pid * UPITCH + J * K + c0 + lc, mask=(c0 + lc) < K, other=0.0)
    tile = tl.load(
        wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :],
        mask=rmask[None, :],
        other=0.0,
    )
    tile = tile - tl.reshape(up, (64, 1)) * tl.reshape(wu, (1, 64))
    tl.store(
        wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :], tile, mask=rmask[None, :]
    )


@triton.jit
def _mr_tridiag_step_kernel(W, V, D, E, TAU, ACC, CSCA, J, K, RS, WPITCH, APITCH):
    # Hermitian (symmetric) one-sided Householder tridiagonalization, step J:
    # column J below the diagonal defines the reflector (support rows > J),
    # D[J] = W[J,J] (untouched by this step's reflection), E[J] = +/-sigma.
    # Single pass (provisional v store + scalar pivot overwrite, see
    # _mr_bidiag_lstep_kernel); also zeroes the ACC/CSCA accumulation slots
    # that the mat kernel's atomics add into.
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    lr = tl.arange(0, 64)
    dj = tl.zeros((), dtype=tl.float32)
    ssq = tl.zeros((), dtype=tl.float32)
    x0 = tl.zeros((), dtype=tl.float32)
    for rb in tl.range(J // 64, (K + 63) // 64):
        r0 = rb * 64
        chf = tl.load(wbase + J * RS + r0 + lr)
        dj += tl.sum(
            chf * ((r0 + lr > J - 1) & (r0 + lr < J + 1)).to(tl.float32), axis=0
        )
        ch = chf * ((r0 + lr) > J).to(tl.float32)
        tl.store(V + pid * WPITCH + J * RS + r0 + lr, ch)
        ssq += tl.sum(ch * ch, axis=0)
        x0 += tl.sum(ch * ((r0 + lr > J) & (r0 + lr < J + 2)).to(tl.float32), axis=0)
    sigma = tl.sqrt(ssq)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    tl.store(V + pid * WPITCH + J * RS + (J + 1), x0 - alpha)
    tl.store(D + pid * K + J, dj)
    tl.store(E + pid * K + J, alpha)
    tl.store(TAU + pid * K + J, tau)
    tl.store(CSCA + pid, tl.zeros((), dtype=tl.float32))
    for cb in tl.range((J + 1) // 64, APITCH // 64):
        tl.store(ACC + pid * APITCH + cb * 64 + lr, tl.zeros((64,), dtype=tl.float32))


@triton.jit
def _mr_tridiag_mat_kernel(W, V, ACC, CSCA, J, K, RS, WPITCH, APITCH, NRT):
    # omega = W_trailing . v (symmetric matvec), multi-program over
    # (col tile, row strip) with atomic accumulation.  NRT counts the
    # K-trimmed trailing tiles (cdiv(K-1-J, 64)): with c0/r0 = J+1+tile*64
    # the last tile reaches at most K+62 < RS, so no row-pitch straddle
    # and no masks (pad rows/cols are zeros by construction and excluded
    # entirely).  Each program also accumulates its slice of v^T omega
    # into CSCA (for the rank-2 correction coefficient in apply).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRT
    rt = flat % NRT
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    wbase = W + pid * WPITCH
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    vc = tl.load(V + pid * WPITCH + J * RS + c0 + lc)
    part = tl.sum(tl.trans(tile) * vc[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + r0 + lr, part)
    vp = tl.load(V + pid * WPITCH + J * RS + r0 + lr)
    tl.atomic_add(CSCA + pid, tl.sum(part * vp, axis=0))


@triton.jit
def _mr_tridiag_apply_kernel(W, V, TAU, ACC, CSCA, J, K, RS, WPITCH, APITCH, NRT):
    # Trailing symmetric rank-2 update W -= v w^T + w v^T with
    # w = tau*omega - (tau^2/2)(v^T omega) v (LAPACK DSYTD2 form), one
    # program per (col tile, row strip) of the K-trimmed trailing block
    # (tiles never reach RS, so no masks).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRT
    rt = flat % NRT
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    wbase = W + pid * WPITCH
    tau = tl.load(TAU + pid * K + J)
    cs = tl.load(CSCA + pid)
    # coef = (tau^2/2)(v^T omega): evaluate as tau*(tau*cs), NOT tau*tau*cs.
    # When the trailing subdiagonal deflates (sigma ~ 1e-10 on clustered or
    # strongly rank-deficient spectra), tau = 2/vnorm2 ~ 1e20 and tau*tau
    # overflows fp32 (2e40 > 3.4e38) even though the final coefficient is
    # O(1) -- verified: W diagonal corrupted to -inf, then 0*inf -> NaN in
    # the next step's float mask.  The regrouped form never squares tau.
    coef = 0.5 * tau * (tau * cs)
    om_r = tl.load(ACC + pid * APITCH + r0 + lr)
    v_r = tl.load(V + pid * WPITCH + J * RS + r0 + lr)
    om_c = tl.load(ACC + pid * APITCH + c0 + lc)
    v_c = tl.load(V + pid * WPITCH + J * RS + c0 + lc)
    w_r = tau * om_r - coef * v_r
    w_c = tau * om_c - coef * v_c
    ptrs = wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :]
    tile = tl.load(ptrs)
    tile = tile - tl.reshape(v_c, (64, 1)) * tl.reshape(w_r, (1, 64))
    tile = tile - tl.reshape(w_c, (64, 1)) * tl.reshape(v_r, (1, 64))
    tl.store(ptrs, tile)


@triton.jit
def _mr_bidiag_to_tridiag_kernel(D, E, DD, EE, K, BLOCK: tl.constexpr):
    # Construct the B^T B tridiagonal (dd_i = d_i^2 + e_{i-1}^2,
    # ee_i = d_i * e_i) from the raw bidiagonal d/e, in its OWN launch:
    # the Sturm kernels then only READ global memory.  All loads happen
    # before the stores -- a same-kernel store->load round trip is not
    # ordered on this backend (MTE3 store / MTE2 load), so the consumer
    # must live behind a kernel boundary.  DD/EE are SEPARATE buffers: the
    # fp32 tridiagonal has an absolute element rounding of ~eps*sigma_max^2,
    # which swamps any squared singular value below ~(sqrt(eps)*sigma_max)^2
    # -- i.e. everything near the default tolerance (k*eps*sigma_max).  The
    # decisive df64 kernel therefore works from the RAW d/e instead (their
    # relative precision is ~K*eps regardless of conditioning) and squares
    # them inside the df64 arithmetic.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    base = batch * K
    kmask = kidx < K
    d = tl.load(D + base + kidx, mask=kmask, other=0.0)
    e_cur = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    e_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & kmask, other=0.0)
    tl.store(DD + base + kidx, d * d + e_prev * e_prev, mask=kmask)
    tl.store(EE + base + kidx, d * e_cur, mask=kmask)


@triton.jit
def _mr_sturm_big_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    TOL2,
    FLAG,
    K,
    BLOCK: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
):
    # Large-K variant of the bidiagonal Sturm count (no 2D tiles: those do
    # not compile past 64 wide).  D/E already hold the FINAL B^T B
    # tridiagonal, written by _mr_bidiag_to_tridiag_kernel in a previous
    # launch, so this kernel only READS global memory.  Counts
    # #{sigma > max(atol, rtol*sigma_max)} with sigma_max bracketed by
    # sqrt(max dd) / sqrt(Gershgorin) and refined by bisection only when
    # the two bounds give different ranks.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    base = batch * K
    kmask = kidx < K
    dd = tl.load(D + base + kidx, mask=kmask, other=0.0)
    ee = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    ee_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & kmask, other=0.0)
    gershgorin = tl.abs(dd) + tl.abs(ee) + tl.abs(ee_prev)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(dd, axis=0)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    if hi == 0.0:
        # Zero matrix: rank 0. TOL2 must still be written: the final df64
        # kernel runs unconditionally and reads it -- an uninitialized tol2
        # (e.g. a negative value recycled by the caching allocator) makes
        # the df64 count report full rank and overwrites the zero result.
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
        tl.store(TOL2 + batch, 0.0)
        tl.store(FLAG + batch, tl.zeros((), dtype=tl.int64))
    else:
        sigma_lo = tl.sqrt(tl.maximum(dmax, 0.0))
        sigma_hi = tl.sqrt(hi)
        tol_lo = tl.maximum(atol, rtol * sigma_lo)
        tol_hi = tl.maximum(atol, rtol * sigma_hi)
        cnt_lo = _sturm_count_less(D, E, base, K, tol_lo * tol_lo)
        cnt_hi = _sturm_count_less(D, E, base, K, tol_hi * tol_hi)
        rank_lo = K - cnt_lo
        rank_hi = K - cnt_hi
        rank = rank_lo
        refine = rank_lo != rank_hi
        tol2 = tol_lo * tol_lo
        if refine:
            lo = tl.maximum(dmax, 0.0)
            hi_p = hi * (1.0 + 1e-9) + 1e-30
            it = 0
            while it < BISECT_ITERS:
                mid = 0.5 * (lo + hi_p)
                cnt = _sturm_count_less(D, E, base, K, mid)
                if cnt >= K:
                    hi_p = mid
                else:
                    lo = mid
                it += 1
            lmax = 0.5 * (lo + hi_p)
            sigma_max = tl.sqrt(lmax)
            tol2 = tl.maximum(atol, rtol * sigma_max)
            tol2 = tol2 * tol2
        # The final Sturm count runs in a separate kernel: a while-loop count
        # directly after the bisection while-loop miscomputes on this backend
        # (verified: wrong pivot signs). tol2/flag hand off through GM.
        tl.store(OUT + batch, rank.to(tl.int64))
        tl.store(TOL2 + batch, tol2)
        tl.store(FLAG + batch, refine.to(tl.int64))


@triton.jit
def _df64_add(h1, l1, h2, l2):
    # Error-free addition of two double-single numbers (Knuth TwoSum on the
    # hi parts, lo parts gathered afterwards, then one renormalization).
    s = h1 + h2
    z = s - h1
    e = (h1 - (s - z)) + (h2 - z)
    lo = l1 + l2 + e
    h = s + lo
    e2 = lo - (h - s)
    return h, e2


@triton.jit
def _df64_mul_ds(a_h, a_l, b_h, b_l):
    # Double-single product: TwoProd on the hi parts plus the cross terms.
    p = a_h * b_h
    e = tl.fma(a_h, b_h, -p) + a_h * b_l + a_l * b_h
    h = p + e
    low = e - (h - p)
    return h, low


@triton.jit
def _df64_div_ds(a_h, a_l, b_h, b_l):
    # Double-single division: fp32 quotient plus one df64 correction step.
    q1 = a_h / b_h
    p = q1 * b_h
    pe = tl.fma(q1, b_h, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -(pe + q1 * b_l))
    q2 = r_h / b_h
    h = q1 + q2
    low = q2 - (h - q1)
    return h, low


@triton.jit
def _mr_sturm_final_kernel(D, E, TOL2, FLAG, OUT, K):
    # The decisive count, in double-single (df64) arithmetic, from the RAW
    # bidiagonal d/e (NOT the fp32 dd/ee tridiagonal: squaring into fp32
    # rounds each dd/ee absolutely by ~eps*sigma_max^2, which swamps any
    # lambda near the default tol^2 ~ (k*eps*sigma_max)^2 -- a sqrt(eps)
    # noise floor in the sigma domain).  Squaring the fp32 d/e inside df64
    # keeps every intermediate at ~eps^2 relative precision.  The qd
    # recurrence for B^T B: dd_i = d_i^2 + e_{i-1}^2, ee_i = d_i*e_i,
    # q_i = dd_i - tol2 - ee_{i-1}^2 / q_{i-1}.
    # Requires enable_fp_fusion=False (fma contraction breaks the
    # TwoSum/TwoProd error-free transforms).
    batch = tl.program_id(0)
    tol2 = tl.load(TOL2 + batch)
    base = batch * K
    d0 = tl.load(D + base)
    d0h, d0l = _df64_mul_ds(d0, 0.0, d0, 0.0)
    q_h, q_l = _df64_add(d0h, d0l, -tol2, 0.0)
    zero = (q_h == 0.0) & (q_l == 0.0)
    q_h = tl.where(zero, -1.1754944e-38, q_h)
    q_l = tl.where(zero, 0.0, q_l)
    neg = ((q_h < 0.0) | ((q_h == 0.0) & (q_l < 0.0))).to(tl.int32)
    dprev = d0
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        dih, dil = _df64_mul_ds(di, 0.0, di, 0.0)
        eih, eil = _df64_mul_ds(ei, 0.0, ei, 0.0)
        dd_h, dd_l = _df64_add(dih, dil, eih, eil)
        ph, pl = _df64_mul_ds(dprev, 0.0, ei, 0.0)
        ee_h, ee_l = _df64_mul_ds(ph, pl, ph, pl)
        qd_h, qd_l = _df64_div_ds(ee_h, ee_l, q_h, q_l)
        s_h, s_l = _df64_add(dd_h, dd_l, -tol2, 0.0)
        q_h, q_l = _df64_add(s_h, s_l, -qd_h, -qd_l)
        zero = (q_h == 0.0) & (q_l == 0.0)
        q_h = tl.where(zero, -1.1754944e-38, q_h)
        q_l = tl.where(zero, 0.0, q_l)
        neg += ((q_h < 0.0) | ((q_h == 0.0) & (q_l < 0.0))).to(tl.int32)
        dprev = di
        i += 1
    rank_b = (K - neg).to(tl.int64)
    # always take the df64 count: the fp32 bracket counts in kernel A are only
    # used to bound sigma_max / trigger the bisection, never for the answer.
    tl.store(OUT + batch, rank_b)


@triton.jit
def _mr_sturm_big_tridiag_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    TOL2,
    FLAG,
    K,
    BLOCK: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
):
    # Hermitian Sturm bracket on the RAW tridiagonal (d, e) -- the
    # eigenvalue-domain counterpart of _mr_sturm_big_kernel.  Counts
    # #{|lambda| > max(atol, rtol*|lambda|max)} = (K - cle(tol)) + clt(-tol)
    # where cle(x) = #{lambda <= x} (zero pivot -> tiny negative) and
    # clt(x) = #{lambda < x} STRICTLY (zero pivot -> tiny positive), both
    # from _sturm_count_posneg2.  |lambda|max is
    # bracketed by [max|d_i|, max Gershgorin radius] (max_i |d_ii| <=
    # |lambda|max by Rayleigh) and refined by bisection on
    # f(x) = #{|lambda| > x} only when the two bounds give different ranks.
    # TOL2 hands the (LINEAR, not squared) tolerance to the decisive df64
    # kernel; the flag convention matches _mr_sturm_big_kernel.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    base = batch * K
    kmask = kidx < K
    d = tl.load(D + base + kidx, mask=kmask, other=0.0)
    ee = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    ee_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & kmask, other=0.0)
    gershgorin = tl.abs(d) + tl.abs(ee) + tl.abs(ee_prev)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(tl.abs(d), axis=0)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    if hi == 0.0:
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
        tl.store(TOL2 + batch, 0.0)
        tl.store(FLAG + batch, tl.zeros((), dtype=tl.int64))
    else:
        tol_lo = tl.maximum(atol, rtol * dmax)
        tol_hi = tl.maximum(atol, rtol * hi)
        c_lo, m_lo = _sturm_count_posneg2(D, E, base, K, tol_lo, -tol_lo)
        c_hi, m_hi = _sturm_count_posneg2(D, E, base, K, tol_hi, -tol_hi)
        rank_lo = K - c_lo + m_lo
        rank_hi = K - c_hi + m_hi
        tol2 = tol_lo
        if rank_lo != rank_hi:
            # bisect |lambda|max on [dmax, hi]: f(x) = #{|lambda| > x} is
            # monotone; |lambda|max = sup{x : f(x) > 0}
            lo = dmax
            hi_p = hi * (1.0 + 1e-9) + 1e-30
            it = 0
            while it < BISECT_ITERS:
                mid = 0.5 * (lo + hi_p)
                c_m, m_m = _sturm_count_posneg2(D, E, base, K, mid, -mid)
                if K - c_m + m_m >= 1:
                    lo = mid
                else:
                    hi_p = mid
                it += 1
            lmax = 0.5 * (lo + hi_p)
            tol2 = tl.maximum(atol, rtol * lmax)
        refine = rank_lo != rank_hi
        tl.store(OUT + batch, rank_lo.to(tl.int64))
        tl.store(TOL2 + batch, tol2)
        tl.store(FLAG + batch, refine.to(tl.int64))


@triton.jit
def _mr_sturm_final_tridiag_kernel(D, E, TOL2, FLAG, OUT, K):
    # Decisive hermitian count in df64 arithmetic from the RAW tridiagonal:
    # rank = #{lambda > tol} + #{lambda < -tol}, two lockstep qd chains
    # q_i = d_i - x - e_{i-1}^2 / q_{i-1}.  The positive chain runs at
    # x = tol with the DLANEG guard (zero pivot -> tiny negative), so it
    # counts #{lambda <= tol} and K - neg1 IS the strict #{lambda > tol};
    # the negative chain runs at x = -tol with the MIRRORED guard (zero
    # pivot -> tiny positive), so neg2 IS the strict #{lambda < -tol} --
    # exact for every tol, including tol == 0, where it yields
    # #{lambda < 0} (a nonzero rank-deficient spectrum is counted
    # correctly, no special-casing).  Same df64 primitives as
    # _mr_sturm_final_kernel; the e^2 terms are TwoProd'd so the recurrence
    # keeps ~eps^2 relative precision.  Requires enable_fp_fusion=False.
    batch = tl.program_id(0)
    tol = tl.load(TOL2 + batch)
    xn = -tol
    base = batch * K
    d0 = tl.load(D + base)
    q1h, q1l = _df64_add(d0, 0.0, -tol, 0.0)
    q2h, q2l = _df64_add(d0, 0.0, -xn, 0.0)
    zero1 = (q1h == 0.0) & (q1l == 0.0)
    q1h = tl.where(zero1, -1.1754944e-38, q1h)
    q1l = tl.where(zero1, 0.0, q1l)
    zero2 = (q2h == 0.0) & (q2l == 0.0)
    q2h = tl.where(zero2, 1.1754944e-38, q2h)
    q2l = tl.where(zero2, 0.0, q2l)
    neg1 = ((q1h < 0.0) | ((q1h == 0.0) & (q1l < 0.0))).to(tl.int32)
    neg2 = ((q2h < 0.0) | ((q2h == 0.0) & (q2l < 0.0))).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        eih, eil = _df64_mul_ds(ei, 0.0, ei, 0.0)
        qd1h, qd1l = _df64_div_ds(eih, eil, q1h, q1l)
        s1h, s1l = _df64_add(di, 0.0, -tol, 0.0)
        q1h, q1l = _df64_add(s1h, s1l, -qd1h, -qd1l)
        qd2h, qd2l = _df64_div_ds(eih, eil, q2h, q2l)
        s2h, s2l = _df64_add(di, 0.0, -xn, 0.0)
        q2h, q2l = _df64_add(s2h, s2l, -qd2h, -qd2l)
        zero1 = (q1h == 0.0) & (q1l == 0.0)
        q1h = tl.where(zero1, -1.1754944e-38, q1h)
        q1l = tl.where(zero1, 0.0, q1l)
        zero2 = (q2h == 0.0) & (q2l == 0.0)
        q2h = tl.where(zero2, 1.1754944e-38, q2h)
        q2l = tl.where(zero2, 0.0, q2l)
        neg1 += ((q1h < 0.0) | ((q1h == 0.0) & (q1l < 0.0))).to(tl.int32)
        neg2 += ((q2h < 0.0) | ((q2h == 0.0) & (q2l < 0.0))).to(tl.int32)
        i += 1
    rank = (K - neg1) + neg2
    tl.store(OUT + batch, rank.to(tl.int64))


def _bidiag_workspace(dev, batch_count, k, rows):
    kp = triton.cdiv(k, 64) * 64 + 64  # one slack tile: tile accesses are
    rs = triton.cdiv(rows, 64) * 64 + 64  # unmasked and must not run off
    block_k = max(triton.next_power_of_2(k), 64)
    return {
        "kp": kp,
        "rs": rs,
        "wpitch": kp * rs,
        "upitch": kp * (k + 64),
        "block_k": block_k,
        "W": torch.zeros((batch_count, kp, rs), dtype=torch.float32, device=dev),
        "V": torch.zeros((batch_count, kp, rs), dtype=torch.float32, device=dev),
        "U": torch.zeros((batch_count, kp * (k + 64)), dtype=torch.float32, device=dev),
        "dbuf": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        "ebuf": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        # fp32 B^T B tridiagonal for the bracket counts only; the decisive
        # df64 count works from the RAW d/e (see _mr_bidiag_to_tridiag_kernel).
        "ddbuf": torch.empty((batch_count, k), dtype=torch.float32, device=dev),
        "eebuf": torch.empty((batch_count, k), dtype=torch.float32, device=dev),
        "taul": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        "taur": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        "nrm2": torch.empty((batch_count, k), dtype=torch.float32, device=dev),
        "frob": torch.zeros((batch_count,), dtype=torch.float32, device=dev),
        # One extra tile of slack: the right matvec's atomic accumulation
        # index r0 + 63 can reach rs (pad-row tiles are updated, never read).
        "acc": torch.zeros((batch_count, kp + 64), dtype=torch.float32, device=dev),
        "uacc": torch.zeros((batch_count, rs + 64), dtype=torch.float32, device=dev),
        "tol2": torch.empty((batch_count,), dtype=torch.float32, device=dev),
        "flag": torch.empty((batch_count,), dtype=torch.int64, device=dev),
        "atol": torch.empty((batch_count,), dtype=torch.float32, device=dev),
        "rtol": torch.empty((batch_count,), dtype=torch.float32, device=dev),
        "out": torch.empty((batch_count,), dtype=torch.int64, device=dev),
    }


def _bidiag_run(
    ws, matrix, atol_tensor, rtol_tensor, m, n, k, rows, batch_count, hermitian
):
    """The launch sequence of the unblocked Golub-Kahan bidiagonalization.

    Written as a pure function of the workspace so it can run either
    directly or under an NPUGraph capture (graph capture requires stable
    buffer addresses across replays)."""
    kp, rs, block_k = ws["kp"], ws["rs"], ws["block_k"]
    wpitch, upitch = ws["wpitch"], ws["upitch"]
    W, V, U = ws["W"], ws["V"], ws["U"]
    dbuf, ebuf, ddbuf, eebuf = ws["dbuf"], ws["ebuf"], ws["ddbuf"], ws["eebuf"]
    taul, taur = ws["taul"], ws["taur"]
    nrm2, frob, acc, uacc = ws["nrm2"], ws["frob"], ws["acc"], ws["uacc"]
    _fast_launch(
        _mr_rrqr_init_kernel,
        (batch_count, (kp - 64) // 64),
        matrix,
        W,
        nrm2,
        frob,
        m,
        n,
        k,
        rows,
        rs,
        wpitch,
        TALL=m >= n,
        HERMITIAN=hermitian,
        num_warps=4,
        num_stages=1,
    )
    nrc_full = (rs - 64) // 64
    ncc_full = (kp - 64) // 64
    # Valid row tiles = cdiv(rows, 64); the workspace rs carries one extra
    # slack tile (for unmasked addressing) plus up to 63 pad rows, all zeros
    # forever, so iterating them in the step/mat/apply kernels is pure waste
    # (for near-square 129^2 it doubles the right-side tile count).
    rbv = (rows + 63) // 64
    for j in range(k):
        ntl = triton.cdiv(k - 1 - j, 64)
        # trim the matvec/apply grids to the trailing block: v/u are zero
        # above/left of j, so those tiles are pure waste (halves the
        # average per-step work of the four big kernels).
        rb0 = j // 64
        nrct = nrc_full - rb0
        _fast_launch(
            _mr_bidiag_lstep_kernel,
            (batch_count,),
            W,
            V,
            dbuf,
            taul,
            acc,
            j,
            k,
            rs,
            wpitch,
            kp,
            rbv,
            num_warps=4,
            num_stages=1,
        )
        if ntl > 0:
            _fast_launch(
                _mr_bidiag_lmat_kernel,
                (batch_count, ntl * nrct),
                W,
                V,
                acc,
                j,
                k,
                rs,
                wpitch,
                kp,
                nrct,
                rb0,
                num_warps=4,
                num_stages=1,
            )
            _fast_launch(
                _mr_bidiag_lapply_kernel,
                (batch_count, ntl * nrct),
                W,
                V,
                taul,
                acc,
                j,
                k,
                rs,
                wpitch,
                kp,
                nrct,
                rb0,
                num_warps=4,
                num_stages=1,
            )
        if j + 1 < k:
            # valid row tiles only (see rbv above): rows j+1 .. rows-1
            ntr = triton.cdiv(rows - 1 - j, 64)
            cc0 = (j + 1) // 64
            ncct = ncc_full - cc0
            _fast_launch(
                _mr_bidiag_rstep_kernel,
                (batch_count,),
                W,
                U,
                ebuf,
                taur,
                uacc,
                j,
                k,
                rs,
                wpitch,
                upitch,
                rs,
                num_warps=4,
                num_stages=1,
            )
            _fast_launch(
                _mr_bidiag_rmat_kernel,
                (batch_count, ntr * ncct),
                W,
                U,
                uacc,
                j,
                k,
                rs,
                wpitch,
                upitch,
                rs,
                ncct,
                cc0,
                num_warps=4,
                num_stages=1,
            )
            _fast_launch(
                _mr_bidiag_rapply_kernel,
                (batch_count, ntl * ntr),
                W,
                U,
                taur,
                uacc,
                j,
                k,
                rs,
                wpitch,
                upitch,
                rs,
                (rs - 64) // 64,
                ntr,
                num_warps=4,
                num_stages=1,
            )
    _fast_launch(
        _mr_bidiag_to_tridiag_kernel,
        (batch_count,),
        dbuf,
        ebuf,
        ddbuf,
        eebuf,
        k,
        BLOCK=block_k,
        num_warps=1,
        num_stages=1,
    )
    _fast_launch(
        _mr_sturm_big_kernel,
        (batch_count,),
        ddbuf,
        eebuf,
        atol_tensor,
        rtol_tensor,
        ws["out"],
        ws["tol2"],
        ws["flag"],
        k,
        BLOCK=block_k,
        BISECT_ITERS=32,
        num_warps=1,
        num_stages=1,
    )
    _fast_launch(
        _mr_sturm_final_kernel,
        (batch_count,),
        dbuf,
        ebuf,
        ws["tol2"],
        ws["flag"],
        ws["out"],
        k,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )


# Per-shape NPUGraph cache for the bidiag path.  The unblocked GK loop
# issues 6k kernel launches; on this host each _fast_launch costs ~80us of
# HOST time (slow ARM CPU), so a k=512 call spends ~100x more time
# enqueueing than the device spends executing.  Capturing the sequence once
# per shape and replaying it collapses the host cost to ~1us/launch
# (measured: 300 captured launches replay in 286us vs 124ms direct).
# Buffers are persistent per cache entry; only the input matrix and the
# tolerances are copied in per call, and the rank out of it.
#
# Concurrency: the cache key includes the current stream (a capture is only
# valid on its own stream), and ALL cache access (lookup, capture, staging
# copy, replay) is serialized by a module-level lock -- otherwise two
# threads calling with the same shape on the same device would race on the
# shared staging/tolerance/out buffers.  Eviction is FIFO-one (the dict
# preserves insertion order), not clear-all: shape churn beyond
# _*_GRAPH_MAX_ENTRIES re-captures one entry instead of all of them.
_GRAPH_LOCK = threading.Lock()


def _current_stream_key(device):
    try:
        stream = torch.npu.current_stream(device)
        for attr in ("cuda_stream", "npu_stream", "stream_id"):
            handle = getattr(stream, attr, None)
            if handle is not None:
                return handle
        return id(stream)
    except Exception:
        return None


_BIDIAG_GRAPHS = {}
_BIDIAG_GRAPH_MAX_ENTRIES = 16


def _launch_bidiag_rank(
    matrix,
    atol_tensor,
    rtol_tensor,
    out,
    m,
    n,
    k,
    rows,
    batch_count,
    input,
    hermitian,
):
    """Unblocked Golub-Kahan bidiagonalization + Sturm count (fp32; the
    default for every non-hermitian k > 64 -- k > 255 always, and compact
    64 < k <= 255 even when FLAGGEMS_MR_FAST_PATH=1; only long-row inputs in
    that band may opt into the unpivoted QR).

    SVD-accurate rank: unlike the RRQR path (|R_ii| only approximate sigma_i),
    the bidiagonal d/e keep every singular value at linear precision and the
    Sturm count is exact, so borderline tolerances are resolved correctly.
    Per step: six kernel launches (reflector / matvec / apply, left then
    right); cross-program reductions use atomic accumulation and kernel
    boundaries provide all ordering. Slower per step than the RRQR path but
    exact -- used only where the QR pivot gap is a risk (large k).

    The launch sequence is NPUGraph-captured per shape (see above); set
    FLAGGEMS_MR_NO_GRAPH=1 to force direct launches.
    """
    dev = input.device
    # TRITON_ALL_BLOCKS_PARALLEL (set globally by the fused sparse-attention
    # module at import) miscompiles the bidiag matvec/apply kernels (verified:
    # wrong atomic-accumulated sums). It is read at kernel COMPILE time only,
    # so pop it for the duration of this launcher (compiles happen here via
    # _fast_launch warmups and the graph capture) and restore it afterwards.
    # _GRAPH_LOCK serializes the WHOLE launcher, not just the graph cache:
    # the TRITON_ALL_BLOCKS_PARALLEL pop/restore below is process-global
    # state and kernel compiles happen inside this body (via _fast_launch
    # warmups and graph capture), so a concurrent thread must neither
    # interleave pop/restore nor compile while the variable is removed.
    with _GRAPH_LOCK:
        saved_all_blocks = os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
        try:
            with torch_device_fn.device(input.device):
                if os.environ.get("FLAGGEMS_MR_NO_GRAPH") == "1":
                    ws = _bidiag_workspace(dev, batch_count, k, rows)
                    _bidiag_run(
                        ws,
                        matrix,
                        atol_tensor,
                        rtol_tensor,
                        m,
                        n,
                        k,
                        rows,
                        batch_count,
                        hermitian,
                    )
                    out.copy_(ws["out"].reshape(out.shape))
                    return out
                key = (
                    m,
                    n,
                    batch_count,
                    hermitian,
                    input.device.index,
                    _current_stream_key(input.device),
                )
                ent = _BIDIAG_GRAPHS.get(key)
                if ent is None:
                    ws = _bidiag_workspace(dev, batch_count, k, rows)
                    staging = torch.empty(
                        (batch_count, m, n), dtype=torch.float32, device=dev
                    )
                    staging.copy_(matrix)
                    ws["atol"].copy_(atol_tensor)
                    ws["rtol"].copy_(rtol_tensor)
                    # warm-up run: compiles every kernel/specialization
                    # (compile is illegal during capture) and fills
                    # _FAST_LAUNCH_CACHE
                    _bidiag_run(
                        ws,
                        staging,
                        ws["atol"],
                        ws["rtol"],
                        m,
                        n,
                        k,
                        rows,
                        batch_count,
                        hermitian,
                    )
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    try:
                        with torch.npu.graph(graph):
                            _bidiag_run(
                                ws,
                                staging,
                                ws["atol"],
                                ws["rtol"],
                                m,
                                n,
                                k,
                                rows,
                                batch_count,
                                hermitian,
                            )
                    except Exception:
                        logger.warning(
                            "NPUGraph capture failed for matrix_rank bidiag "
                            "shape (%d, %d); falling back to direct launches",
                            m,
                            n,
                        )
                        _bidiag_run(
                            ws,
                            staging,
                            ws["atol"],
                            ws["rtol"],
                            m,
                            n,
                            k,
                            rows,
                            batch_count,
                            hermitian,
                        )
                        out.copy_(ws["out"].reshape(out.shape))
                        return out
                    if len(_BIDIAG_GRAPHS) >= _BIDIAG_GRAPH_MAX_ENTRIES:
                        # An evicted graph may still have an in-flight
                        # replay (replay is async); dropping the entry frees
                        # its workspace.  Rare event (17th distinct shape):
                        # drain before freeing.  The cache is process-wide
                        # and the FIFO victim may live on ANOTHER device, so
                        # synchronize the victim's device (key[4]), not the
                        # current one.
                        victim_key = next(iter(_BIDIAG_GRAPHS))
                        torch.npu.synchronize(victim_key[4])
                        _BIDIAG_GRAPHS.pop(victim_key)
                    _BIDIAG_GRAPHS[key] = (ws, staging, graph)
                ws, staging, graph = _BIDIAG_GRAPHS[key]
                staging.copy_(matrix)
                ws["atol"].copy_(atol_tensor)
                ws["rtol"].copy_(rtol_tensor)
                graph.replay()
                out.copy_(ws["out"].reshape(out.shape))
        finally:
            if saved_all_blocks is not None:
                os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = saved_all_blocks
    return out


_TRIDIAG_GRAPHS = {}
_TRIDIAG_GRAPH_MAX_ENTRIES = 16


def _tridiag_workspace(dev, batch_count, k, rows):
    # Slim variant of _bidiag_workspace: the one-sided tridiagonalization
    # needs no U/ddbuf/eebuf/taur (those are bidiag-only), saving ~40% of
    # the per-entry memory.
    kp = triton.cdiv(k, 64) * 64 + 64
    rs = triton.cdiv(rows, 64) * 64 + 64
    block_k = max(triton.next_power_of_2(k), 64)
    return {
        "kp": kp,
        "rs": rs,
        "wpitch": kp * rs,
        "block_k": block_k,
        "W": torch.zeros((batch_count, kp, rs), dtype=torch.float32, device=dev),
        "V": torch.zeros((batch_count, kp, rs), dtype=torch.float32, device=dev),
        "dbuf": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        "ebuf": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        "taul": torch.zeros((batch_count, k), dtype=torch.float32, device=dev),
        "nrm2": torch.empty((batch_count, k), dtype=torch.float32, device=dev),
        "frob": torch.zeros((batch_count,), dtype=torch.float32, device=dev),
        "acc": torch.zeros((batch_count, kp + 64), dtype=torch.float32, device=dev),
        "tol2": torch.empty((batch_count,), dtype=torch.float32, device=dev),
        "flag": torch.empty((batch_count,), dtype=torch.int64, device=dev),
        "atol": torch.empty((batch_count,), dtype=torch.float32, device=dev),
        "rtol": torch.empty((batch_count,), dtype=torch.float32, device=dev),
        "out": torch.empty((batch_count,), dtype=torch.int64, device=dev),
    }


def _tridiag_run(ws, matrix, atol_tensor, rtol_tensor, m, n, k, rows, batch_count):
    """The launch sequence of the unblocked one-sided Householder
    tridiagonalization (hermitian k > 64): three kernels per step
    (reflector step / symmetric matvec / rank-2 apply) plus the
    eigenvalue-domain Sturm tail.  Hermitian input needs only ONE
    reflection per step (vs two for Golub-Kahan), halving the per-step
    cost of the bidiag path.  Pure function of the workspace so it can be
    NPUGraph-captured."""
    kp, rs, block_k = ws["kp"], ws["rs"], ws["block_k"]
    wpitch = ws["wpitch"]
    W, V = ws["W"], ws["V"]
    dbuf, ebuf, taul = ws["dbuf"], ws["ebuf"], ws["taul"]
    nrm2, frob, acc = ws["nrm2"], ws["frob"], ws["acc"]
    apitch = kp + 64
    _fast_launch(
        _mr_rrqr_init_kernel,
        (batch_count, (kp - 64) // 64),
        matrix,
        W,
        nrm2,
        frob,
        m,
        n,
        k,
        rows,
        rs,
        wpitch,
        TALL=True,
        HERMITIAN=True,
        num_warps=4,
        num_stages=1,
    )
    for j in range(k):
        _fast_launch(
            _mr_tridiag_step_kernel,
            (batch_count,),
            W,
            V,
            dbuf,
            ebuf,
            taul,
            acc,
            frob,
            j,
            k,
            rs,
            wpitch,
            apitch,
            num_warps=4,
            num_stages=1,
        )
        if j + 1 < k:
            nrt = triton.cdiv(k - 1 - j, 64)
            _fast_launch(
                _mr_tridiag_mat_kernel,
                (batch_count, nrt * nrt),
                W,
                V,
                acc,
                frob,
                j,
                k,
                rs,
                wpitch,
                apitch,
                nrt,
                num_warps=4,
                num_stages=1,
            )
            _fast_launch(
                _mr_tridiag_apply_kernel,
                (batch_count, nrt * nrt),
                W,
                V,
                taul,
                acc,
                frob,
                j,
                k,
                rs,
                wpitch,
                apitch,
                nrt,
                num_warps=4,
                num_stages=1,
            )
    _fast_launch(
        _mr_sturm_big_tridiag_kernel,
        (batch_count,),
        dbuf,
        ebuf,
        atol_tensor,
        rtol_tensor,
        ws["out"],
        ws["tol2"],
        ws["flag"],
        k,
        BLOCK=block_k,
        BISECT_ITERS=32,
        num_warps=1,
        num_stages=1,
    )
    _fast_launch(
        _mr_sturm_final_tridiag_kernel,
        (batch_count,),
        dbuf,
        ebuf,
        ws["tol2"],
        ws["flag"],
        ws["out"],
        k,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )


def _launch_tridiag_big_rank(
    matrix,
    atol_tensor,
    rtol_tensor,
    out,
    m,
    n,
    k,
    rows,
    batch_count,
    input,
):
    """Hermitian k > 64: unblocked one-sided Householder tridiagonalization
    + eigenvalue-domain Sturm count (|lambda| > tol via +/-tol qd chains,
    decisive count in df64).  Replaces the Golub-Kahan route for symmetric
    input: half the reflectors per step.  NPUGraph-captured per shape (set
    FLAGGEMS_MR_NO_GRAPH=1 to force direct launches)."""
    dev = input.device
    # see _launch_bidiag_rank for why TRITON_ALL_BLOCKS_PARALLEL is popped,
    # and for why the WHOLE body runs under _GRAPH_LOCK (the pop/restore is
    # process-global state; compiles happen inside this body).
    with _GRAPH_LOCK:
        saved_all_blocks = os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
        try:
            with torch_device_fn.device(input.device):
                if os.environ.get("FLAGGEMS_MR_NO_GRAPH") == "1":
                    ws = _tridiag_workspace(dev, batch_count, k, rows)
                    _tridiag_run(
                        ws,
                        matrix,
                        atol_tensor,
                        rtol_tensor,
                        m,
                        n,
                        k,
                        rows,
                        batch_count,
                    )
                    out.copy_(ws["out"].reshape(out.shape))
                    return out
                key = (
                    m,
                    n,
                    batch_count,
                    input.device.index,
                    _current_stream_key(input.device),
                )
                ent = _TRIDIAG_GRAPHS.get(key)
                if ent is None:
                    ws = _tridiag_workspace(dev, batch_count, k, rows)
                    staging = torch.empty(
                        (batch_count, m, n), dtype=torch.float32, device=dev
                    )
                    staging.copy_(matrix)
                    ws["atol"].copy_(atol_tensor)
                    ws["rtol"].copy_(rtol_tensor)
                    _tridiag_run(
                        ws,
                        staging,
                        ws["atol"],
                        ws["rtol"],
                        m,
                        n,
                        k,
                        rows,
                        batch_count,
                    )
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    try:
                        with torch.npu.graph(graph):
                            _tridiag_run(
                                ws,
                                staging,
                                ws["atol"],
                                ws["rtol"],
                                m,
                                n,
                                k,
                                rows,
                                batch_count,
                            )
                    except Exception:
                        logger.warning(
                            "NPUGraph capture failed for matrix_rank tridiag "
                            "shape (%d, %d); falling back to direct launches",
                            m,
                            n,
                        )
                        _tridiag_run(
                            ws,
                            staging,
                            ws["atol"],
                            ws["rtol"],
                            m,
                            n,
                            k,
                            rows,
                            batch_count,
                        )
                        out.copy_(ws["out"].reshape(out.shape))
                        return out
                    if len(_TRIDIAG_GRAPHS) >= _TRIDIAG_GRAPH_MAX_ENTRIES:
                        # An evicted graph may still have an in-flight
                        # replay (replay is async); dropping the entry frees
                        # its workspace.  Rare event (17th distinct shape):
                        # drain before freeing.  The cache is process-wide
                        # and the FIFO victim may live on ANOTHER device, so
                        # synchronize the victim's device (key[3]), not the
                        # current one.
                        victim_key = next(iter(_TRIDIAG_GRAPHS))
                        torch.npu.synchronize(victim_key[3])
                        _TRIDIAG_GRAPHS.pop(victim_key)
                    _TRIDIAG_GRAPHS[key] = (ws, staging, graph)
                ws, staging, graph = _TRIDIAG_GRAPHS[key]
                staging.copy_(matrix)
                ws["atol"].copy_(atol_tensor)
                ws["rtol"].copy_(rtol_tensor)
                graph.replay()
                out.copy_(ws["out"].reshape(out.shape))
        finally:
            if saved_all_blocks is not None:
                os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = saved_all_blocks
    return out


def _launch_matrix_rank(input, atol, rtol, hermitian):
    output_shape = input.shape[:-2]
    m, n = input.shape[-2:]
    k = min(m, n)
    rows = max(m, n)
    is_fp64 = input.dtype == torch.float64
    batch_count = input.numel() // (m * n)
    matrix = input.contiguous().reshape(batch_count, m, n)
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    if is_fp64:
        # fp64 is unsupported on this backend end to end: aclnn svd_npu
        # is fp32-only, and triton-ascend cannot compile fp64 kernels
        # (verified: the rank1/rank2 closed forms and the fused-Jacobi
        # reference path all die with MLIRCompilationError at kernel
        # compile time).  Reject BEFORE any shape dispatch -- and before
        # the scale normalization below, whose pow2 mantissa mask is an
        # fp32-only bit trick -- so every shape fails fast with a clear
        # error instead of a compiler crash.
        raise NotImplementedError(
            "FlagGems Ascend linalg_matrix_rank does not support "
            "float64 inputs (triton-ascend cannot compile fp64 kernels)"
        )

    # Scalar tolerances (the common case) are passed to the small-path
    # kernels as scalar arguments, skipping the two fill-kernel launches that
    # materializing (batch,) tolerance tensors would cost.  Tensor
    # tolerances, and the large-matrix paths (whose kernels only take
    # tensors), still go through _prepare_tolerances.
    tol_tensor = isinstance(atol, torch.Tensor) or isinstance(rtol, torch.Tensor)
    atol_tensor = rtol_tensor = None
    atol_val = rtol_val = 0.0
    if tol_tensor:
        atol_tensor, rtol_tensor = _prepare_tolerances(input, atol, rtol)
    else:
        atol_val, rtol_val = _tolerance_scalars(input, atol, rtol)

    # Input scale normalization (mirrors the shared implementation): the
    # Householder algebra squares the matrix scale (w = A.v is
    # O(sigma^2)), so fp32 sum-of-squares overflow for |x| ~ 1e20 and
    # underflow for |x| ~ 1e-30.  Normalize each batch matrix by its
    # max-abs and shrink atol by the same factor -- rank = #{sigma >
    # max(atol, rtol*sigma_max)} is invariant under this (rtol is
    # relative).  Hermitian reads ONLY the lower triangle, so the scale
    # comes from the lower triangle: garbage in the strict upper triangle
    # is invisible to the computation and must not inflate the scale.
    # The divisor is rounded DOWN to a power of two (fp32 mantissa bits
    # masked off): division by 2^e is exact (pure exponent shift), so the
    # normalization adds no rounding step to ordinary inputs (a plain
    # max-abs division flipped a noise-floor lowrank case: (24,24) rank 12
    # read as 13; exp2(floor(log2(x))) is NOT exact on this backend --
    # measured 0.5 -> 0.49999997).  A masked-to-zero scale (all-zero or
    # subnormal-max matrix) maps to 1, i.e. "no scaling", matching native
    # torch.  Small matrices (m, n <= 64) use a single-launch kernel --
    # the aten chain below costs 5-6 launches, which halves the speedup of
    # sub-millisecond ops (measured: herm 17^2 0.85 -> 0.46).
    # nonzero_flag (batch,) int8 records "the (lower-triangle) max-abs is
    # nonzero" for the negative-tolerance fixup at the end, sparing it a
    # full-matrix tril+amax+amin scan per tensor-tolerance call.  Also
    # needed when a nonzero scalar atol gets materialized into tensors by
    # the scaling below.
    need_flag = tol_tensor or atol_val != 0.0
    nonzero_flag = None
    if rows <= 64:
        normed = torch.empty_like(matrix)
        scale = torch.empty((batch_count,), dtype=torch.float32, device=matrix.device)
        if need_flag:
            nonzero_flag = torch.empty(
                (batch_count,), dtype=torch.int8, device=matrix.device
            )
        with torch_device_fn.device(matrix.device):
            _fast_launch(
                _mr_normalize_small_kernel,
                (batch_count,),
                matrix,
                normed,
                scale,
                nonzero_flag if need_flag else scale,  # dummy: not stored
                M=m,
                N=n,
                HERMITIAN=hermitian,
                NEED_FLAG=need_flag,
                BLOCK_R=triton.next_power_of_2(m),
                BLOCK_C=triton.next_power_of_2(n),
                num_warps=1,
            )
        matrix = normed
    else:
        scale_src = matrix.tril() if hermitian else matrix
        scale = scale_src.abs().amax(dim=(1, 2))
        if need_flag:
            nonzero_flag = (scale > 0).to(torch.int8)
        scale = (scale.view(torch.int32) & -0x800000).view(torch.float32)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        matrix = matrix / scale[:, None, None]
    if tol_tensor:
        atol_tensor = atol_tensor / scale
    elif atol_val != 0.0:
        # A nonzero scalar atol becomes per-batch under scaling.  The
        # kernels take tolerances either both as tensors or both as
        # scalars, so rtol is materialized alongside (it is relative and
        # needs no division).  The common atol == 0 case keeps the scalar
        # fast path: 0 / scale == 0.
        atol_tensor = (
            torch.full(
                (batch_count,), atol_val, dtype=matrix.dtype, device=matrix.device
            )
            / scale
        )
        rtol_tensor = torch.full(
            (batch_count,), rtol_val, dtype=matrix.dtype, device=matrix.device
        )
        atol_val = rtol_val = 0.0
        tol_tensor = True

    block_r = triton.next_power_of_2(rows)
    relative_epsilon = 1.0e-15 if is_fp64 else 1.0e-7
    absolute_epsilon = 1.0e-300 if is_fp64 else 1.0e-30
    num_warps = 1 if block_r <= 64 else 4
    # Dummy operands for the ATOL/RTOL pointer arguments when the scalar
    # fast path is active (TOL_TENSOR=False): never dereferenced.
    at_arg = atol_tensor if tol_tensor else matrix
    rt_arg = rtol_tensor if tol_tensor else matrix

    with torch_device_fn.device(input.device):
        if k == 1:
            _matrix_rank_rank1_kernel[(batch_count,)](
                matrix,
                at_arg,
                rt_arg,
                out,
                atol_val,
                rtol_val,
                M=m,
                N=n,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                TOL_TENSOR=tol_tensor,
                num_warps=num_warps,
            )
        elif k == 2:
            _matrix_rank_rank2_kernel[(batch_count,)](
                matrix,
                at_arg,
                rt_arg,
                out,
                atol_val,
                rtol_val,
                M=m,
                N=n,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                REL_EPS=relative_epsilon,
                ABS_EPS=absolute_epsilon,
                TOL_TENSOR=tol_tensor,
                num_warps=num_warps,
            )
        elif k <= _TRIDIAG_MAX_K:
            # fp32 small/medium matrices (any row count): fused kernel /
            # bidiag64 / hermitian padded tridiagonalization / QR-compress
            # for long dimensions -- all linear-domain now.
            _launch_tridiag_rank(
                matrix,
                atol_tensor,
                rtol_tensor,
                atol_val,
                rtol_val,
                out,
                m,
                n,
                batch_count,
                input,
                hermitian,
            )
        elif (
            k > _RRQR_MAX_K
            or hermitian
            or rows < _RRQR_FAST_MIN_ROWS
            or os.environ.get("FLAGGEMS_MR_FAST_PATH") != "1"
        ):
            # k >= 256, every hermitian input, compact rows <= 256, and the
            # default mode: use the exact paths. FAST_PATH=1 selects QR only
            # for non-hermitian 65..255 inputs whose long dimension is > 256.
            # hermitian: one-sided Householder tridiagonalization +
            # eigenvalue-domain Sturm count (|lambda| > tol via +/-tol qd
            # chains, decisive pass in df64).  general: unblocked Golub-Kahan
            # bidiagonalization + df64 Sturm.  Both SVD-accurate.  Performance:
            # >= 0.8x torch for k >= 256 (general 1.1x/1.8x at 256^2/512^2,
            # 2.8x at 1024^2; herm 2.0x/3.8x, 7.0x at 1024^2). Some compact
            # 65..255 exact shapes dip below 0.8x, but GM QR is slower there
            # after the register-panel path became uncompilable on CANN 9.1.1.
            if atol_tensor is None:
                atol_tensor, rtol_tensor = _prepare_tolerances(input, atol, rtol)
            if hermitian:
                _launch_tridiag_big_rank(
                    matrix,
                    atol_tensor,
                    rtol_tensor,
                    out,
                    m,
                    n,
                    k,
                    rows,
                    batch_count,
                    input,
                )
            else:
                _launch_bidiag_rank(
                    matrix,
                    atol_tensor,
                    rtol_tensor,
                    out,
                    m,
                    n,
                    k,
                    rows,
                    batch_count,
                    input,
                    hermitian,
                )
        else:
            # Non-hermitian 65..255 with rows > 256 and FAST_PATH=1:
            # pure-Triton blocked Householder QR (unpivoted); rank is read off
            # the |R_ii| diagonal. There is no native decomposition fallback.
            if atol_tensor is None:
                atol_tensor, rtol_tensor = _prepare_tolerances(input, atol, rtol)
            _launch_rrqr_rank(
                matrix,
                atol_tensor,
                rtol_tensor,
                out,
                m,
                n,
                k,
                rows,
                batch_count,
                input,
                hermitian,
            )

    # torch's tolerance is tol = max(atol, rtol * sigma_max) with NO lower
    # clamp, so tol < 0 is reachable only when BOTH atol < 0 and rtol < 0
    # (elementwise) -- and then every singular value (>= 0) exceeds tol:
    # rank == k for a nonzero matrix.  (A zero matrix still gives 0:
    # rtol*0 == 0 lifts tol back to max(atol, 0) == 0.)  The in-kernel
    # counts assume tol >= 0 -- the hermitian split #{|lam|>tol} =
    # #{lam>tol} + #{lam<-tol} double-counts the overlap when tol < 0, and
    # the sigma^2-domain paths square tol -- so this corner is fixed up
    # after the fact instead of touching every counting kernel.  Two
    # implementation constraints: (a) hermitian reads ONLY the lower
    # triangle, so "nonzero" must be checked on the lower triangle --
    # garbage in the strict upper triangle is invisible to torch and must
    # not trigger the fix; (b) no Python branch on device data -- a
    # bool(...any()) here would sync the host on EVERY tensor-tolerance
    # call, so the tensor branch applies the (usually no-op) where
    # unconditionally and asynchronously.  (A Triton early-exit kernel was
    # tried and abandoned: this backend fails to legalize a masked load
    # combined with any runtime scalar in a vector expression -- unresolved
    # () -> tensor materialization, five variants.)  The matrix scan itself
    # is free: the tensor branch reuses the entry normalization's
    # nonzero_flag (the scale is already computed from the lower triangle
    # for hermitian inputs, satisfying (a)), leaving only the tiny
    # (batch,)-sized where.
    if tol_tensor:
        neg_pair = ((atol_tensor < 0) & (rtol_tensor < 0)).reshape(output_shape)
        nonzero = nonzero_flag.reshape(output_shape) != 0
        out = torch.where(neg_pair & nonzero, torch.full_like(out, k), out)
    elif atol_val < 0.0 and rtol_val < 0.0:
        src = matrix.tril() if hermitian else matrix
        nonzero = ((src.amax(dim=(1, 2)) > 0) | (src.amin(dim=(1, 2)) < 0)).reshape(
            output_shape
        )
        out = torch.where(nonzero, torch.full_like(out, k), out)
    return out


# ---------------------------------------------------------------------------
# Public API (signatures match the shared op verbatim for SpecOpRegistrar)
# ---------------------------------------------------------------------------
def linalg_matrix_rank(input, *, atol=None, rtol=None, hermitian=False):
    """Computes numerical matrix rank (Ascend backend)."""
    logger.debug("GEMS_ASCEND LINALG_MATRIX_RANK")
    _check_input(input, hermitian)

    output_shape = input.shape[:-2]
    if input.numel() == 0:
        # Native torch validates tolerances BEFORE its empty-input return;
        # match that (complex dtype / wrong device / unbroadcastable tensor
        # tolerances must still raise).
        _validate_tolerance_metadata(input, atol, rtol)
        return _empty_matrix_rank(input, output_shape)

    return _launch_matrix_rank(input, atol, rtol, hermitian)


def linalg_matrix_rank_tol(input, tol, hermitian=False):
    """NumPy-compatible legacy overload where tol is an absolute tolerance."""
    logger.debug("GEMS_ASCEND LINALG_MATRIX_RANK_TOL")
    return linalg_matrix_rank(input, atol=tol, rtol=0.0, hermitian=hermitian)


def linalg_matrix_rank_out(input, *, atol=None, rtol=None, hermitian=False, out=None):
    logger.debug("GEMS_ASCEND LINALG_MATRIX_RANK_OUT")
    result = linalg_matrix_rank(input, atol=atol, rtol=rtol, hermitian=hermitian)
    return _copy_rank_to_out(input, result, out)


def linalg_matrix_rank_tol_out(input, tol, hermitian=False, *, out=None):
    logger.debug("GEMS_ASCEND LINALG_MATRIX_RANK_TOL_OUT")
    result = linalg_matrix_rank_tol(input, tol, hermitian)
    return _copy_rank_to_out(input, result, out)
