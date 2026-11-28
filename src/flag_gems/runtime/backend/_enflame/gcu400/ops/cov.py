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

# On the enflame backend every fp32 dot precision request (ieee/tf32/tf32x3)
# lowers to the same matmul_fp32_fp32 intrinsic, so requesting tf32 buys
# nothing. The fast paths therefore keep the dot operands in the input dtype
# (exact fp16/bf16 products with an fp32 accumulator, which is accuracy-neutral
# or better than rounding through tf32) and only the short-reduction arm
# retains the 3-pass request for upstream parity.
_DOT_PRECISION_HIGH = tl.constexpr("tf32x3")

# Whole-row mean loads are capped so one program still fits the DSM budget;
# wider observation axes fall back to the chunked loop.
_MEAN_ROW_CAP = 8192
# Single-program "tiny" arm budget (next_pow2 rows x next_pow2 cols elements):
# the in-register fp32 centering intermediates dominate, so fp32 gets a much
# tighter cap than fp16/bf16 (measured: fp32 64x1024 requires 1056768 bytes).
_TINY_ELEMS_FP32 = 16384
_TINY_ELEMS_FP16 = 65536


@triton.jit
def _cov_mean_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    n_rows,
    n_cols,
    w_sum,
    stride_xn,
    stride_xm,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    WHOLE: tl.constexpr,
):
    # Each program reduces one row (variable) to its weighted mean over the
    # observation axis: avg[i] = sum_k w[k] * x[i, k] / w_sum. With no weights
    # this is the plain mean and w_sum == n_cols. WHOLE covers the whole row
    # in one masked load (the original 128-wide chunk loop needed 32 tiny
    # transactions per 4096-column row and dominated small matrices); wider
    # rows than the cap keep the chunked form.
    row = tl.program_id(0)
    if WHOLE:
        offs = tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride_xn + offs * stride_xm, mask=mask, other=0.0)
        x = x.to(tl.float32)
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = x * w
        mean = tl.sum(x) / w_sum
        tl.store(mean_ptr + row, mean)
    else:
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for m_start in range(0, n_cols, BLOCK_M):
            offs = m_start + tl.arange(0, BLOCK_M)
            mask = offs < n_cols
            x = tl.load(
                x_ptr + row * stride_xn + offs * stride_xm, mask=mask, other=0.0
            )
            x = x.to(tl.float32)
            if HAS_W:
                w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                x = x * w
            acc += x
        mean = tl.sum(acc) / w_sum
        tl.store(mean_ptr + row, mean)


@triton.jit
def _cov_weight_kernel(
    wf_ptr,
    wa_ptr,
    w_ptr,
    n_cols,
    HAS_WF: tl.constexpr,
    HAS_WA: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Builds the combined float32 observation weight w = fweights * aweights,
    # keeping whichever is provided. fweights arrives as an integer tensor, so
    # the int->float32 conversion happens here inside the kernel rather than via
    # a host-side ``.to(float32)``. That host cast would dispatch to FlagGems'
    # own _to_copy, which is unstable across the many shapes/dtypes exercised in
    # a full test run; keeping the conversion in-kernel sidesteps it entirely.
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_cols
    acc = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    if HAS_WF:
        wf = tl.load(wf_ptr + offs, mask=mask, other=0).to(tl.float32)
        acc = acc * wf
    if HAS_WA:
        wa = tl.load(wa_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * wa
    tl.store(w_ptr + offs, acc, mask=mask)


@triton.jit
def _cov_tiny_kernel(
    x_ptr,
    c_ptr,
    n_rows,
    n_cols,
    fact,
    w_sum,
    stride_xn,
    stride_xm,
    stride_cn,
    stride_cm,
    NATIVE: tl.constexpr,
    HIGH_PRECISION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Single-program whole-matrix covariance for small inputs: one masked load
    # of the entire (rows x observations) matrix, means and centering stay in
    # registers, one dot, one store. This halves the fixed cost (one launch
    # instead of two) for the shapes where both kernels are launch-bound.
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < n_rows
    mask_k = offs_k < n_cols
    x = tl.load(
        x_ptr + offs_n[:, None] * stride_xn + offs_k[None, :] * stride_xm,
        mask=mask_n[:, None] & mask_k[None, :],
        other=0.0,
    )
    xf = x.to(tl.float32)
    s = tl.sum(xf, axis=1)
    mean = s / w_sum
    xc = xf - mean[:, None]
    xc = tl.where(mask_k[None, :], xc, 0.0)
    if NATIVE:
        acc = tl.dot(xc.to(x.dtype), tl.trans(xc.to(x.dtype)))
    elif HIGH_PRECISION:
        acc = tl.dot(xc, tl.trans(xc), input_precision=_DOT_PRECISION_HIGH)
    else:
        acc = tl.dot(xc, tl.trans(xc), input_precision="tf32")
    c = acc / fact
    tl.store(
        c_ptr + offs_n[:, None] * stride_cn + offs_n[None, :] * stride_cm,
        c,
        mask=mask_n[:, None] & mask_n[None, :],
    )


@triton.jit
def _cov_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    c_ptr,
    n_rows,
    n_cols,
    fact,
    stride_xn,
    stride_xm,
    stride_cn,
    stride_cm,
    HAS_W: tl.constexpr,
    HIGH_PRECISION: tl.constexpr,
    NATIVE: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 2D-tiled covariance:
    #   c[i, j] = sum_k w[k] * (x[i, k] - mean[i]) * (x[j, k] - mean[j]) / fact.
    # Centering (and the optional weighting) is fused into the GEMM to avoid an
    # extra global-memory round trip over a materialized centered matrix. The
    # accumulator stays in float32.
    #
    # Vendor tuning knobs (gcu400):
    #   NATIVE     dot operands keep the input dtype (fp16/bf16 MMA with an
    #              fp32 accumulator); the fp32-cast + tf32 request of the
    #              generic path is kept for the weighted / short-reduction arms.
    #   SYMMETRIC  one program per upper-triangle tile, storing both c and c^T;
    #              halves the dot work and the x tile traffic once the full
    #              grid would need more than one wave on the 48 SPs.
    #   EVEN       n_rows % BLOCK_N == 0 and n_cols % BLOCK_M == 0: drops every
    #              mask (masked loads/`other` predication measurably tax the
    #              GCU load path).
    if SYMMETRIC:
        # Decode the linear tile id into the upper-triangle (pid_i, pid_j).
        pid = tl.program_id(0)
        pid_i = 0
        t = pid
        while t > pid_i:
            t -= pid_i + 1
            pid_i += 1
        pid_j = t
    else:
        pid_i = tl.program_id(0)
        pid_j = tl.program_id(1)

    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)

    if EVEN:
        mi = tl.load(mean_ptr + offs_i)
        mj = tl.load(mean_ptr + offs_j)
    else:
        mask_i = offs_i < n_rows
        mask_j = offs_j < n_rows
        mi = tl.load(mean_ptr + offs_i, mask=mask_i, other=0.0)
        mj = tl.load(mean_ptr + offs_j, mask=mask_j, other=0.0)

    acc = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)

    for m_start in range(0, n_cols, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        if EVEN:
            xi = tl.load(
                x_ptr + offs_i[:, None] * stride_xn + offs_m[None, :] * stride_xm
            )
            xj = tl.load(
                x_ptr + offs_j[:, None] * stride_xn + offs_m[None, :] * stride_xm
            )
            if NATIVE:
                xi_centered = xi - mi[:, None].to(xi.dtype)
                xj_centered = xj - mj[:, None].to(xj.dtype)
                if HAS_W:
                    w = tl.load(w_ptr + offs_m).to(tl.float32)
                    xi_centered = (xi_centered.to(tl.float32) * w[None, :]).to(xi.dtype)
            else:
                xi_centered = xi.to(tl.float32) - mi[:, None]
                xj_centered = xj.to(tl.float32) - mj[:, None]
                if HAS_W:
                    w = tl.load(w_ptr + offs_m).to(tl.float32)
                    xi_centered = xi_centered * w[None, :]
        else:
            mask_m = offs_m < n_cols
            mask_i2 = offs_i < n_rows
            mask_j2 = offs_j < n_rows
            xi = tl.load(
                x_ptr + offs_i[:, None] * stride_xn + offs_m[None, :] * stride_xm,
                mask=mask_i2[:, None] & mask_m[None, :],
                other=0.0,
            ).to(tl.float32)
            xj = tl.load(
                x_ptr + offs_j[:, None] * stride_xn + offs_m[None, :] * stride_xm,
                mask=mask_j2[:, None] & mask_m[None, :],
                other=0.0,
            ).to(tl.float32)
            xi_centered = xi - mi[:, None]
            xj_centered = xj - mj[:, None]
            # The weight is folded into one operand so the dot product yields
            # the weighted cross-products sum_k w[k] * xi_c[k] * xj_c[k].
            if HAS_W:
                w = tl.load(w_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
                xi_centered = xi_centered * w[None, :]
            # Re-mask padding lanes to 0 (not -mean) so the dot product stays
            # exact for observation counts that are not a multiple of BLOCK_M.
            xi_centered = tl.where(mask_m[None, :], xi_centered, 0.0)
            xj_centered = tl.where(mask_m[None, :], xj_centered, 0.0)

        if NATIVE:
            acc += tl.dot(xi_centered, tl.trans(xj_centered))
        elif HIGH_PRECISION:
            acc += tl.dot(
                xi_centered, tl.trans(xj_centered), input_precision=_DOT_PRECISION_HIGH
            )
        else:
            acc += tl.dot(xi_centered, tl.trans(xj_centered), input_precision="tf32")

    c = acc / fact
    if EVEN:
        tl.store(c_ptr + offs_i[:, None] * stride_cn + offs_j[None, :] * stride_cm, c)
        if SYMMETRIC:
            if pid_i != pid_j:
                tl.store(
                    c_ptr + offs_j[:, None] * stride_cn + offs_i[None, :] * stride_cm,
                    tl.trans(c),
                )
    else:
        mask_ij = (offs_i[:, None] < n_rows) & (offs_j[None, :] < n_rows)
        tl.store(
            c_ptr + offs_i[:, None] * stride_cn + offs_j[None, :] * stride_cm,
            c,
            mask=mask_ij,
        )
        if SYMMETRIC:
            if pid_i != pid_j:
                tl.store(
                    c_ptr + offs_j[:, None] * stride_cn + offs_i[None, :] * stride_cm,
                    tl.trans(c),
                    mask=tl.trans(mask_ij),
                )


def _cov_pick_blocks(n_rows, n_cols, out_dtype):
    # Tile geometry for the unweighted fast path, distilled from the variant
    # sweep on GCU400 (48 SPs, DSM <= 448KB/program):
    #   * BLOCK_N=64 halves the x tile traffic per output element versus the
    #     generic 32 and keeps MMA utilization high; 32 stays for row counts
    #     that do not divide 64 (masked tail would waste half a tile).
    #   * BLOCK_K=512 amortizes the dot pipeline for fp32 everywhere and for
    #     fp16/bf16 up to 256 rows; at 512 rows the fp16 MMA is fast enough
    #     that the smaller 256-chunk pipelines better (measured crossover).
    #   * The symmetric triangle wins once the full square grid needs more
    #     than one wave (grid_n >= 2 halves dot work without losing waves).
    if n_rows >= 64 and n_rows % 64 == 0:
        block_n = 64
    else:
        block_n = 32
    if out_dtype == torch.float32:
        if n_cols % 512 == 0:
            block_k = 512
        elif n_cols % 256 == 0:
            block_k = 256
        else:
            block_k = 128
    else:
        if n_cols % 512 == 0 and n_rows <= 256:
            block_k = 512
        elif n_cols % 256 == 0:
            block_k = 256
        else:
            block_k = 128
    return block_n, block_k


def cov(
    input: torch.Tensor,
    *,
    correction: int = 1,
    fweights: torch.Tensor = None,
    aweights: torch.Tensor = None,
) -> torch.Tensor:
    """Estimates the covariance matrix of the variables given by ``input``.

    Rows of ``input`` are the variables and columns are the observations, so a
    (N, M) input yields an (N, N) covariance matrix where ``C[i, j]`` is the
    covariance of variables ``i`` and ``j``. ``correction`` is the difference
    between the number of observations and the divisor (Bessel's correction, 1
    by default). ``fweights`` and ``aweights`` are optional frequency and
    analytic weights over the observations. The computation runs in float32
    internally and the result is cast back to the input dtype, matching the
    precision behaviour expected from a stable Triton implementation.
    """
    logger.debug("GEMS_ENFLAME COV")

    if input.dim() > 2:
        raise ValueError(
            f"cov(): expected input to have two or fewer dimensions but got {input.dim()}"
        )

    # torch.cov treats a 0D/1D input as a single variable observed across the
    # elements, returning a 0D scalar (the variance). A 2D input with a single
    # row is likewise one variable and also yields a scalar.
    x = input.view(1, -1) if input.dim() < 2 else input
    x = x.contiguous()
    n_rows, n_cols = x.shape
    device = x.device
    out_dtype = input.dtype
    single_variable = n_rows == 1

    # Build the combined observation weight vector w = fweights * aweights,
    # keeping whichever is provided. Weights live in float32 for stability. The
    # int->float32 conversion of fweights is done inside _cov_weight_kernel
    # rather than via a host-side ``.to(float32)``, which would dispatch to
    # FlagGems' unstable _to_copy (see the kernel's comment).
    has_wf = fweights is not None
    has_wa = aweights is not None
    has_w = has_wf or has_wa
    w = None
    if has_w:
        wf = fweights.contiguous() if has_wf else None
        wa = (
            aweights.to(device=device, dtype=torch.float32).contiguous()
            if has_wa
            else None
        )
        w = torch.empty(n_cols, device=device, dtype=torch.float32)
        WEIGHT_BLOCK = 1024
        grid_w = triton.cdiv(n_cols, WEIGHT_BLOCK)
        _cov_weight_kernel[(grid_w,)](
            wf,
            wa,
            w,
            n_cols,
            HAS_WF=has_wf,
            HAS_WA=has_wa,
            BLOCK_M=WEIGHT_BLOCK,
        )

    # w_sum and the divisor `fact` follow torch.cov exactly:
    #   no weights          -> fact = n_cols - correction
    #   fweights only        -> fact = w_sum - correction
    #   aweights (any)       -> fact = w_sum - correction * sum(w * aweights) / w_sum
    # `fact` is clamped at 0, matching torch's degenerate-weight handling.
    if has_w:
        w_sum = float(w.sum().item())
        if not has_wa:
            fact = w_sum - correction
        else:
            fact = w_sum - correction * float((w * wa).sum().item()) / w_sum
    else:
        w_sum = float(n_cols)
        fact = float(n_cols - correction)
    if fact < 0.0:
        fact = 0.0

    # Per-variable (weighted) means, kept in float32 for numerical stability.
    # One program per row; a whole-row load avoids the generic 128-wide chunk
    # loop whose many tiny transactions dominate the mean kernel otherwise.
    # Covariance accumulates in float32 inside the kernel for stability; the
    # output buffer is allocated in the target dtype so the final tl.store casts
    # directly, avoiding a separate .to() round trip (which also matches how
    # corrcoef writes its result).
    cov_mat = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
    # Precision policy keyed on the reduction length n_cols, which is what
    # governs the relative error of the centered cross-products sum. The fast
    # paths keep dot operands in the input dtype (exact fp16/bf16 products);
    # short reductions retain the exact 3-pass fp32 request of the generic
    # implementation.
    high_precision = n_cols < 32

    tiny_cap = _TINY_ELEMS_FP32 if out_dtype == torch.float32 else _TINY_ELEMS_FP16
    tiny_elems = triton.next_power_of_2(max(n_rows, 1)) * triton.next_power_of_2(
        max(n_cols, 1)
    )
    use_tiny = not has_w and n_rows <= 64 and n_cols <= 1024 and tiny_elems <= tiny_cap

    if not use_tiny:
        mean = torch.empty(n_rows, device=device, dtype=torch.float32)
        if n_cols <= _MEAN_ROW_CAP:
            mean_block = triton.next_power_of_2(max(n_cols, 1))
            mean_whole = True
        else:
            mean_block = _MEAN_ROW_CAP
            mean_whole = False
        _cov_mean_kernel[(n_rows,)](
            x,
            w,
            mean,
            n_rows,
            n_cols,
            w_sum,
            x.stride(0),
            x.stride(1),
            HAS_W=has_w,
            BLOCK_M=mean_block,
            WHOLE=mean_whole,
            num_warps=4,
        )

    if use_tiny:
        # Small matrix: one launch, one load pass, register-resident means.
        _cov_tiny_kernel[(1,)](
            x,
            cov_mat,
            n_rows,
            n_cols,
            fact,
            w_sum,
            x.stride(0),
            x.stride(1),
            cov_mat.stride(0),
            cov_mat.stride(1),
            NATIVE=True,
            HIGH_PRECISION=high_precision,
            BLOCK_N=max(16, triton.next_power_of_2(max(n_rows, 1))),
            BLOCK_K=max(16, triton.next_power_of_2(max(n_cols, 1))),
            num_warps=4,
        )
    elif not has_w:
        block_n, block_k = _cov_pick_blocks(n_rows, n_cols, out_dtype)
        grid_n = triton.cdiv(n_rows, block_n)
        symmetric = grid_n >= 2
        even = (n_rows % block_n == 0) and (n_cols % block_k == 0)
        native = even and not high_precision
        if symmetric:
            grid = (grid_n * (grid_n + 1) // 2,)
        else:
            grid = (grid_n, grid_n)
        _cov_kernel[grid](
            x,
            w,
            mean,
            cov_mat,
            n_rows,
            n_cols,
            fact,
            x.stride(0),
            x.stride(1),
            cov_mat.stride(0),
            cov_mat.stride(1),
            HAS_W=False,
            HIGH_PRECISION=high_precision,
            NATIVE=native,
            SYMMETRIC=symmetric,
            EVEN=even,
            BLOCK_N=block_n,
            BLOCK_M=block_k,
            num_warps=4,
        )
    else:
        # Weighted covariance keeps the generic masked fp32 structure: the
        # weight tensor multiplies one centered operand inside the dot loop.
        grid_n = triton.cdiv(n_rows, 32)
        _cov_kernel[(grid_n, grid_n)](
            x,
            w,
            mean,
            cov_mat,
            n_rows,
            n_cols,
            fact,
            x.stride(0),
            x.stride(1),
            cov_mat.stride(0),
            cov_mat.stride(1),
            HAS_W=True,
            HIGH_PRECISION=high_precision,
            NATIVE=False,
            SYMMETRIC=False,
            EVEN=False,
            BLOCK_N=32,
            BLOCK_M=128,
            num_warps=4,
        )

    # A single variable (1D input, or a 2D input with one row) yields a 0D
    # scalar variance, matching torch.cov.
    if single_variable:
        return cov_mat.view(())
    return cov_mat
