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
# matrix_exp_backward: grad = tr(G^T exp(A)) w.r.t. A is the top-right block
# of exp([[A^T, G], [0, A^T]]). The batched exponential is computed by
# _block_exp below, which reuses the gcu400 linalg_matrix_exp kernels
# (Pade 13 + scaling/squaring, identical numerics) but launches them with
# configs tuned for this op's shapes. The generic flag_gems.ops.bmm path is
# NOT used: generic bmm_kernel is launched through
# runtime.get_tuned_config("bmm"), whose gcu400 tune_configs.yaml entry keys
# BLOCK_M/BLOCK_N/BLOCK_K for the vendor bmm kernel signature and therefore
# starves the generic kernel of its TILE_* constexprs.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

from .linalg_matrix_exp import (
    _MATRIX_EXP_SMALL_N_MAX,
    _THETA_18,
    _get_t18_coeff,
    _matrix_exp_add_kernel,
    _matrix_exp_bmm_kernel,
    _matrix_exp_lincomb_kernel,
    _matrix_exp_norm_kernel,
    _matrix_exp_small_kernel,
    _matrix_exp_square_kernel,
)

logger = logging.getLogger(__name__)


@triton.jit
def _assemble_kernel(
    OUT,  # (B, 2n, 2n) block buffer, compute dtype (fp32 for half inputs)
    SELF,  # (B, n, n) contiguous input, original dtype
    GRAD,  # (B, n, n) contiguous grad, original dtype
    B,
    n,
    n2,
    BLOCK_SIZE: tl.constexpr,
):
    # Build block matrix M = [[A^T, G], [0, A^T]] for each batch.
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * n2 * n2
    mask = offs < total

    idx = offs
    col = idx % n2
    tmp = idx // n2
    row = tmp % n2
    b = tmp // n2

    top = row < n
    left = col < n
    r_in = tl.where(top, row, row - n)
    c_in = tl.where(left, col, col - n)
    src = b * n * n + r_in * n + c_in
    src_t = b * n * n + c_in * n + r_in  # A^T[r_in, c_in] = A[c_in, r_in]

    at_val = tl.load(SELF + src_t, mask=mask, other=0.0)
    g_val = tl.load(GRAD + src, mask=mask, other=0.0)

    # top-left: A^T ; top-right: G ; bottom-left: 0 ; bottom-right: A^T
    val = tl.where(
        top & left,
        at_val,
        tl.where(top & (~left), g_val, tl.where((~top) & (~left), at_val, 0.0)),
    )
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _assemble_tiled_kernel(
    OUT,  # (B, 2n, 2n) block buffer, compute dtype
    SELF,  # (B, n, n) contiguous input, original dtype
    GRAD,  # (B, n, n) contiguous grad, original dtype
    n,
    n2,
    TR: tl.constexpr,
    TC: tl.constexpr,
):
    # Tiled variant of _assemble_kernel for the >=128 wide block matrices.
    # Plain 2D affine tiles (no tl.trans): the [64, 128] tile keeps 256B
    # contiguous segments on the transposed A^T loads and 512B on the stores,
    # 1.8x faster than the flat kernel at (1, 1024, 1024) fp32 (60 vs 107us).
    # [128, 256]-class tiles are off limits: their 2D masked loads lower to
    # the tops_cvt_v0_si8_to_v1_si8 intrinsic and hit the P03 ld.lld
    # R_DTU_ADDR16_*_ICALL relocation failure.
    pid = ext.program_id(0)
    num_c = tl.cdiv(n2, TC)
    tiles_r = tl.cdiv(n2, TR)
    pid_b = pid // (tiles_r * num_c)
    rem = pid % (tiles_r * num_c)
    pid_r = rem // num_c
    pid_c = rem % num_c

    rows = pid_r * TR + tl.arange(0, TR)
    cols = pid_c * TC + tl.arange(0, TC)
    msk = (rows[:, None] < n2) & (cols[None, :] < n2)
    top = rows[:, None] < n
    left = cols[None, :] < n
    r_in = tl.where(rows[:, None] < n, rows[:, None], rows[:, None] - n)
    c_in = tl.where(cols[None, :] < n, cols[None, :], cols[None, :] - n)

    base_s = pid_b * n * n
    at_val = tl.load(SELF + base_s + c_in * n + r_in, mask=msk, other=0.0)
    g_val = tl.load(GRAD + base_s + r_in * n + c_in, mask=msk, other=0.0)

    val = tl.where(
        top & left,
        at_val,
        tl.where(top & (~left), g_val, tl.where((~top) & (~left), at_val, 0.0)),
    )
    out_ptrs = OUT + pid_b * n2 * n2 + rows[:, None] * n2 + cols[None, :]
    tl.store(out_ptrs, val.to(OUT.dtype.element_ty), mask=msk)


@triton.jit
def _extract_kernel(
    OUT,  # (B, n, n) grad output, original dtype
    INP,  # (B, 2n, 2n) exponential of the block matrix, compute dtype
    B,
    n,
    n2,
    BLOCK_SIZE: tl.constexpr,
):
    # Extract the top-right (0:n, n:2n) block of each batch.
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * n * n
    mask = offs < total

    idx = offs
    col = idx % n
    tmp = idx // n
    row = tmp % n
    b = tmp // n

    src = b * n2 * n2 + row * n2 + (col + n)
    val = tl.load(INP + src, mask=mask, other=0.0)
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _norm_stage1_kernel(
    BLOCK,  # (B, N, N) block buffer
    P,  # (B, num_c, num_r, BLOCK_N) column-sum partials
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 1-norm (max column sum), stage 1: each program computes the column sums
    # of one (BLOCK_M, BLOCK_N) tile and stores the BLOCK_N-vector; the
    # per-column partials are reduced by _norm_stage2_kernel in ascending
    # tile order, so the column sums match the upstream _matrix_exp_norm_kernel
    # bit-for-bit (the upstream kernel is a single program per batch and
    # costs 372us at (1, 1024, 1024) fp32; this two-stage form ~38us).
    pid = ext.program_id(0)
    num_c = tl.cdiv(N, BLOCK_N)
    num_r = tl.cdiv(N, BLOCK_M)
    pid_b = pid // (num_c * num_r)
    rem = pid % (num_c * num_r)
    pid_c = rem // num_r
    pid_r = rem % num_r

    cols = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = pid_r * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    tile = tl.load(
        BLOCK + pid_b * N * N + rows[:, None] * N + cols[None, :],
        mask=mask,
        other=0.0,
    )
    col_acc = tl.sum(tl.abs(tile), axis=0)
    tl.store(
        P + ((pid_b * num_c + pid_c) * num_r + pid_r) * BLOCK_N + tl.arange(0, BLOCK_N),
        col_acc,
    )


@triton.jit
def _norm_stage2_kernel(
    P,  # (B, num_c, num_r, BLOCK_N) column-sum partials
    S,  # (B,) int32 per-batch scaling exponents
    N,
    THETA,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 1-norm stage 2: accumulate the row-tile partials per column chunk in
    # ascending order, take the chunk max, and derive s = ceil(log2(norm /
    # theta)) with the same clamps as the upstream kernel.
    pid_b = ext.program_id(0)
    num_c = tl.cdiv(N, BLOCK_N)
    num_r = tl.cdiv(N, BLOCK_M)
    offs = tl.arange(0, BLOCK_N)
    norm = tl.zeros((), dtype=P.dtype.element_ty)
    for chunk in range(0, num_c):
        acc = tl.zeros([BLOCK_N], dtype=P.dtype.element_ty)
        for row_tile in range(0, num_r):
            acc += tl.load(
                P + ((pid_b * num_c + chunk) * num_r + row_tile) * BLOCK_N + offs
            )
        norm = tl.maximum(norm, tl.max(acc, axis=0))
    s = tl.maximum(tl.ceil(tl.log2(norm / THETA)), 0.0)
    s = tl.minimum(s, 4096.0)
    s = tl.where(norm != norm, 0.0, s)
    tl.store(S + pid_b, s.to(tl.int32))


def _block_exp(block):
    """Batched matrix exponential of the assembled (B, 2n, 2n) block buffer.

    Reuses the gcu400 linalg_matrix_exp kernels with launch configs tuned for
    this op's shapes: 128-wide GEMM tiles for n >= 128 (probe: 2.5-3x over
    the 64-wide default at n = 256/512/1024) and a two-stage parallel 1-norm.
    """
    batch_count, n, _ = block.shape
    dtype = block.dtype
    device = block.device
    coeff = _get_t18_coeff(dtype, device)
    theta = _THETA_18[dtype]

    if n <= _MATRIX_EXP_SMALL_N_MAX:
        out = torch.empty_like(block)
        _matrix_exp_small_kernel[(batch_count,)](
            block, out, coeff, n, theta, BLOCK_N=16, num_warps=1
        )
        return out

    num_chunks = triton.cdiv(n, 64)
    partials = (
        torch.empty(
            batch_count * num_chunks * num_chunks * 64, dtype=dtype, device=device
        )
        if n >= 256
        else None
    )
    s_vals = torch.empty(batch_count, dtype=torch.int32, device=device)
    grid_norm = (batch_count * num_chunks * num_chunks,)

    # Probe-anchored tile selection (gcu400, do_bench kernel mode):
    #   - GEMM chain: 128-wide tiles win from n = 128 up (2.5-3x at 256/512/
    #     1024; ~2x at 128 for batch >= 8), 64 below.
    #   - Elementwise lincomb/add tiles: 128 wins except when the grid would
    #     stay single-wave-starved (n = 128 with small batch), where 64 keeps
    #     4x more programs in flight.
    tile = 128 if n >= 128 else 64
    elem_tile = 128 if (n >= 256 or batch_count >= 8) else 64
    num_tiles = triton.cdiv(n, tile) * triton.cdiv(n, tile)
    grid_mat = (batch_count * num_tiles,)
    grid_elem = (batch_count * triton.cdiv(n, elem_tile) ** 2,)
    mat_numel = batch_count * n * n

    a2 = torch.empty_like(block)
    a3 = torch.empty_like(block)
    a6 = torch.empty_like(block)
    b_mats = torch.empty(5, batch_count, n, n, dtype=dtype, device=device)
    t_mat = torch.empty_like(block)
    result = torch.empty_like(block)

    with torch_device_fn.device(device):
        if n >= 256:
            # Two-stage parallel 1-norm: the upstream kernel is a single
            # program per batch (372us at (1, 1024, 1024) fp32; ~10x here).
            _norm_stage1_kernel[grid_norm](
                block, partials, n, BLOCK_N=64, BLOCK_M=64, num_warps=4
            )
            _norm_stage2_kernel[(batch_count,)](
                partials, s_vals, n, theta, BLOCK_N=64, BLOCK_M=64, num_warps=4
            )
        else:
            # Small matrices: one launch beats the two-stage form (the extra
            # launch gap dominates the kernel-time saving on short chains).
            _matrix_exp_norm_kernel[(batch_count,)](
                block, s_vals, n, theta, BLOCK_M=64, BLOCK_N=64, num_warps=4
            )

        _matrix_exp_bmm_kernel[grid_mat](
            block,
            block,
            block,
            a2,
            s_vals,
            n,
            SCALE_A=True,
            SCALE_B=True,
            HAS_ACC=False,
            BLOCK_M=tile,
            BLOCK_N=tile,
            BLOCK_K=tile,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            a2,
            block,
            block,
            a3,
            s_vals,
            n,
            SCALE_A=False,
            SCALE_B=True,
            HAS_ACC=False,
            BLOCK_M=tile,
            BLOCK_N=tile,
            BLOCK_K=tile,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            a3,
            a3,
            a3,
            a6,
            s_vals,
            n,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=False,
            BLOCK_M=tile,
            BLOCK_N=tile,
            BLOCK_K=tile,
            num_warps=4,
        )
        _matrix_exp_lincomb_kernel[grid_elem](
            block,
            a2,
            a3,
            a6,
            s_vals,
            coeff,
            b_mats,
            n,
            mat_numel,
            BLOCK_M=elem_tile,
            BLOCK_N=elem_tile,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            b_mats[0],
            b_mats[4],
            b_mats[3],
            b_mats[3],
            s_vals,
            n,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=True,
            BLOCK_M=tile,
            BLOCK_N=tile,
            BLOCK_K=tile,
            num_warps=4,
        )
        _matrix_exp_add_kernel[grid_elem](
            b_mats[2],
            b_mats[3],
            t_mat,
            n,
            BLOCK_M=elem_tile,
            BLOCK_N=elem_tile,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            t_mat,
            b_mats[3],
            b_mats[1],
            result,
            s_vals,
            n,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=True,
            BLOCK_M=tile,
            BLOCK_N=tile,
            BLOCK_K=tile,
            num_warps=4,
        )

        s_max = max(s_vals.tolist())
        if s_max > 0:
            tmp = torch.empty_like(result)
            for step in range(s_max):
                _matrix_exp_square_kernel[grid_mat](
                    result,
                    tmp,
                    s_vals,
                    step,
                    n,
                    BLOCK_M=tile,
                    BLOCK_N=tile,
                    BLOCK_K=tile,
                    num_warps=4,
                )
                result, tmp = tmp, result

    return result


def matrix_exp_backward(self, grad):
    logger.debug("GEMS_ENFLAME MATRIX_EXP_BACKWARD")
    assert self.shape == grad.shape, "self and grad must have the same shape"
    assert (
        self.ndim >= 2 and self.shape[-1] == self.shape[-2]
    ), "matrix_exp_backward expects square matrices"

    orig_shape = self.shape
    orig_dtype = self.dtype
    n = orig_shape[-1]
    n2 = 2 * n

    # Compute in float32 for half-precision inputs (float32/float64 kept as-is).
    if orig_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = orig_dtype

    self_c = self.reshape(-1, n, n).contiguous()
    grad_c = grad.reshape(-1, n, n).contiguous()
    batch = self_c.shape[0]

    if n == 0 or batch == 0:
        return torch.empty_like(self)

    # Assemble the block matrix M = [[A^T, G], [0, A^T]]. The tiled kernel is
    # 1.8x faster from n2 = 256 up and for batched small cases, but the flat
    # kernel keeps more small programs in flight for the single-matrix n2=128
    # case (6.5us vs 14.7us), so dispatch on the probed crossover.
    block = torch.empty((batch, n2, n2), dtype=compute_dtype, device=self.device)
    with torch_device_fn.device(self.device):
        if n2 >= 256 or batch >= 8:
            grid_asm = (batch * triton.cdiv(n2, 64) * triton.cdiv(n2, 128),)
            _assemble_tiled_kernel[grid_asm](
                block, self_c, grad_c, n, n2, TR=64, TC=128, num_warps=2
            )
        else:
            total = block.numel()
            grid_asm = (triton.cdiv(total, 1024),)
            _assemble_kernel[grid_asm](
                block, self_c, grad_c, batch, n, n2, BLOCK_SIZE=1024
            )

        expm = _block_exp(block)

        out = torch.empty((batch, n, n), dtype=orig_dtype, device=self.device)
        total_out = out.numel()
        grid_out = (triton.cdiv(total_out, 1024),)
        _extract_kernel[grid_out](out, expm, batch, n, n2, BLOCK_SIZE=1024)

    return out.reshape(orig_shape)
