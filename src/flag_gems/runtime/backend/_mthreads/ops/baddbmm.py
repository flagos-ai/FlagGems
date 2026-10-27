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
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.ops.add import add, add_func
from flag_gems.ops.mul import mul
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

# Reuse the warp-specialized bmm pipeline: baddbmm is a batched GEMM whose
# consumer only gains the extra beta * bias term before the store, so it shares
# the TME producer, the tile shape and the column-panel rasterization with bmm.
from .bmm import (  # noqa: E402
    _WS_BLOCK_K,
    _WS_BLOCK_M,
    _WS_BLOCK_N,
    _WS_NUM_WARPS,
    _WS_PANEL_WIDTH,
    _WS_STAGES,
    TLE_WS_ON,
    _ws_bmm_producer,
    bmm,
    bmm_sqmma,
    is_sqmma_compatible,
)
from .mm import _ws_rasterization_2d_column  # noqa: E402

try:
    import triton.experimental.tle.language as tle
except ImportError:  # pragma: no cover - depends on the installed Triton build
    tle = None

logger = logging.getLogger(__name__)

# A bias that varies along M forces one block_m x block_n epilogue staging
# buffer in shared memory, which does not fit next to the 3-stage 256x256x64
# TME ring on a 192 KiB part; a bias broadcast along M loads as a cheap 1-D
# vector and keeps that ring.  The narrower, deeper-K ring below measured
# fastest for the M-varying case while keeping the epilogue fused.
_WS_FULLBIAS_BLOCK_M = 256
_WS_FULLBIAS_BLOCK_N = 128
_WS_FULLBIAS_BLOCK_K = 32
_WS_FULLBIAS_STAGES = 5

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "baddbmm_mthreads_expand.yaml")
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("baddbmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
    flagtune_op_name="baddbmm",
    flagtune_expand_op_name="baddbmm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(runtime.get_heuristic_config("baddbmm"))
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    O,
    bias,
    alpha,
    beta,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    bias_batch_stride: tl.constexpr,
    bias_M_stride: tl.constexpr,
    bias_N_stride: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N
    bias += pid_b * bias_batch_stride

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=mask_a)
        b = tl.load(b_ptrs, mask=mask_b)
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

    bias_ptrs = bias + offs_m[:, None] * bias_M_stride + offs_n[None, :] * bias_N_stride

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    else:
        mask_c = True
        if not DIVISIBLE_M:
            mask_c &= offs_m[:, None] < M
        if not DIVISIBLE_N:
            mask_c &= offs_n[None, :] < N

    bi = tl.load(bias_ptrs, mask=mask_c)
    out = accumulator * alpha + bi * beta
    o = out.to(bi.dtype)
    tl.store(o_ptrs, o, mask=mask_c)


def _baddbmm_launch(bias, A, B, beta, alpha, out):
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    bbias = torch.broadcast_to(bias, (batch, M, N))
    bias_batch_stride = bbias.stride(0)
    bias_M_stride = bbias.stride(1)
    bias_N_stride = bbias.stride(-1)

    grid = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        baddbmm_kernel[grid](
            A,
            B,
            out,
            bbias,
            alpha,
            beta,
            M,
            N,
            K,
            bias_batch_stride=bias_batch_stride,
            bias_M_stride=bias_M_stride,
            bias_N_stride=bias_N_stride,
        )


def _baddbmm_sqmma(bias, A, B, beta, alpha):
    batch, M, K = A.shape
    A, B = A.contiguous(), B.contiguous()
    product = bmm_sqmma(A, B, A.dtype, batch, M, B.shape[2], K)
    return add(
        mul(product, alpha), mul(torch.broadcast_to(bias, (batch, M, B.shape[2])), beta)
    )


# ---------------------------------------------------------------------------
# Warp-specialized SQMMA for baddbmm.  Same TLE producer/consumer tile pipeline
# as bmm, with the beta * bias + alpha * (A @ B) epilogue folded into the
# consumer, so the fused op costs one launch instead of a batched GEMM plus
# three full-tensor elementwise passes.
# ---------------------------------------------------------------------------


@triton.jit
def _ws_baddbmm_consumer(
    a_reader,
    b_reader,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha,
    beta,
    batch_index,
    stride_ib,
    stride_im,
    stride_in,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BIAS_M_BROADCAST: tl.constexpr,
):
    offset_m = pid_m * block_m + tl.arange(0, block_m)
    offset_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    bias_base = bias_ptr + batch_index * stride_ib
    if BIAS_M_BROADCAST:
        bias = tl.load(
            bias_base + offset_n * stride_in,
            mask=offset_n < N,
            other=0.0,
        )[None, :]
    else:
        bias_ptrs = (
            bias_base + offset_m[:, None] * stride_im + offset_n[None, :] * stride_in
        )
        bias = tl.load(bias_ptrs, mask=mask, other=0.0)

    result = (alpha * acc + beta * bias).to(c_ptr.dtype.element_ty)
    c_block_ptr = (
        c_ptr + batch_index * (M * N) + N * offset_m[:, None] + offset_n[None, :]
    )
    tl.store(c_block_ptr, result, mask=mask)


@triton.jit
def baddbmm_sqmma_ws_kernel(
    a_desc,
    b_desc,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha,
    beta,
    stride_ib,
    stride_im,
    stride_in,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
    full_panel: tl.constexpr,
    BIAS_M_BROADCAST: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
    batch_index = tl.program_id(axis=2)
    grid_x = tl.num_programs(axis=0)
    grid_y = tl.num_programs(axis=1)
    block_idx = raw_bx + raw_by * grid_x
    pid_n, pid_m = _ws_rasterization_2d_column(
        block_idx,
        grid_x,
        grid_y,
        panel_width,
        full_panel,
    )

    a_smem = tle.gpu.alloc(
        (pipeline_stages, block_m, block_k),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (pipeline_stages, block_k, block_n),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="baddbmm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="baddbmm_b", b=b_smem)

    tle.gpu.warp_specialize(
        [
            (
                _ws_baddbmm_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    bias_ptr,
                    M,
                    N,
                    K,
                    alpha,
                    beta,
                    batch_index,
                    stride_ib,
                    stride_im,
                    stride_in,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                    BIAS_M_BROADCAST,
                ),
            ),
            (
                _ws_bmm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    M,
                    K,
                    batch_index,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def baddbmm_sqmma_ws(bias, A, B, batch, M, N, K, alpha, beta, C=None):
    logger.debug("GEMS_MTHREADS BADDBMM_SQMMA_WS")
    device = A.device
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    assert A.dtype == B.dtype, "Mat A and Mat B should have the same dtype"
    if C is None:
        C = torch.empty((batch, M, N), dtype=A.dtype, device=device)
    bbias = torch.broadcast_to(bias, (batch, M, N))
    stride_ib, stride_im, stride_in = bbias.stride()
    # A bias broadcast along M loads as a 1-D vector and leaves room for the
    # mm/bmm 3-stage ring; a bias that varies along M needs a block_m x block_n
    # epilogue staging buffer, so the narrower deeper-K ring is used instead.
    if stride_im == 0:
        block_m, block_n, block_k, stages = (
            _WS_BLOCK_M,
            _WS_BLOCK_N,
            _WS_BLOCK_K,
            _WS_STAGES,
        )
    else:
        block_m, block_n, block_k, stages = (
            _WS_FULLBIAS_BLOCK_M,
            _WS_FULLBIAS_BLOCK_N,
            _WS_FULLBIAS_BLOCK_K,
            _WS_FULLBIAS_STAGES,
        )
    # The SQMMA consumer stores straight into a row-major buffer of the operand
    # dtype, so `C` must already match the operands; callers gate on that.
    desc_a = TensorDescriptor.from_tensor(A.reshape(batch * M, K), [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(B.reshape(batch * K, N), [block_k, block_n])
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m), batch)
    with torch_device_fn.device(A.device):
        baddbmm_sqmma_ws_kernel[grid](
            desc_a,
            desc_b,
            C,
            bbias,
            M,
            N,
            K,
            alpha,
            beta,
            stride_ib,
            stride_im,
            stride_in,
            block_m,
            block_n,
            block_k,
            _WS_PANEL_WIDTH,
            stages,
            grid[0] % _WS_PANEL_WIDTH == 0,
            stride_im == 0,
            num_warps=_WS_NUM_WARPS,
            num_stages=stages,
        )
    return C


# sqmma has a fixed launch/memory floor; measured crossover on MTT S5000 is ~3.2 GFLOPs.
_SQMMA_MIN_FLOPS = 3.2e9

# The warp-specialized pipeline costs more per launch than it saves whenever
# the launch is too small -- fewer than a 2x2 grid of tiles, fewer than eight
# 64-wide K tiles or fewer than 16 CTAs (the same bounds measured for bmm).
# These are the scalar bounds; the M-varying bias uses the narrower 128-column
# ring and may run on half as many N tiles.
_WS_MIN_M = 2 * _WS_BLOCK_M
_WS_MIN_N = 2 * _WS_BLOCK_N
_WS_FULLBIAS_MIN_N = 2 * _WS_FULLBIAS_BLOCK_N
_WS_MIN_K = 512
_WS_MIN_OUTPUT_TILES = 16


def _use_ws(bias, A, B, M, N, K):
    """Warp-specialized fused baddbmm needs TLE plus enough work per launch."""
    # Reject on scalars first: small shapes must not pay for the bias view or
    # the operand layout checks below.
    if not (
        TLE_WS_ON
        and A.dtype in (torch.float16, torch.bfloat16)
        and M >= _WS_MIN_M
        and N >= _WS_FULLBIAS_MIN_N
        and K >= _WS_MIN_K
    ):
        return False
    if not is_sqmma_compatible(A, B, N, K):
        return False
    batch = A.shape[0]
    if N < _WS_MIN_N and torch.broadcast_to(bias, (batch, M, N)).stride(1) == 0:
        return False
    return (
        triton.cdiv(M, _WS_BLOCK_M) * triton.cdiv(N, _WS_BLOCK_N) * batch
        >= _WS_MIN_OUTPUT_TILES
    )


def _use_sqmma(A, B, N, K):
    return (
        is_sqmma_compatible(A, B, N, K)
        and A.shape[0] * A.shape[1] * N * K * 2 >= _SQMMA_MIN_FLOPS
    )


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_MTHREADS BADDBMM_FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        if _use_ws(bias, A, B, M, N, K):
            return baddbmm_sqmma_ws(bias, A, B, batch, M, N, K, alpha, beta)
        if _use_sqmma(A, B, N, K):
            return _baddbmm_sqmma(bias, A, B, beta, alpha)
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        _baddbmm_launch(bias, A, B, beta, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_MTHREADS BADDBMM_BACKWARD")
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def compute_bias_grad(d_output, beta, bias):
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def compute_A_grad(d_output, B, alpha):
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = bmm(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = bmm(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = bmm(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = bmm(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS_MTHREADS BADDBMM_OUT")
    batch, M, K = A.shape
    _, _, N = B.shape
    assert (
        out.shape == (batch, M, N) and out.dtype == A.dtype
    ), "Incompatible output shape or dtype for baddbmm.out"
    # The SQMMA consumer stores straight into a row-major buffer of the operand
    # dtype, so a contiguous `out` of that dtype can be the destination itself
    # instead of paying for a temporary product plus three elementwise passes.
    if out.is_contiguous() and _use_ws(bias, A, B, M, N, K):
        return baddbmm_sqmma_ws(bias, A, B, batch, M, N, K, alpha, beta, C=out)
    if _use_sqmma(A, B, N, K):
        A, B = A.contiguous(), B.contiguous()
        product = bmm_sqmma(A, B, A.dtype, batch, M, N, K)
        scaled = mul(product, alpha)
        add_func(scaled, torch.broadcast_to(bias, (batch, M, N)), beta, out0=out)
        return out
    _baddbmm_launch(
        bias.contiguous(),
        A.contiguous(),
        B.contiguous(),
        beta,
        alpha,
        out,
    )
    return out


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(
        bias.contiguous(),
        A.contiguous(),
        B.contiguous(),
        beta,
        alpha,
    )
