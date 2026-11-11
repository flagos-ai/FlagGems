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

"""Triton implementation of linear: out = input @ weight.T + bias.

Reference semantics (torch.nn.functional.linear):
  input:  (..., K)
  weight: (N, K)
  bias:   (N,)
  out:    (..., N)
"""

import logging

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.ops.linear import linear as default_linear
from flag_gems.runtime import torch_device_fn

# Reuse the warp-specialized GEMM building blocks.  linear is the same GEMM as
# addmm, but the weight stays in its native [N, K] layout: the consumer asks
# wgmma for the transposed B operand, so no per-call weight transpose copy is
# needed.  Only the epilogue (vector bias) and the B ring shape differ.
from .addmm import is_sqmma_compatible
from .mm import (  # noqa: E402
    _WS_BLOCK_K,
    _WS_BLOCK_M,
    _WS_BLOCK_N,
    _WS_NUM_WARPS,
    _WS_PANEL_WIDTH,
    _WS_STAGES,
    TLE_WS_ON,
    _ws_rasterization_2d_column,
)

try:
    import triton.experimental.tle.language as tle
except ImportError:  # pragma: no cover - depends on the installed Triton build
    tle = None

logger = logging.getLogger(__name__)


@triton.jit
def _linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    USE_TF32: tl.constexpr,
    X_CG: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight is (N, K); load tile as (BLOCK_K, BLOCK_N) for tl.dot
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                if X_CG:
                    x = tl.load(x_ptrs, cache_modifier=".cg")
                else:
                    x = tl.load(x_ptrs)
            else:
                x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
            if EVEN_N:
                w = tl.load(w_ptrs)
            else:
                w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            kmask = (k * BLOCK_K + offs_k) < K
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & kmask[:, None], other=0.0)
        if FP32_INPUT:
            if USE_TF32:
                acc = tl.dot(x, w, acc, input_precision="tf32")
            else:
                acc = tl.dot(x, w, acc, input_precision="ieee")
        else:
            acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if EVEN_N:
        bias = tl.load(b_ptr + offs_n)
    else:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    if EVEN_M and EVEN_N:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty))
    else:
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


# ---------------------------------------------------------------------------
# Warp-specialized SQMMA for linear.  A producer warp group drives TME copies
# into a pipelined shared-memory ring while a consumer runs SQMMA, so the next
# K tile is already in flight during the MMA.  The weight tile is staged as
# [block_n, block_k] and fed to wgmma with trans_b=True, which removes the
# O(NK) transpose copy that forwarding to addmm would otherwise pay.
# ---------------------------------------------------------------------------

# The 256x256 ring leaves 2048^2 with only 64 CTAs, so mid-size shapes switch
# to the 256x128 ring that addmm already uses for its full-bias epilogue.
# Both configurations fit the 192 KiB shared-memory budget.
_WS_LINEAR_MIN_M = 2048
_WS_LINEAR_MIN_N = 2048
_WS_LINEAR_LARGE_M = 4096
_WS_LINEAR_LARGE_N = 4096
_WS_LINEAR_MID_BLOCK_M = 256
_WS_LINEAR_MID_BLOCK_N = 128
_WS_LINEAR_MID_BLOCK_K = 32
_WS_LINEAR_MID_STAGES = 5


@triton.jit
def _ws_linear_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    K: tl.constexpr,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offset_m = pid_m * block_m
    offset_n = pid_n * block_n
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        offset_k = k_iter * block_k
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            (block_m, block_k),
            (offset_m, offset_k),
        )
        tle.gpu.copy(
            b_desc,
            b_slot.b,
            (block_n, block_k),
            (offset_n, offset_k),
        )
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _ws_linear_consumer(
    a_reader,
    b_reader,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
        acc = tle.gpu.wgmma(
            a_wait.slot.a,
            b_wait.slot.b,
            acc,
            trans_b=True,
        )
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offset_n, mask=offset_n < N, other=0.0)
        acc = acc + bias[None, :]

    c_block_ptr = c_ptr + N * offset_m[:, None] + offset_n[None, :]
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def linear_sqmma_ws_kernel(
    a_desc,
    b_desc,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
    full_panel: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
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
        (pipeline_stages, block_n, block_k),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="linear_a", a=a_smem)
    b_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="linear_b", b=b_smem)

    tle.gpu.warp_specialize(
        [
            (
                _ws_linear_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    bias_ptr,
                    M,
                    N,
                    K,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                    HAS_BIAS,
                ),
            ),
            (
                _ws_linear_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    K,
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


def _is_linear_ws_compatible(x, weight, bias, M, N, K):
    """Warp-specialized SQMMA needs the TLE frontend, large half-precision
    shapes and a row-major weight that wgmma can read with trans_b."""
    return (
        TLE_WS_ON
        and M >= _WS_LINEAR_MIN_M
        and N >= _WS_LINEAR_MIN_N
        and x.is_contiguous()
        and weight.is_contiguous()
        and is_sqmma_compatible(x, weight, N, K)
        and (
            bias is None
            or (
                bias.dim() == 1
                and bias.shape[0] == N
                and bias.dtype == x.dtype
                and bias.is_contiguous()
            )
        )
    )


def _linear_sqmma_ws(x, weight, bias, y, M, N, K):
    logger.debug("GEMS_MTHREADS LINEAR_SQMMA_WS")
    if M >= _WS_LINEAR_LARGE_M and N >= _WS_LINEAR_LARGE_N:
        block_m, block_n, block_k, stages = (
            _WS_BLOCK_M,
            _WS_BLOCK_N,
            _WS_BLOCK_K,
            _WS_STAGES,
        )
    else:
        block_m, block_n, block_k, stages = (
            _WS_LINEAR_MID_BLOCK_M,
            _WS_LINEAR_MID_BLOCK_N,
            _WS_LINEAR_MID_BLOCK_K,
            _WS_LINEAR_MID_STAGES,
        )
    has_bias = bias is not None
    # The consumer only dereferences bias_ptr when HAS_BIAS is set; reuse the
    # output pointer to avoid materializing a zeros(N) bias.
    bias_ptr = bias if has_bias else y
    desc_a = TensorDescriptor.from_tensor(x, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(weight, [block_n, block_k])
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m), 1)
    with torch_device_fn.device(x.device):
        linear_sqmma_ws_kernel[grid](
            desc_a,
            desc_b,
            y,
            bias_ptr,
            M,
            N,
            K,
            block_m,
            block_n,
            block_k,
            _WS_PANEL_WIDTH,
            stages,
            grid[0] % _WS_PANEL_WIDTH == 0,
            has_bias,
            num_warps=_WS_NUM_WARPS,
            num_stages=stages,
        )


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def linear(input, weight, bias=None):
    logger.debug("GEMS_MTHREADS LINEAR")
    if (
        not isinstance(input, torch.Tensor)
        or input.device.type != "musa"
        or input.dtype not in _SUPPORTED_DTYPES
        or weight.dtype != input.dtype
        or weight.dim() != 2
        or input.dim() < 1
    ):
        return default_linear(input, weight, bias)

    in_shape = input.shape
    orig_dim = input.dim()
    x = input
    if orig_dim == 1:
        x = x.reshape(1, -1)
    elif orig_dim > 2:
        x = x.reshape(-1, x.shape[-1])
    M, K = x.shape
    N = weight.shape[0]

    # Warp-specialized SQMMA fast path.  Keeping the weight in [N, K] layout
    # and using wgmma's trans_b operand avoids the per-call transpose copy.
    if _is_linear_ws_compatible(x, weight, bias, M, N, K):
        out_shape = in_shape[:-1] + (N,)
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
        _linear_sqmma_ws(x, weight, bias, out.reshape(M, N), M, N, K)
        return out

    # Tile dispatch on MTT S5000 (microbenchmark-verified):
    #  - large fp32 (tf32) squares: 256x128x32/w8/ns2/g4 (4096^3: 6.44->3.24ms)
    #  - fp16/bf16 M>=2048: 64x64x32/w4/g4 + .cg x-loads (4096^3: 1.68->1.48ms)
    #  - fp16/bf16 M=1024: 64x32x64/w4/g4 (0.0487->0.0357ms)
    #  - fp32 1024: 64x64x32/w4/g8 (unchanged)
    #  - M<1024 shapes (384s + correctness): 32x32x32 (fp32) / 32x32x64 (fp16/bf16)
    #    raise CTA count from 36 to 144, ~2x faster on the 384 half workloads.
    if input.dtype == torch.float32:
        if M >= 2048 and N >= 2048:
            bm, bn, bk, nw, ns, gm, xcg = 256, 128, 32, 8, 2, 4, False
        elif M >= 1024:
            bm, bn, bk, nw, ns, gm, xcg = 64, 64, 32, 4, 1, 8, False
        else:
            bm, bn, bk, nw, ns, gm, xcg = 32, 32, 32, 4, 1, 8, False
    else:
        if M >= 2048:
            bm, bn, bk, nw, ns, gm, xcg = 64, 64, 32, 4, 1, 4, True
        elif M >= 1024:
            bm, bn, bk, nw, ns, gm, xcg = 64, 32, 64, 4, 1, 4, False
        else:
            bm, bn, bk, nw, ns, gm, xcg = 32, 32, 64, 4, 1, 8, False

    # Keep the reduced-precision TF32 path limited to the large FP32 GEMM
    # configuration above.  The small-M configurations are accuracy-sensitive
    # and therefore use the IEEE dot product.
    use_tf32 = input.dtype == torch.float32 and M >= 2048 and N >= 2048

    out_shape = in_shape[:-1] + (N,)
    out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    if orig_dim == 1:
        y = out.reshape(1, N)
    elif orig_dim > 2:
        y = out.reshape(M, N)
    else:
        y = out

    if bias is None:
        b = torch.zeros(N, device=input.device, dtype=input.dtype)
    else:
        b = bias

    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _linear_kernel[grid](
        x,
        weight,
        b,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        EVEN_M=(M % bm == 0),
        EVEN_N=(N % bn == 0),
        EVEN_K=(K % bk == 0),
        FP32_INPUT=(input.dtype == torch.float32),
        USE_TF32=use_tf32,
        X_CG=xcg,
        num_warps=nw,
        num_stages=ns,
    )
    return out
