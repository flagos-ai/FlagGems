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

logger = logging.getLogger(__name__)


os.environ.setdefault("MACA_PATH", "/opt/maca")


# ---------------------------------------------------------------------------
# Kernel 1: grad_input = grad_output @ weight
#   A = grad_output [M, K] (stride go_sm, go_sk)
#   B = weight      [K, N] (stride w_sk, w_sn)
#   C = grad_input  [M, N]
# ---------------------------------------------------------------------------
@triton.jit
def _gi_kernel(
    go_ptr,
    w_ptr,
    gi_ptr,
    M,
    N,
    K,
    go_sm,
    go_sk,
    w_sk,
    w_sn,
    gi_sm,
    gi_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # wrap output indices so hot-loop loads are always in-bounds
    ram = (rm % M).to(tl.int64)
    rbn = (rn % N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    prev = (K // BLOCK_K) * BLOCK_K
    for k0 in range(0, prev, BLOCK_K):
        rk = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(go_ptr + ram[:, None] * go_sm + rk[None, :] * go_sk)
        b = tl.load(w_ptr + rk[:, None] * w_sk + rbn[None, :] * w_sn)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    if prev < K:
        rk = (prev + tl.arange(0, BLOCK_K)).to(tl.int64)
        mk = rk < K
        a = tl.load(
            go_ptr + ram[:, None] * go_sm + rk[None, :] * go_sk,
            mask=mk[None, :],
            other=0.0,
        )
        b = tl.load(
            w_ptr + rk[:, None] * w_sk + rbn[None, :] * w_sn,
            mask=mk[:, None],
            other=0.0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    out = acc.to(gi_ptr.dtype.element_ty)
    msk = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(gi_ptr + rm[:, None] * gi_sm + rn[None, :] * gi_sn, out, mask=msk)


# ---------------------------------------------------------------------------
# Kernel 2: grad_weight = grad_output^T @ input  (+ optional fused grad_bias)
#   a[k, m] = go[m, k]; b = input [M, N]; c = grad_weight [K, N]
#   bias[k] = sum_m go[m, k] (computed by the pid_n == 0 program of each k-block)
# ---------------------------------------------------------------------------
@triton.jit
def _gw_kernel(
    go_ptr,
    inp_ptr,
    gw_ptr,
    bias_ptr,
    K,
    N,
    M,
    go_sm,
    go_sk,
    inp_sm,
    inp_sn,
    gw_sk,
    gw_sn,
    COMPUTE_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_k - group_id * GROUP_M, GROUP_M)
    pid_k = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rak = (rk % K).to(tl.int64)
    rbn = (rn % N).to(tl.int64)
    rk = rk.to(tl.int64)
    rn = rn.to(tl.int64)
    rm = tl.arange(0, BLOCK_M).to(tl.int64)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    bias_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        mm = m0 + rm
        mmask = mm < M
        # Load go as [BLOCK_M, BLOCK_K] (coalesced along k, stride 1) and
        # input as [BLOCK_M, BLOCK_N]; a = trans(go_tile) is an in-kernel
        # register/smem transpose, avoiding stride-K column loads entirely.
        go_tile = tl.load(
            go_ptr + mm[:, None] * go_sm + rak[None, :] * go_sk,
            mask=mmask[:, None],
            other=0.0,
        )
        b = tl.load(
            inp_ptr + mm[:, None] * inp_sm + rbn[None, :] * inp_sn,
            mask=mmask[:, None],
            other=0.0,
        )
        acc = tl.dot(tl.trans(go_tile), b, acc, out_dtype=tl.float32, allow_tf32=False)
        if COMPUTE_BIAS and pid_n == 0:
            # fp16/bf16 tl.sum would round in the low-precision dtype (and the
            # MACA backend fails to lower bf16 reductions); accumulate fp32.
            bias_acc += tl.sum(go_tile.to(tl.float32), axis=0)

    out = acc.to(gw_ptr.dtype.element_ty)
    msk = (rk < K)[:, None] & (rn < N)[None, :]
    tl.store(gw_ptr + rk[:, None] * gw_sk + rn[None, :] * gw_sn, out, mask=msk)
    if COMPUTE_BIAS and pid_n == 0:
        tl.store(bias_ptr + rk, bias_acc.to(bias_ptr.dtype.element_ty), mask=(rk < K))


# ---------------------------------------------------------------------------
# Kernel 3: grad_bias = sum_m grad_output[m, :]
# ---------------------------------------------------------------------------
@triton.jit
def _gb_kernel(
    go_ptr,
    bias_ptr,
    K,
    M,
    go_sm,
    go_sk,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rk = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    rak = (rk % K).to(tl.int64)
    rk = rk.to(tl.int64)
    rm = tl.arange(0, BLOCK_M).to(tl.int64)

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        mm = m0 + rm
        tile = tl.load(
            go_ptr + mm[:, None] * go_sm + rak[None, :] * go_sk,
            mask=(mm < M)[:, None],
            other=0.0,
        )
        acc += tl.sum(tile.to(tl.float32), axis=0)
    tl.store(bias_ptr + rk, acc.to(bias_ptr.dtype.element_ty), mask=(rk < K))


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
def _configs(dtype, M):
    if dtype == torch.float16:
        if M >= 256:
            # Overlap-pair sweep winner at the big shapes: larger GI tiles
            # (fewer, bigger CTAs) overlap the GW kernel better than the
            # smaller 64x128 tiles (fp16 s3: 1188us vs 1216us overlapped).
            gi = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=16, warps=8, stages=4)
        else:
            gi = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=16, warps=4, stages=6)
        gw = dict(BLOCK_K=128, BLOCK_N=128, BLOCK_M=64, warps=8, stages=2)
    else:  # float32 / bfloat16: 4-byte operands need smaller tiles (64KB smem limit)
        if M <= 64:
            # tiny batch: shrink tiles to raise CTA count (occupancy) on 104 SMs
            gi = dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=16, warps=4, stages=4)
            gw = dict(BLOCK_K=64, BLOCK_N=64, BLOCK_M=32, warps=4, stages=2)
        elif M <= 128:
            gi = dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=16, warps=4, stages=4)
            gw = dict(BLOCK_K=128, BLOCK_N=64, BLOCK_M=32, warps=4, stages=2)
        else:
            gi = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, warps=4, stages=6)
            gw = dict(BLOCK_K=128, BLOCK_N=64, BLOCK_M=32, warps=4, stages=2)
    return gi, gw


_SIDE_STREAM = None
_SIDE_EVENT = None


def _side_stream():
    # Cache one side stream + completion event per process (each eval/benchmark
    # subprocess is pinned to a single device).  Creating them per call costs
    # ~20-25us of host time, which is large relative to the small-shape
    # kernels, so overlap is only worthwhile for big batches (gated in run()).
    global _SIDE_STREAM, _SIDE_EVENT
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream(device=torch.cuda.current_device())
        _SIDE_EVENT = torch.cuda.Event()
    return _SIDE_STREAM, _SIDE_EVENT


def linear_backward(input, grad_output, weight, output_mask):
    mask = tuple(bool(m) for m in output_mask)

    in_features = input.shape[-1]
    out_features = weight.shape[0]
    batch_dims = input.shape[:-1]
    M = 1
    for d in batch_dims:
        M *= d
    K = out_features
    N = in_features

    if input.dim() == 2:
        x2 = input
        go2 = grad_output
    else:
        x2 = input.reshape(M, N)
        go2 = grad_output.reshape(M, K)

    gi = gw = gb = None

    # GI (go @ w) and GW (go^T @ x) are independent matmuls that both read
    # grad_output; overlap them on a second stream so the bandwidth-bound
    # cases can use idle SMs.  The stream/event are cached (no per-call
    # allocation), so even small batches with combined CTA counts above the
    # 104-SM capacity benefit; only the bias-only/GI-only paths stay serial.
    if mask[0] and (mask[1] or mask[2]):
        gi = torch.empty((M, N), device=input.device, dtype=input.dtype)
        gi_cfg, gw_cfg = _configs(input.dtype, M)
        grid_gi = (
            triton.cdiv(M, gi_cfg["BLOCK_M"]) * triton.cdiv(N, gi_cfg["BLOCK_N"]),
        )
        if mask[1]:
            gw = torch.empty((K, N), device=input.device, dtype=weight.dtype)
        if mask[2]:
            gb = torch.empty((K,), device=input.device, dtype=weight.dtype)

        side, ev = _side_stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            if mask[1]:
                grid_gw = (
                    triton.cdiv(K, gw_cfg["BLOCK_K"])
                    * triton.cdiv(N, gw_cfg["BLOCK_N"]),
                )
                bias_ptr = gb if (mask[2] and gb is not None) else gw
                _gw_kernel[grid_gw](
                    go2,
                    x2,
                    gw,
                    bias_ptr,
                    K,
                    N,
                    M,
                    go2.stride(0),
                    go2.stride(1),
                    x2.stride(0),
                    x2.stride(1),
                    gw.stride(0),
                    gw.stride(1),
                    COMPUTE_BIAS=bool(mask[2]),
                    BLOCK_K=gw_cfg["BLOCK_K"],
                    BLOCK_N=gw_cfg["BLOCK_N"],
                    BLOCK_M=gw_cfg["BLOCK_M"],
                    GROUP_M=8,
                    num_warps=gw_cfg["warps"],
                    num_stages=gw_cfg["stages"],
                )
            else:
                grid_gb = (triton.cdiv(K, 128),)
                _gb_kernel[grid_gb](
                    go2,
                    gb,
                    K,
                    M,
                    go2.stride(0),
                    go2.stride(1),
                    BLOCK_K=128,
                    BLOCK_M=32,
                    num_warps=4,
                    num_stages=2,
                )
            ev.record(side)
        _gi_kernel[grid_gi](
            go2,
            weight,
            gi,
            M,
            N,
            K,
            go2.stride(0),
            go2.stride(1),
            weight.stride(0),
            weight.stride(1),
            gi.stride(0),
            gi.stride(1),
            BLOCK_M=gi_cfg["BLOCK_M"],
            BLOCK_N=gi_cfg["BLOCK_N"],
            BLOCK_K=gi_cfg["BLOCK_K"],
            GROUP_M=8,
            num_warps=gi_cfg["warps"],
            num_stages=gi_cfg["stages"],
        )
        torch.cuda.current_stream().wait_event(ev)
        if input.dim() > 2:
            gi = gi.view(*batch_dims, N)
        return gi, gw, gb

    if mask[0]:
        gi = torch.empty((M, N), device=input.device, dtype=input.dtype)
        cfg, _ = _configs(input.dtype, M)
        grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
        _gi_kernel[grid](
            go2,
            weight,
            gi,
            M,
            N,
            K,
            go2.stride(0),
            go2.stride(1),
            weight.stride(0),
            weight.stride(1),
            gi.stride(0),
            gi.stride(1),
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=8,
            num_warps=cfg["warps"],
            num_stages=cfg["stages"],
        )
        if input.dim() > 2:
            gi = gi.view(*batch_dims, N)

    if mask[1] or mask[2]:
        if mask[1]:
            gw = torch.empty((K, N), device=input.device, dtype=weight.dtype)
        if mask[2]:
            gb = torch.empty((K,), device=input.device, dtype=weight.dtype)

        if mask[1]:
            _, cfg = _configs(weight.dtype, M)
            grid = (triton.cdiv(K, cfg["BLOCK_K"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
            bias_ptr = gb if (mask[2] and gb is not None) else gw
            _gw_kernel[grid](
                go2,
                x2,
                gw,
                bias_ptr,
                K,
                N,
                M,
                go2.stride(0),
                go2.stride(1),
                x2.stride(0),
                x2.stride(1),
                gw.stride(0),
                gw.stride(1),
                COMPUTE_BIAS=bool(mask[2]),
                BLOCK_K=cfg["BLOCK_K"],
                BLOCK_N=cfg["BLOCK_N"],
                BLOCK_M=cfg["BLOCK_M"],
                GROUP_M=8,
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
            )
        else:
            grid = (triton.cdiv(K, 128),)
            _gb_kernel[grid](
                go2,
                gb,
                K,
                M,
                go2.stride(0),
                go2.stride(1),
                BLOCK_K=128,
                BLOCK_M=32,
                num_warps=4,
                num_stages=2,
            )

    return gi, gw, gb
