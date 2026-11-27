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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# activation string -> integer code consumed by the kernel
_ACT_MAP = {"none": 0, None: 0, "relu": 1, "silu": 2}


@libentry()
@triton.jit
def _mixed_dtypes_linear_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    ncols,
    stride_xm,
    stride_wn,
    stride_om,
    INT4: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,  # 0: none, 1: relu, 2: silu
    EPI_SCALE: tl.constexpr,
    EVEN_K: tl.constexpr,  # K % BK == 0: no K masking in the hot loop
    EVEN_N: tl.constexpr,  # weight tile columns exact: no N masking
    SPLIT4: tl.constexpr,  # int4: two half-width dots + epilogue interleave
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BM), BM)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :]

    if INT4 and SPLIT4:
        # Two 4-bit weights per byte: low nibble -> even output column,
        # high nibble -> odd output column. Compute the even/odd halves as
        # two half-width dots and interleave once in the epilogue; an
        # in-loop tl.interleave on the dequantized tile is a layout-chain
        # pathology (~80% of kernel time on GCU400) and is avoided here.
        offs_nb = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        w_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_nb[None, :]
        nb_mask = offs_nb < ncols
        if not EPI_SCALE:
            # bf16: per-element scale before the dot keeps the reference
            # rounding boundary; even/odd scale columns, hoisted (the tile
            # column index never changes across the K loop).
            s_lo = tl.load(scale_ptr + 2 * offs_nb, mask=nb_mask, other=0.0)
            s_hi = tl.load(scale_ptr + 2 * offs_nb + 1, mask=nb_mask, other=0.0)
            s_lo_f = s_lo.to(tl.float32)
            s_hi_f = s_hi.to(tl.float32)
        acc_lo = tl.zeros((BM, BN // 2), dtype=tl.float32)
        acc_hi = tl.zeros((BM, BN // 2), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BK)):
            if EVEN_K:
                a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            else:
                a = tl.load(
                    x_ptrs, mask=m_mask[:, None] & (offs_k < K)[None, :], other=0.0
                )
            if EVEN_K and EVEN_N:
                wb = tl.load(w_ptrs)
            else:
                wb = tl.load(
                    w_ptrs, mask=(offs_k[:, None] < K) & nb_mask[None, :], other=0
                )
            wb32 = wb.to(tl.int32)
            lo = (wb32 & 0xF) - 8
            hi = ((wb32 >> 4) & 0xF) - 8
            if EPI_SCALE:
                wf_lo = lo.to(a.dtype)
                wf_hi = hi.to(a.dtype)
            else:
                wf_lo = (lo.to(tl.float32) * s_lo_f[None, :]).to(a.dtype)
                wf_hi = (hi.to(tl.float32) * s_hi_f[None, :]).to(a.dtype)
            acc_lo = tl.dot(a, wf_lo, acc_lo)
            acc_hi = tl.dot(a, wf_hi, acc_hi)
            x_ptrs += BK
            w_ptrs += BK * stride_wn
            if not EVEN_K:
                offs_k += BK
        acc = tl.interleave(acc_lo, acc_hi)
    else:
        if INT4:
            offs_nb = pid_n * (BN // 2) + tl.arange(0, BN // 2)
            w_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_nb[None, :]
            w_mask_n = (offs_nb < ncols)[None, :]
        else:
            w_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_n[None, :]
            w_mask_n = n_mask[None, :]
        if not EPI_SCALE:
            # bf16: keep the per-tile scale multiply (reference rounding
            # boundary) but hoist the loop-invariant scale load out.
            if EVEN_N:
                s = tl.load(scale_ptr + offs_n)
            else:
                s = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
            s_f = s.to(tl.float32)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BK)):
            if EVEN_K:
                a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            else:
                a = tl.load(
                    x_ptrs, mask=m_mask[:, None] & (offs_k < K)[None, :], other=0.0
                )
            if EVEN_K and EVEN_N:
                wb = tl.load(w_ptrs)
            else:
                wb = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & w_mask_n, other=0)
            wb32 = wb.to(tl.int32)
            if INT4:
                # FasterTransformer biased int4 converter: subtract 8.
                lo = (wb32 & 0xF) - 8
                hi = ((wb32 >> 4) & 0xF) - 8
                wi = tl.interleave(lo, hi)  # [BK, BN]
            else:
                # FasterTransformer biased int8 converter: subtract 128.
                wi = wb32 - 128
            if EPI_SCALE:
                # fp16: fold the per-column scale into the epilogue (fewer
                # in-loop ops); algebraically identical and within fp16
                # tolerance.
                wf = wi.to(a.dtype)
            else:
                # bf16: keep the per-tile scale multiply to match the
                # reference rounding boundary under strict bf16 tolerance.
                wf = (wi.to(tl.float32) * s_f[None, :]).to(a.dtype)
            acc = tl.dot(a, wf, acc)
            x_ptrs += BK
            w_ptrs += BK * stride_wn
            if not EVEN_K:
                offs_k += BK

    if EPI_SCALE:
        if EVEN_N:
            s_epi = tl.load(scale_ptr + offs_n)
        else:
            s_epi = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc * s_epi.to(tl.float32)[None, :]

    if HAS_BIAS:
        if EVEN_N:
            b = tl.load(bias_ptr + offs_n)
        else:
            b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b.to(tl.float32)[None, :]
    if ACT == 1:
        # gcu400: a select-family op (tl.maximum/tl.where) applied directly
        # to the fp32 tl.dot loop accumulator silently miscompiles (values
        # off by ~1e4); multiplying by a 0/1 float mask is the only safe
        # form. With HAS_BIAS the add before the select hides the bug, which
        # is why only the no-bias relu specialization fails.
        acc = acc * (acc > 0).to(tl.float32)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if EVEN_N:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def _pick_config(M, N, ncols, is_fp16):
    """Tile/launch hyper-parameters, tuned on GCU400 (Libra, 48 SP).

    Small-M shapes are SP-occupancy/MMA-slot bound: BM shrinks to the mma
    minimum so padded rows do not waste tensor-core slots, and BN aims at
    ~one wave of programs. BM never drops below the M bucket boundary —
    two M tiles would re-stream the weight matrix twice. Larger M grows BM
    with it. BK=512 halves loop trips for K=4096-class shapes; K not
    divisible by BK simply keeps the masked arm.
    Returns (BM, BN, BK, num_warps, num_stages).
    """
    int4 = 2 * ncols == N
    if int4:
        if M <= 16:
            return 16, 256, 512, 1, 2
        if M <= 32:
            return 32, 256, 512, 1, 2
        if M <= 64:
            # bf16 needs BN=256 here; (64,256,512,nw2) exceeds the 917504B
            # DSM budget on the bf16 split arm at compile time.
            return (64, 128, 512, 1, 2) if is_fp16 else (64, 256, 512, 1, 2)
        return 128, 128, 256, 1, 3
    if M <= 16:
        if N >= 8192:
            return 16, 256, 512, 2, 2
        # fp16 MMA keeps full rate at BM=16; bf16 measured faster at BM=32.
        return (16 if is_fp16 else 32), 128, 512, 1, 4
    if M <= 32:
        if N >= 8192:
            return 32, 256, 512, 2, 2
        return 32, 128, 512, 1, 4
    if M <= 64:
        return 64, 256, 512, 2, 2
    return 128, 128, 256, 1, 3


def mixed_dtypes_linear(input, weight, scale, bias=None, activation=None):
    # Constant-value logging is disabled (semantics-preserving fix4 rewrite)

    # The kernel's strides are hardcoded for a contiguous layout (stride_xm=K / stride_wn=ncols);
    # materialize non-contiguous views into a contiguous copy first, to avoid reading input / weight / scale at wrong
    # addresses.
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input must be float16 or bfloat16, got {input.dtype}"
    assert weight.dtype == torch.uint8, f"weight must be uint8, got {weight.dtype}"
    assert (
        scale.dtype == input.dtype
    ), f"scale dtype {scale.dtype} must match input dtype {input.dtype}"
    act = activation if activation is not None else "none"
    assert act in _ACT_MAP, f"unsupported activation: {activation}"

    orig = input.shape
    K = orig[-1]
    M = input.numel() // K
    N = scale.shape[0]
    ncols = weight.shape[1]
    # int8 mode: one byte per output column; int4-packed mode: two per byte.
    assert ncols == N or 2 * ncols == N, (
        f"weight columns {ncols} must equal scale size {N} (int8) "
        f"or half of it (packed int4)"
    )
    int4 = ncols != N

    out = torch.empty((*orig[:-1], N), dtype=input.dtype, device=input.device)

    is_fp16 = input.dtype == torch.float16
    epi_scale = is_fp16
    BM, BN, BK, nw, ns = _pick_config(M, N, ncols, is_fp16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    with torch_device_fn.device(input.device):
        _mixed_dtypes_linear_kernel[grid](
            input,
            weight,
            scale,
            bias if bias is not None else input,
            out,
            M,
            N,
            K,
            ncols,
            K,
            ncols,
            N,
            INT4=int4,
            HAS_BIAS=(bias is not None),
            ACT=_ACT_MAP[act],
            EPI_SCALE=epi_scale,
            EVEN_K=(K % BK == 0),
            EVEN_N=(ncols % (BN // 2 if int4 else BN) == 0),
            SPLIT4=int4,
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP_M=8,
            num_warps=nw,
            num_stages=ns,
        )
    return out
