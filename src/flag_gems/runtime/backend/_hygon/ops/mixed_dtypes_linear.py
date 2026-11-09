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

"""Mixed-dtype linear (uint8 int8 / packed int4 weights) for FlagGems.

Reference semantics (from the authoritative pytest `_reference`):

    k_size, ncols = weight.shape                      # weight is (K, ncols)
    if ncols == N:      unpacked = weight.int() - 128            # int8
    else:               unpacked = stack([lo-8, hi-8], -1).reshape(K, N)  # int4
    fp16:  acc = input.float() @ unpacked.float();  acc = acc * scale.float()
    other: deq = (unpacked.float() * scale.float()[:, None]).to(input.dtype)
           acc = input.float() @ deq.float()
    acc = acc + bias.float()  (optional)
    acc = relu(acc) | acc * sigmoid(acc)  (optional)
    out = acc.to(input.dtype)

The bfloat16 reference pre-rounds (weight * scale) to bfloat16, so that path
keeps the per-element scale and rounding inside the K-loop; rounding once in the
epilogue was measured to break the pytest atol (1e-4 * K) on cancelled outputs.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_ACT_MAP = {None: 0, "none": 0, "relu": 1, "silu": 2}

# Bounded configuration search keyed on the problem shape.  The round-2 profile
# showed these workloads launch only 32 CTAs (0.4 waves on 80 SMs) with 4 warps
# per CTA, i.e. almost no latency hiding, so the search concentrates on output
# tile width (CTA count) and pipeline depth.
_CONFIGS = [
    triton.Config({"BM": 16, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 16, "BN": 64, "BK": 64}, num_warps=4, num_stages=5),
    triton.Config({"BM": 16, "BN": 64, "BK": 64}, num_warps=4, num_stages=6),
    triton.Config({"BM": 16, "BN": 32, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 16, "BN": 64, "BK": 128}, num_warps=4, num_stages=3),
    triton.Config({"BM": 16, "BN": 128, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 16, "BN": 128, "BK": 64}, num_warps=4, num_stages=4),
    triton.Config({"BM": 32, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 32, "BN": 64, "BK": 64}, num_warps=4, num_stages=5),
    triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_CONFIGS,
    key=["M", "N", "K", "INT4", "EPI_SCALE"],
    warmup=10,
    rep=50,
)
@triton.jit
def _mixed_linear_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    K,
    P,
    stride_xm,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,  # 0 none, 1 relu, 2 silu
    INT4: tl.constexpr,
    EPI_SCALE: tl.constexpr,
    EVEN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # 2-D grid with the M tile fastest-varying: concurrent CTAs then cover the
    # same weight column tile for every M tile, so one weight tile is fetched
    # once into L2 and reused across the M tiles instead of being re-read in a
    # later wave (the A working set is small enough to stay in L2).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    n_mask = offs_n < N

    if EVEN:
        # Exact tile boundaries: clamp the row indices instead of masking so
        # both tile loads stay unmasked contiguous loads (surplus rows are
        # discarded by the masked store).
        rows = tl.minimum(offs_m, M - 1)
        x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :]
        if INT4:
            offs_nb = pid_n * (BN // 2) + tl.arange(0, BN // 2)
            w_ptrs = w_ptr + offs_k[:, None] * P + offs_nb[None, :]
        else:
            w_ptrs = w_ptr + offs_k[:, None] * P + offs_n[None, :]
    else:
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
        if INT4:
            offs_nb = pid_n * (BN // 2) + tl.arange(0, BN // 2)
            w_ptrs = w_ptr + offs_k[:, None] * P + offs_nb[None, :]
            wcol_mask = (offs_nb < P)[None, :]
        else:
            w_ptrs = w_ptr + offs_k[:, None] * P + offs_n[None, :]
            wcol_mask = n_mask[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if EVEN:
        for _ in range(0, K // BK):
            a = tl.load(x_ptrs)
            wb = tl.load(w_ptrs)
            wb32 = wb.to(tl.int32)
            if INT4:
                lo = (wb32 & 0xF) - 8
                hi = ((wb32 >> 4) & 0xF) - 8
                wi = tl.interleave(lo, hi)
            else:
                wi = wb32 - 128
            if EPI_SCALE:
                wf = wi.to(a.dtype)
            else:
                s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
                wf = (wi.to(tl.float32) * s.to(tl.float32)[None, :]).to(a.dtype)
            acc = tl.dot(a, wf, acc)
            x_ptrs += BK
            w_ptrs += BK * P
    else:
        for _ in range(0, tl.cdiv(K, BK)):
            k_mask = offs_k < K
            a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            wb = tl.load(w_ptrs, mask=k_mask[:, None] & wcol_mask, other=0)
            wb32 = wb.to(tl.int32)
            if INT4:
                lo = (wb32 & 0xF) - 8
                hi = ((wb32 >> 4) & 0xF) - 8
                wi = tl.interleave(lo, hi)
            else:
                wi = wb32 - 128
            if EPI_SCALE:
                wf = wi.to(a.dtype)
            else:
                s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
                wf = (wi.to(tl.float32) * s.to(tl.float32)[None, :]).to(a.dtype)
            acc = tl.dot(a, wf, acc)
            x_ptrs += BK
            w_ptrs += BK * P
            offs_k += BK

    if EPI_SCALE:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc * s.to(tl.float32)[None, :]
    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b.to(tl.float32)[None, :]
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)

    o_ptrs = o_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        o_ptrs,
        acc.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def mixed_dtypes_linear(input, weight, scale, bias=None, activation=None):
    logger.debug("GEMS_HYGON MIXED_DTYPES_LINEAR")
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input must be float16 or bfloat16, got {input.dtype}"
    assert weight.dtype == torch.uint8, f"weight must be uint8, got {weight.dtype}"
    assert (
        scale.dtype == input.dtype
    ), f"scale dtype {scale.dtype} must match input dtype {input.dtype}"
    act = "none" if activation is None else activation
    assert act in _ACT_MAP, f"unsupported activation: {activation}"

    orig = input.shape
    K = int(orig[-1])
    M = int(input.numel()) // K
    N = int(scale.shape[0])
    ncols = int(weight.shape[-1])
    assert ncols == N or 2 * ncols == N, (
        f"weight columns {ncols} must equal scale size {N} (int8) "
        f"or half of it (packed int4)"
    )
    int4 = ncols != N

    out = torch.empty((*orig[:-1], N), dtype=input.dtype, device=input.device)
    if M == 0 or N == 0:
        return out

    epi_scale = input.dtype == torch.float16

    # Exact tile boundaries for every config in the search list -> the K-loop
    # runs a mask-free variant with unmasked contiguous tile loads.
    even = K % 128 == 0 and N % 128 == 0 and (not int4 or ncols % 64 == 0)

    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    _mixed_linear_kernel[grid](
        input,
        weight,
        scale,
        bias if bias is not None else input,
        out,
        M,
        N,
        K,
        ncols,
        input.stride(-2),
        HAS_BIAS=bias is not None,
        ACT=_ACT_MAP[act],
        INT4=int4,
        EPI_SCALE=epi_scale,
        EVEN=even,
    )
    return out
