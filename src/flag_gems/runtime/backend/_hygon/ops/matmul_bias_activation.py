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

# Fused matmul + bias + ReLU: out = relu(input @ weight + bias)
#
# Contract (FlagGems pytest): input (M, K), weight (K, N), bias (N,) or (1, N)
#   -> fresh contiguous out (M, N) with out.dtype == input.dtype.
#   reference: torch.relu(torch.mm(input, weight) + bias) computed in float64
#   tolerance: atol = 1e-4 * K, rtol = {fp16: 1e-3, bf16: 1.6e-2, fp32: 1.3e-6}
#
# Design:
#   * one fused Triton kernel (no extra bias/relu passes),
#   * fp32 accumulation for every input dtype, single cast on store,
#   * grouped 1-D program ordering (GROUP_M) so a group of CTAs walks the same
#     A row panel, keeping it hot in L2,
#   * compile-time EVEN_M / EVEN_N / EVEN_K chosen for the selected tile, so the
#     common case has no per-element mask arithmetic at all,
#   * explicit contiguity flags so the innermost axis load stays vectorized even
#     if integer-stride specialization is unavailable,
#   * NaN-propagating ReLU (torch.relu semantics: relu(NaN) = NaN, relu(-inf) = 0),
#   * one autotune space with a dtype tag in the key; each dtype only sees the
#     tiles registered for it, and the very large fp32 tiles are gated to large
#     shapes so they cannot be mis-selected for the tiny shapes,
#   * steady-state calls relaunch the tuner-selected tile directly, skipping the
#     autotuner/heuristics dispatch that dominates the small shapes.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_LOWP_TUPLES = [
    (128, 128, 32, 8, 3),
    (128, 128, 32, 8, 2),
    (128, 128, 64, 8, 2),
    (128, 128, 32, 4, 3),
    (128, 256, 32, 8, 2),
    (256, 128, 32, 8, 2),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 64, 64, 8, 2),
    (64, 128, 64, 8, 2),
    (64, 64, 64, 4, 3),
    (256, 64, 32, 8, 3),
    (64, 256, 32, 8, 3),
    # low-occupancy-cost tiles for small / wave-quantized shapes
    (128, 128, 32, 4, 2),
    (128, 64, 32, 4, 2),
    (64, 128, 32, 4, 2),
    (64, 64, 32, 4, 2),
    (64, 64, 32, 4, 3),
    # wide square tile: 32 KB of shared (two resident CTAs, same as 128x128x32)
    # but the MMA operands are re-read for half as many instructions per MAC.
    (256, 256, 16, 8, 2),
]

# fp32 tiles: (BLOCK_M + BLOCK_N) * BLOCK_K * 4 * stages <= 64 KB
_FP32_TUPLES = [
    (64, 64, 32, 4, 3),
    (128, 64, 32, 8, 2),
    (64, 128, 32, 8, 2),
    (128, 128, 16, 8, 3),
    # 32 KB shared with a large tile: same footprint as the 64x64 winner but 4x
    # the tile area, so fewer LDS fragments per MAC.
    (128, 128, 16, 8, 2),
    (128, 128, 16, 4, 2),
    (128, 128, 32, 8, 2),
    (256, 128, 16, 8, 2),
    (128, 256, 16, 8, 2),
    (64, 64, 64, 4, 2),
    (256, 64, 16, 8, 2),
    # widest legal fp32 tile (64 KB of shared): half the LDS operand bytes per
    # MMA versus 128x128, at the cost of one resident CTA instead of two.
    (256, 256, 16, 8, 2),
]

_LOWP_TILES = set(_LOWP_TUPLES)
_FP32_TILES = set(_FP32_TUPLES)
# big fp32 tiles are only offered to large shapes (tuner regret guard)
_FP32_LARGE_ONLY = {
    (128, 128, 16, 8, 2),
    (128, 128, 16, 4, 2),
    (256, 128, 16, 8, 2),
    (128, 256, 16, 8, 2),
}
# wide tiles only pay off once the grid is deep enough that resident CTAs are
# scheduled for a long time; on the small shapes they would only add regret.
_WIDE_ONLY = {(256, 256, 16, 8, 2)}

TAG_LOWP = 33
TAG_FP32 = 65

GROUP_M = 8


def _mk(tile):
    return triton.Config(
        {"BLOCK_M": tile[0], "BLOCK_N": tile[1], "BLOCK_K": tile[2]},
        num_warps=tile[3],
        num_stages=tile[4],
    )


_ALL_CONFIGS = []
_seen_tiles = set()
for _tile in _LOWP_TUPLES + _FP32_TUPLES:
    if _tile in _seen_tiles:
        continue
    _seen_tiles.add(_tile)
    _ALL_CONFIGS.append(_mk(_tile))


def _config_key(c):
    return (
        c.kwargs["BLOCK_M"],
        c.kwargs["BLOCK_N"],
        c.kwargs["BLOCK_K"],
        c.num_warps,
        c.num_stages,
    )


def _filter(configs, named_args):
    """Keep only the tiles registered for the active dtype and shape size."""
    fp32 = named_args.get("tag", TAG_LOWP) == TAG_FP32
    group = _FP32_TILES if fp32 else _LOWP_TILES
    large = named_args.get("M", 0) * named_args.get("N", 0) >= 512 * 512
    kept = []
    for c in configs:
        key = _config_key(c)
        if key not in group:
            continue
        if not large and (key in _FP32_LARGE_ONLY or key in _WIDE_ONLY):
            continue
        kept.append(c)
    return kept or list(configs)


@triton.autotune(
    configs=_ALL_CONFIGS,
    key=["M", "N", "K", "tag"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args, **kw: _filter(
            configs, named_args
        )
    },
    warmup=20,
    rep=50,
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def _mba_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    tag,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    A_K_CONTIG: tl.constexpr,
    B_N_CONTIG: tl.constexpr,
    BIAS_CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped ordering: consecutive pids walk GROUP_M row panels before the
    # column panel advances, so the A panel is reused out of L2.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if A_K_CONTIG:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    else:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    if B_N_CONTIG:
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :])
    else:
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    m_mask = offs_am < M
    n_mask = offs_bn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_mask = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if BIAS_CONTIG:
        bias = tl.load(bias_ptr + offs_bn, mask=n_mask, other=0.0)
    else:
        bias = tl.load(bias_ptr + offs_bn * stride_bias, mask=n_mask, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)
    # torch.relu semantics: NaN propagates, -inf maps to 0.
    acc = tl.where(acc > 0.0, acc, tl.where(acc == acc, 0.0, acc))

    # out is always freshly allocated and contiguous -> last axis stride is 1.
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :])
    c = acc.to(c_ptr.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


# The tuner wraps the jitted kernel twice (autotune -> heuristics); keep a
# handle on the raw JITFunction so steady-state calls skip both dispatchers.
_JIT_KERNEL = _mba_kernel.fn.fn
# (tag, M, N, K, strides) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_LAUNCH_CACHE = {}


def _broadcastable_to(s1, s2):
    r1, r2 = len(s1), len(s2)
    if r1 == 0:
        return True
    if r2 == 0 or r1 > r2:
        return False
    d = r2 - r1
    for i in range(r1):
        if s1[i] == 1 or s1[i] == s2[d + i]:
            continue
        return False
    return True


def matmul_bias_activation(input, weight, bias):
    logger.debug("GEMS_HYGON MATMUL_BIAS_ACTIVATION")
    a_shape = input.shape
    if a_shape[1] != weight.shape[0]:
        raise AssertionError("Incompatible dimensions")
    M = a_shape[0]
    K = a_shape[1]
    N = weight.shape[1]
    if not _broadcastable_to(bias.shape, (M, N)):
        raise AssertionError("Incompatible input shape")

    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    if M == 0 or N == 0:
        return out

    a_stride_m, a_stride_k = input.stride()
    b_stride_k, b_stride_n = weight.stride()
    bias_stride = 0 if bias.numel() == 1 else bias.stride(-1)
    tag = TAG_FP32 if input.dtype is torch.float32 else TAG_LOWP
    a_k_contig = a_stride_k == 1
    b_n_contig = b_stride_n == 1
    bias_contig = bias_stride == 1

    # Steady state: launch the tile the tuner selected for this exact metadata
    # signature directly (no autotuner/heuristics dispatch on the hot path).
    key = (tag, M, N, K, a_stride_m, a_stride_k, b_stride_k, b_stride_n, bias_stride)
    cfg = _LAUNCH_CACHE.get(key)
    if cfg is not None:
        bm, bn, bk, warps, stages = cfg
        grid = (((M + bm - 1) // bm) * ((N + bn - 1) // bn),)
        _JIT_KERNEL[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            a_stride_m,
            a_stride_k,
            b_stride_k,
            b_stride_n,
            bias_stride,
            N,
            tag,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=GROUP_M,
            EVEN_M=(M % bm) == 0,
            EVEN_N=(N % bn) == 0,
            EVEN_K=(K % bk) == 0,
            A_K_CONTIG=a_k_contig,
            B_N_CONTIG=b_n_contig,
            BIAS_CONTIG=bias_contig,
            num_warps=warps,
            num_stages=stages,
        )
        return out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _mba_kernel[grid](
        input,
        weight,
        bias,
        out,
        M,
        N,
        K,
        a_stride_m,
        a_stride_k,
        b_stride_k,
        b_stride_n,
        bias_stride,
        out.stride(0),
        tag,
        GROUP_M=GROUP_M,
        A_K_CONTIG=a_k_contig,
        B_N_CONTIG=b_n_contig,
        BIAS_CONTIG=bias_contig,
    )
    best = _mba_kernel.best_config
    _LAUNCH_CACHE[key] = (
        best.kwargs["BLOCK_M"],
        best.kwargs["BLOCK_N"],
        best.kwargs["BLOCK_K"],
        best.num_warps,
        best.num_stages,
    )
    return out
