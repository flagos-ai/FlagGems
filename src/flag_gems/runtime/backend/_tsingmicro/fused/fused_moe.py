# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from the vLLM project (https://github.com/vllm-project/vllm).
# Source files under vllm/model_executor/layers/:
#   fused_moe/fused_moe.py      – Triton kernels, dispatch, fused_experts_impl
#   fused_moe/activation.py     – MoEActivation enum, apply_moe_activation
#   fused_moe/utils.py          – _fp8_quantize, _int8_quantize, moe_kernel_quantize_input
#   fused_moe/config.py         – _get_config_dtype_str
#   quantization/utils/mxfp4_utils.py   – dequant_mxfp4
#   quantization/utils/mxfp6_utils.py   – dequant_mxfp6
#   quantization/utils/ocp_mx_utils.py  – OCP_MX_BLOCK_SIZE


import functools
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import yaml

# Prefer vendor moe_align; moe_sum stays on the common FlagGems path.
# silu_and_mul lives in its own module so pointwise_dynamic codegen does not
# scrape this large fused_moe.py (relative imports / other @triton.jit bodies).
from flag_gems.fused.moe_sum import moe_sum
from flag_gems.runtime import device, torch_device_fn

from .moe_align_block_size import moe_align_block_size
from .silu_and_mul import silu_and_mul_out
from .silu_and_mul_with_clamp import silu_and_mul_with_clamp_out

logger = logging.getLogger(__name__)

# OCP MX quantization helpers (requires amd-quark)

OCP_MX_BLOCK_SIZE = 32
# H100/Qwen-style MoE tuning thresholds. GEMM tile changes become reliably
# positive from 4096 tokens; direct_sum is kept separate because it is a
# reduction-layout decision even though it currently shares the same cutoff.
MOE_GEMM_TUNING_MIN_TOKENS = 4096
MOE_DIRECT_SUM_MIN_TOKENS = 4096
_HALF_GEMM_TILE_M = 128
_HALF_GEMM_TILE_K = 64
_HALF_GEMM2_TILE_N = 256
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")


# --- TX81 compact grouped GEMM (from fused_moe0818) ---
def _sync_device(device: torch.device) -> None:
    if device.type != "cpu":
        torch.get_device_module(device.type).synchronize()


class _PipeTimer:
    """Optional stage timer. FUSED_MOE_PIPELINE_PROFILE=0 to disable (default on)."""

    def __init__(self, enabled: bool, device: torch.device, tag: str):
        self.enabled = enabled
        self.device = device
        self.tag = tag
        self.stages: list[tuple[str, float]] = []
        self.meta: dict[str, Any] = {}
        if enabled:
            _sync_device(device)
            self._t0 = self._last = time.perf_counter()
        else:
            self._t0 = self._last = 0.0

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        _sync_device(self.device)
        now = time.perf_counter()
        self.stages.append((name, (now - self._last) * 1000.0))
        self._last = now

    def report(self) -> None:
        if not self.enabled:
            return
        _sync_device(self.device)
        total = (time.perf_counter() - self._t0) * 1000.0
        print("-" * 60, flush=True)
        print(f"[moe profile] {self.tag}  total={total:.4f} ms", flush=True)
        for k, v in self.meta.items():
            print(f"  meta.{k}: {v}", flush=True)
        for name, ms in self.stages:
            pct = 100.0 * ms / total if total > 0 else 0.0
            print(f"  {name:<22} {ms:10.4f} ms  ({pct:5.1f}%)", flush=True)
        print("-" * 60, flush=True)


@triton.jit
def moe_compact_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    seg_starts_ptr,
    seg_ends_ptr,
    seg_experts_ptr,
    a_rows_ptr,
    weight_ptr,
    dest_ids_ptr,
    N,
    K,
    num_segments,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GATHER_A: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    WEIGHT_BY_DEST: tl.constexpr,
    WRITE_DIRECT: tl.constexpr,
    USE_INT8_W8A16: tl.constexpr,
):
    """Persistent CTAs: gather A, optional INT8 dequant in SPM, optional router w, scatter C."""
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    seg_id = pid
    while seg_id < num_segments:
        start = tl.load(seg_starts_ptr + seg_id).to(tl.int64)
        end = tl.load(seg_ends_ptr + seg_id).to(tl.int64)
        expert = tl.load(seg_experts_ptr + seg_id).to(tl.int64)
        m_size = end - start
        if (expert >= 0) and (m_size > 0):
            offs_n = tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            for mid in range(0, tl.cdiv(m_size, BLOCK_M)):
                offs_m = start + mid * BLOCK_M + tl.arange(0, BLOCK_M)
                mask_m = offs_m < end
                if GATHER_A:
                    a_row = tl.load(a_rows_ptr + offs_m, mask=mask_m, other=0).to(
                        tl.int64
                    )
                else:
                    a_row = offs_m
                if WRITE_DIRECT:
                    dest_row = tl.load(dest_ids_ptr + offs_m, mask=mask_m, other=0).to(
                        tl.int64
                    )
                else:
                    dest_row = offs_m

                acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                if NUM_N_TILES > 1:
                    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                if NUM_N_TILES > 2:
                    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                if NUM_N_TILES > 3:
                    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                a_ptrs = (
                    a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
                )
                for k in range(0, tl.cdiv(K, BLOCK_K)):
                    k_offs = k * BLOCK_K + offs_k
                    k_mask = k_offs < K
                    a = tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    b_k_off = k * BLOCK_K * stride_bk

                    offs_bn = offs_n
                    b = tl.load(
                        b_ptr
                        + expert * stride_be
                        + b_k_off
                        + offs_bn[None, :] * stride_bn
                        + offs_k[:, None] * stride_bk,
                        mask=k_mask[:, None] & (offs_bn[None, :] < N),
                        other=0.0,
                    )
                    if USE_INT8_W8A16:
                        b = b.to(COMPUTE_DTYPE)
                    acc0 = tl.dot(a, b, acc=acc0)
                    if NUM_N_TILES > 1:
                        offs_bn = BLOCK_N + offs_n
                        b = tl.load(
                            b_ptr
                            + expert * stride_be
                            + b_k_off
                            + offs_bn[None, :] * stride_bn
                            + offs_k[:, None] * stride_bk,
                            mask=k_mask[:, None] & (offs_bn[None, :] < N),
                            other=0.0,
                        )
                        if USE_INT8_W8A16:
                            b = b.to(COMPUTE_DTYPE)
                        acc1 = tl.dot(a, b, acc=acc1)
                    if NUM_N_TILES > 2:
                        offs_bn = 2 * BLOCK_N + offs_n
                        b = tl.load(
                            b_ptr
                            + expert * stride_be
                            + b_k_off
                            + offs_bn[None, :] * stride_bn
                            + offs_k[:, None] * stride_bk,
                            mask=k_mask[:, None] & (offs_bn[None, :] < N),
                            other=0.0,
                        )
                        if USE_INT8_W8A16:
                            b = b.to(COMPUTE_DTYPE)
                        acc2 = tl.dot(a, b, acc=acc2)
                    if NUM_N_TILES > 3:
                        offs_bn = 3 * BLOCK_N + offs_n
                        b = tl.load(
                            b_ptr
                            + expert * stride_be
                            + b_k_off
                            + offs_bn[None, :] * stride_bn
                            + offs_k[:, None] * stride_bk,
                            mask=k_mask[:, None] & (offs_bn[None, :] < N),
                            other=0.0,
                        )
                        if USE_INT8_W8A16:
                            b = b.to(COMPUTE_DTYPE)
                        acc3 = tl.dot(a, b, acc=acc3)
                    a_ptrs += BLOCK_K * stride_ak

                if USE_INT8_W8A16:
                    offs_bn = offs_n
                    s0 = tl.load(
                        b_scale_ptr + expert * stride_bse + offs_bn * stride_bsn,
                        mask=offs_bn < N,
                        other=1.0,
                    ).to(tl.float32)
                    acc0 *= s0[None, :]
                    if NUM_N_TILES > 1:
                        offs_bn = BLOCK_N + offs_n
                        s1 = tl.load(
                            b_scale_ptr + expert * stride_bse + offs_bn * stride_bsn,
                            mask=offs_bn < N,
                            other=1.0,
                        ).to(tl.float32)
                        acc1 *= s1[None, :]
                    if NUM_N_TILES > 2:
                        offs_bn = 2 * BLOCK_N + offs_n
                        s2 = tl.load(
                            b_scale_ptr + expert * stride_bse + offs_bn * stride_bsn,
                            mask=offs_bn < N,
                            other=1.0,
                        ).to(tl.float32)
                        acc2 *= s2[None, :]
                    if NUM_N_TILES > 3:
                        offs_bn = 3 * BLOCK_N + offs_n
                        s3 = tl.load(
                            b_scale_ptr + expert * stride_bse + offs_bn * stride_bsn,
                            mask=offs_bn < N,
                            other=1.0,
                        ).to(tl.float32)
                        acc3 *= s3[None, :]

                if MUL_ROUTED_WEIGHT:
                    if WEIGHT_BY_DEST:
                        w = tl.load(weight_ptr + dest_row, mask=mask_m, other=0.0).to(
                            tl.float32
                        )
                    else:
                        w = tl.load(weight_ptr + offs_m, mask=mask_m, other=0.0).to(
                            tl.float32
                        )
                    acc0 *= w[:, None]
                    if NUM_N_TILES > 1:
                        acc1 *= w[:, None]
                    if NUM_N_TILES > 2:
                        acc2 *= w[:, None]
                    if NUM_N_TILES > 3:
                        acc3 *= w[:, None]

                offs_bn = offs_n
                tl.store(
                    c_ptr
                    + dest_row[:, None] * stride_cm
                    + offs_bn[None, :] * stride_cn,
                    acc0.to(COMPUTE_DTYPE),
                    mask=mask_m[:, None] & (offs_bn[None, :] < N),
                )
                if NUM_N_TILES > 1:
                    offs_bn = BLOCK_N + offs_n
                    tl.store(
                        c_ptr
                        + dest_row[:, None] * stride_cm
                        + offs_bn[None, :] * stride_cn,
                        acc1.to(COMPUTE_DTYPE),
                        mask=mask_m[:, None] & (offs_bn[None, :] < N),
                    )
                if NUM_N_TILES > 2:
                    offs_bn = 2 * BLOCK_N + offs_n
                    tl.store(
                        c_ptr
                        + dest_row[:, None] * stride_cm
                        + offs_bn[None, :] * stride_cn,
                        acc2.to(COMPUTE_DTYPE),
                        mask=mask_m[:, None] & (offs_bn[None, :] < N),
                    )
                if NUM_N_TILES > 3:
                    offs_bn = 3 * BLOCK_N + offs_n
                    tl.store(
                        c_ptr
                        + dest_row[:, None] * stride_cm
                        + offs_bn[None, :] * stride_cn,
                        acc3.to(COMPUTE_DTYPE),
                        mask=mask_m[:, None] & (offs_bn[None, :] < N),
                    )
        seg_id += n_prog


def _pick_grouped_block_n(n: int, cap: int) -> int:
    """Power-of-2 BN only（TX81: tl.arange 长度必须是 2 的幂）。"""
    best = None
    bn = 16
    while bn <= cap:
        tiles = (n + bn - 1) // bn
        if tiles <= 4:
            waste = (tiles * bn - n) / float(tiles * bn)
            cand = (waste, tiles, bn)
            if best is None or cand < best:
                best = cand
        bn *= 2
    return best[2] if best is not None else min(16, cap)


def _floor_pow2(x: int) -> int:
    """TX81 arange 约束：把非法长度收到 ≤x 的最大 2 的幂。"""
    if x <= 1:
        return 1
    p = 1
    while (p << 1) <= x:
        p <<= 1
    return p


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return int(v)


# TX81 hardware shared-memory ceiling (~2.87 MiB). Soft limit leaves headroom
# for Triton prologue / scale vectors / masks beyond the A/B tile estimate.
_TX81_SPM_HARD_LIMIT = 3_014_656
_TX81_SPM_SOFT_LIMIT = 1_600_000


def _estimate_grouped_spm(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a_nbytes: int,
    compute_nbytes: int,
    use_int8: bool,
    num_n_tiles: int = 1,
) -> int:
    """Estimate SPM for moe_compact_grouped_gemm_kernel.

    INT8 path does ``b.to(COMPUTE_DTYPE)``: Triton keeps pipelined A/B in
    compute dtype across ``num_stages``, plus one int8 B staging buffer.
    Multi-N-tile kernels keep ``num_n_tiles`` fp32 accumulators that can spill.
    """
    a_bytes = block_m * block_k * a_nbytes
    # The kernel issues num_n_tiles independent [BLOCK_K, BLOCK_N] B loads per
    # k-step (NUM_N_TILES=2..4). Each occupies its own pipeline buffer, so the
    # B term must scale with num_n_tiles; counting a single tile under-estimated
    # SPM by ~(num_n_tiles-1)*BLOCK_K*BLOCK_N*compute_nbytes, which overflowed
    # the TX81 hard limit on M3.1 (hidden=6144: BN=4096 -> 2 tiles).
    b_compute = num_n_tiles * block_k * block_n * compute_nbytes
    spm = num_stages * (a_bytes + b_compute)
    if use_int8:
        spm += block_k * block_n  # int8 B staging
    # fp32 acc spill budget (conservative; TX81 may park large acc in SPM)
    spm += num_n_tiles * block_m * block_n * 4
    return spm


def _pick_grouped_block_k(
    k: int,
    block_m: int,
    block_n: int,
    num_stages: int,
    spm_limit: int = _TX81_SPM_SOFT_LIMIT,
    a_nbytes: int = 2,
    b_nbytes: int = 2,
    use_int8: bool = False,
    num_n_tiles: int = 1,
) -> int:
    # b_nbytes retained for call-site compat; INT8 uses compute_nbytes=b_nbytes
    # after dequant (caller should pass A.element_size(), not 1).
    compute_nbytes = b_nbytes
    for bk in (512, 256, 128, 64, 32):
        if bk > k:
            continue
        if (
            _estimate_grouped_spm(
                block_m,
                block_n,
                bk,
                num_stages,
                a_nbytes,
                compute_nbytes,
                use_int8,
                num_n_tiles,
            )
            <= spm_limit
        ):
            return bk
    return 32


def _fit_grouped_tiles_to_spm(
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a_nbytes: int,
    compute_nbytes: int,
    use_int8: bool,
    n_cap: int,
    fixed_bk: bool,
) -> tuple[int, int, int, int, int]:
    """Shrink BN / stages / BK until estimated SPM fits TX81 soft limit.

    Returns (block_m, block_n, block_k, num_stages, num_n_tiles).
    """
    # Never allocate BN larger than the next pow2 covering N — BN=4096 with
    # N=2048 was the Qwen3.6-A3B W8A8 OOM trigger.
    max_bn = _floor_pow2(max(N, 1))
    block_n = min(_floor_pow2(max(block_n, 1)), max_bn, _floor_pow2(max(n_cap, 1)))
    block_n = max(block_n, 16) if N >= 16 else max(_floor_pow2(max(N, 1)), 1)
    block_m = _floor_pow2(max(block_m, 1))
    block_k = _floor_pow2(max(block_k, 1))
    num_stages = max(int(num_stages), 1)

    def _tiles(bn: int) -> int:
        return min((N + bn - 1) // bn, 4)

    def _ok(bm: int, bn: int, bk: int, stages: int) -> bool:
        return (
            _estimate_grouped_spm(
                bm, bn, bk, stages, a_nbytes, compute_nbytes, use_int8, _tiles(bn)
            )
            <= _TX81_SPM_SOFT_LIMIT
        )

    # 1) drop stages
    while num_stages > 1 and not _ok(block_m, block_n, block_k, num_stages):
        num_stages -= 1

    # 2) shrink BN (pow2) then re-pick BK if not fixed
    while block_n > 16 and not _ok(block_m, block_n, block_k, num_stages):
        block_n //= 2
        if not fixed_bk:
            block_k = _pick_grouped_block_k(
                K,
                block_m,
                block_n,
                num_stages,
                a_nbytes=a_nbytes,
                b_nbytes=compute_nbytes,
                use_int8=use_int8,
                num_n_tiles=_tiles(block_n),
            )

    # 3) shrink BK
    while block_k > 32 and not _ok(block_m, block_n, block_k, num_stages):
        block_k //= 2

    # 4) last resort: shrink BM
    while block_m > 1 and not _ok(block_m, block_n, block_k, num_stages):
        block_m //= 2

    if not _ok(block_m, block_n, block_k, num_stages):
        logger.warning(
            "grouped GEMM SPM still high after shrink: "
            "BM=%d BN=%d BK=%d stages=%d est=%d limit=%d int8=%s N=%d K=%d",
            block_m,
            block_n,
            block_k,
            num_stages,
            _estimate_grouped_spm(
                block_m,
                block_n,
                block_k,
                num_stages,
                a_nbytes,
                compute_nbytes,
                use_int8,
                _tiles(block_n),
            ),
            _TX81_SPM_SOFT_LIMIT,
            use_int8,
            N,
            K,
        )

    return block_m, block_n, block_k, num_stages, _tiles(block_n)


def _default_grouped_ctas(n_seg: int) -> int:
    # TX81 核数有限：过度 oversubscribe（256）会拖慢胖段 GEMM
    if n_seg <= 16:
        return n_seg
    return min(n_seg, 64)


def _pick_grouped_block_m(avg_rows_per_seg: float) -> int:
    # TX81 cap: BLOCK_M>=32 overflows the 3MB SPM limit for the gemm2 fat-BN
    # shape (acc = NUM_N_TILES*BM*BN*4 scales with BM) AND produces wrong
    # numerics on this Triton backend. Clamp to 16; the OOM-retry in
    # _moe_compact_grouped_gemm is a secondary safety net. Override with
    # FUSED_MOE_G1_BM / FUSED_MOE_G2_BM if ever needed.
    if avg_rows_per_seg <= 2.0:
        return 2
    if avg_rows_per_seg <= 4.0:
        return 4
    if avg_rows_per_seg <= 8.0:
        return 8
    return 16


def _meta_fast_threshold() -> int:
    # 超过该规模不用 Python 扫段；默认走 fast CPU（TX81 上 GPU argsort 反而慢）
    return int(os.environ.get("FUSED_MOE_META_GPU_THRESHOLD", "2048"))


def _pack_compact_meta(
    valid_ids: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    seg_experts: torch.Tensor,
    top_k: int,
    meta_source: str,
) -> dict[str, Any]:
    n_valid = int(valid_ids.numel())
    n_mm = int(seg_starts.numel())
    return {
        "n_post": n_valid,
        "n_valid": n_valid,
        "n_mm": n_mm,
        "avg_rows_per_seg": n_valid / max(n_mm, 1),
        "valid_ids": valid_ids,
        "a_rows_hidden": (valid_ids // top_k).contiguous(),
        "a_rows_flat": valid_ids,
        "seg_starts": seg_starts,
        "seg_ends": seg_ends,
        "seg_experts": seg_experts,
        "meta_source": meta_source,
    }


def _segmentize_sorted_experts(
    flats: torch.Tensor, exps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """exps 已按 expert 排好序 → starts/ends/experts（与 flats 同 device）。"""
    n_valid = int(flats.numel())
    change = torch.empty(n_valid, dtype=torch.bool, device=flats.device)
    change[0] = True
    if n_valid > 1:
        change[1:] = exps[1:] != exps[:-1]
    # bool index 通常比 nonzero 轻
    seg_starts_i64 = torch.arange(n_valid, device=flats.device, dtype=torch.int64)[
        change
    ]
    n_mm = int(seg_starts_i64.numel())
    seg_starts = seg_starts_i64.to(torch.int32)
    seg_ends = torch.empty(n_mm, dtype=torch.int32, device=flats.device)
    if n_mm > 1:
        seg_ends[:-1] = seg_starts[1:]
    seg_ends[-1] = n_valid
    seg_experts = exps[seg_starts_i64].to(torch.int32)
    return seg_starts, seg_ends, seg_experts


def _build_moe_compact_meta_from_topk_cpu_python(
    topk_ids: torch.Tensor, device: torch.device
) -> Optional[dict[str, Any]]:
    """Tiny decode：CPU + Python 扫段。"""
    topk_cpu = topk_ids.detach().to("cpu")
    _M, top_k = topk_cpu.shape
    flat_e = topk_cpu.reshape(-1).to(torch.int64)
    n = int(flat_e.numel())
    if n == 0:
        return None

    flats = torch.arange(n, dtype=torch.int64)
    valid = flat_e >= 0
    if not bool(valid.any()):
        return None
    flats = flats[valid]
    exps = flat_e[valid]
    order = torch.argsort(exps, stable=True)
    flats = flats[order]
    exps = exps[order]

    exps_l = exps.tolist()
    flats_l = flats.tolist()
    n_valid = len(flats_l)
    starts: list[int] = []
    ends: list[int] = []
    experts: list[int] = []
    i = 0
    while i < n_valid:
        e = int(exps_l[i])
        j = i + 1
        while j < n_valid and int(exps_l[j]) == e:
            j += 1
        starts.append(i)
        ends.append(j)
        experts.append(e)
        i = j

    valid_ids = torch.tensor(flats_l, device=device, dtype=torch.int64)
    return _pack_compact_meta(
        valid_ids,
        torch.tensor(starts, device=device, dtype=torch.int32),
        torch.tensor(ends, device=device, dtype=torch.int32),
        torch.tensor(experts, device=device, dtype=torch.int32),
        top_k,
        "topk_ids_cpu",
    )


def _build_moe_compact_meta_from_topk_cpu_fast(
    topk_ids: torch.Tensor, device: torch.device
) -> Optional[dict[str, Any]]:
    """
    Prefill：CPU 向量化 sort/建段 + 一次 H2D。
    19k 行上通常远快于 TX81 上的 device argsort（此前 GPU meta ~45ms）。
    """
    top_k = int(topk_ids.shape[1])
    topk_cpu = topk_ids.detach().to("cpu")
    flat_e = topk_cpu.reshape(-1).to(torch.int64)
    n = int(flat_e.numel())
    if n == 0:
        return None
    flats = torch.arange(n, dtype=torch.int64)
    valid = flat_e >= 0
    if not bool(valid.any()):
        return None
    flats = flats[valid]
    exps = flat_e[valid]
    order = torch.argsort(exps, stable=True)
    flats = flats[order]
    exps = exps[order]
    if int(flats.numel()) == 0:
        return None
    seg_starts, seg_ends, seg_experts = _segmentize_sorted_experts(flats, exps)
    # 合并 H2D
    payload = torch.cat(
        [
            flats.to(torch.int64),
            seg_starts.to(torch.int64),
            seg_ends.to(torch.int64),
            seg_experts.to(torch.int64),
        ]
    ).to(device, non_blocking=False)
    n_valid = int(flats.numel())
    n_mm = int(seg_starts.numel())
    valid_ids = payload[:n_valid]
    off = n_valid
    seg_starts_d = payload[off : off + n_mm].to(torch.int32)
    off += n_mm
    seg_ends_d = payload[off : off + n_mm].to(torch.int32)
    off += n_mm
    seg_experts_d = payload[off : off + n_mm].to(torch.int32)
    return _pack_compact_meta(
        valid_ids,
        seg_starts_d,
        seg_ends_d,
        seg_experts_d,
        top_k,
        "topk_ids_cpu_fast",
    )


def _build_moe_compact_meta_from_topk_gpu(
    topk_ids: torch.Tensor, device: torch.device
) -> Optional[dict[str, Any]]:
    """Device 向量化路径（可用 FUSED_MOE_META_DEVICE=gpu 强制）。"""
    top_k = int(topk_ids.shape[1])
    flat_e = topk_ids.detach().reshape(-1).to(device=device, dtype=torch.int64)
    n = int(flat_e.numel())
    if n == 0:
        return None
    flats = torch.arange(n, device=device, dtype=torch.int64)
    valid = flat_e >= 0
    if not bool(valid.any().item()):
        return None
    flats = flats[valid]
    exps = flat_e[valid]
    order = torch.argsort(exps, stable=True)
    flats = flats[order]
    exps = exps[order]
    if int(flats.numel()) == 0:
        return None
    seg_starts, seg_ends, seg_experts = _segmentize_sorted_experts(flats, exps)
    return _pack_compact_meta(
        flats.contiguous(),
        seg_starts.contiguous(),
        seg_ends.contiguous(),
        seg_experts.contiguous(),
        top_k,
        "topk_ids_gpu",
    )


def _build_moe_compact_meta_from_topk(
    topk_ids: torch.Tensor, device: torch.device
) -> Optional[dict[str, Any]]:
    """
    Env:
      FUSED_MOE_META_GPU_THRESHOLD  默认 2048，超过则走 fast 路径
      FUSED_MOE_META_DEVICE=cpu|gpu|cpu_fast  强制指定（默认 auto）
    """
    n = int(topk_ids.numel())
    mode = os.environ.get("FUSED_MOE_META_DEVICE", "auto").lower()
    if mode == "gpu":
        return _build_moe_compact_meta_from_topk_gpu(topk_ids, device)
    if mode == "cpu":
        return _build_moe_compact_meta_from_topk_cpu_python(topk_ids, device)
    if mode == "cpu_fast":
        return _build_moe_compact_meta_from_topk_cpu_fast(topk_ids, device)
    # auto
    if n >= _meta_fast_threshold() and device.type != "cpu":
        return _build_moe_compact_meta_from_topk_cpu_fast(topk_ids, device)
    return _build_moe_compact_meta_from_topk_cpu_python(topk_ids, device)


def _moe_compact_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C_out: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    seg_experts: torch.Tensor,
    config: dict[str, Any],
    a_rows: Optional[torch.Tensor] = None,
    router_weights: Optional[torch.Tensor] = None,
    dest_ids: Optional[torch.Tensor] = None,
    weight_by_dest: bool = False,
    B_scale: Optional[torch.Tensor] = None,
) -> None:
    E, N, K = B.size(0), B.size(1), B.size(2)
    num_segments = int(seg_starts.numel())
    if num_segments == 0:
        return

    use_int8 = B_scale is not None and B.dtype == torch.int8
    if use_int8:
        if B_scale.numel() != E * N:
            raise ValueError(
                f"INT8 B_scale numel {B_scale.numel()} != E*N={E * N} "
                f"(got shape {tuple(B_scale.shape)})"
            )
        B_scale = B_scale.reshape(E, N).contiguous()

    if A.dtype == torch.bfloat16:
        compute_dtype = tl.bfloat16
    elif A.dtype == torch.float16:
        compute_dtype = tl.float16
    else:
        compute_dtype = tl.float32

    a_nbytes = A.element_size()
    # After INT8→compute dequant, B tiles occupy compute-dtype SPM (not 1 byte).
    compute_nbytes = A.element_size()

    block_m = int(config.get("GROUPED_BLOCK_M", 16))
    n_cap = int(config.get("GROUPED_BLOCK_N_CAP", 2048))
    num_stages = int(config.get("num_stages", 2))
    num_warps = int(config.get("num_warps", 1))
    # Cap BN by actual N so Hy3's BN=4096 default cannot OOM on N=2048 (Qwen A3B).
    n_cap = min(n_cap, max(_floor_pow2(max(N, 1)), 16))
    block_n = int(config.get("GROUPED_BLOCK_N", 0)) or _pick_grouped_block_n(N, n_cap)
    block_n = min(_floor_pow2(block_n), _floor_pow2(max(N, 1)))
    fixed_bk = bool(int(config.get("GROUPED_BLOCK_K", 0)))
    if fixed_bk:
        block_k = _floor_pow2(int(config["GROUPED_BLOCK_K"]))
    else:
        block_k = _pick_grouped_block_k(
            K,
            block_m,
            block_n,
            num_stages,
            a_nbytes=a_nbytes,
            b_nbytes=compute_nbytes,
            use_int8=use_int8,
            num_n_tiles=min((N + block_n - 1) // block_n, 4),
        )

    num_n_tiles = (N + block_n - 1) // block_n
    if num_n_tiles > 4:
        block_n = _pick_grouped_block_n(N, max(n_cap, (N + 3) // 4))
        block_n = min(_floor_pow2(block_n), _floor_pow2(max(N, 1)))
        num_n_tiles = min((N + block_n - 1) // block_n, 4)
        if not fixed_bk:
            block_k = _pick_grouped_block_k(
                K,
                block_m,
                block_n,
                num_stages,
                a_nbytes=a_nbytes,
                b_nbytes=compute_nbytes,
                use_int8=use_int8,
                num_n_tiles=num_n_tiles,
            )

    block_m, block_n, block_k, num_stages, num_n_tiles = _fit_grouped_tiles_to_spm(
        N,
        K,
        block_m,
        block_n,
        block_k,
        num_stages,
        a_nbytes,
        compute_nbytes,
        use_int8,
        n_cap,
        fixed_bk,
    )

    config["GROUPED_BLOCK_N_RESOLVED"] = block_n
    config["GROUPED_BLOCK_K_RESOLVED"] = block_k
    config["GROUPED_NUM_N_TILES"] = num_n_tiles
    config["GROUPED_NUM_STAGES_RESOLVED"] = num_stages
    config["GROUPED_USE_INT8"] = use_int8

    gather_a = a_rows is not None
    mul_w = router_weights is not None
    write_direct = dest_ids is not None
    if not gather_a:
        a_rows = seg_starts.new_empty((1,), dtype=torch.int64)
    if not mul_w:
        router_weights = A.new_empty((1,))
        weight_by_dest = False
    if not write_direct:
        dest_ids = seg_starts.new_empty((1,))
    if not use_int8:
        B_scale = A.new_empty((1,))
        stride_bse = 0
        stride_bsn = 0
    else:
        stride_bse = B_scale.stride(0)
        stride_bsn = B_scale.stride(1)

    if os.environ.get("FUSED_MOE_GROUPED_CTAS"):
        n_ctas = min(num_segments, int(os.environ["FUSED_MOE_GROUPED_CTAS"]))
    else:
        n_ctas = _default_grouped_ctas(num_segments)
    n_ctas = max(n_ctas, 1)
    config["GROUPED_NUM_CTAS"] = n_ctas

    # --- Adaptive SPM-safe launch (safety net) -----------------------------
    # Primary SPM control is the BLOCK_M<=16 cap in _pick_grouped_block_m; this
    # loop is a defensive fallback for any residual config that still exceeds the
    # TX81 shared-memory limit: on OutOfResources shrink BLOCK_N (never below the
    # smallest pow2 still covering all N with NUM_N_TILES<=4), then num_stages,
    # then BLOCK_K, then BLOCK_M.
    def _ceil_pow2(x):
        return 1 << (max(int(x), 1) - 1).bit_length()

    _bn_floor = max(16, min(_floor_pow2(max(N, 1)), _ceil_pow2((N + 3) // 4)))
    _bn, _bk, _st = block_n, block_k, num_stages
    _nt = min((N + _bn - 1) // _bn, 4)

    def _launch(bn, bk, st, nt):
        moe_compact_grouped_gemm_kernel[(n_ctas,)](
            A,
            B,
            C_out,
            B_scale,
            seg_starts,
            seg_ends,
            seg_experts,
            a_rows,
            router_weights,
            dest_ids,
            N,
            K,
            num_segments,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C_out.stride(-2) if C_out.ndim >= 2 else C_out.stride(0),
            C_out.stride(-1),
            stride_bse,
            stride_bsn,
            BLOCK_M=block_m,
            BLOCK_N=bn,
            BLOCK_K=bk,
            NUM_N_TILES=nt,
            COMPUTE_DTYPE=compute_dtype,
            GATHER_A=gather_a,
            MUL_ROUTED_WEIGHT=mul_w,
            WEIGHT_BY_DEST=bool(weight_by_dest),
            WRITE_DIRECT=write_direct,
            USE_INT8_W8A16=use_int8,
            num_warps=num_warps,
            num_stages=st,
        )

    while True:
        try:
            _launch(_bn, _bk, _st, _nt)
            break
        except triton.runtime.errors.OutOfResources:
            if _bn > _bn_floor:
                _bn = max(_bn_floor, _bn // 2)
                _nt = min((N + _bn - 1) // _bn, 4)
                continue
            if _st > 1:
                _st -= 1
                continue
            if _bk > 32:
                _bk //= 2
                continue
            if block_m > 1:
                block_m //= 2
                continue
            raise

    config["GROUPED_BLOCK_N_RESOLVED"] = _bn
    config["GROUPED_BLOCK_K_RESOLVED"] = _bk
    config["GROUPED_NUM_N_TILES"] = _nt
    config["GROUPED_NUM_STAGES_RESOLVED"] = _st
    # Cache the working tiles so later calls start from a known-good config.
    config["GROUPED_BLOCK_N"] = _bn
    config["GROUPED_BLOCK_N_CAP"] = _bn
    if _st != num_stages:
        config["num_stages"] = _st
    if _bk != block_k:
        config["GROUPED_BLOCK_K"] = _bk


@triton.jit
def _moe_sum_topk_kernel(
    inp_ptr,
    out_ptr,
    num_tokens,
    topk,
    hidden,
    stride_it,
    stride_ik,
    stride_ih,
    stride_ot,
    stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """2D tile reduce: [M, topk, K] -> [M, K]. No autotune (TX81)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < num_tokens
    mask_n = offs_n < hidden
    mask = mask_m[:, None] & mask_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(topk):
        ptrs = (
            inp_ptr
            + offs_m[:, None] * stride_it
            + k * stride_ik
            + offs_n[None, :] * stride_ih
        )
        acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + offs_m[:, None] * stride_ot + offs_n[None, :] * stride_oh,
        acc.to(COMPUTE_DTYPE),
        mask=mask,
    )


def _moe_sum_topk(cache: torch.Tensor, out: torch.Tensor) -> None:
    """cache [M, topk, K] -> out [M, K]. Avoid generic moe_sum autotune (~1s on TX81)."""
    num_tokens, topk, hidden = cache.shape
    if num_tokens == 0:
        return
    if cache.dtype == torch.bfloat16:
        compute_dtype = tl.bfloat16
    elif cache.dtype == torch.float16:
        compute_dtype = tl.float16
    else:
        compute_dtype = tl.float32
    block_m = 16
    block_n = 256
    grid = (
        triton.cdiv(num_tokens, block_m),
        triton.cdiv(hidden, block_n),
    )
    _moe_sum_topk_kernel[grid](
        cache,
        out,
        num_tokens,
        topk,
        hidden,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        COMPUTE_DTYPE=compute_dtype,
        num_warps=1,
        num_stages=1,
    )


@functools.lru_cache(maxsize=1)
def get_embedded_moe_configs():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "utils", "configs", "fused_moe_config.yaml"
    )
    if not os.path.exists(config_path):
        return {}, {}
    with open(config_path, "r") as f:
        # JSON keys are strings, values are dicts where keys are M and values are configs
        data = yaml.safe_load(f)

        fallback = data.get("_FALLBACK", {})

        # We need to convert the innermost keys (which are stringified integers for M) back to integers.
        # Ensure we map the lists back to config dicts.
        keys_order = [
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        ]
        parsed_data = {}
        for dev, configs in data.items():
            if dev == "_FALLBACK":
                continue
            parsed_data[dev] = {}
            for k, m_dict in configs.items():
                parsed_dict = {}
                for m, v in m_dict.items():
                    if isinstance(v, list):
                        parsed_dict[int(m)] = dict(zip(keys_order, v))
                    else:
                        parsed_dict[int(m)] = v
                parsed_data[dev][k] = parsed_dict

        return parsed_data, fallback


def dequant_mxfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP4 tensor via quark.torch.kernel.mx.dq_mxfp4."""
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("amd-quark is required for MX-FP4") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def dequant_mxfp6(
    x: torch.Tensor,
    scale: torch.Tensor,
    float_dtype: torch.dtype,
    quant_dtype: str,
) -> torch.Tensor:
    """Dequantize MXFP6 tensor via quark hw_emulation."""
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group,
        )
        from quark.torch.utils.pack import create_pack_method
    except ImportError as err:
        raise ImportError("amd-quark is required for MX-FP6") from err

    pack_method = create_pack_method(None, dtype=quant_dtype)
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2 ** (scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    return dequantize_fp4_fp6_per_group(
        unpacked_x,
        scale,
        axis=-1,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    ).to(float_dtype)


# Activation quantization helpers


@functools.lru_cache(maxsize=1)
def _get_device_name() -> str:
    """Return the normalised device name (spaces replaced by underscores).

    Matches the naming convention used by vLLM for its per-device config files.
    H800 falls back to H100_80GB_HBM3 (same SM 9.0 architecture).
    """
    try:
        name = torch_device_fn.get_device_name().replace(" ", "_")
    except AttributeError:
        name = device.name
    # Normalise the H200 product family to a single key, following vLLM.
    if "H200" in name.split("_"):
        name = "NVIDIA_H200"
    # H800 has the same SM 9.0 as H100; use H100 configs as fallback.
    embedded_configs, fallback_mapping = get_embedded_moe_configs()
    if name in embedded_configs:
        return name
    # Fallback mapping for devices whose tuning profiles are equivalent.
    fallback = fallback_mapping.get(name)
    if fallback and fallback in embedded_configs:
        logger.info("Device %s not in config table, falling back to %s", name, fallback)
        return fallback
    return name


@functools.lru_cache(maxsize=1)
def _is_h20() -> bool:
    """Whether the current device is an NVIDIA H20 (gates the FP8 MoE swap_ab default)."""
    return "H20" in _get_device_name().split("_")


def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    Looks up pre-tuned configs from the embedded table (ported from vLLM)
    for the current GPU device. Returns None if no matching config is found.
    """
    device_name = _get_device_name()
    embedded_configs, _ = get_embedded_moe_configs()
    device_table = embedded_configs.get(device_name)
    if device_table is None:
        logger.debug(
            "No embedded MoE configs for device %s. Will use default config.",
            device_name,
        )
        return None

    _block_n = block_n if block_n else 0
    _block_k = block_k if block_k else 0
    key = f"{E},{N},{dtype},{_block_n},{_block_k}"
    configs = device_table.get(key)
    if configs is not None:
        logger.debug(
            "Using embedded MoE config for device=%s, key=%s", device_name, key
        )
        return configs
    logger.debug(
        "No embedded MoE config for device=%s, key=%s. Will use default config.",
        device_name,
        key,
    )
    return None


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    E: int,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
    return_is_embedded: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], bool]:
    if gemm_stage not in ("gemm1", "gemm2"):
        raise ValueError(f"Unsupported MoE GEMM stage: {gemm_stage}")
    _, _, config_n = w2_shape
    if dtype == "int4_w4a16":
        config_n = config_n * 2
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    configs = get_moe_configs(E, config_n, dtype, block_n, block_k)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))].copy()
        is_embedded = True
    else:
        if gemm_stage == "gemm1":
            _, N, K = w1_shape
        else:
            _, N, K = w2_shape
        config = get_default_config(
            M,
            E,
            N,
            K,
            top_k,
            dtype,
            block_shape,
            gemm_stage=gemm_stage,
            enable_gemm_fast_path=enable_gemm_fast_path,
        )
        is_embedded = False
    if return_is_embedded:
        return config, is_embedded
    return config


def _get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    ocp_mx_scheme: str | None,
) -> None | torch.dtype | str:
    """Map quantization flags to the corresponding dtype."""
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif ocp_mx_scheme == "w_mxfp4_a_mxfp4":
        return "mxfp4"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e3m2", "w_mxfp6_e3m2_a_mxfp6_e3m2"}:
        return "mxfp6_e3m2"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e2m3", "w_mxfp6_e2m3_a_mxfp6_e2m3"}:
        return "mxfp6_e2m3"
    elif ocp_mx_scheme in {"w_mxfp4", "w_mxfp6_e3m2", "w_mxfp6_e2m3"}:
        return torch.bfloat16
    elif ocp_mx_scheme in {"w_mxfp4_a_fp8", "w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"}:
        return torch.float8_e4m3fn

    return None


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        return {}
    if not use_moe_wna16_cuda:
        if num_valid_tokens // real_top_k == 1:
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            block_size_n = 1024

        block_size_k = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    """Default Triton config for fused MoE kernel.

    Heuristic selection aligned with vLLM v0.17.0 defaults, tuned on H20/H100.
    Key insight: for high-expert-count MoE (e.g. DeepSeek-V3 E=256), each
    expert sees very few tokens, so small BLOCK_SIZE_M (16) is critical.
    """
    is_fp8_blockwise = dtype == "fp8_w8a8" and block_shape is not None
    if gemm_stage not in ("gemm1", "gemm2"):
        raise ValueError(f"Unsupported MoE GEMM stage: {gemm_stage}")

    if is_fp8_blockwise:
        avg_tokens_per_expert = M * max(topk, 1) // max(E, 1)
        is_large_m = M >= 16384
        if avg_tokens_per_expert <= 16:
            block_m = 16
        elif avg_tokens_per_expert <= 32:
            block_m = 32
        elif avg_tokens_per_expert <= 64 or not is_large_m:
            block_m = 64
        else:
            block_m = 128

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 8 if (is_large_m and avg_tokens_per_expert > 16) else 1,
            "num_warps": 8 if (is_large_m and block_m > 32) else 4,
            "num_stages": 4 if M >= 1024 else 3,
            # H20: swap_ab puts the larger N dim on the wgmma M-axis, a measured
            # win on H20 for small token tiles (block_m<=32) but a regression for
            # large tiles (e.g. 16k/32k-token prefill), so gate on both.
            "SWAP_AB": _is_h20() and block_m <= 32,
        }
    elif dtype is None or dtype in _PLAIN_HALF_CONFIG_DTYPES:
        # Routed rows per expert drives block_m.  Each token contributes topk
        # rows to the expert-sorted GEMM input, so M * topk / E is the relevant
        # density for high-expert-count MoE routing.
        routed_tokens_per_expert = M * max(topk, 1) // max(E, 1)
        tokens_per_expert = M // max(E, 1)

        if routed_tokens_per_expert <= 16:
            block_m = 16
        elif routed_tokens_per_expert <= 64:
            block_m = 64
        else:
            block_m = 128

        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        block_k = 128 if M <= 64 else 64

        if N >= 4096:
            block_n = 128 if M <= 128 else 256
        else:
            block_n = 64 if M <= 64 else 128

        can_use_gemm_fast_path = (
            enable_gemm_fast_path
            and M >= MOE_GEMM_TUNING_MIN_TOKENS
            and block_m == _HALF_GEMM_TILE_M
            and block_k == _HALF_GEMM_TILE_K
        )

        use_gemm2_fast_path = (
            gemm_stage == "gemm2"
            and can_use_gemm_fast_path
            and N % _HALF_GEMM2_TILE_N == 0
        )
        use_gemm1_fast_path = (
            gemm_stage == "gemm1" and can_use_gemm_fast_path and N % block_n == 0
        )

        if gemm_stage == "gemm2" and enable_gemm_fast_path:
            block_n = (
                _HALF_GEMM2_TILE_N if use_gemm2_fast_path else (64 if M <= 64 else 128)
            )

        # Prefer 4 warps for small tiles; only use 8 for large M
        num_warps = 4 if M <= 128 else 8
        num_stages = 3

        if use_gemm1_fast_path:
            group_m = 1
            num_stages = 4
        elif use_gemm2_fast_path:
            group_m = 2
            num_stages = 4

        smem_per_stage = (block_m * block_k + block_k * block_n) * 2
        while num_stages > 2 and smem_per_stage * num_stages > 200_000:
            num_stages -= 1

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        if use_gemm1_fast_path:
            config["PAIR_GATE_UP_DOT"] = True
    else:
        tokens_per_expert = M // max(E, 1)

        if tokens_per_expert <= 2:
            block_m = 16
        elif tokens_per_expert <= 4:
            block_m = 32
        elif tokens_per_expert <= 16:
            block_m = 64
        else:
            block_m = 128

        # Tile sizing
        if N >= 4096:
            block_n = 128 if M <= 128 else 256
        elif N >= 1024:
            block_n = 64 if M <= 64 else 128
        else:
            block_n = 64 if M <= 64 else 128

        if dtype == "fp8_w8a8":
            block_k = 128
        elif M <= 64:
            block_k = 128
        else:
            block_k = 64

        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        # Prefer 4 warps for small tiles; only use 8 for large M
        num_warps = 4 if M <= 128 else 8
        num_stages = 3

        smem_per_stage = (block_m * block_k + block_k * block_n) * 2
        while num_stages > 2 and smem_per_stage * num_stages > 200_000:
            num_stages -= 1

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def _get_config_dtype_str(
    dtype: Optional[torch.dtype] = None,
    use_fp8_w8a8: bool = False,
    use_fp8_w8a16: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    """Return dtype suffix for MoE JSON filenames (vLLM-compatible).

    Unquantized fp16/bf16 → None (filename omits dtype).
    Quantized paths → e.g. int8_w8a16, fp8_w8a8.
    """
    if use_fp8_w8a8:
        return "fp8_w8a8"
    if use_fp8_w8a16:
        return "fp8_w8a16"
    if use_int8_w8a8:
        return "int8_w8a8"
    if use_int8_w8a16:
        return "int8_w8a16"
    if use_int4_w4a16:
        return "int4_w4a16"
    if ocp_mx_scheme is not None:
        return None
    if dtype == torch.float:
        return "float32"
    return None


# MoE activation enum


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated: gate * activation(up), input [..., 2*d] -> output [..., d]
    SILU = "silu"
    GELU = "gelu"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
    SWIGLUSTEP = "swiglustep"

    # Non-gated: input [..., d] -> output [..., d]
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")

    def without_mul(self) -> "MoEActivation":
        """Return the non-gated variant."""
        _without_mul: dict[MoEActivation, MoEActivation] = {
            MoEActivation.SILU: MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU: MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2: MoEActivation.RELU2_NO_MUL,
        }
        return _without_mul.get(self, self)

    @classmethod
    def from_str(cls, s: str) -> "MoEActivation":
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")

    @staticmethod
    def adjust_N_for_activation(N: int, activation: "MoEActivation") -> int:
        """Return N for non-gated, N // 2 for gated activations."""
        return N if not activation.is_gated else N // 2


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    clamp_limit: float | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
) -> torch.Tensor:
    """Apply MoE activation (pure PyTorch / FlagGems Triton)."""
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation.value} expects 2x ratio: "
            f"{output.size(-1) * 2} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) == input.size(-1), (
            f"{activation.value} expects equal sizes: "
            f"{output.size(-1)} vs {input.size(-1)}"
        )

    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI):
        N = output.size(-1)
        x, y = input[:, :N], input[:, N:]
        if clamp_limit is None:
            silu_and_mul_out(x, y, output)
        else:
            silu_and_mul_with_clamp_out(x, y, output, clamp_limit)
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        if gemm1_alpha is None or gemm1_beta is None or clamp_limit is None:
            raise ValueError(
                "swigluoai_uninterleave requires gemm1_alpha/gemm1_beta/clamp_limit"
            )
        alpha = float(gemm1_alpha)
        beta = float(gemm1_beta)
        limit = float(clamp_limit)
        N = output.size(-1)
        gate = torch.clamp(input[..., :N], max=limit)
        up = torch.clamp(input[..., N:], min=-limit, max=limit)
        output.copy_(gate * torch.sigmoid(alpha * gate) * (up + beta))
    elif activation == MoEActivation.GELU:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(F.gelu(gate) * up)
    elif activation == MoEActivation.SWIGLUSTEP:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(torch.sigmoid(gate) * up)
    elif activation == MoEActivation.RELU2:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(F.relu(gate).square() * up)

    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 quantization: per-tensor, per-token, or block-wise."""
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = finfo.min
    eps = 1e-10

    if block_shape is not None:
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        if A.ndim == 2 and A.stride(-1) == 1:
            from flag_gems.ops.per_token_group_quant_fp8 import (
                per_token_group_quant_fp8,
            )

            return per_token_group_quant_fp8(
                A,
                group_size=block_k,
                eps=eps,
                dtype=fp8_dtype,
                column_major_scales=False,
                scale_ue8m0=False,
            )
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = (
            A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        )
        scale = amax / fp8_max
        A_q = (A_groups.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / fp8_max
        min_scale = torch.tensor(
            1.0 / (fp8_max * 512.0), dtype=torch.float32, device=A.device
        )
        scale = scale.clamp(min=min_scale)
        A_q = (A_flat.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        if A_scale is not None:
            scale = (
                A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
            )
            A_q = (A.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, A_scale
        else:
            amax = A.abs().amax().clamp(min=eps).to(torch.float32)
            scale = amax / fp8_max
            iscale = 1.0 / scale
            A_q = (A.float() * iscale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, scale.view(1)


def _int8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 quantization: per-tensor, per-token, or block-wise."""
    iinfo = torch.iinfo(torch.int8)
    int8_max = iinfo.max
    int8_min = iinfo.min
    eps = 1e-10

    if block_shape is not None:
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = (
            A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        )
        scale = amax / int8_max
        A_q = (
            (A_groups.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        )
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / int8_max
        A_q = (A_flat.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        assert A_scale is not None, "int8 per-tensor requires A_scale"
        scale = A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
        A_q = (A.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        return A_q, A_scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
    ocp_mx_scheme: str | None = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantize MoE input activations before GEMM."""
    if ocp_mx_scheme is not None:
        if ocp_mx_scheme in {"w_mxfp4", "w_mxfp4_a_mxfp4"}:
            pass
        elif ocp_mx_scheme.endswith("a_fp8"):
            qA, qA_scale = _fp8_quantize(A, A_scale, per_act_token=False)
            A = (qA.float() * qA_scale.float()).to(A.dtype)
            return A, None

    if quant_dtype is None:
        return A, A_scale
    elif quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    else:
        return A, A_scale


def _ensure_block_size_k_divisible(
    size_k: int, block_size_k: int, group_size: int
) -> int:
    """Find largest block_size_k that divides size_k and is divisible by group_size."""
    if size_k % block_size_k == 0 and block_size_k % group_size == 0:
        return block_size_k

    max_search = min(block_size_k, size_k)
    start = (max_search // group_size) * group_size
    for candidate in range(start, group_size - 1, -group_size):
        if size_k % candidate == 0:
            return candidate

    if size_k % group_size == 0:
        return group_size

    return size_k


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """Fused MoE kernel for GPTQ/AWQ (WNA16) quantized weights."""
    # Map pid to C block (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for first blocks of A and B
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Cast to int64 to prevent overflow in stride*offset products
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # Accumulate C block in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # Write back output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_K: tl.constexpr,
    N_DIVISIBLE_BY_BLOCK_N: tl.constexpr,
    PAIR_GATE_UP_DOT: tl.constexpr,
    DIRECT_SUM: tl.constexpr,
    OUT_TOP_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
):
    """Fused MoE kernel: token × expert GEMM with quantization support and optional SiLU fusion."""
    # Map pid to C block (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    # Adjust N for FUSE_SILU. If fused, the actual output dimension is N // 2
    N_out = N // 2 if FUSE_SILU else N
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for first blocks of A and B
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    if not naive_block_assignment:
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    offs_token = offs_token.to(tl.int64)  # prevent int32 overflow

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if not N_DIVISIBLE_BY_BLOCK_N:
        offs_bn = offs_bn % N_out
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)

    if FUSE_SILU and PAIR_GATE_UP_DOT:
        if off_experts == -1:
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N_out,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
            return

        offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
        offs_pair_bn = tl.where(
            offs_pair < BLOCK_SIZE_N,
            pid_n * BLOCK_SIZE_N + offs_pair,
            N_out + pid_n * BLOCK_SIZE_N + offs_pair - BLOCK_SIZE_N,
        )
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        b_pair_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_pair_bn[None, :] * stride_bn)
        )
        pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                if N_DIVISIBLE_BY_BLOCK_N:
                    b_pair = tl.load(b_pair_ptrs)
                else:
                    b_pair = tl.load(
                        b_pair_ptrs, mask=offs_pair_bn[None, :] < N, other=0.0
                    )
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b_pair = tl.load(
                    b_pair_ptrs,
                    mask=(offs_k[:, None] < k_remaining) & (offs_pair_bn[None, :] < N),
                    other=0.0,
                )
            pair_acc += tl.dot(a, b_pair)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_pair_ptrs += BLOCK_SIZE_K * stride_bk

        if HAS_BIAS:
            pair_bias_ptrs = (
                b_bias_ptr + off_experts * stride_bbe + (offs_pair_bn * stride_bbn)
            )
            pair_bias = tl.load(pair_bias_ptrs, mask=offs_pair_bn < N, other=0.0)
            pair_acc += pair_bias[None, :]

        gate_up = tl.trans(
            tl.reshape(pair_acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N)),
            (0, 2, 1),
        )
        gate_acc, up_acc = tl.split(gate_up)
        gate_sig = tl.sigmoid(gate_acc)
        accumulator = (
            gate_acc.to(compute_type)
            * gate_sig.to(compute_type)
            * up_acc.to(compute_type)
        )

    elif FUSE_SILU:
        offs_bn_gate = offs_bn
        offs_bn_up = offs_bn + N_out

        b_expert_base = b_ptr + off_experts * stride_be
        b_ptrs_gate = b_expert_base + (
            offs_k[:, None] * stride_bk + offs_bn_gate[None, :] * stride_bn
        )
        b_ptrs_up = b_expert_base + (
            offs_k[:, None] * stride_bk + offs_bn_up[None, :] * stride_bn
        )

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:  # block-wise
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                # Use scalar scale load for hardware broadcast when block size fits within quantization group.
                if BLOCK_SIZE_N <= group_n:
                    offs_bsn_gate_idx = (pid_n * BLOCK_SIZE_N) % N_out // group_n
                    offs_bsn_up_idx = (
                        (pid_n * BLOCK_SIZE_N) % N_out + N_out
                    ) // group_n
                else:
                    offs_bsn_gate_idx = offs_bn_gate // group_n
                    offs_bsn_up_idx = offs_bn_up // group_n
                b_scale_gate_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bsn_gate_idx * stride_bsn
                )
                b_scale_up_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bsn_up_idx * stride_bsn
                )
            elif per_channel_quant:  # channel-wise
                b_scale_gate_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn_gate[None, :] * stride_bsn
                )
                b_scale_gate = tl.load(b_scale_gate_ptrs)
                b_scale_up_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn_up[None, :] * stride_bsn
                )
                b_scale_up = tl.load(b_scale_up_ptrs)
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:  # tensor-wise
                a_scale = tl.load(a_scale_ptr)
                b_scale_gate = tl.load(b_scale_ptr + off_experts)
                b_scale_up = b_scale_gate

        # Pass 1: Sequential execution of gate projection to minimize peak register pressure.
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Eliminate masking overhead when K is perfectly aligned with BLOCK_SIZE_K.
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b_gate = tl.load(b_ptrs_gate)
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b_gate = tl.load(
                    b_ptrs_gate, mask=offs_k[:, None] < k_remaining, other=0.0
                )

            if use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    b_scale_val = tl.load(b_scale_gate_ptrs + offs_ks * stride_bsk)

                    # Pre-compute combined scale to reduce arithmetic overhead via the associative property.
                    if BLOCK_SIZE_N <= group_n:
                        combined_scale = a_scale[:, None] * b_scale_val
                    else:
                        combined_scale = a_scale[:, None] * b_scale_val[None, :]
                    acc_gate += tl.dot(a, b_gate) * combined_scale
                else:
                    if use_fp8_w8a8:
                        acc_gate = tl.dot(a, b_gate, acc=acc_gate)
                    else:
                        acc_gate += tl.dot(a, b_gate)
            else:
                acc_gate += tl.dot(a, b_gate)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs_gate += BLOCK_SIZE_K * stride_bk

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                pass
            elif per_channel_quant:
                acc_gate = acc_gate * a_scale * b_scale_gate
            else:
                acc_gate = acc_gate * a_scale * b_scale_gate

        # Pass 2: Sequential up projection; operand A is reloaded with high L1 hit rate.
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Apply mask elimination during the up projection stage.
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b_up = tl.load(b_ptrs_up)
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b_up = tl.load(b_ptrs_up, mask=offs_k[:, None] < k_remaining, other=0.0)

            if use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    b_scale_val = tl.load(b_scale_up_ptrs + offs_ks * stride_bsk)

                    # Apply pre-computed scale merging to reduce multiplication overhead.
                    if BLOCK_SIZE_N <= group_n:
                        combined_scale = a_scale[:, None] * b_scale_val
                    else:
                        combined_scale = a_scale[:, None] * b_scale_val[None, :]
                    acc_up += tl.dot(a, b_up) * combined_scale
                else:
                    if use_fp8_w8a8:
                        acc_up = tl.dot(a, b_up, acc=acc_up)
                    else:
                        acc_up += tl.dot(a, b_up)
            else:
                acc_up += tl.dot(a, b_up)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs_up += BLOCK_SIZE_K * stride_bk

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                pass
            elif per_channel_quant:
                acc_up = acc_up * a_scale * b_scale_up
            else:
                acc_up = acc_up * a_scale * b_scale_up

        # SiLU activation fusion
        accumulator = tl.fdiv(acc_gate, (1.0 + tl.exp(-acc_gate))) * acc_up

    else:
        if off_experts == -1:
            # Expert not in current EP rank, write zeros
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N_out,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
            return
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        if use_int8_w8a16:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:  # block-wise
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                # Use scalar scale load for hardware broadcast when block size fits within quantization group.
                if BLOCK_SIZE_N <= group_n:
                    offs_bsn = (pid_n * BLOCK_SIZE_N) % N_out // group_n
                else:
                    offs_bsn = offs_bn // group_n
                b_scale_ptrs = (
                    b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
                )
            elif per_channel_quant:  # channel-wise
                b_scale_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn[None, :] * stride_bsn
                )
                b_scale = tl.load(b_scale_ptrs)
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:  # tensor-wise
                a_scale = tl.load(a_scale_ptr)
                b_scale = tl.load(b_scale_ptr + off_experts)

        if HAS_BIAS:
            bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
            bias = tl.load(bias_ptrs, mask=(offs_bn < N_out), other=0.0)

        # Accumulate C block in fp32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if SWAP_AB:
            accumulator_nm = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Eliminate masking overhead when K is perfectly aligned with BLOCK_SIZE_K.
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

            if use_int8_w8a16:
                # In-kernel dequant (zp=0): load INT8 tile → cast to compute
                # dtype → bf16/fp16 dot. Per-channel scale applied after K-loop.
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    if SWAP_AB:
                        b_scale_val = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                        if BLOCK_SIZE_N <= group_n:
                            combined_scale_nm = b_scale_val * a_scale[None, :]
                        else:
                            combined_scale_nm = b_scale_val[:, None] * a_scale[None, :]
                        accumulator_nm += (
                            tl.dot(tl.trans(b), tl.trans(a)) * combined_scale_nm
                        )
                    else:
                        b_scale_val = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                        # Pre-compute combined scale to reduce arithmetic overhead via the associative property.
                        if BLOCK_SIZE_N <= group_n:
                            combined_scale = a_scale[:, None] * b_scale_val
                        else:
                            combined_scale = a_scale[:, None] * b_scale_val[None, :]
                        accumulator += tl.dot(a, b) * combined_scale
                else:
                    if use_fp8_w8a8:
                        if SWAP_AB:
                            accumulator_nm = tl.dot(
                                tl.trans(b), tl.trans(a), acc=accumulator_nm
                            )
                        else:
                            accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        if SWAP_AB:
                            accumulator_nm += tl.dot(tl.trans(b), tl.trans(a))
                        else:
                            accumulator += tl.dot(a, b)
            else:
                if SWAP_AB:
                    accumulator_nm += tl.dot(tl.trans(b), tl.trans(a))
                else:
                    accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if SWAP_AB:
            accumulator = tl.trans(accumulator_nm)

        # Dequantization
        if use_int8_w8a16:
            accumulator = accumulator * b_scale
        elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
            accumulator = accumulator * a_scale * b_scale

        if HAS_BIAS:
            accumulator += bias[None, :]

    # Router weight multiplication (must be in fp32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if DIRECT_SUM:
        offs_c = offs_token // OUT_TOP_K
    else:
        offs_c = offs_token
    c_ptrs = c_ptr + stride_cm * offs_c[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None]
    if not N_DIVISIBLE_BY_BLOCK_N:
        c_mask = c_mask & (offs_cn[None, :] < N_out)
    if DIRECT_SUM:
        # Kernel completion provides the only ordering needed here.
        tl.atomic_add(c_ptrs, accumulator, sem="relaxed", mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
    B_bias: torch.Tensor | None = None,
    FUSE_SILU: bool = False,
    direct_sum: bool = False,
    out_top_k: int = 1,
) -> None:
    """Launch the fused_moe_kernel Triton kernel."""
    logger.debug("GEMS_TSINGMICRO INVOKE_FUSED_MOE_TRITON_KERNEL")
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]

    #  FUSE_SILU means B.size(1) contains both Gate and Up. N is halved.
    actual_N = B.size(1) // 2 if FUSE_SILU else B.size(1)
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(actual_N, META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    swap_AB = config.pop("SWAP_AB", False)
    pair_gate_up_dot = config.pop("PAIR_GATE_UP_DOT", False)
    # Force disable SWAP_AB in fusion mode
    if FUSE_SILU:
        swap_AB = False

    fused_moe_kernel[grid](
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),  # N
        B.size(2),  # K
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SWAP_AB=swap_AB,
        K_DIVISIBLE_BY_BLOCK_K=(B.size(2) % BLOCK_SIZE_K == 0),
        N_DIVISIBLE_BY_BLOCK_N=(actual_N % config["BLOCK_SIZE_N"] == 0),
        PAIR_GATE_UP_DOT=pair_gate_up_dot,
        DIRECT_SUM=direct_sum,
        OUT_TOP_K=out_top_k,
        FUSE_SILU=FUSE_SILU,
        **config,
    )


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
    FUSE_SILU: bool = False,
    direct_sum: bool = False,
    out_top_k: int = 1,
) -> None:
    """Dispatch to the appropriate fused MoE kernel based on quantization flags."""
    logger.debug("GEMS_TSINGMICRO DISPATCH_FUSED_MOE_KERNEL")
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    # M = A.size(0)
    # num_tokens = M * top_k

    if False:
        # TODO: Other precision-specific implementations
        # use_fp8_w8a8,
        # use_int8_w8a8,
        # use_int8_w8a16,
        # use_int4_w4a16,
        pass
    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None
        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )
    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
            FUSE_SILU=FUSE_SILU,
            direct_sum=direct_sum,
            out_top_k=out_top_k,
        )


def _use_compact_grouped_moe(
    *,
    activation_enum: MoEActivation,
    apply_router_weight_on_input: bool,
    expert_map: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: Optional[str],
    block_shape: Optional[list[int]],
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
) -> bool:
    """Plain bf16/fp16 or INT8-w8a16 (SPM dequant) + silu: skip moe_align."""
    if os.environ.get("FUSED_MOE_DENSE", "1") == "0":
        return False
    if activation_enum not in (
        MoEActivation.SILU,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    ):
        return False
    if apply_router_weight_on_input:
        return False
    if expert_map is not None:
        return False
    if use_fp8_w8a8 or use_int8_w8a8 or use_int4_w4a16:
        return False
    if ocp_mx_scheme is not None or block_shape is not None:
        return False
    if a1_scale is not None or a2_scale is not None:
        return False
    if w1_bias is not None or w2_bias is not None:
        return False
    if use_int8_w8a16:
        return w1_scale is not None and w2_scale is not None
    if w1_scale is not None or w2_scale is not None:
        return False
    return True


_DEFAULT_MOE_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "moe_config"
)


def _moe_config_device_name() -> str:
    override = os.environ.get("FUSED_MOE_CONFIG_DEVICE")
    if override:
        return override.replace(" ", "_")
    return str(getattr(device, "name", "txda")).replace(" ", "_")


def _moe_config_search_dirs() -> list[str]:
    """Search order: VLLM_TUNED_CONFIG_FOLDER, FUSED_MOE_CONFIG_DIR, bundled moe_config/."""
    dirs: list[str] = []
    vllm_folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
    if vllm_folder:
        dirs.append(vllm_folder)
    extra = os.environ.get("FUSED_MOE_CONFIG_DIR")
    if extra and extra not in dirs:
        dirs.append(extra)
    dirs.append(_DEFAULT_MOE_CONFIG_DIR)
    return dirs


def _get_grouped_config_filename(
    E: int,
    N: int,
    device_name: str,
    dtype: str | None = None,
    block_shape: list[int] | None = None,
) -> str:
    """vLLM get_config_file_name: dtype/block_shape suffix only when set."""
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return (
        f"E={E},N={N},device_name={device_name}"
        f"{dtype_selector}{block_shape_selector}.json"
    )


def _json_tile_to_grouped(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map auto-tune JSON (BLOCK_SIZE_*) onto grouped-GEMM knobs."""
    bm = int(cfg.get("GROUPED_BLOCK_M", cfg.get("BLOCK_SIZE_M", 16)))
    bn = int(cfg.get("GROUPED_BLOCK_N", cfg.get("BLOCK_SIZE_N", 0)))
    bk = int(cfg.get("GROUPED_BLOCK_K", cfg.get("BLOCK_SIZE_K", 0)))
    warps = int(cfg.get("num_warps", 1))
    stages = int(cfg.get("num_stages", 2))
    n_cap = int(cfg.get("GROUPED_BLOCK_N_CAP", bn if bn else 2048))
    return {
        "BLOCK_SIZE_M": bm,
        "GROUPED_BLOCK_M": bm,
        "GROUPED_BLOCK_N": bn,
        "GROUPED_BLOCK_N_CAP": max(n_cap, bn),
        "GROUPED_BLOCK_K": bk,
        "num_warps": warps,
        "num_stages": stages,
        "config_from_file": True,
    }


@functools.lru_cache(maxsize=64)
def _load_grouped_moe_config_table(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int,
    block_k: int,
) -> Optional[tuple[str, dict[int, dict[str, Any]]]]:
    """Load tuned JSON from VLLM_TUNED_CONFIG_FOLDER / moe_config (vLLM naming)."""
    device_name = _moe_config_device_name()
    block_shape = [block_n, block_k] if block_n and block_k else None
    name = _get_grouped_config_filename(E, N, device_name, dtype, block_shape)
    for folder in _moe_config_search_dirs():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        data.pop("triton_version", None)
        table: dict[int, dict[str, Any]] = {}
        for k, v in data.items():
            if isinstance(v, dict) and str(k).isdigit():
                table[int(k)] = v
        if table:
            logger.debug("Using grouped MoE config %s", path)
            return path, table
    return None


def _pick_compact_grouped_config(
    E: int,
    N_out: int,
    num_tokens: int,
    dtype: Optional[str],
    fallback: dict[str, Any],
    block_shape: list[int] | None = None,
) -> dict[str, Any]:
    """vLLM-style: closest num_tokens key in E,N JSON; else fallback."""
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    loaded = _load_grouped_moe_config_table(E, N_out, dtype, block_n, block_k)
    if loaded is None:
        cfg = dict(fallback)
        cfg["config_from_file"] = False
        return cfg
    path, table = loaded
    nearest = min(table.keys(), key=lambda x: abs(x - num_tokens))
    cfg = _json_tile_to_grouped(table[nearest])
    cfg["config_file"] = path
    cfg["config_M"] = nearest
    return cfg


def _default_compact_grouped_configs(
    N: int, K: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prefill-winning tiles; Hy3 short K_in uses a fatter G2 BN (capped by K)."""
    config1 = {
        "BLOCK_SIZE_M": 16,
        "GROUPED_BLOCK_M": 16,
        "GROUPED_BLOCK_N": 0,
        "GROUPED_BLOCK_N_CAP": 2048,
        "GROUPED_BLOCK_K": 0,
        "num_warps": 1,
        "num_stages": 3,
    }
    g2_k_in = N // 2
    # G2 GEMM N-dim is K (hidden). Never request BN > K (pow2), or INT8
    # dequant SPM exceeds TX81 limit (seen on Qwen3.6-A3B W8A8, K=2048).
    g2_bn_cap = _floor_pow2(max(K, 1))
    if g2_k_in <= 256 and K >= 2048:
        config2 = {
            "BLOCK_SIZE_M": 16,
            "GROUPED_BLOCK_M": 16,
            "GROUPED_BLOCK_N": g2_bn_cap,
            "GROUPED_BLOCK_N_CAP": g2_bn_cap,
            "GROUPED_BLOCK_K": 0,
            "num_warps": 1,
            # stages=3 + fat BN blew SPM on INT8; start at 2, fit logic can drop further
            "num_stages": 2,
        }
    else:
        config2 = {
            "BLOCK_SIZE_M": 16,
            "GROUPED_BLOCK_M": 16,
            "GROUPED_BLOCK_N": min(2048, g2_bn_cap),
            "GROUPED_BLOCK_N_CAP": min(2048, g2_bn_cap),
            "GROUPED_BLOCK_K": 128,
            "num_warps": 1,
            "num_stages": 2,
        }
    return config1, config2


def _apply_grouped_tile_env(config1: dict[str, Any], config2: dict[str, Any]) -> None:
    """Optional overrides: FUSED_MOE_G1_BM/BN/BK/WARPS/STAGES (G2 likewise)."""
    for cfg, prefix in ((config1, "G1"), (config2, "G2")):
        bm = _env_int(f"FUSED_MOE_{prefix}_BM")
        bn = _env_int(f"FUSED_MOE_{prefix}_BN")
        bk = _env_int(f"FUSED_MOE_{prefix}_BK")
        warps = _env_int(f"FUSED_MOE_{prefix}_WARPS")
        stages = _env_int(f"FUSED_MOE_{prefix}_STAGES")
        if bm is not None:
            cfg["GROUPED_BLOCK_M"] = bm
        if bn is not None:
            cfg["GROUPED_BLOCK_N"] = bn
            cfg["GROUPED_BLOCK_N_CAP"] = max(
                int(cfg.get("GROUPED_BLOCK_N_CAP", bn)), bn
            )
        if bk is not None:
            cfg["GROUPED_BLOCK_K"] = bk
        if warps is not None:
            cfg["num_warps"] = warps
        if stages is not None:
            cfg["num_stages"] = stages


def _invoke_compact_grouped_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    compact_meta: dict[str, Any],
    config: dict[str, Any],
    *,
    a_row_mode: str,
    topk_weights: Optional[torch.Tensor] = None,
    mul_routed_weight: bool = False,
    B_scale: Optional[torch.Tensor] = None,
) -> None:
    if a_row_mode == "hidden":
        a_rows = compact_meta["a_rows_hidden"].clamp(0, A.size(0) - 1)
    else:
        a_rows = compact_meta["a_rows_flat"].clamp(0, A.size(0) - 1)
    router_w = None
    weight_by_dest = False
    if mul_routed_weight:
        assert topk_weights is not None
        router_w = topk_weights.reshape(-1)
        weight_by_dest = True
    _moe_compact_grouped_gemm(
        A,
        B,
        C.reshape(-1, C.size(-1)),
        compact_meta["seg_starts"],
        compact_meta["seg_ends"],
        compact_meta["seg_experts"],
        config=config,
        a_rows=a_rows,
        router_weights=router_w,
        dest_ids=compact_meta["valid_ids"],
        weight_by_dest=weight_by_dest,
        B_scale=B_scale,
    )


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: float | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    # FlagGems 4.2 compatibility: old callers used num_experts=
    num_experts: Optional[int] = None,
) -> torch.Tensor:
    logger.debug("GEMS_TSINGMICRO FUSED_EXPERTS_IMPL")
    if num_experts is not None and global_num_experts == -1:
        global_num_experts = num_experts

    assert activation in (
        "silu",
        "swigluoai",
        "swigluoai_uninterleave",
    ), f"Only silu/swigluoai/swigluoai_uninterleave supported, got {activation}"

    activation_enum = MoEActivation.from_str(activation)

    # Check constraints
    if use_int4_w4a16:
        # INT4 stored unpacked in INT8 containers (full K dim)
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme.startswith("w_mxfp4"):
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme.startswith("w_mxfp6"):
            assert (
                hidden_states.size(1) == (w1.size(2) * 4) // 3
            ), "hidden size mismatch"
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # TX81/tsingmicro: no int8 GEMM / no dynamic int8 activations.
    # Keep signed INT8 weights + per-channel FP32 scale (zp=0) in HBM, and let
    # fused_moe_kernel dequant per tile: load int8 → cast to bf16/fp16 → dot,
    # then multiply per-channel scale (use_int8_w8a16 path). This cuts weight
    # DRAM traffic vs eager full-tensor dequant.
    if use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        if w1_scale is None or w2_scale is None:
            raise ValueError(
                "INT8/INT4 MoE on tsingmicro requires w1_scale and w2_scale "
                "(symmetric, zero_point=0)"
            )
        if w1_zp is not None or w2_zp is not None:
            raise NotImplementedError(
                "tsingmicro INT8 kernel-dequant path assumes zero_point=0; "
                "got non-None w1_zp/w2_zp"
            )
        if use_int4_w4a16:
            # INT4 still needs host unpack/dequant; not covered by int8 tile path.
            compute_dtype = hidden_states.dtype
            w1 = (w1.to(torch.float32) * w1_scale.to(torch.float32).unsqueeze(-1)).to(
                compute_dtype
            )
            w2 = (w2.to(torch.float32) * w2_scale.to(torch.float32).unsqueeze(-1)).to(
                compute_dtype
            )
            w1_scale = None
            w2_scale = None
            use_int4_w4a16 = False
            use_int8_w8a16 = False
            use_int8_w8a8 = False
            per_channel_quant = False
            a1_scale = None
            a2_scale = None
            logger.info(
                "GEMS_TSINGMICRO FUSED_EXPERTS_IMPL: INT4 host-dequant → %s",
                compute_dtype,
            )
        else:
            # W8A8 checkpoint → W8A16 compute: act stays bf16/fp16; weights int8.
            if use_int8_w8a8:
                use_int8_w8a8 = False
                use_int8_w8a16 = True
                a1_scale = None
                a2_scale = None
            per_channel_quant = True
            logger.info(
                "GEMS_TSINGMICRO FUSED_EXPERTS_IMPL: INT8 weights kept; "
                "in-kernel dequant + %s GEMM (use_int8_w8a16)",
                hidden_states.dtype,
            )

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    CHUNK_SIZE: int = 32 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )
    is_plain_half_config = config_dtype is None and hidden_states.dtype in (
        torch.float16,
        torch.bfloat16,
    )
    is_fp8_blockwise = config_dtype == "fp8_w8a8" and block_shape is not None

    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_moe_config = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
        E=E,
        return_is_embedded=True,
    )

    base_config, is_embedded_config = get_moe_config(M)

    # cache1 and cache3 share memory (non-overlapping lifetime)
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # cache2 needs separate memory (concurrent with cache1)
    activation_out_dim = MoEActivation.adjust_N_for_activation(N, activation_enum)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # Dequantize OCP MX weights (TODO: skip on platforms with native MX)
        if ocp_mx_scheme.startswith("w_mxfp4"):
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e3m2"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e2m3"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    direct_sum_supported = is_plain_half_config or is_fp8_blockwise

    # Check if we can safely fuse the activation with the first GEMM pass.
    # FUSE_SILU / PAIR_GATE_UP_DOT paths do not yet cast int8 weights to
    # compute_type before tl.dot; force the standard GEMM path for w8a16.
    can_use_fused_silu = (
        activation_enum in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)
        and w1_bias is None
        and expert_map is None  # Fused kernel doesn't handle EP -1 experts
        and not use_int8_w8a16
        and gemm1_clamp_limit is None  # Fused SILU has no clamp support
    )

    # Plain half-precision: skip moe_align padding; compact meta + grouped GEMM.
    use_compact_grouped = _use_compact_grouped_moe(
        activation_enum=activation_enum,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=expert_map,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        block_shape=block_shape,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    if use_compact_grouped:
        fallback_w1, fallback_w2 = _default_compact_grouped_configs(N, K)
        pipe = _PipeTimer(
            os.environ.get("FUSED_MOE_PIPELINE_PROFILE", "0")
            != "0",  # profile : "1") != "0"
            hidden_states.device,
            "fused_experts_pipeline",
        )
        last_meta: Optional[dict[str, Any]] = None
        auto_bm = 16
        config1 = fallback_w1
        config2 = fallback_w2
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            tokens_in_chunk, _ = curr_hidden_states.size()
            if tokens_in_chunk == 0:
                break
            if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
                intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
                intermediate_cache2 = intermediate_cache2[
                    : tokens_in_chunk * topk_ids.size(1)
                ]
                intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            compact_meta = _build_moe_compact_meta_from_topk(
                curr_topk_ids, device=hidden_states.device
            )
            pipe.mark("build_compact_meta")
            if compact_meta is None:
                out_hidden_states[begin_chunk_idx:end_chunk_idx].zero_()
                continue

            # w1 GEMM N = w1.size(1); w2 GEMM N = w2.size(1)=K. Closest M in JSON.
            cfg1 = _pick_compact_grouped_config(
                E, N, tokens_in_chunk, config_dtype, fallback_w1, block_shape
            )
            cfg2 = _pick_compact_grouped_config(
                E, K, tokens_in_chunk, config_dtype, fallback_w2, block_shape
            )
            if not cfg1.get("config_from_file"):
                cfg1["GROUPED_BLOCK_M"] = _pick_grouped_block_m(
                    float(compact_meta["avg_rows_per_seg"])
                )
            if not cfg2.get("config_from_file"):
                cfg2["GROUPED_BLOCK_M"] = _pick_grouped_block_m(
                    float(compact_meta["avg_rows_per_seg"])
                )
            _apply_grouped_tile_env(cfg1, cfg2)
            auto_bm = int(cfg1["GROUPED_BLOCK_M"])

            _invoke_compact_grouped_moe(
                curr_hidden_states,
                w1,
                intermediate_cache1,
                compact_meta,
                cfg1,
                a_row_mode="hidden",
                B_scale=w1_scale if use_int8_w8a16 else None,
            )
            pipe.mark("gemm1")
            apply_moe_activation(
                activation_enum,
                intermediate_cache2,
                intermediate_cache1.view(-1, N),
                clamp_limit=gemm1_clamp_limit,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
            )
            pipe.mark("silu_and_mul")
            _invoke_compact_grouped_moe(
                intermediate_cache2,
                w2,
                intermediate_cache3,
                compact_meta,
                cfg2,
                a_row_mode="flat",
                topk_weights=curr_topk_weights,
                mul_routed_weight=True,
                B_scale=w2_scale if use_int8_w8a16 else None,
            )
            pipe.mark("gemm2")
            _moe_sum_topk(
                intermediate_cache3,
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
            )
            pipe.mark("moe_sum")
            last_meta = compact_meta
            config1, config2 = cfg1, cfg2

        if last_meta is not None:
            pipe.meta.update(
                {
                    "M": num_tokens,
                    "E": E,
                    "K": K,
                    "N": N,
                    "top_k": top_k_num,
                    "n_valid": last_meta["n_valid"],
                    "n_mm": last_meta["n_mm"],
                    "avg_rows": round(float(last_meta["avg_rows_per_seg"]), 3),
                    "meta_source": last_meta.get("meta_source"),
                    "bm": auto_bm,
                    "warps": config1.get("num_warps"),
                    "bn_g1": config1.get("GROUPED_BLOCK_N_RESOLVED"),
                    "bn_g2": config2.get("GROUPED_BLOCK_N_RESOLVED"),
                    "bk_g1": config1.get("GROUPED_BLOCK_K_RESOLVED"),
                    "bk_g2": config2.get("GROUPED_BLOCK_K_RESOLVED"),
                    "ctas_g1": config1.get("GROUPED_NUM_CTAS"),
                    "ctas_g2": config2.get("GROUPED_NUM_CTAS"),
                    "int8_w8a16": bool(use_int8_w8a16),
                    "cfg_w1_file": config1.get("config_file"),
                    "cfg_w2_file": config2.get("config_file"),
                    "cfg_w1_M": config1.get("config_M"),
                    "cfg_w2_M": config2.get("config_M"),
                }
            )
        pipe.report()
        return out_hidden_states

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust cache size for last chunk
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            base_config, is_embedded_config = get_moe_config(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        SPARSITY_FACTOR = 4
        naive_block_assignment = (
            expert_map is None
            and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
            and not (
                (use_int8_w8a16 or use_int4_w4a16)
                and block_shape is not None
                and block_shape[1] > 0
            )
        )

        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                base_config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
                # ignore_invalid_experts=True,
            )
        else:
            max_num_tokens_padded = topk_ids.numel() * base_config["BLOCK_SIZE_M"]
            expert_ids = curr_topk_ids.view(-1)
            num_tokens_post_padded = torch.empty(
                (1), dtype=torch.int32, device=topk_ids.device
            )
            num_tokens_post_padded.fill_(max_num_tokens_padded)
            sorted_token_ids = None

        # 1. Extract a unified boolean flag for GEMM1 fusion and select config
        do_fuse_silu = can_use_fused_silu and not naive_block_assignment
        use_half_gemm_fast_paths = not is_embedded_config and is_plain_half_config

        gemm1_config = base_config
        if do_fuse_silu and use_half_gemm_fast_paths:
            gemm1_config, _ = get_moe_config(
                tokens_in_chunk,
                gemm_stage="gemm1",
                enable_gemm_fast_path=True,
            )

        # 2. Dynamically determine the differing parameters based on the fusion flag
        if do_fuse_silu:
            # Output goes directly to cache 2 with adjusted dimensions
            out_cache = intermediate_cache2.view(
                tokens_in_chunk, top_k_num, activation_out_dim
            )
            # Fused kernel weight handling depends on apply_router_weight_on_input
            if apply_router_weight_on_input:
                weights_arg = curr_topk_weights
            else:
                weights_arg = None
        else:
            # Standard path outputs to cache 1
            out_cache = intermediate_cache1
            # Standard path always passes the weights
            weights_arg = curr_topk_weights

        # 3. Unified GEMM1 dispatch call to eliminate redundant code blocks
        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            out_cache,  # Dynamically assigned output buffer
            a1q_scale,
            w1_scale,
            w1_zp,
            weights_arg,  # Dynamically assigned weights argument
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            gemm1_config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
            FUSE_SILU=do_fuse_silu,  # Master switch for the kernel
        )

        # 4. Apply activation separately if the fused path was not taken
        if not do_fuse_silu:
            apply_moe_activation(
                activation_enum,
                intermediate_cache2,
                intermediate_cache1.view(-1, N),
                clamp_limit=gemm1_clamp_limit,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
            )

        # 5. Quantize activated intermediate for GEMM2
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        # 6. Select GEMM2 config and output buffer/reduction path
        gemm2_config = base_config
        if use_half_gemm_fast_paths:
            gemm2_config, _ = get_moe_config(
                tokens_in_chunk,
                gemm_stage="gemm2",
                enable_gemm_fast_path=True,
            )
        use_direct_sum = (
            not is_embedded_config
            and direct_sum_supported
            and tokens_in_chunk >= MOE_DIRECT_SUM_MIN_TOKENS
            and expert_map is None
            and not apply_router_weight_on_input
        )
        if use_direct_sum:
            gemm2_output = out_hidden_states[begin_chunk_idx:end_chunk_idx].view(
                tokens_in_chunk, 1, K
            )
            gemm2_output.zero_()
        else:
            gemm2_output = intermediate_cache3

        # 7. Dispatch GEMM2
        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            gemm2_output,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            gemm2_config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
            FUSE_SILU=False,
            direct_sum=use_direct_sum,
            out_top_k=top_k_num,
        )

        # 8. Reduce GEMM2 top-k outputs unless direct_sum wrote final output directly
        if not use_direct_sum:
            moe_sum(
                intermediate_cache3.view(*intermediate_cache3.size()),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
            )

    return out_hidden_states


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: float | None = None,
) -> None:
    """
    In-place fused MoE: writes output directly into ``hidden_states``.


    Same semantics as ``fused_experts_impl(..., inplace=True)``.
    Returns None (the result is stored in ``hidden_states``).
    """
    logger.debug("GEMS_TSINGMICRO INPLACE_FUSED_EXPERTS")
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: float | None = None,
) -> torch.Tensor:
    """
    Out-of-place fused MoE: allocates and returns a new output tensor.

    Same semantics as ``fused_experts_impl(..., inplace=False)``.
    """
    logger.debug("GEMS_TSINGMICRO OUTPLACE_FUSED_EXPERTS")
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
