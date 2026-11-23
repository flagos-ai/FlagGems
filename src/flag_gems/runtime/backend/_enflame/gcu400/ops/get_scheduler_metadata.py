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
import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.utils.device_info import get_device_capability, get_device_info

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def tile_size_fwd_sm8x(
    sm86_or_89: bool,
    headdim: int,
    headdim_v: int,
    is_causal: bool,
    is_local: bool,
    element_size: int = 2,
    paged_kv: bool = False,
    varlen_and_split: bool = False,
    softcap: bool = False,
    append_kv: bool = False,
):
    logger.debug("GEMS_ENFLAME TILE_SIZE_FWD_SM8X")
    if element_size == 2:  # fp16/bf16
        if headdim <= 64:
            kBlockM = 128
            kBlockN = 80 if varlen_and_split else (96 if is_local else 112)
            kNWarps = 4
            kStages = 1
            Q_in_regs = False

        elif headdim <= 96:
            kBlockM = 128
            kBlockN = 48 if (varlen_and_split or is_local) else 64
            kNWarps = 4
            kStages = 1
            Q_in_regs = False

        elif headdim <= 128:
            use_8_warps = sm86_or_89 or varlen_and_split
            kBlockM = 128
            if use_8_warps:
                kBlockN = (
                    (96 if is_local else 112)
                    if varlen_and_split
                    else (96 if is_local else 128)
                )
            else:
                kBlockN = 48 if is_local else 64
            kNWarps = 8 if use_8_warps else 4
            kStages = 1
            Q_in_regs = use_8_warps

        elif headdim <= 192:
            kBlockN_64 = append_kv or is_local or varlen_and_split or paged_kv
            kBlockM = 128
            kBlockN = 64 if kBlockN_64 else 96
            kNWarps = 8
            kStages = 1 if sm86_or_89 else 2
            Q_in_regs = not kBlockN_64

        else:  # headdim > 192
            kBlockM = 128
            if sm86_or_89:
                if append_kv:
                    kBlockN = 32
                elif varlen_and_split or is_local:
                    kBlockN = 48
                else:
                    kBlockN = 64
            else:
                if append_kv:
                    kBlockN = 48
                elif varlen_and_split or is_local:
                    kBlockN = 64
                else:
                    kBlockN = 96
            kNWarps = 8
            kStages = 1
            Q_in_regs = sm86_or_89 and not append_kv
    else:
        kBlockM = 128
        kBlockN = 64
        kNWarps = 8
        kStages = 2
        Q_in_regs = False

    return kBlockM, kBlockN, kNWarps, kStages, Q_in_regs


@lru_cache(maxsize=256)
def tile_size_fwd_sm90(
    headdim: int,
    headdim_v: int,
    is_causal: bool,
    is_local: bool,
    element_size: int = 2,
    v_colmajor: bool = False,
    paged_kv_non_TMA: bool = False,
    softcap: bool = False,
    use_one_mma_wg: bool = False,
):
    logger.debug("GEMS_ENFLAME TILE_SIZE_FWD_SM90")
    if element_size == 2:
        if headdim <= 64:
            if headdim_v == 512:
                return 64, 64
            elif headdim_v == 256:
                return 128, 112
            else:
                use_blockN_128 = is_causal or is_local
                return 192, (128 if use_blockN_128 else 192)
        elif headdim <= 96:
            return 192, (128 if (is_local or paged_kv_non_TMA) else 144)
        elif headdim <= 128:
            if use_one_mma_wg:
                return 64, (128 if (is_causal or is_local or paged_kv_non_TMA) else 176)
            else:
                return 128, (
                    128 if (is_causal or is_local or paged_kv_non_TMA) else 176
                )
        elif headdim <= 192:
            return 128, (
                96
                if (paged_kv_non_TMA or is_local)
                else (128 if headdim_v <= 128 else 112)
            )
        else:
            return 128, (64 if is_local else 80)
    else:
        if headdim <= 64:
            return 192, 160
        elif headdim <= 96:
            return 192, 128
        elif headdim <= 128:
            return 128, (
                160
                if paged_kv_non_TMA
                else (192 if (v_colmajor or (softcap and is_local)) else 224)
            )
        elif headdim <= 192:
            return 128, (128 if ((paged_kv_non_TMA or softcap) and is_local) else 160)
        else:
            return 128, (64 if is_local else 128)


@lru_cache(maxsize=256)
def get_optimal_block_mn(
    device,
    headdim,
    headdim_v,
    is_causal,
    is_local,
    has_softcap,
    element_size=2,
    paged_kv=False,
    pagedkv_tma: bool = False,
    varlen_and_split=False,
    append_kv=False,
):
    logger.debug("GEMS_ENFLAME GET_OPTIMAL_BLOCK_MN")
    major, minor = get_device_capability()
    arch = major * 10 + minor

    if arch >= 90:
        paged_kv_non_TMA = bool(paged_kv and (not pagedkv_tma))
        kBlockM, kBlockN = tile_size_fwd_sm90(
            headdim=headdim,
            headdim_v=headdim_v,
            is_causal=is_causal,
            is_local=is_local,
            element_size=element_size,
            v_colmajor=False,
            paged_kv_non_TMA=paged_kv_non_TMA,
            softcap=has_softcap,
            use_one_mma_wg=False,
        )
        return kBlockM, kBlockN
    else:
        kBlockM, kBlockN, kNWarps, kStages, Q_in_regs = tile_size_fwd_sm8x(
            sm86_or_89=arch == 86 or arch == 89,
            headdim=headdim,
            headdim_v=headdim_v,
            is_causal=is_causal,
            is_local=is_local,
            element_size=element_size,
            paged_kv=paged_kv,
            varlen_and_split=varlen_and_split,
            softcap=has_softcap,
            append_kv=append_kv,
        )
        return kBlockM, kBlockN


@lru_cache(maxsize=256)
def round_up_headdim(headdim: int) -> int:
    logger.debug("GEMS_ENFLAME ROUND_UP_HEADDIM")
    if headdim <= 64:
        return 64
    if headdim <= 96:
        return 96
    if headdim <= 128:
        return 128
    if headdim <= 192:
        return 192
    if headdim <= 256:
        return 256
    return 256


@lru_cache(maxsize=256)
def round_up_headdimv(headdim_v: int) -> int:
    logger.debug("GEMS_ENFLAME ROUND_UP_HEADDIMV")
    if headdim_v <= 64:
        return 64
    if headdim_v <= 96:
        return 96
    if headdim_v <= 128:
        return 128
    if headdim_v <= 192:
        return 192
    if headdim_v <= 256:
        return 256
    return 512


def get_pagedkv_tma(
    arch: int,
    page_size: int,
    has_page_table: bool,
    leftpad_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k_new: int,
    num_heads: int,
    num_heads_k: int,
    d_rounded: int,
    dv_rounded: int,
    is_causal: bool,
    is_local: bool,
    element_size: int,
    softcap: bool,
):
    logger.debug("GEMS_ENFLAME GET_PAGEDKV_TMA")
    if (
        arch < 90
        or (not has_page_table)
        or (leftpad_k is not None)
        or (max_seqlen_k_new > 0)
    ):
        return False
    kBlockM, kBlockN = tile_size_fwd_sm90(
        headdim=d_rounded,
        headdim_v=dv_rounded,
        is_causal=is_causal,
        is_local=is_local,
        element_size=element_size,
        v_colmajor=False,
        paged_kv_non_TMA=False,
        softcap=softcap,
        use_one_mma_wg=False,
    )
    if page_size % kBlockN != 0:
        return False
    seqlen_q_packgqa = max_seqlen_q * (num_heads // num_heads_k)
    return seqlen_q_packgqa > kBlockM


@lru_cache(maxsize=256)
def use_one_mma_wg(
    arch: int,
    headdim: int,
    seqlen_q: int,
    pack_gqa: bool,
    num_heads: int,
    num_heads_k: int,
) -> bool:
    logger.debug("GEMS_ENFLAME USE_ONE_MMA_WG")
    if arch < 90 or headdim != 128:
        return False

    qhead_per_khead = 1 if not pack_gqa else num_heads // num_heads_k
    effective_seqlen_q = seqlen_q * qhead_per_khead

    return effective_seqlen_q <= 64


@lru_cache(maxsize=256)
def should_pack_gqa(
    varlen_q: bool,
    seqlen_q: int,
    qhead_per_khead: int,
    blockM: int,
) -> bool:
    logger.debug("GEMS_ENFLAME SHOULD_PACK_GQA")
    if varlen_q:
        return True

    def round_up(a: int, b: int) -> int:
        return (a + b - 1) // b * b

    nopack_eff = float(seqlen_q) / float(round_up(seqlen_q, blockM))
    pack_eff = float(seqlen_q * qhead_per_khead) / float(
        round_up(seqlen_q * qhead_per_khead, blockM)
    )
    return nopack_eff < 0.9 * pack_eff


@lru_cache(maxsize=256)
def get_num_splits(
    batch_size: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    d_rounded: int,
    dv_rounded: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_seqlen_k_new: int,
    arch: int,
    num_sm: int,
    is_causal: bool,
    is_local: bool,
    has_softcap: float,
    is_varlen: bool,
    has_page_table: bool,
    pack_gqa: bool,
    window_size_left: int,
    window_size_right: int,
    element_size: int = 2,  # fp16/bf16 = 2, fp8 = 1
    max_splits: int = 128,
    use_dynamic_split: bool = False,
) -> int:
    logger.debug("GEMS_ENFLAME GET_NUM_SPLITS")
    pagedkv_tma = False
    append_kv = max_seqlen_k_new > 0

    if arch >= 90:
        uomw = use_one_mma_wg(
            arch=arch,
            headdim=headdim,
            seqlen_q=max_seqlen_q,
            pack_gqa=pack_gqa,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
        )
        kBlockM, kBlockN = tile_size_fwd_sm90(
            headdim=d_rounded,
            headdim_v=dv_rounded,
            is_causal=is_causal,
            is_local=is_local,
            element_size=element_size,
            v_colmajor=False,
            paged_kv_non_TMA=(has_page_table and not pagedkv_tma),
            softcap=(has_softcap > 0.0),
            use_one_mma_wg=uomw,
        )
    else:
        sm86_or_89 = arch == 86 or arch == 89
        kBlockM, kBlockN, _, _, _ = tile_size_fwd_sm8x(
            sm86_or_89=sm86_or_89,
            headdim=d_rounded,
            headdim_v=dv_rounded,
            is_causal=is_causal,
            is_local=is_local,
            element_size=element_size,
            paged_kv=has_page_table,
            varlen_and_split=is_varlen,
            softcap=(has_softcap > 0.0),
            append_kv=append_kv,
        )

    seqlen_q_packgqa = max_seqlen_q * (num_heads // num_heads_k)

    if is_local:
        seqlen_k_loaded = max(
            0,
            min(max_seqlen_k, window_size_left + window_size_right + 1 + kBlockM),
        )
    else:
        seqlen_k_loaded = max_seqlen_k

    num_n_blocks = (seqlen_k_loaded + kBlockN - 1) // kBlockN
    num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) // kBlockM

    size_one_kv_head = max_seqlen_k * (headdim + headdim_v) * element_size

    effective_batch = 1 if use_dynamic_split else batch_size
    total_mblocks = effective_batch * num_heads_k * num_m_blocks

    return _vllm_num_splits_heuristic(
        total_mblocks=total_mblocks,
        num_sm=num_sm,
        num_n_blocks=num_n_blocks,
        num_m_blocks=num_m_blocks,
        size_one_kv_head=size_one_kv_head,
        is_causal_or_local=is_causal or is_local,
        max_splits=max_splits,
    )


def _vllm_num_splits_heuristic(
    total_mblocks: int,
    num_sm: int,
    num_n_blocks: int,
    num_m_blocks: int,
    size_one_kv_head: int,
    is_causal_or_local: bool,
    max_splits: int,
) -> int:
    if total_mblocks >= 0.8 * num_sm:
        size_l2 = 50 * 1024 * 1024
        if (
            size_one_kv_head > size_l2
            and num_m_blocks >= num_sm * 2
            and not is_causal_or_local
        ):
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        else:
            return 1

    if num_n_blocks <= 4:
        return 1

    max_splits = min(max_splits, num_sm, num_n_blocks)

    max_efficiency = 0.0
    efficiencies = []

    for num_splits in range(1, max_splits + 1):
        n_waves = float(total_mblocks * num_splits) / num_sm
        eff = n_waves / math.ceil(n_waves)
        if eff > max_efficiency:
            max_efficiency = eff
        efficiencies.append(eff)

    for num_splits in range(1, max_splits + 1):
        if efficiencies[num_splits - 1] >= 0.85 * max_efficiency:
            return num_splits

    return 1


@triton.jit
def _prepare_pass1_kernel(
    num_m_blocks_ptr,
    num_n_blocks_ptr,
    total_blocks_ptr,
    seqlen_k_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    cu_seqlens_k_new_ptr,
    seqused_q_ptr,
    seqused_k_ptr,
    leftpad_k_ptr,
    batch,
    qhead_per_khead,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k_new: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    # HAS_XXX is used to implement static branch in kernel
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    HAS_LEFT_PAD: tl.constexpr,
    HAS_K_NEW: tl.constexpr,
    HAS_CU_SEQLENS_K_NEW: tl.constexpr,
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_SIZE_B
    b_offs = b_start + tl.arange(0, BLOCK_SIZE_B)
    mask = b_offs < batch

    if HAS_SEQUSED_Q:
        q_len = tl.load(seqused_q_ptr + b_offs, mask=mask, other=0)
    elif HAS_CU_SEQLENS_Q:
        cur = tl.load(cu_seqlens_q_ptr + b_offs, mask=mask, other=0)
        nxt = tl.load(cu_seqlens_q_ptr + b_offs + 1, mask=mask, other=0)
        q_len = nxt - cur
    else:
        q_len = tl.full(
            [BLOCK_SIZE_B], max_seqlen_q, dtype=tl.int32
        )  # max_seqlen_q constexpr
    q_len = q_len * qhead_per_khead
    m_blocks = (q_len + BLOCK_M - 1) // BLOCK_M

    if HAS_SEQUSED_K:
        k_len = tl.load(seqused_k_ptr + b_offs, mask=mask, other=0)
    elif HAS_CU_SEQLENS_K:
        cur = tl.load(cu_seqlens_k_ptr + b_offs, mask=mask, other=0)
        nxt = tl.load(cu_seqlens_k_ptr + b_offs + 1, mask=mask, other=0)
        k_len = nxt - cur
    else:
        k_len = tl.load(seqlen_k_ptr + b_offs, mask=mask, other=0)
    left = tl.load(leftpad_k_ptr + b_offs, mask=mask, other=0) if HAS_LEFT_PAD else 0

    if HAS_K_NEW:
        if HAS_CU_SEQLENS_K_NEW:
            cur_new = tl.load(cu_seqlens_k_new_ptr + b_offs, mask=mask, other=0)
            nxt_new = tl.load(cu_seqlens_k_new_ptr + b_offs + 1, mask=mask, other=0)
            k_len += nxt_new - cur_new
        else:
            k_len += max_seqlen_k_new
    k_len = k_len - left
    n_blocks = (k_len + BLOCK_N - 1) // BLOCK_N

    tl.store(num_m_blocks_ptr + b_offs, m_blocks, mask=mask)
    tl.store(num_n_blocks_ptr + b_offs, n_blocks, mask=mask)
    total = m_blocks * n_blocks
    tl.atomic_add(total_blocks_ptr, tl.sum(total, axis=0))


@triton.jit
def _prepare_pass2_kernel(
    num_n_blocks_per_seq_ptr,
    scheduler_metadata_ptr,
    total_blocks_ptr,
    num_batch,
    num_head,
    num_sm,
    num_splits_static,
    SPLIT_OFFSET: tl.constexpr,
    HAS_SEMAPHORE: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_start = pid * BLOCK_SIZE_B
    b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
    b_mask = b_offsets < num_batch

    # Zero the semaphore slot (metadata[0]) inside the kernel, saving one host->device scalar write
    if HAS_SEMAPHORE:
        if pid == 0:
            tl.store(scheduler_metadata_ptr, 0)

    # total_blocks is accumulated on the device side by pass1; reading it directly inside the kernel here
    # avoids one full-device synchronization caused by .item() on the host.
    total_blocks = tl.load(total_blocks_ptr).to(tl.float32)

    blocks_per_sm_float = tl.ceil(total_blocks * 1.1 * num_head / num_sm)
    blocks_per_sm = blocks_per_sm_float.to(tl.int32)

    blocks_per_sm = tl.maximum(1, blocks_per_sm)

    num_n_blocks = tl.load(num_n_blocks_per_seq_ptr + b_offsets, mask=b_mask, other=0)
    num_splits_dynamic = (num_n_blocks + blocks_per_sm - 1) // blocks_per_sm

    num_splits_dynamic = tl.minimum(num_splits_dynamic, num_splits_static)
    num_splits_dynamic = tl.maximum(1, num_splits_dynamic)

    # Write the splits segment directly into the finally returned metadata, saving one device->device copy
    tl.store(
        scheduler_metadata_ptr + SPLIT_OFFSET + b_offsets,
        num_splits_dynamic,
        mask=b_mask,
    )


def get_pack_gqa(
    arch: int,
    has_page_table: bool,
    pagedkv_tma: bool,
    num_splits: int,
    num_heads: int,
    num_heads_k: int,
    # SM90-specific params for heuristic
    varlen_q: bool,
    seqlen_q: int,
    d_rounded: int,
    dv_rounded: int,
    is_causal: bool,
    is_local: bool,
    element_size: int,
    softcap: bool,
) -> bool:
    logger.debug("GEMS_ENFLAME GET_PACK_GQA")
    if arch < 90 or (has_page_table and not pagedkv_tma) or num_splits > 1:
        return True
    if num_heads == num_heads_k:
        return False
    kBlockM, _ = tile_size_fwd_sm90(
        headdim=d_rounded,
        headdim_v=dv_rounded,
        is_causal=is_causal,
        is_local=is_local,
        element_size=element_size,
        v_colmajor=False,
        paged_kv_non_TMA=(has_page_table and not pagedkv_tma),
        softcap=softcap,
        use_one_mma_wg=False,
    )
    qhead_per_khead = num_heads // num_heads_k
    return should_pack_gqa(varlen_q, seqlen_q, qhead_per_khead, kBlockM)


def _resolve_schedule_plan(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    cu_seqlens_q: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    page_size: Optional[int],
    max_seqlen_k_new: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    has_softcap: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
) -> dict:
    """Resolve the host-side schedule constants shared by the device op.

    Only scalar heuristics live here (tile sizes, split cap, metadata layout
    offset), so the FA3 parameter normalization is written once instead of
    being copied into every entry point. The per-sequence integer arithmetic
    is deliberately NOT shared: the device Triton kernels and the CPU
    reference used as the accuracy oracle implement it independently, so a
    drift in one of them is visible to the test.
    """
    # check parameters
    supported_dtypes = (torch.half, torch.bfloat16)
    assert (
        qkv_dtype in supported_dtypes
    ), "FlashAttention only supports fp16 and bf16 data type"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    # is_causal & window_size implementation
    effective_is_causal = is_causal
    effective_window_left = window_size_left if window_size_left >= 0 else -1
    effective_window_right = window_size_right

    if effective_window_left >= max_seqlen_k - 1:
        effective_window_left = -1
    if effective_window_right >= max_seqlen_q - 1:
        effective_window_right = -1

    if (
        max_seqlen_q == 1
        and effective_window_left == -1
        and effective_window_right == -1
    ):
        if (headdim <= 64 or headdim > 128) or page_size is None:
            effective_is_causal = False

    if effective_is_causal:
        effective_window_right = 0

    final_is_causal = effective_window_left < 0 and effective_window_right == 0
    final_is_local = (
        effective_window_left >= 0 or effective_window_right >= 0
    ) and not final_is_causal

    major, minor = get_device_capability()
    arch = major * 10 + minor
    num_sm = get_device_info().sm_count - sm_margin

    softcap = 1.0 if has_softcap else 0.0

    element_size = qkv_dtype.itemsize

    has_page_table = page_size is not None

    d_rounded = round_up_headdim(headdim)
    dv_rounded = round_up_headdimv(headdim_v)

    pagedkv_tma = get_pagedkv_tma(
        arch=arch,
        page_size=page_size if page_size is not None else 1,
        has_page_table=has_page_table,
        leftpad_k=leftpad_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_new=max_seqlen_k_new,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        d_rounded=d_rounded,
        dv_rounded=dv_rounded,
        is_causal=final_is_causal,
        is_local=final_is_local,
        element_size=element_size,
        softcap=has_softcap,
    )

    # blockM/blockN are resolved further below, after num_splits and pack_gqa are
    # known: both feed back into the tile heuristic (varlen_and_split, append_kv),
    # so computing them here would only be overwritten.

    # GQA
    varlen_q_flag = cu_seqlens_q is not None or seqused_q is not None
    pack_gqa = (
        pack_gqa
        if pack_gqa is not None
        else get_pack_gqa(
            arch=arch,
            has_page_table=has_page_table,
            pagedkv_tma=pagedkv_tma,
            num_splits=num_splits,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            varlen_q=varlen_q_flag,
            seqlen_q=max_seqlen_q,
            d_rounded=d_rounded,
            dv_rounded=dv_rounded,
            is_causal=final_is_causal,
            is_local=final_is_local,
            element_size=element_size,
            softcap=has_softcap,
        )
    )
    qhead_per_khead = (
        1 if not pack_gqa else (num_heads + num_heads_k - 1) // num_heads_k
    )
    num_head_k = num_heads_k if pack_gqa else num_heads

    # dynamic split depends ONLY on batch_size, regardless of num_splits_static
    use_dynamic_split = batch_size <= 992

    if num_splits <= 0:
        element_size = qkv_dtype.itemsize
        is_fp16 = qkv_dtype == torch.float16
        is_bf16 = qkv_dtype == torch.bfloat16

        if not (is_fp16 or is_bf16):
            raise ValueError(
                f"unsupported qkv dtype: {qkv_dtype}. FlashAttention supports "
                "torch.float16 and torch.bfloat16 only"
            )

        eff_num_splits = get_num_splits(
            batch_size=batch_size,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            headdim=headdim,
            headdim_v=headdim_v,
            d_rounded=d_rounded,
            dv_rounded=dv_rounded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_k_new=max_seqlen_k_new,
            arch=arch,
            num_sm=num_sm,
            is_causal=final_is_causal,
            is_local=final_is_local,
            has_softcap=softcap,
            is_varlen=True,
            has_page_table=has_page_table,
            pack_gqa=pack_gqa,
            window_size_left=effective_window_left,
            window_size_right=effective_window_right,
            element_size=element_size,
            use_dynamic_split=use_dynamic_split,
        )
    else:
        eff_num_splits = num_splits

    eff_num_splits = min(eff_num_splits, 256, num_sm)

    #  Always enable PackGQA for Split
    pack_gqa = True if eff_num_splits > 1 else pack_gqa

    # Recompute qhead_per_khead/num_head_k for the kernels
    qhead_per_khead = (
        1 if not pack_gqa else (num_heads + num_heads_k - 1) // num_heads_k
    )
    num_head_k = num_heads_k if pack_gqa else num_heads

    is_varlen = True
    if arch >= 90:
        uomw = use_one_mma_wg(
            arch=arch,
            headdim=headdim,
            seqlen_q=max_seqlen_q,
            pack_gqa=pack_gqa,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
        )
        blockM, blockN = tile_size_fwd_sm90(
            headdim=round_up_headdim(headdim),
            headdim_v=round_up_headdimv(headdim_v),
            is_causal=final_is_causal,
            is_local=final_is_local,
            element_size=element_size,
            v_colmajor=False,
            paged_kv_non_TMA=(has_page_table and not pagedkv_tma),
            softcap=has_softcap,
            use_one_mma_wg=uomw,
        )
    else:
        blockM, blockN = get_optimal_block_mn(
            device=None,
            headdim=headdim,
            headdim_v=headdim_v,
            is_causal=final_is_causal,
            is_local=final_is_local,
            has_softcap=has_softcap,
            element_size=element_size,
            paged_kv=has_page_table,
            pagedkv_tma=pagedkv_tma,
            varlen_and_split=is_varlen and (eff_num_splits > 1),
            append_kv=(max_seqlen_k_new > 0),
        )

    # Semaphore slot: is_varlen is always True, so only arch and splits are considered
    scheduler_needs_semaphore = arch >= 90 or eff_num_splits > 1
    split_offset = int(scheduler_needs_semaphore)
    alloc_size = split_offset + int(use_dynamic_split) * batch_size

    return {
        "arch": arch,
        "num_sm": num_sm,
        "block_m": blockM,
        "block_n": blockN,
        "pack_gqa": pack_gqa,
        "qhead_per_khead": qhead_per_khead,
        "num_head_k": num_head_k,
        "num_splits": eff_num_splits,
        "use_dynamic_split": use_dynamic_split,
        "split_offset": split_offset,
        "alloc_size": alloc_size,
    }


def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    seqused_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME Get Scheduler Metadata")
    device = seqused_k.device
    dtype = torch.int32

    plan = _resolve_schedule_plan(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        leftpad_k=leftpad_k,
        page_size=page_size,
        max_seqlen_k_new=max_seqlen_k_new,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        has_softcap=has_softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    blockM = plan["block_m"]
    blockN = plan["block_n"]
    qhead_per_khead = plan["qhead_per_khead"]
    num_head_k = plan["num_head_k"]
    eff_num_splits = plan["num_splits"]
    use_dynamic_split = plan["use_dynamic_split"]
    split_offset = plan["split_offset"]
    alloc_size = plan["alloc_size"]
    num_sm = plan["num_sm"]

    seqlen_k = seqused_k
    seqlen_knew = (
        torch.full((batch_size,), max_seqlen_k_new, dtype=dtype, device=device)
        if max_seqlen_k_new > 0
        else None
    )

    BLOCK_SIZE_B = 128
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B),)

    if alloc_size <= 0:
        return torch.empty((0,), dtype=torch.int32, device=device)

    if not use_dynamic_split:
        # With no dynamic splitting (batch > 992) the metadata keeps only the semaphore slot, whose
        # content is always 0: pass1 only feeds the num_n_blocks/total_blocks statistics of pass2, so
        # it need not launch while one zeros call suffices, saving a launch and three scratch buffers.
        return torch.zeros(alloc_size, dtype=dtype, device=device)

    # A single zeros produces both the returned metadata (front segment) and 3 scratch buffers (back segments):
    # On GCU every torch tensor creation has a fixed launch overhead; after merging, only 3 device-side operations
    # remain in total
    # (zeros + pass1 + pass2). The metadata happens to be zero-initialized, so the semaphore slot needs no extra write.
    pool = torch.zeros(alloc_size + 2 * batch_size + 1, dtype=dtype, device=device)
    scheduler_metadata = pool[:alloc_size]
    num_m_blocks = pool[alloc_size : alloc_size + batch_size]
    num_n_blocks = pool[alloc_size + batch_size : alloc_size + 2 * batch_size]
    total_blocks = pool[alloc_size + 2 * batch_size :]

    _prepare_pass1_kernel[grid](
        num_m_blocks,
        num_n_blocks,
        total_blocks,
        seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        leftpad_k,
        batch_size,
        qhead_per_khead,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_new=max_seqlen_k_new,
        BLOCK_M=blockM,
        BLOCK_N=blockN,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        HAS_CU_SEQLENS_Q=cu_seqlens_q is not None,
        HAS_CU_SEQLENS_K=cu_seqlens_k is not None,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=True,
        HAS_LEFT_PAD=leftpad_k is not None,
        HAS_K_NEW=seqlen_knew is not None,
        HAS_CU_SEQLENS_K_NEW=cu_seqlens_k_new is not None,
    )

    if use_dynamic_split:
        # pass2 writes the splits segment directly into the returned metadata, saving a device->device copy
        _prepare_pass2_kernel[grid](
            num_n_blocks,
            scheduler_metadata,
            total_blocks,
            num_batch=batch_size,
            num_head=num_head_k,
            num_sm=num_sm,
            num_splits_static=eff_num_splits,
            SPLIT_OFFSET=split_offset,
            HAS_SEMAPHORE=False,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
        )

    return scheduler_metadata


def get_scheduler_metadata_ref(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    seqused_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> torch.Tensor:
    """Pure-host reference for :func:`get_scheduler_metadata`, used as test oracle.

    CUDA FA3 (``torch.ops._vllm_fa3_C.get_scheduler_metadata``) is the only
    upstream oracle and it does not exist off CUDA, so the metadata sequence is
    recomputed here with plain Python integer arithmetic on host scalars while
    the device path uses Triton kernels. The two entry points share only the
    scalar heuristics (:func:`_resolve_schedule_plan`); the per-sequence
    arithmetic is intentionally written twice so a porting drift in either side
    shows up as a mismatch.

    Covered: metadata length/layout, per-sequence split counts.
    Not covered: the tile-size and ``num_splits`` heuristics themselves, and the
    ``sm``-semaphore protocol of SM90+ (arch >= 90), where the caller must keep
    skipping.
    """
    plan = _resolve_schedule_plan(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        leftpad_k=leftpad_k,
        page_size=page_size,
        max_seqlen_k_new=max_seqlen_k_new,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        has_softcap=has_softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )

    alloc_size = plan["alloc_size"]
    if alloc_size <= 0:
        return torch.empty((0,), dtype=torch.int32)

    # The leading segment is the splits segment (possibly with 1 semaphore slot) and stays 0 without dynamic splitting
    metadata = [0] * alloc_size
    if not plan["use_dynamic_split"]:
        return torch.tensor(metadata, dtype=torch.int32)

    block_m = plan["block_m"]
    block_n = plan["block_n"]
    qhead_per_khead = plan["qhead_per_khead"]
    num_head_k = plan["num_head_k"]
    num_splits_static = plan["num_splits"]
    num_sm = plan["num_sm"]
    split_offset = plan["split_offset"]

    def _to_host_list(tensor, fallback_len):
        if tensor is None:
            return None
        return [int(v) for v in tensor.detach().to("cpu").reshape(-1).tolist()]

    seqused_k_host = _to_host_list(seqused_k, batch_size) or [0] * batch_size
    seqused_q_host = _to_host_list(seqused_q, batch_size)
    cu_seqlens_q_host = _to_host_list(cu_seqlens_q, batch_size + 1)
    cu_seqlens_k_new_host = _to_host_list(cu_seqlens_k_new, batch_size + 1)
    leftpad_k_host = _to_host_list(leftpad_k, batch_size)

    # q_len: scaled by qhead/khead after pack_gqa, then rounded up to block_m
    num_m_blocks_per_seq = []
    num_n_blocks_per_seq = []
    for batch_index in range(batch_size):
        if seqused_q_host is not None:
            q_len = seqused_q_host[batch_index]
        elif cu_seqlens_q_host is not None:
            q_len = cu_seqlens_q_host[batch_index + 1] - cu_seqlens_q_host[batch_index]
        else:
            q_len = max_seqlen_q
        q_len *= qhead_per_khead
        num_m_blocks_per_seq.append(_cdiv(q_len, block_m))

        k_len = seqused_k_host[batch_index]
        if max_seqlen_k_new > 0:
            if cu_seqlens_k_new_host is not None:
                k_len += (
                    cu_seqlens_k_new_host[batch_index + 1]
                    - cu_seqlens_k_new_host[batch_index]
                )
            else:
                k_len += max_seqlen_k_new
        if leftpad_k_host is not None:
            k_len -= leftpad_k_host[batch_index]
        num_n_blocks_per_seq.append(_cdiv(k_len, block_n))

    total_blocks = sum(
        m_blocks * n_blocks
        for m_blocks, n_blocks in zip(num_m_blocks_per_seq, num_n_blocks_per_seq)
    )

    # total_blocks takes part in the kernel fp32 estimate of how many blocks each SM handles
    total_blocks_f32 = np.float32(total_blocks)
    blocks_per_sm_f32 = np.float32(
        np.float32(total_blocks_f32 * np.float32(1.1)) * np.float32(num_head_k)
    ) / np.float32(num_sm)
    blocks_per_sm = max(1, int(np.ceil(np.float64(blocks_per_sm_f32))))

    for batch_index in range(batch_size):
        dynamic_splits = _cdiv(num_n_blocks_per_seq[batch_index], blocks_per_sm)
        dynamic_splits = min(dynamic_splits, num_splits_static)
        dynamic_splits = max(1, dynamic_splits)
        metadata[split_offset + batch_index] = dynamic_splits

    return torch.tensor(metadata, dtype=torch.int32)


def _cdiv(numerator: int, denominator: int) -> int:
    """Upward integer division with the kernel's non-negative operand domain."""
    return -(-int(numerator) // int(denominator))
