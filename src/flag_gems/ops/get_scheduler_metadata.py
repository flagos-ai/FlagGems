import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


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


def get_optimal_block_mn(
    device,
    headdim,
    headdim_v,
    is_causal,
    is_local,
    has_softcap,
    element_size=2,
    paged_kv=False,
    varlen_and_split=False,
    append_kv=False,
):
    arch_cap = torch_device_fn.get_device_capability(device)
    arch = arch_cap[0] * 10 + arch_cap[1]
    sm86_or_89 = arch == 86 or arch == 89

    kBlockM, kBlockN, kNWarps, kStages, Q_in_regs = tile_size_fwd_sm8x(
        sm86_or_89=sm86_or_89,
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


def round_up_headdim(headdim: int) -> int:
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


def round_up_headdimv(headdim_v: int) -> int:
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


def use_one_mma_wg(
    arch: int,
    headdim: int,
    seqlen_q: int,
    pack_gqa: bool,
    num_heads: int,
    num_heads_k: int,
) -> bool:
    if arch < 90 or headdim != 128:
        return False

    qhead_per_khead = 1 if not pack_gqa else num_heads // num_heads_k
    effective_seqlen_q = seqlen_q * qhead_per_khead

    return effective_seqlen_q <= 64


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
    element_size: int = 2,  # fp16/bf16 = 2, fp8 = 1
    max_splits: int = 128,
    use_dynamic_split: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> int:
    pagedkv_tma = False
    append_kv = max_seqlen_k_new > 0

    if arch >= 90:
        # TODO: tile_size_fwd_sm90
        kBlockM, kBlockN = get_optimal_block_mn(
            device=0,
            headdim=d_rounded,
            headdim_v=dv_rounded,
            is_causal=is_causal,
            is_local=is_local,
            has_softcap=has_softcap,
            element_size=element_size,
            paged_kv=has_page_table and not pagedkv_tma,
            varlen_and_split=is_varlen,
            append_kv=append_kv,
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
            0, min(max_seqlen_k, window_size_right + window_size_left + 1 + kBlockM)
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
def _prepare_scheduler_kernel(
    seqlen_q_static: tl.constexpr,
    seqlen_k_static: tl.constexpr,
    seqlen_k_new_static: tl.constexpr,
    num_batch: tl.constexpr,
    num_head: tl.constexpr,
    qhead_per_khead: tl.constexpr,
    num_sm: tl.constexpr,
    num_splits_static: tl.constexpr,
    blockm_divisor: tl.constexpr,
    blockn_divisor: tl.constexpr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    cu_seqlens_k_new_ptr,
    seqused_q_ptr,
    seqused_k_ptr,
    leftpad_k_ptr,
    total_blocks_ptr,
    temp_num_n_ptr,
    num_splits_dynamic_ptr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_CU_SEQLENS_K_NEW: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    HAS_LEFT_PAD: tl.constexpr,
    HAS_SEMAPHORE: tl.constexpr,
):
    total_blocks = 0

    b = 0
    while b < num_batch:
        seqlen_q = 0
        if HAS_SEQUSED_Q:
            seqlen_q = tl.load(seqused_q_ptr + b)
        elif HAS_CU_SEQLENS_Q:
            cur = tl.load(cu_seqlens_q_ptr + b)
            nxt = tl.load(cu_seqlens_q_ptr + b + 1)
            seqlen_q = nxt - cur
        else:
            seqlen_q = seqlen_q_static

        seqlen_q *= qhead_per_khead
        num_m_blocks = (seqlen_q + blockm_divisor - 1) // blockm_divisor

        seqlen_k = 0
        if HAS_SEQUSED_K:
            seqlen_k = tl.load(seqused_k_ptr + b)
        elif HAS_CU_SEQLENS_K:
            cur_k = tl.load(cu_seqlens_k_ptr + b)
            nxt_k = tl.load(cu_seqlens_k_ptr + b + 1)
            seqlen_k = nxt_k - cur_k
        else:
            seqlen_k = seqlen_k_static

        seqlen_k_new = 0
        if HAS_CU_SEQLENS_K_NEW:
            cur_kn = tl.load(cu_seqlens_k_new_ptr + b)
            nxt_kn = tl.load(cu_seqlens_k_new_ptr + b + 1)
            seqlen_k_new = nxt_kn - cur_kn
        else:
            seqlen_k_new = seqlen_k_new_static

        leftpad_k = 0
        if HAS_LEFT_PAD:
            leftpad_k = tl.load(leftpad_k_ptr + b)

        seqlen_k_total = seqlen_k - leftpad_k + seqlen_k_new
        num_n_blocks = (seqlen_k_total + blockn_divisor - 1) // blockn_divisor

        total_blocks += num_m_blocks * num_n_blocks
        tl.store(temp_num_n_ptr + b, num_n_blocks)

        b += 1

    if HAS_SEMAPHORE:
        tl.store(total_blocks_ptr, 0)

    total_blocks_f = total_blocks.to(tl.float32)
    blocks_per_sm_f = tl.ceil(total_blocks_f * 1.1 * num_head / num_sm)
    blocks_per_sm_int32 = blocks_per_sm_f.to(tl.int32)
    one = 1  # Python int
    blocks_per_sm = tl.where(blocks_per_sm_int32 > one, blocks_per_sm_int32, one)

    b = 0
    while b < num_batch:
        num_n = tl.load(temp_num_n_ptr + b)

        ns = (num_n + blocks_per_sm - 1) // blocks_per_sm
        ns = tl.where(ns < one, one, ns)
        ns = tl.where(ns > num_splits_static, num_splits_static, ns)

        tl.store(num_splits_dynamic_ptr + b, ns)
        b += 1


def get_pack_gqa(
    arch: int,
    has_page_table: bool,
    pagedkv_tma: bool,
    num_splits: int,
    num_heads: int,
    num_heads_k: int,
) -> bool:
    if arch < 90 or (has_page_table and not pagedkv_tma) or num_splits > 1:
        return True

    if num_heads == num_heads_k:
        return False

    # TODO implement tile_size_fwd_sm90 and should_pack_gqa (Hopper+ only)
    return False


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
    disable_split_k: bool = True,
) -> torch.Tensor:
    device = seqused_k.device
    dtype = torch.int32

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

    arch_cap = torch_device_fn.get_device_capability(device)
    arch = arch_cap[0] * 10 + arch_cap[1]
    num_sm = (
        torch_device_fn.get_device_properties(device).multi_processor_count - sm_margin
    )

    softcap = 1.0 if has_softcap else 0.0

    element_size = qkv_dtype.itemsize

    has_page_table = page_size is not None

    # TODO implement get_pagedkv_tma function (Hopper+ only)
    pagedkv_tma = False

    # GQA
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
        )
    )
    qhead_per_khead = (
        1 if not pack_gqa else (num_heads + num_heads_k - 1) // num_heads_k
    )
    num_head_k = num_heads_k if pack_gqa else num_heads

    # TODO: implement use_one_mma_wg (Hopper+ only)

    if num_splits <= 0:
        element_size = qkv_dtype.itemsize
        d_rounded = round_up_headdim(headdim)
        dv_rounded = round_up_headdimv(headdim_v)

        use_dynamic_split_for_heuristic = batch_size <= 992
        is_varlen_for_heuristic = (
            (cu_seqlens_q is not None)
            or (cu_seqlens_k is not None)
            or (seqused_q is not None)
            or (seqused_k is not None)
            or (leftpad_k is not None)
        )

        heuristic_num_splits = get_num_splits(
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
            is_varlen=is_varlen_for_heuristic,
            has_page_table=(page_size is not None),
            element_size=element_size,
            use_dynamic_split=use_dynamic_split_for_heuristic,
            window_size_left=effective_window_left,
            window_size_right=effective_window_right,
        )
    else:
        heuristic_num_splits = num_splits

    if heuristic_num_splits > 1:
        pack_gqa = True
        qhead_per_khead = (num_heads + num_heads_k - 1) // num_heads_k
        num_head_k = num_heads_k

    kernel_launched = batch_size <= 992

    scheduler_needs_semaphore = arch >= 90 or heuristic_num_splits > 1
    use_dynamic_split = batch_size <= 992

    alloc_size = int(scheduler_needs_semaphore) + int(use_dynamic_split) * batch_size

    if alloc_size == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)

    scheduler_metadata = torch.empty(alloc_size, dtype=torch.int32, device=device)
    semaphore_ptr = scheduler_metadata[:1] if scheduler_needs_semaphore else None
    splits_array_ptr = (
        scheduler_metadata[int(scheduler_needs_semaphore) :]
        if use_dynamic_split
        else None
    )

    if kernel_launched:
        if semaphore_ptr is not None:
            semaphore_ptr.zero_()

        temp_num_n = torch.empty(batch_size, dtype=dtype, device=device)
        num_splits_static_for_kernel = min(heuristic_num_splits, 256, num_sm)

        d_rounded_kernel = round_up_headdim(headdim)
        dv_rounded_kernel = (
            d_rounded_kernel if headdim_v == headdim else round_up_headdimv(headdim_v)
        )
        varlen_and_split_for_kernel = (
            (cu_seqlens_q is not None)
            or (cu_seqlens_k is not None)
            or (seqused_q is not None)
            or (seqused_k is not None)
            or (leftpad_k is not None)
        ) and (heuristic_num_splits > 1)
        blockM_kernel, blockN_kernel = get_optimal_block_mn(
            device=device,
            headdim=d_rounded_kernel,
            headdim_v=dv_rounded_kernel,
            is_causal=final_is_causal,
            is_local=final_is_local,
            has_softcap=has_softcap,
            element_size=element_size,
            paged_kv=has_page_table,
            varlen_and_split=varlen_and_split_for_kernel,
            append_kv=(max_seqlen_k_new > 0),
        )

        grid = (1,)
        _prepare_scheduler_kernel[grid](
            max_seqlen_q,
            max_seqlen_k,
            max_seqlen_k_new,
            batch_size,
            num_head_k,
            qhead_per_khead,
            num_sm,
            num_splits_static_for_kernel,
            blockM_kernel,
            blockN_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_k_new,
            seqused_q,
            seqused_k,
            leftpad_k,
            semaphore_ptr,
            temp_num_n,
            splits_array_ptr,
            HAS_CU_SEQLENS_Q=cu_seqlens_q is not None,
            HAS_CU_SEQLENS_K=cu_seqlens_k is not None,
            HAS_CU_SEQLENS_K_NEW=cu_seqlens_k_new is not None,
            HAS_SEQUSED_Q=seqused_q is not None,
            HAS_SEQUSED_K=seqused_k is not None,
            HAS_LEFT_PAD=leftpad_k is not None,
            HAS_SEMAPHORE=semaphore_ptr is not None,
        )
    else:
        if semaphore_ptr is not None:
            semaphore_ptr.zero_()

        if splits_array_ptr is not None:
            final_num_splits_for_fill = heuristic_num_splits
            final_num_splits_for_fill = heuristic_num_splits
            if disable_split_k:
                final_num_splits_for_fill = 1
            final_num_splits_for_fill = min(final_num_splits_for_fill, 256, num_sm)

            splits_array_ptr.fill_(final_num_splits_for_fill)

    return scheduler_metadata
