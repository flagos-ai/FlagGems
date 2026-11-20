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

# Ascend launch policy for dense, split-KV and variable-length FlashAttention.
import logging
import math

import torch
import triton

from flag_gems.ops.flash_api import CHECK_DEVICE, fwd_params
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .flash_kernel import flash_fwd_splitkv_combine_kernel, flash_varlen_fwd_kernel

logger = logging.getLogger(__name__)


def _mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    lse=None,
    optimize=False,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.device == k.device == v.device
    assert 0 <= p_dropout < 1
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    # max_num_pages_per_seq = page_table.size(1)
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (total_q, num_heads, head_size)
        assert out.device == q_device

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if lse is not None:
        assert lse.device == q_device
        assert lse.dtype == torch.float32
        assert lse.shape == (num_heads, total_q)
        assert lse.is_contiguous(), "LSE must be contiguous"

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = not is_causal and (window_size_left >= 0 or window_size_right >= 0)
    if is_local:
        if window_size_left < 0:
            window_size_left = max_seqlen_k
        if window_size_right < 0:
            window_size_right = max_seqlen_k

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and total_q == batch_size
        and lse is None
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
        # o_batch_stride = out.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        0 < head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        alibi_slopes_batch_stride = (
            alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
        )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=v.dtype)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=v.dtype)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        if lse is None:
            lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = (
                None
                if optimize
                else torch.empty((2,), dtype=torch.int64, device=q_device)
            )

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.zeros(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = None if optimize else torch.empty((), device=q_device)

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            seqused_k is None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
            k.stride(0) if is_paged else 0,  # k_page_stride,
        )

        logger.debug("kernel: flash_varlen_fwd")
        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            batch_size,
            num_heads,
        )
        kernel = flash_varlen_fwd_kernel[grid]
        args = tuple(getattr(params, k) for k in params.__slots__)

        # Bound on-chip storage independently of the GPU workload buckets.
        # M=128, D=72 padded to 128 needs >192 KiB UB on Ascend910.
        # A single pipeline stage also avoids multiplying local buffers.
        cfg_params = {
            "BLOCK_M": 16 if max_seqlen_q <= 16 or is_dropout else 32,
            "BLOCK_N": 32,
            "BLOCK_K": max(16, triton.next_power_of_2(head_size)),
            "num_warps": 4,
            "num_stages": 1,
        }

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
        if total_q > 0:
            kernel(*args, **cfg_params)

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.transpose(1, 2).reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = (
            None if optimize else torch.empty((), dtype=torch.int64, device=q_device)
        )
    return out, q, k, v, lse, philox_args, unused, p


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    return _mha_varlan_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        page_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
    )


def mha_varlan_fwd_opt(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    return _mha_varlan_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        page_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
        lse=lse,
        optimize=True,
    )


def mha_fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    p_dropout,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    disable_splitkv=False,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_dtype = q.dtype
    q_device = q.device
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert k.shape == v.shape
    assert q.device == k.device == v.device
    assert k.shape[0] == q.shape[0] and k.shape[-1] == q.shape[-1]
    assert 0 <= p_dropout < 1
    assert softcap <= 0 or p_dropout == 0
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    batch_size, seqlen_q, num_heads, head_size = q.size()
    _, seqlen_k, num_heads_k, _ = k.size()

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (batch_size, seqlen_q, num_heads, head_size)
        assert out.device == q_device

    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_k:
        window_size_right = -1
    if seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = not is_causal and (window_size_left >= 0 or window_size_right >= 0)
    if is_local:
        if window_size_left < 0:
            window_size_left = seqlen_k
        if window_size_right < 0:
            window_size_right = seqlen_k

    seqlenq_ngroups_swapped = False

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    assert (
        0 < head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"

    def splits_heuristic(num_tasks, num_sms, n_blocks):
        # splits when wave efficiency is low
        n_waves = triton.cdiv(num_tasks, num_sms)
        eff = (num_tasks / num_sms) / n_waves
        if eff > 0.8 or n_waves > 1:
            return 1

        min_blocks_per_split = 2
        best_splits = min(
            triton.cdiv(n_blocks, min_blocks_per_split),
            int(math.floor(1.0 / eff)),
            num_sms,
            16,  # Bound the combine kernel's on-chip storage on Ascend.
        )

        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
        )

        if out is None:
            out = torch.empty_like(q, dtype=v.dtype)

        # Set dropout params
        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.zeros(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        M_LOG2E = 1.4426950408889634074
        if softcap > 0.0:
            is_softcap = True
            adjusted_scale_softmax = softcap
            adjusted_softcap = softmax_scale / softcap
            adjusted_scale_softmax_log2e = softcap * M_LOG2E
        else:
            is_softcap = False
            adjusted_softcap = 0.0
            adjusted_scale_softmax = softmax_scale
            adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

        # Set alibi params
        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float,)
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
                batch_size,
                num_heads,
            )
            alibi_slopes_batch_stride = (
                alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
            )
            is_alibi = True
        else:
            alibi_slopes_batch_stride = 0
            is_alibi = False

        # One Ascend kernel handles dense rows and optional KV partitions.
        def dispatch(B, H, Q, K, D, params):
            if B == 0 or Q == 0:
                return
            block_m = 16 if Q <= 16 or is_dropout else 32
            block_n = 32
            cfg = {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": max(16, triton.next_power_of_2(D)),
                "num_warps": 4,
                "num_stages": 1,
                "DENSE": True,
            }
            if not is_dropout and not is_local and not disable_splitkv and K > 0:
                num_cores = torch_device_fn.get_device_properties(
                    q_device
                ).multi_processor_count
                n_blocks = triton.cdiv(K, block_n)
                n_splits = splits_heuristic(
                    B * H * triton.cdiv(Q, block_m), num_cores, n_blocks
                )
                if n_splits > 1:
                    lse_splits = torch.empty(
                        (n_splits, B, H, Q), dtype=torch.float, device=q_device
                    )
                    out_splits = torch.empty(
                        (n_splits, B, H, Q, D), dtype=torch.float, device=q_device
                    )
                    params.o_ptr = out_splits
                    params.softmax_lse_ptr = lse_splits
                    flash_varlen_fwd_kernel[(triton.cdiv(Q, block_m), n_splits, B * H)](
                        *params.args(),
                        SPLIT_KV=True,
                        blocks_per_split=triton.cdiv(n_blocks, n_splits),
                        **cfg,
                    )
                    combine_m = 4 if D >= 128 else (8 if D >= 64 else 16)
                    flash_fwd_splitkv_combine_kernel[
                        (triton.cdiv(B * H * Q, combine_m),)
                    ](
                        out_ptr=out,
                        lse_ptr=lse,
                        out_splits_ptr=out_splits,
                        lse_splits_ptr=lse_splits,
                        head_size=D,
                        out_split_stride=out_splits.stride(0),
                        lse_split_stride=lse_splits.stride(0),
                        out_b_stride=out.stride(0),
                        out_s_stride=out.stride(1),
                        out_h_stride=out.stride(2),
                        n_splits=n_splits,
                        BLOCK_M=combine_m,
                        BLOCK_K=cfg["BLOCK_K"],
                        q_total=B * H * Q,
                        MAX_N_SPLITS=triton.next_power_of_2(n_splits),
                        num_heads=H,
                        seqlen_q=Q,
                        num_warps=4,
                        num_stages=1,
                    )
                    return
            flash_varlen_fwd_kernel[(triton.cdiv(Q, block_m), B, H)](
                *params.args(), **cfg
            )

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q.stride(0),  # q_batch_stride,
            k.stride(0),  # k_batch_stride,
            v.stride(0),  # v_batch_stride,
            out.stride(0),  # o_batch_stride,
            False,  # is_cu_seqlens_q,
            None,  # cu_seqlens_q_ptr,
            False,  # is_cu_seqlens_k,
            None,  # cu_seqlens_k_ptr,
            False,  # is_seqused_k,
            None,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            0,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            seqlen_q,  # seqlen_q,
            seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            False,  # is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            0,  # total_q,
            None,  # page_table_ptr,
            0,  # page_table_batch_stride,
            0,  # block_size,
            0,  # k_page_stride,
        )

        dispatch(batch_size, num_heads, seqlen_q, seqlen_k, head_size, params)

        unused = torch.empty((), dtype=torch.int64, device=q_device)

    return out, q, k, v, lse, philox_args, unused, p
