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
import os

import torch
import triton

import flag_gems
from flag_gems import runtime
from flag_gems.ops.flash_kernel import (
    block_m_splitkv_heuristic,
    block_n_splitkv_heuristic,
    flash_fwd_kernel,
    flash_fwd_splitkv_combine_kernel,
    flash_fwd_splitkv_kernel,
    flash_varlen_fwd_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
_debug = False


# ---------------------------------------------------------------------------
# Phase-aware tile selection for the paged varlen forward kernel.
#
# A scheduler step normally mixes long prefill chunks with single-token decodes.
# Their best tile shapes are far apart: a 256-row tile is ~2.7x faster than the
# 128-row one for prefill on BI-V150, while a decode row only fills 1 of the
# 256 lanes.  Any statistic derived from the batch average therefore picks a
# tile that is wrong for both phases, so mixed batches are launched twice (one
# pass per phase) and each pass skips the rows owned by the other phase.
# ---------------------------------------------------------------------------
_PD_SPLIT = os.environ.get("FLAGOS_FLASH_PD_SPLIT", "1") != "0"
_DECODE_SWAP = os.environ.get("FLAGOS_FLASH_DECODE_SWAP", "1") != "0"
_PREFILL_COMPACT = os.environ.get("FLAGOS_FLASH_PREFILL_COMPACT", "0") != "0"
_SPLITKV = os.environ.get("FLAGOS_FLASH_DECODE_SPLITKV", "0") != "0"
_SPLITKV_NUM = int(os.environ.get("FLAGOS_FLASH_SPLITKV_NUM", "8"))
_FLASH_DEBUG = os.environ.get("FLAGOS_FLASH_DEBUG", "0") == "1"
_DEBUG_SEEN = set()
# Per-step cache for the decode-row compaction: (data_ptr, numel, _version).
_DEC_CACHE = {}
_DIRECT_GQA = True
_DUMMY_PHILOX = {}
_DUMMY_EMPTY = {}


def _get_dummy_philox(device):
    idx = device.index if device.index is not None else 0
    t = _DUMMY_PHILOX.get(idx)
    if t is None:
        t = torch.zeros((2,), dtype=torch.int64, device=device)
        _DUMMY_PHILOX[idx] = t
    return t


def _get_dummy_empty(device):
    idx = device.index if device.index is not None else 0
    t = _DUMMY_EMPTY.get(idx)
    if t is None:
        t = torch.empty((), dtype=torch.float32, device=device)
        _DUMMY_EMPTY[idx] = t
    return t


def _debug_tile(tag, max_seqlen_q, batch_size, total_q, head_size, cfg):
    if not _FLASH_DEBUG:
        return
    key = (tag, max_seqlen_q, batch_size, cfg["BLOCK_M"], cfg["BLOCK_N"])
    if key in _DEBUG_SEEN:
        return
    _DEBUG_SEEN.add(key)
    print(
        f"[flash_varlen] {tag} max_q={max_seqlen_q} batch={batch_size} "
        f"total_q={total_q} d={head_size} BM={cfg['BLOCK_M']} BN={cfg['BLOCK_N']} "
        f"warps={cfg['num_warps']} stages={cfg['num_stages']}",
        flush=True,
    )


_VARLEN_TILES = {
    # phase: (BLOCK_M, BLOCK_N, num_warps, num_stages)
    "decode": (16, 32, 4, 1),
    "prefill_xl": (256, 64, 16, 3),
    "prefill_l": (128, 64, 8, 3),
    "prefill_m": (64, 32, 4, 3),
}


def _varlen_tile(max_seqlen_q, head_size, decode=False):
    """Tile shape for one phase of ``flash_varlen_fwd_kernel``."""
    if decode:
        bm, bn, nw, ns = _VARLEN_TILES["decode"]
    elif max_seqlen_q >= 256 and head_size <= 128:
        bm, bn, nw, ns = _VARLEN_TILES["prefill_xl"]
    elif max_seqlen_q >= 64 and head_size <= 128:
        bm, bn, nw, ns = _VARLEN_TILES["prefill_l"]
    else:
        bm, bn, nw, ns = _VARLEN_TILES["prefill_m"]
    return {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "num_stages": ns}


def CHECK_DEVICE(x):
    assert x.device.type == flag_gems.device


class fwd_params:
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "is_softcap",
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        # dropout
        "is_dropout",
        "p_dropout",
        "rp_dropout",
        "p_dropout_in_uint8_t",
        "philox_args",
        "return_softmax",
        # masking
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        "is_paged",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
        "k_page_stride",
    )

    def __init__(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        # scaling factors
        is_softcap,
        softcap,
        scale_softmax,
        scale_softmax_log2,
        # dropout
        is_dropout,
        p_dropout,
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        # masking
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
        k_page_stride,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_k_rounded = seqlen_k_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.is_softcap = is_softcap
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        # dropout
        self.is_dropout = is_dropout
        self.p_dropout = p_dropout
        self.rp_dropout = rp_dropout
        self.p_dropout_in_uint8_t = p_dropout_in_uint8_t
        self.philox_args = philox_args
        self.return_softmax = return_softmax
        # masking
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        self.is_paged = is_paged
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size
        self.k_page_stride = k_page_stride

    def args(self):
        return tuple(getattr(self, k) for k in self.__slots__)


_SEQLEN_Q_SLOT = fwd_params.__slots__.index("seqlen_q")


def _split_prefix(cu_seqlens_q, max_seqlen_q):
    """Return (n_decode, token_offset) when decodes form a prefix, else None."""
    if max_seqlen_q <= 1:
        return None
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    lens = q_lens.tolist()
    n_dec = 0
    while n_dec < len(lens) and lens[n_dec] == 1:
        n_dec += 1
    if n_dec == 0 or n_dec == len(lens):
        return None
    if any(length <= 1 for length in lens[n_dec:]):
        return None
    return n_dec, int(cu_seqlens_q[n_dec].item())


def _launch_prefill_sliced(
    q,
    out,
    page_table,
    seqused_k,
    cu_seqlens_q,
    num_heads,
    head_size,
    n_dec,
    base,
    cfg,
    base_args,
):
    """Prefill-only launch over the batch suffix (no idle decode CTAs)."""
    slots = fwd_params.__slots__
    pre_args = list(base_args)

    def _set(name, value):
        pre_args[slots.index(name)] = value

    _set("q_ptr", q[base:])
    _set("o_ptr", out[base:])
    _set("cu_seqlens_q_ptr", cu_seqlens_q[n_dec:] - base)
    _max_q = cfg.pop("_max_q")
    _set("seqlen_q", _max_q)
    _set("b", cu_seqlens_q.numel() - 1 - n_dec)
    _set("total_q", q.shape[0] - base)
    if seqused_k is not None:
        _set("seqused_k_ptr", seqused_k[n_dec:])
    if page_table is not None:
        _set("page_table_ptr", page_table[n_dec:])
        _set("page_table_batch_stride", page_table.stride(0))
    n_pre = cu_seqlens_q.numel() - 1 - n_dec
    flash_varlen_fwd_kernel[(triton.cdiv(_max_q, cfg["BLOCK_M"]), n_pre, num_heads)](
        *pre_args, **cfg, SPLIT_MODE=1
    )


def _launch_decode_swapped(
    q,
    out,
    block_table,
    seqused_k,
    cu_seqlens_q,
    num_heads,
    num_heads_k,
    head_size,
    cfg,
    base_args,
):
    """Run the single-token rows of a mixed batch as a compacted swapped batch.

    A decode row fills one lane of a BLOCK_M-row tile (1/16 utilisation) when it
    shares a launch with prefill chunks.  Gathering the decode rows and applying
    the same GQA swap that the pure-decode path uses puts q_groups rows in the
    tile instead, and one KV head per program keeps K/V loads shared.
    """
    q_groups = num_heads // num_heads_k
    # The metadata buffers are reused across steps and written in place, so
    # (data_ptr, numel, _version) identifies a step's contents exactly.  Caching
    # removes the per-layer .item()/nonzero synchronisation (42 per step).
    _key = (cu_seqlens_q.data_ptr(), cu_seqlens_q.numel(), cu_seqlens_q._version)
    if _DEC_CACHE.get("key") == _key:
        n_dec, dec_idx, rows = _DEC_CACHE["val"]
    else:
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        dec_mask = q_lens == 1
        from flag_gems.ops.nonzero import nonzero

        dec_idx = nonzero(dec_mask).flatten()
        n_dec = dec_idx.numel()
        if n_dec == 0:
            _DEC_CACHE["key"] = _key
            _DEC_CACHE["val"] = (0, None, None)
            return
        rows = cu_seqlens_q.index_select(0, dec_idx).to(torch.int64)
        _DEC_CACHE["key"] = _key
        _DEC_CACHE["val"] = (n_dec, dec_idx, rows)
    if n_dec == 0:
        return
    q_dec = q.index_select(0, rows)
    q_sw = (
        q_dec.reshape(n_dec, num_heads_k, q_groups, head_size)
        .transpose(1, 2)
        .reshape(n_dec * q_groups, num_heads_k, head_size)
    )
    o_sw = torch.empty_like(q_sw)
    lse_sw = torch.empty(
        (num_heads_k, n_dec * q_groups), dtype=torch.float, device=q.device
    )
    bt_dec = block_table.index_select(0, dec_idx) if block_table is not None else None
    sk_dec = seqused_k.index_select(0, dec_idx) if seqused_k is not None else None

    slots = fwd_params.__slots__
    dec_args = list(base_args)

    def _set(name, value):
        dec_args[slots.index(name)] = value

    _set("q_ptr", q_sw)
    _set("o_ptr", o_sw)
    _set("softmax_lse_ptr", lse_sw)
    _set("is_cu_seqlens_q", False)
    _set("seqlen_q", q_groups)
    _set("b", n_dec)
    _set("h", num_heads_k)
    _set("h_hk_ratio", 1)
    _set("q_row_stride", q_sw.stride(-3))
    _set("q_head_stride", q_sw.stride(-2))
    _set("o_row_stride", o_sw.stride(-3))
    _set("o_head_stride", o_sw.stride(-2))
    _set("q_batch_stride", q_sw.stride(0) * q_groups)
    _set("o_batch_stride", o_sw.stride(0) * q_groups)
    _set("seqlenq_ngroups_swapped", True)
    # Each tile row is a *head*, not a query position, so the bottom-right
    # causal alignment must be switched off (same as the pure-decode swap path).
    _set("is_causal", False)
    _set("total_q", n_dec * q_groups)
    if sk_dec is not None:
        _set("is_seqused_k", True)
        _set("seqused_k_ptr", sk_dec)
    if bt_dec is not None:
        _set("page_table_ptr", bt_dec)
        _set("page_table_batch_stride", bt_dec.stride(0))

    flash_varlen_fwd_kernel[
        (triton.cdiv(q_groups, cfg["BLOCK_M"]), n_dec, num_heads_k)
    ](*dec_args, **cfg, SPLIT_MODE=0)

    # o_sw rows are laid out as [seq][group][kv_head]; rebuild [seq][head] with
    # head = kv_head * q_groups + group (the inverse of the gather above).
    out_sw = (
        o_sw.view(n_dec, q_groups, num_heads_k, head_size)
        .transpose(1, 2)
        .reshape(n_dec, num_heads, head_size)
    )
    out.index_copy_(0, rows, out_sw)


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
    need_lse=True,
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

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0

    # Address query heads directly in the original tensor. No CPU inspection,
    # compaction, temporary Q/O, metadata cache, or scatter is needed.
    direct_gqa = (
        _DIRECT_GQA
        and flag_gems.vendor_name == "iluvatar"
        and is_paged
        and seqused_k is not None
        and cu_seqlens_q is not None
        and 1 < num_heads // num_heads_k <= 16
        and alibi_slopes is None
        and window_size_left < 0
        and window_size_right <= 0
        and p_dropout == 0
        and softcap == 0
        and not return_softmax
    )

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and not direct_gqa
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
        head_size <= 256
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

    # Decode fast path: one query token per sequence => Flash-Decoding style
    # split over the KV range (bandwidth-bound on BI-V150).
    if (
        _SPLITKV
        and is_paged
        and cu_seqlens_q is not None
        and seqused_k is not None
        and max_seqlen_q == 1
        and num_heads > num_heads_k
        and p_dropout == 0
        and not return_softmax
    ):
        from flag_gems.ops.flash_decode_splitkv import flash_decode_splitkv

        if out is None:
            out = torch.empty_like(q)
        with torch_device_fn.device(q_device):
            flash_decode_splitkv(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                seqused_k,
                page_table,
                softmax_scale,
                num_splits=_SPLITKV_NUM,
            )
            lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)
        return out, q, k, v, lse, None, None, None

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

        lse = (
            torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)
            if (need_lse or zero_tensors or not direct_gqa or _PREFILL_COMPACT)
            else None
        )

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
                _get_dummy_philox(q_device)
                if lse is None
                else torch.empty((2,), dtype=torch.int64, device=q_device)
            )

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = (
                _get_dummy_empty(q_device)
                if lse is None
                else torch.empty((), device=q_device)
            )

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse if lse is not None else p,  # softmax_lse_ptr,
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

        # Head-interleaved pages use the raw tensor pointer and the explicit
        # page, token and head strides already populated above.
        if flag_gems.vendor_name == "iluvatar" and (
            not is_paged or k.stride(-2) == k.shape[-1]
        ):
            params.k_ptr = k.view(k.shape[0], k.shape[1], -1)
            params.v_ptr = v.view(v.shape[0], v.shape[1], -1)
        logger.debug("kernel: flash_varlen_fwd")
        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            batch_size,
            num_heads,
        )
        kernel = flash_varlen_fwd_kernel[grid]
        args = tuple(getattr(params, k) for k in params.__slots__)

        block_k = triton.next_power_of_2(head_size)
        # Phase-aware dispatch for the paged varlen kernel.
        #
        # The scheduler packs long prefill chunks together with single-token
        # decodes into one batch.  Their optimal tile shapes are far apart: a
        # 256-row tile is ~2.7x faster than the 128-row one for prefill on
        # BI-V150, while a decode row only fills 1 of the 256 lanes.  Anything
        # derived from the batch average therefore lands on a tile that is wrong
        # for both phases, so mixed batches are launched twice -- once for rows
        # with q_len > 1 and once for rows with q_len == 1 -- and each launch
        # bails out on the rows owned by the other phase (SPLIT_MODE).
        split_pd = (
            _PD_SPLIT
            and flag_gems.vendor_name == "iluvatar"
            and is_paged
            and cu_seqlens_q is not None
            and not seqlenq_ngroups_swapped
            and max_seqlen_q > 1
            and batch_size > 1
            and not return_softmax
            and not is_dropout
        )
        use_page_tiles = (
            direct_gqa
            and num_heads == 16
            and num_heads_k == 2
            and head_size == 128
            and q_dtype == torch.bfloat16
            and block_size == 16
            and k.stride() == v.stride()
            and page_table.stride(1) == 1
        )
        prefill_group = (
            8 if use_page_tiles and max_seqlen_q >= 32 and not _PREFILL_COMPACT else 1
        )
        write_lse = lse is not None

        def prefill_tile():
            cfg = _varlen_tile(max_seqlen_q, head_size, decode=False)
            cfg["BLOCK_K"] = block_k
            cfg["WRITE_LSE"] = write_lse
            if prefill_group > 1:
                cfg.update(
                    BLOCK_M=128 if max_seqlen_q < 128 else 256,
                    BLOCK_N=128,
                    num_warps=8 if max_seqlen_q < 128 else 16,
                    num_stages=1,
                    PREFILL_GQA_GROUP=prefill_group,
                )
            return cfg

        def launch_direct_decode(cfg):
            from flag_gems.ops.flash_decode_splitkv import (
                flash_decode_splitkv,
                get_decode_workspace,
            )

            workspace = get_decode_workspace(q.device, num_heads_k, q_groups, head_size)
            if (
                workspace is not None
                and batch_size <= 64
                and num_heads == 16
                and num_heads_k == 2
                and head_size == 128
            ):
                max_k_offset = sum(
                    (size - 1) * stride for size, stride in zip(k.shape, k.stride())
                )
                page_tiles = use_page_tiles and max_k_offset < (1 << 31)
                merge_width = 0
                if page_tiles:
                    if batch_size <= 2:
                        splits, dec_warps = 32, 2
                    elif batch_size <= 4:
                        splits, dec_warps = 16, 2
                    elif batch_size <= 12:
                        splits, dec_warps = 8, 2
                    elif batch_size <= 20:
                        splits, dec_warps = 4, 2
                    elif batch_size <= 26:
                        splits, dec_warps = 8, 1
                    elif batch_size <= 32:
                        splits, dec_warps = 16, 1
                    elif batch_size == 48:
                        splits, dec_warps = 4, 1
                    elif batch_size <= 60:
                        splits, dec_warps = 8, 1
                    else:
                        splits, dec_warps = 4, 2
                    if batch_size in (4, 8, 31, 32):
                        merge_width = 32
                else:
                    splits = (
                        32
                        if batch_size <= 2
                        else 16 if batch_size <= 4 else 8 if batch_size <= 32 else 4
                    )
                    dec_warps = 4
                flash_decode_splitkv(
                    q,
                    k,
                    v,
                    out,
                    cu_seqlens_q,
                    seqused_k,
                    page_table,
                    softmax_scale,
                    num_splits=splits,
                    block_n=32,
                    num_stages=1,
                    num_warps=dec_warps,
                    page_tiles=page_tiles,
                    workspace=workspace,
                    lse_out=lse,
                    merge_width=merge_width,
                )
                return
            direct_args = list(args)
            direct_args[fwd_params.__slots__.index("is_causal")] = False
            flash_varlen_fwd_kernel[(1, batch_size, num_heads_k)](
                *direct_args, **cfg, DIRECT_GQA=True, WRITE_LSE=write_lse
            )

        long_meta = None
        if (
            use_page_tiles
            and is_causal
            and not _PREFILL_COMPACT
            and (batch_size == 1 or split_pd)
            and 1 <= batch_size <= 64
            and (128 if max_seqlen_k >= 2048 else 1024) <= max_seqlen_q <= total_q
            and max_seqlen_q * 2 >= total_q
            and 1024 <= max_seqlen_k <= 32768
            and softmax_scale > 0
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
            and cu_seqlens_q.stride(0) == seqused_k.stride(0) == 1
            and cu_seqlens_q.dtype == seqused_k.dtype == page_table.dtype == torch.int32
            and all(
                sum((n - 1) * s for n, s in zip(t.shape, t.stride())) < (1 << 31)
                for t in (q, out)
            )
            and not torch.cuda.is_current_stream_capturing()
        ):
            from flag_gems.fused.iluvatar_dense_prefill import dense_long_query

            long_meta = dense_long_query(
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q,
                seqused_k,
                page_table,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
            )

        if direct_gqa and max_seqlen_q == 1:
            dec_cfg = _varlen_tile(1, head_size, decode=True)
            dec_cfg["BLOCK_K"] = block_k
            launch_direct_decode(dec_cfg)
        elif split_pd:
            pre_cfg = prefill_tile()
            _debug_tile(
                "prefill", max_seqlen_q, batch_size, total_q, head_size, pre_cfg
            )
            logger.debug("flash_varlen_fwd prefill pass: %s", pre_cfg)
            _split = (
                _split_prefix(cu_seqlens_q, max_seqlen_q) if _PREFILL_COMPACT else None
            )
            if long_meta is not None:
                q_tile = pre_cfg["BLOCK_M"] // prefill_group
                # With the longest sequence removed, no remaining sequence can
                # own more than half of the packed Q rows.
                rows = triton.cdiv(min(max_seqlen_q, total_q // 2), q_tile)
                tiles = min(
                    rows * batch_size, triton.cdiv(total_q, q_tile) + batch_size - 1
                )
                pre_cfg["PREFILL_BATCH_WIDTH"] = triton.next_power_of_2(batch_size)
                flash_varlen_fwd_kernel[(num_heads // prefill_group, tiles, 1)](
                    *args,
                    **pre_cfg,
                    SPLIT_MODE=1,
                    EXCLUDE_SELECTED=True,
                    selected_meta=long_meta,
                )
            elif _split is not None:
                _n_dec, _base = _split
                pre_kw = dict(pre_cfg)
                pre_kw["_max_q"] = max_seqlen_q
                _launch_prefill_sliced(
                    q,
                    out,
                    page_table,
                    seqused_k,
                    cu_seqlens_q,
                    num_heads,
                    head_size,
                    _n_dec,
                    _base,
                    pre_kw,
                    args,
                )
            else:
                pre_args = list(args)
                pre_args[_SEQLEN_Q_SLOT] = max_seqlen_q
                q_tile = pre_cfg["BLOCK_M"] // prefill_group
                rows = triton.cdiv(max_seqlen_q, q_tile)
                grid = (rows, batch_size, num_heads // prefill_group)
                if (
                    direct_gqa
                    and num_heads == 16
                    and num_heads_k == 2
                    and head_size == 128
                    and 8 <= batch_size <= 64
                    and (
                        (max_seqlen_q >= 256 and max_seqlen_k <= 8192)
                        or (max_seqlen_q >= 32 and total_q <= 512)
                    )
                ):
                    # Each nonempty sequence adds less than one tile of rounding.
                    tiles = min(
                        rows * batch_size, triton.cdiv(total_q, q_tile) + batch_size - 1
                    )
                    grid = (num_heads // prefill_group, tiles, 1)
                    pre_cfg["PREFILL_BATCH_WIDTH"] = triton.next_power_of_2(batch_size)
                flash_varlen_fwd_kernel[grid](*pre_args, **pre_cfg, SPLIT_MODE=1)

            dec_cfg = _varlen_tile(1, head_size, decode=True)
            dec_cfg["BLOCK_K"] = block_k
            if direct_gqa:
                launch_direct_decode(dec_cfg)
            elif _DECODE_SWAP:
                _launch_decode_swapped(
                    q,
                    out,
                    page_table,
                    seqused_k,
                    cu_seqlens_q,
                    num_heads,
                    num_heads_k,
                    head_size,
                    dec_cfg,
                    args,
                )
            else:
                dec_args = list(args)
                dec_args[_SEQLEN_Q_SLOT] = 1
                logger.debug("flash_varlen_fwd decode pass: %s", dec_cfg)
                flash_varlen_fwd_kernel[(1, batch_size, num_heads)](
                    *dec_args, **dec_cfg, SPLIT_MODE=2
                )
        elif (
            flag_gems.vendor_name == "iluvatar"
            and max_seqlen_q > 1
            and not seqlenq_ngroups_swapped
        ):
            # Single prefill batch (typical when one long prompt owns the whole
            # token budget): use the large prefill tile directly.
            if long_meta is None:
                cfg_params = prefill_tile()
                q_tile = cfg_params["BLOCK_M"] // prefill_group
                heads = num_heads // prefill_group
                _debug_tile(
                    "single", max_seqlen_q, batch_size, total_q, head_size, cfg_params
                )
                logger.debug("flash_varlen_fwd single prefill pass: %s", cfg_params)
                flash_varlen_fwd_kernel[
                    (triton.cdiv(max_seqlen_q, q_tile), batch_size, heads)
                ](*args, **cfg_params, SPLIT_MODE=0)
        elif flag_gems.vendor_name == "iluvatar" and (
            max_seqlen_q <= 1 or seqlenq_ngroups_swapped
        ):
            # Pure decode step: one query token per sequence (the GQA swap turns
            # it into q_groups rows per CTA).  Small tile with a single pipeline
            # stage keeps shared-memory use low so several CTAs fit per SM.
            cfg_params = _varlen_tile(1, head_size, decode=True)
            cfg_params["BLOCK_K"] = block_k
            _debug_tile(
                "decode", max_seqlen_q, batch_size, total_q, head_size, cfg_params
            )
            flash_varlen_fwd_kernel[
                (
                    triton.cdiv(max_seqlen_q, cfg_params["BLOCK_M"]),
                    batch_size,
                    num_heads,
                )
            ](*args, **cfg_params, SPLIT_MODE=0)
        else:
            # Single-phase batch (pure prefill or pure decode).
            total_rows = total_q * num_heads
            num_sms = torch_device_fn.get_device_properties(
                flag_gems.device
            ).multi_processor_count
            avg_rows_per_sm = total_rows / num_sms
            avg_rows_per_batch = total_q / batch_size
            avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
            if avg_rows_per_cta > 64:
                varlen_fwd_config_str = "mha_block_128"
            elif avg_rows_per_cta > 32:
                varlen_fwd_config_str = "mha_block_64"
            elif avg_rows_per_cta > 16:
                varlen_fwd_config_str = "mha_block_32"
            else:
                varlen_fwd_config_str = "mha_block_16"
            if flag_gems.vendor_name == "mthreads":
                varlen_fwd_config_str = "mha_block_32"

            cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
            cfg_params = {
                "BLOCK_M": cfg["BLOCK_M"](args),
                "BLOCK_N": cfg["BLOCK_N"](args),
                "BLOCK_K": block_k,
                "num_warps": cfg["num_warps"](args),
                "num_stages": 1 if not is_paged else cfg["num_stages"](args),
            }

            logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
            kernel(*args, **cfg_params, SPLIT_MODE=0)

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
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = torch.empty((), dtype=torch.int64, device=q_device)
    return out, q, k, v, lse, philox_args, unused, p


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
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.emtpty((0, 0), device=q_device, dtype=torch.int32)

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

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
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
        head_size <= 256
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
            # philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)
            philox_args = None

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            # p = torch.empty((), device=q_device)
            p = None
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
            cu_seqlens_k is not None,  # is_cu_seqlens_k,
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

        # Head-interleaved pages use the raw tensor pointer and the explicit
        # page, token and head strides already populated above.
        if flag_gems.vendor_name == "iluvatar" and (
            not is_paged or k.stride(-2) == k.shape[-1]
        ):
            params.k_ptr = k.view(k.shape[0], k.shape[1], -1)
            params.v_ptr = v.view(v.shape[0], v.shape[1], -1)
        logger.debug("kernel: flash_varlen_fwd")
        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            batch_size,
            num_heads,
        )
        kernel = flash_varlen_fwd_kernel[grid]
        args = tuple(getattr(params, k) for k in params.__slots__)

        # We assess which phase the requests are likely to be in and set the config accordingly.
        total_rows = total_q * num_heads
        num_sms = torch_device_fn.get_device_properties(
            flag_gems.device
        ).multi_processor_count
        avg_rows_per_sm = total_rows / num_sms
        avg_rows_per_batch = total_q / batch_size
        avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
        # Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
        # This is a rough heuristic and may not be accurate for all scenarios.
        if avg_rows_per_cta > 64:
            varlen_fwd_config_str = "mha_block_128"
        elif avg_rows_per_cta > 32:
            varlen_fwd_config_str = "mha_block_64"
        elif avg_rows_per_cta > 16:
            varlen_fwd_config_str = "mha_block_32"
        else:
            varlen_fwd_config_str = "mha_block_16"
        if flag_gems.vendor_name == "mthreads":
            varlen_fwd_config_str = "mha_block_32"

        cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
        cfg_params = {
            "BLOCK_M": cfg["BLOCK_M"](args),
            "BLOCK_N": cfg["BLOCK_N"](args),
            "BLOCK_K": triton.next_power_of_2(head_size),
            "num_warps": cfg["num_warps"](args),
            "num_stages": 1 if not is_paged else cfg["num_stages"](args),
        }

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
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
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        # unused = torch.empty((), dtype=torch.int64, device=q_device)
        unused = None
    return out, q, k, v, lse, philox_args, unused, p


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
        CHECK_DEVICE(out)

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
    is_local = window_size_left >= 0 and window_size_right >= 0

    seqlenq_ngroups_swapped = (
        seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k

    if seqlenq_ngroups_swapped:
        logger.debug("q_kg swapped.")
        q = q.reshape(batch_size, num_heads_k, q_groups, head_size).transpose(1, 2)
        seqlen_q = q_groups
        num_heads = num_heads_k

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert head_size == head_size_rounded, "head_size must be rounded to 32"

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
        )

        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
        )

        if out is not None:
            if seqlenq_ngroups_swapped:
                out = out.reshape(
                    batch_size, num_heads_k, q_groups, head_size
                ).transpose(1, 2)
        else:
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
            p = torch.empty(
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

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Do kernel dispatching
        def dispatch(B, H, Q, K, D, params):
            num_sms = torch_device_fn.get_device_properties(
                "cuda"
            ).multi_processor_count

            # Try bh parallel
            # if B * H > 0.8 * num_sms:
            #     kernel = flash_fwd_bh_parallel_kernel[(H, B)]
            #     # Yield kernel and prefilled args
            #     return kernel, default_args, None, None

            # Try splitkv
            if not is_dropout and not is_local and not disable_splitkv:
                BM = block_m_splitkv_heuristic(D)
                n_tasks = B * H * triton.cdiv(seqlen_q, BM)
                BN = block_n_splitkv_heuristic(D)
                n_blocks = triton.cdiv(seqlen_k, BN)
                n_splits = splits_heuristic(n_tasks, num_sms, n_blocks)

                if n_splits > 1:
                    logger.debug("kernel: flash_fwd_splitkv")
                    lse_splits = torch.empty(
                        (n_splits, B, H, Q), dtype=torch.float, device=q_device
                    )
                    out_splits = torch.empty(
                        (n_splits, B, H, Q, D), dtype=torch.float, device=q_device
                    )
                    grid = lambda args: (
                        triton.cdiv(Q, args["BLOCK_M"]),
                        n_splits,
                        B * H,
                    )
                    splitkv_kernel = flash_fwd_splitkv_kernel[grid]
                    params.o_ptr = out_splits
                    params.softmax_lse_ptr = lse_splits
                    extra_args = {"blocks_per_split": triton.cdiv(n_blocks, n_splits)}
                    kernel = splitkv_kernel(*params.args(), **extra_args)

                    if D >= 128:
                        BLOCK_M = 4
                    elif D >= 64:
                        BLOCK_M = 8
                    else:
                        BLOCK_M = 16
                    BLOCK_K = triton.next_power_of_2(D)
                    grid = lambda args: (triton.cdiv(B * H * Q, BLOCK_M),)
                    combine_kernel = flash_fwd_splitkv_combine_kernel[grid]
                    combine_args = {
                        "out_ptr": out,
                        "lse_ptr": lse,
                        "head_size": head_size,
                        "out_split_stride": out_splits.stride(0),
                        "lse_split_stride": lse_splits.stride(0),
                        "out_b_stride": out.stride(0),
                        "out_s_stride": out.stride(-3),
                        # Head stride is the second-to-last dim of the BSHD view;
                        # stride(-1) is the contiguous head-size stride.
                        "out_h_stride": out.stride(-2),
                        "out_splits_ptr": out_splits,
                        "lse_splits_ptr": lse_splits,
                        "n_splits": n_splits,
                        "BLOCK_M": BLOCK_M,
                        "BLOCK_K": BLOCK_K,
                        "q_total": B * H * Q,
                        "MAX_N_SPLITS": triton.next_power_of_2(n_splits),
                        "num_heads": H,
                        "seqlen_q": Q,
                    }
                    combine_kernel(**combine_args)
                    return kernel

            # Last option: flash_fwd
            logger.debug("kernel: flash_fwd")
            grid = lambda args: (
                triton.cdiv(Q, args["BLOCK_M"]),
                H * B,
            )
            kernel = flash_fwd_kernel[grid]
            kernel = kernel(*params.args())
            return kernel

        if _debug:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=torch.float32,
                device=q_device,
            )
            return_softmax = True

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

        # Move TxD to last dims for correct stride in Triton tt.load
        if flag_gems.vendor_name == "iluvatar":
            params.q_ptr = q.transpose(1, 2)
            params.k_ptr = k.transpose(1, 2)
            params.v_ptr = v.transpose(1, 2)
        kernel = dispatch(batch_size, num_heads, seqlen_q, seqlen_k, head_size, params)

        if _debug:
            print(f"{kernel.name} shared memory:", kernel.metadata.shared)
            print(f"{kernel.name} num_warps:", kernel.metadata.num_warps)
            print(f"{kernel.name} num_stages:", kernel.metadata.num_stages)
            # print(kernel.asm['ttgir'])

        if seqlenq_ngroups_swapped:
            out = out.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            q = q.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            lse = lse.reshape((batch_size, num_heads_k * seqlen_q, 1))

        unused = torch.empty((), dtype=torch.int64, device=q_device)

    return out, q, k, v, lse, philox_args, unused, p
