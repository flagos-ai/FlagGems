"""Shared code for experimental Hopper FA3 WS kernels.

This file is mechanically split from ``flash_kernel_v3.py`` so the experimental
WS variants can live as standalone kernel modules while preserving the proven
TLE helper implementations.
"""

import os

import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

try:
    import triton.experimental.tle.language as tle

    TLE_FA3_AVAILABLE = True
except Exception:
    tle = None
    TLE_FA3_AVAILABLE = False
_FA3_TLE_FAMILY_LONG = 0
_FA3_TLE_FAMILY_SHORT = 1
_FA3_TLE_FAMILY_SPLITKV = 2
_FA3_TLE_FAMILY_MIXED = 3
_FA3_TLE_FAMILY_DECODE = 4
_FA3_TLE_FAMILY_PAGED_DECODE = 5
_FA3_TLE_FAMILY_SERVE = 6
_FA3_TLE_FAMILY_PAGED_SERVE = 7
_FA3_TLE_FAMILY_DIRECT = 8
_FA3_TLE_FAMILY_WS_SIMPLE = 9
_FA3_TLE_FAMILY_WS_SHORT = 10
_FA3_TLE_FAMILY_WS_SYNC_DECODE = 11
_FA3_TLE_FAMILY_WS_PIPE2_DECODE = 12
_FA3_TLE_FAMILY_WS_SYNC_SMALL = 13
_FA3_TLE_FAMILY_WS_SYNC_PAGED_DECODE = 14
_FA3_TLE_FAMILY_WS_PIPE2_PAGED_DECODE = 15
_FA3_TLE_FAMILY_AUTO = -1

_FA3_TLE_PAGED_GATHER_LEGACY = 0
_FA3_TLE_PAGED_GATHER_BLOCKWISE = 1
_FA3_TLE_PAGED_GATHER_AUTO = 2
_FA3_TLE_PAGED_GATHER_NAMES = {
    _FA3_TLE_PAGED_GATHER_LEGACY: "legacy",
    _FA3_TLE_PAGED_GATHER_BLOCKWISE: "blockwise",
    _FA3_TLE_PAGED_GATHER_AUTO: "auto",
}


def fa3_tle_paged_gather_mode() -> int:
    value = os.getenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", "auto").strip().lower()
    allowed = {
        "legacy": _FA3_TLE_PAGED_GATHER_LEGACY,
        "blockwise": _FA3_TLE_PAGED_GATHER_BLOCKWISE,
        "auto": _FA3_TLE_PAGED_GATHER_AUTO,
    }
    if value not in allowed:
        raise RuntimeError(
            "invalid FLAG_GEMS_FA3_TLE_PAGED_GATHER="
            f"{value!r}; expected one of {', '.join(sorted(allowed))}"
        )
    return allowed[value]


def fa3_tle_paged_gather_name(mode: int) -> str:
    return _FA3_TLE_PAGED_GATHER_NAMES.get(mode, f"unknown({mode})")

_FA3_TLE_BUCKET_LONG = 0
_FA3_TLE_BUCKET_SHORT = 1
_FA3_TLE_BUCKET_SPLITKV = 2
_FA3_TLE_BUCKET_MIXED_LONG = 3
_FA3_TLE_BUCKET_MIXED_SHORT = 4
_FA3_TLE_BUCKET_DECODE = 5
_FA3_TLE_BUCKET_PAGED_DECODE = 6
_FA3_TLE_BUCKET_SERVE_SHORT = 7
_FA3_TLE_BUCKET_PAGED_SERVE_SHORT = 8
_FA3_TLE_BUCKET_PAGED_SMALL = 9
_FA3_TLE_BUCKET_PAGED_MEDIUM = 10
_FA3_TLE_BUCKET_DIRECT_DECODE = 11
_FA3_TLE_BUCKET_DIRECT_PAGED_DECODE = 12
_FA3_TLE_BUCKET_DIRECT_SMALL = 13
_FA3_TLE_BUCKET_WS_SMALL_DENSE = 14
_FA3_TLE_BUCKET_WS_DECODE = 15
_FA3_TLE_BUCKET_WS_PAGED_DECODE = 16
_FA3_TLE_BUCKET_WS_SHORT_SMALL_DENSE = 17
_FA3_TLE_BUCKET_WS_SHORT_DECODE = 18
_FA3_TLE_BUCKET_WS_SHORT_PAGED_DECODE = 19
_FA3_TLE_BUCKET_WS_SYNC_DECODE = 20
_FA3_TLE_BUCKET_WS_PIPE2_DECODE = 21
_FA3_TLE_BUCKET_WS_SYNC_SMALL = 22
_FA3_TLE_BUCKET_WS_SYNC_PAGED_DECODE = 23
_FA3_TLE_BUCKET_WS_PIPE2_PAGED_DECODE = 24

_FA3_TLE_FORCE_PATHS = {
    "auto": _FA3_TLE_FAMILY_AUTO,
    "long": _FA3_TLE_FAMILY_LONG,
    "short": _FA3_TLE_FAMILY_SHORT,
    "splitkv": _FA3_TLE_FAMILY_SPLITKV,
    "mixed": _FA3_TLE_FAMILY_MIXED,
    "decode": _FA3_TLE_FAMILY_DECODE,
    "paged_decode": _FA3_TLE_FAMILY_PAGED_DECODE,
    "serve": _FA3_TLE_FAMILY_SERVE,
    "paged_serve": _FA3_TLE_FAMILY_PAGED_SERVE,
    "direct": _FA3_TLE_FAMILY_DIRECT,
    "ws_simple": _FA3_TLE_FAMILY_WS_SIMPLE,
    "ws_short": _FA3_TLE_FAMILY_WS_SHORT,
    "ws_sync_decode": _FA3_TLE_FAMILY_WS_SYNC_DECODE,
    "ws_pipe2_decode": _FA3_TLE_FAMILY_WS_PIPE2_DECODE,
    "ws_sync_small": _FA3_TLE_FAMILY_WS_SYNC_SMALL,
    "ws_sync_paged_decode": _FA3_TLE_FAMILY_WS_SYNC_PAGED_DECODE,
    "ws_pipe2_paged_decode": _FA3_TLE_FAMILY_WS_PIPE2_PAGED_DECODE,
}

_FA3_TLE_WS_CANDIDATE_NAMES = {
    _FA3_TLE_FAMILY_WS_SYNC_DECODE: "ws_sync_decode",
    _FA3_TLE_FAMILY_WS_PIPE2_DECODE: "ws_pipe2_decode",
    _FA3_TLE_FAMILY_WS_SYNC_SMALL: "ws_sync_small",
    _FA3_TLE_FAMILY_WS_SYNC_PAGED_DECODE: "ws_sync_paged_decode",
    _FA3_TLE_FAMILY_WS_PIPE2_PAGED_DECODE: "ws_pipe2_paged_decode",
}

_FA3_TLE_WS_CANDIDATE_BUCKETS = {
    _FA3_TLE_FAMILY_WS_SYNC_DECODE: _FA3_TLE_BUCKET_WS_SYNC_DECODE,
    _FA3_TLE_FAMILY_WS_PIPE2_DECODE: _FA3_TLE_BUCKET_WS_PIPE2_DECODE,
    _FA3_TLE_FAMILY_WS_SYNC_SMALL: _FA3_TLE_BUCKET_WS_SYNC_SMALL,
    _FA3_TLE_FAMILY_WS_SYNC_PAGED_DECODE: _FA3_TLE_BUCKET_WS_SYNC_PAGED_DECODE,
    _FA3_TLE_FAMILY_WS_PIPE2_PAGED_DECODE: _FA3_TLE_BUCKET_WS_PIPE2_PAGED_DECODE,
}

_FA3_TLE_WS_PIPE2_BUCKETS = (
    _FA3_TLE_BUCKET_WS_PIPE2_DECODE,
    _FA3_TLE_BUCKET_WS_PIPE2_PAGED_DECODE,
)

_FA3_TLE_WS_PAGED_DECODE_BUCKETS = (
    _FA3_TLE_BUCKET_WS_SHORT_PAGED_DECODE,
    _FA3_TLE_BUCKET_WS_SYNC_PAGED_DECODE,
    _FA3_TLE_BUCKET_WS_PIPE2_PAGED_DECODE,
)

_FA3_TLE_WS_SYNC_BUCKETS = (
    _FA3_TLE_BUCKET_WS_SHORT_SMALL_DENSE,
    _FA3_TLE_BUCKET_WS_SHORT_DECODE,
    _FA3_TLE_BUCKET_WS_SHORT_PAGED_DECODE,
    _FA3_TLE_BUCKET_WS_SYNC_DECODE,
    _FA3_TLE_BUCKET_WS_SYNC_SMALL,
    _FA3_TLE_BUCKET_WS_SYNC_PAGED_DECODE,
)


def _next_power_of_2_host(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _fa3_tle_config(
    *,
    family_id,
    block_m,
    block_n,
    num_buffers_kv,
    num_mma_groups,
    num_mma_warps,
    use_tma_qo,
    use_tma_kv,
):
    return triton.Config(
        {
            "FAMILY_ID": family_id,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": num_buffers_kv,
            "NUM_MMA_WARPS": num_mma_warps,
            "NUM_MMA_GROUPS": num_mma_groups,
            "Q_STAGE_CAPACITY": _next_power_of_2_host(num_mma_groups),
            "KV_STAGE_CAPACITY": _next_power_of_2_host(num_buffers_kv),
            "USE_TMA_QO": use_tma_qo,
            "USE_TMA_KV": use_tma_kv,
        },
        num_warps=4,
    )


def _fa3_tle_configs():
    return [
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_LONG,
            block_m=128,
            block_n=128,
            num_buffers_kv=2,
            num_mma_groups=2,
            num_mma_warps=8,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_LONG,
            block_m=128,
            block_n=64,
            num_buffers_kv=1,
            num_mma_groups=2,
            num_mma_warps=8,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SIMPLE,
            block_m=64,
            block_n=64,
            num_buffers_kv=1,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=False,
            use_tma_kv=False,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SIMPLE,
            block_m=64,
            block_n=128,
            num_buffers_kv=1,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=False,
            use_tma_kv=False,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SIMPLE,
            block_m=128,
            block_n=64,
            num_buffers_kv=1,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=False,
            use_tma_kv=False,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SIMPLE,
            block_m=128,
            block_n=128,
            num_buffers_kv=1,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=False,
            use_tma_kv=False,
        ),
    ]


def _fa3_tle_config_smem_bytes(cfg, head_dim: int) -> int:
    block_k = _next_power_of_2_host(head_dim)
    block_m = cfg.kwargs["BLOCK_M"]
    block_n = cfg.kwargs["BLOCK_N"]
    num_groups = cfg.kwargs["NUM_MMA_GROUPS"]
    bm_split = block_m // num_groups
    q_stage = cfg.kwargs["Q_STAGE_CAPACITY"]
    kv_stage = cfg.kwargs["KV_STAGE_CAPACITY"]
    elems = q_stage * bm_split * block_k
    elems += 2 * kv_stage * block_n * block_k
    return elems * 2


def _prune_fa3_tle_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    shape_bucket = kwargs.get(
        "SHAPE_BUCKET", nargs.get("SHAPE_BUCKET", _FA3_TLE_BUCKET_LONG)
    )
    force_family_id = kwargs.get(
        "FORCE_FAMILY_ID", nargs.get("FORCE_FAMILY_ID", _FA3_TLE_FAMILY_AUTO)
    )

    kept = []
    for cfg in configs:
        family_id = cfg.kwargs["FAMILY_ID"]
        block_n = cfg.kwargs["BLOCK_N"]
        block_m = cfg.kwargs["BLOCK_M"]
        num_groups = cfg.kwargs["NUM_MMA_GROUPS"]

        if force_family_id in (
            _FA3_TLE_FAMILY_LONG,
            _FA3_TLE_FAMILY_SHORT,
            _FA3_TLE_FAMILY_SPLITKV,
            _FA3_TLE_FAMILY_WS_SIMPLE,
        ) and family_id != force_family_id:
            continue
        if force_family_id == _FA3_TLE_FAMILY_MIXED:
            if shape_bucket == _FA3_TLE_BUCKET_MIXED_LONG:
                if family_id != _FA3_TLE_FAMILY_LONG:
                    continue
            elif family_id not in (
                _FA3_TLE_FAMILY_SHORT,
                _FA3_TLE_FAMILY_SPLITKV,
            ):
                continue

        if force_family_id == _FA3_TLE_FAMILY_AUTO:
            if shape_bucket in (_FA3_TLE_BUCKET_LONG, _FA3_TLE_BUCKET_MIXED_LONG):
                if family_id != _FA3_TLE_FAMILY_LONG:
                    continue
            elif shape_bucket == _FA3_TLE_BUCKET_SHORT:
                if family_id != _FA3_TLE_FAMILY_SHORT:
                    continue
            elif shape_bucket in (
                _FA3_TLE_BUCKET_WS_SMALL_DENSE,
                _FA3_TLE_BUCKET_WS_DECODE,
                _FA3_TLE_BUCKET_WS_PAGED_DECODE,
            ):
                if family_id != _FA3_TLE_FAMILY_WS_SIMPLE:
                    continue
            else:
                if family_id not in (
                    _FA3_TLE_FAMILY_SHORT,
                    _FA3_TLE_FAMILY_SPLITKV,
                ):
                    continue

        if family_id == _FA3_TLE_FAMILY_WS_SIMPLE:
            if shape_bucket == _FA3_TLE_BUCKET_WS_DECODE and block_m != 64:
                continue
            if (
                shape_bucket == _FA3_TLE_BUCKET_WS_PAGED_DECODE
                and (block_m != 64 or block_n != 64)
            ):
                continue
            if shape_bucket == _FA3_TLE_BUCKET_WS_SMALL_DENSE and block_m < 64:
                continue
            if head_dim >= 192 and block_n > 64:
                continue
            if is_paged and block_n > 64:
                continue

        if head_dim > 128 and block_n > 64:
            continue
        if (
            head_dim >= 128
            and family_id not in (_FA3_TLE_FAMILY_LONG, _FA3_TLE_FAMILY_WS_SIMPLE)
            and block_m > 64
        ):
            continue
        if head_dim > 128 and cfg.kwargs["NUM_BUFFERS_KV"] > 1 and block_n >= 64:
            continue
        if head_dim > 192 and block_n > 64:
            continue
        if block_m % num_groups != 0:
            continue
        if is_paged and block_n > 128:
            continue
        if _fa3_tle_config_smem_bytes(cfg, head_dim) > 220 * 1024:
            continue
        kept.append(cfg)

    if kept:
        return kept
    return [configs[0]]


def _fa3_ws_short_configs():
    return [
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SHORT,
            block_m=64,
            block_n=64,
            num_buffers_kv=1,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SHORT,
            block_m=64,
            block_n=128,
            num_buffers_kv=1,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SHORT,
            block_m=64,
            block_n=64,
            num_buffers_kv=2,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_WS_SHORT,
            block_m=64,
            block_n=128,
            num_buffers_kv=2,
            num_mma_groups=1,
            num_mma_warps=4,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
    ]


def _prune_fa3_ws_short_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    shape_bucket = kwargs.get(
        "WS_SHORT_SHAPE_BUCKET",
        nargs.get("WS_SHORT_SHAPE_BUCKET", _FA3_TLE_BUCKET_WS_SHORT_SMALL_DENSE),
    )

    kept = []
    for cfg in configs:
        block_n = cfg.kwargs["BLOCK_N"]
        num_buffers_kv = cfg.kwargs["NUM_BUFFERS_KV"]

        if shape_bucket in _FA3_TLE_WS_PIPE2_BUCKETS:
            if num_buffers_kv != 2:
                continue
        elif shape_bucket in _FA3_TLE_WS_SYNC_BUCKETS:
            if num_buffers_kv != 1:
                continue

        if shape_bucket in _FA3_TLE_WS_PAGED_DECODE_BUCKETS and block_n != 64:
            continue
        if head_dim >= 192 and block_n != 64:
            continue
        if is_paged and head_dim > 128 and block_n != 64:
            continue
        if _fa3_tle_config_smem_bytes(cfg, head_dim) > 220 * 1024:
            continue
        kept.append(cfg)

    if kept:
        return kept

    want_pipe2 = shape_bucket in _FA3_TLE_WS_PIPE2_BUCKETS
    for cfg in configs:
        if (cfg.kwargs["NUM_BUFFERS_KV"] == 2) == want_pipe2:
            return [cfg]
    return [configs[0]]


@triton.jit
def _apply_softcap_v3(S, softcap, IS_SOFTCAP: tl.constexpr):
    if IS_SOFTCAP:
        S = tl_extra_shim.tanh(S * softcap)
    return S


@triton.jit
def _apply_alibi_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    IS_CAUSAL: tl.constexpr,
    IS_ALIBI: tl.constexpr,
    alibi_slope,
):
    if IS_ALIBI:
        if IS_CAUSAL:
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias
    return S


@triton.jit
def _apply_mask_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    IS_EVEN_MN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if IS_CAUSAL or IS_LOCAL or (not IS_EVEN_MN):
        col_lb = tl.maximum(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = tl.minimum(
            max_seqlen_k - 1,
            row_idx + max_seqlen_k - max_seqlen_q + window_size_right,
        )
        if IS_CAUSAL:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)
        if IS_LOCAL:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )
        if (not IS_LOCAL) and (not IS_CAUSAL) and (not IS_EVEN_MN):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)
    return S


@triton.jit
def _softmax_online_deferred(
    S,
    m_prev,
    l_prev,
    softmax_scale_log2e: tl.constexpr,
    IS_BORDER: tl.constexpr,
):
    m_new = tl.maximum(m_prev, tl.max(S, 1))
    if IS_BORDER:
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    else:
        m_safe = m_new

    alpha = tl.math.exp2((m_prev - m_safe) * softmax_scale_log2e)
    m_scaled = tl.where(m_new == float("-inf"), 0.0, m_safe * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - m_scaled[:, None])
    l_new = l_prev * alpha + tl.sum(P, 1)
    return alpha, P, m_new, l_new


@triton.jit
def _virtual_to_cache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    BOUNDARY_CHECK: tl.constexpr = False,
):
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if BOUNDARY_CHECK:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int32)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int32)
    return page_block_index * block_size + page_offset


@triton.jit
def _paged_blockwise_cache_indices(
    n_start,
    offsets,
    max_virtual_index,
    page_table_ptr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr = True,
):
    logical_idx = n_start + offsets
    if PAGED_GATHER_MODE == 0 or (block_size != 16 and block_size != 32):
        return _virtual_to_cache(
            logical_idx,
            max_virtual_index,
            page_table_ptr,
            block_size,
            BOUNDARY_CHECK=BOUNDARY_CHECK,
        )

    cache_idx = logical_idx.to(tl.int32)
    if block_size == 16:
        for page_base in tl.static_range(0, BLOCK_N, 16):
            first_idx = n_start + page_base
            page_idx = first_idx // 16
            if BOUNDARY_CHECK:
                page_block = tl.load(
                    page_table_ptr + page_idx,
                    mask=first_idx < max_virtual_index,
                    other=0,
                ).to(tl.int32)
            else:
                page_block = tl.load(page_table_ptr + page_idx).to(tl.int32)
            in_page = (offsets >= page_base) & (offsets < page_base + 16)
            candidate = page_block * 16 + (logical_idx % 16)
            cache_idx = tl.where(in_page, candidate, cache_idx)
    else:
        for page_base in tl.static_range(0, BLOCK_N, 32):
            first_idx = n_start + page_base
            page_idx = first_idx // 32
            if BOUNDARY_CHECK:
                page_block = tl.load(
                    page_table_ptr + page_idx,
                    mask=first_idx < max_virtual_index,
                    other=0,
                ).to(tl.int32)
            else:
                page_block = tl.load(page_table_ptr + page_idx).to(tl.int32)
            in_page = (offsets >= page_base) & (offsets < page_base + 32)
            candidate = page_block * 32 + (logical_idx % 32)
            cache_idx = tl.where(in_page, candidate, cache_idx)
    return cache_idx


@triton.jit
def _decode_apply_alibi(
    scores,
    col_idx,
    row_idx,
    q_len,
    k_len,
    IS_CAUSAL: tl.constexpr,
    IS_ALIBI: tl.constexpr,
    alibi_slope,
):
    if IS_ALIBI:
        if IS_CAUSAL:
            scores += alibi_slope * (-k_len + 1 + col_idx).to(tl.float32)
        else:
            scores += -alibi_slope * tl.abs(col_idx - k_len + q_len - row_idx).to(
                tl.float32
            )
    return scores


@triton.jit
def _decode_apply_mask(
    scores,
    col_idx,
    row_idx,
    q_len,
    k_len,
    window_size_left,
    window_size_right,
    IS_BORDER: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if IS_BORDER:
        scores = tl.where(col_idx < k_len, scores, float("-inf"))
    if IS_CAUSAL or IS_LOCAL:
        col_lb = tl.maximum(0, row_idx + k_len - q_len - window_size_left)
        col_rb = tl.minimum(k_len - 1, row_idx + k_len - q_len + window_size_right)
        if IS_CAUSAL:
            scores = tl.where(col_idx <= col_rb, scores, float("-inf"))
        if IS_LOCAL:
            scores = tl.where(
                (col_idx >= col_lb) & (col_idx <= col_rb),
                scores,
                float("-inf"),
            )
    return scores


def _heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@triton.jit
def _buf_phase_tle(count, num_buffers: tl.constexpr):
    buf = count % num_buffers
    phase_idx = count // num_buffers
    return buf, phase_idx


@triton.jit
def _persistent_tile_coords(tile_idx, num_pid_m, batch_size):
    m_block = tile_idx % num_pid_m
    hb = tile_idx // num_pid_m
    bid = hb % batch_size
    hid = hb // batch_size
    return m_block, bid, hid


@triton.jit
def _fence_async_shared_cta():
    tl.inline_asm_elementwise(
        "mov.u32 $0, 0x0; membar.cta; fence.proxy.async.shared::cta;",
        constraints="=r",
        args=(),
        dtype=(tl.int32,),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _copy_paged_kv_tile_to_smem(
    src_base,
    row_stride,
    page_table_ptr_b,
    smem_tile,
    n_offset,
    k_len,
    d: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    for row_base in tl.static_range(0, BLOCK_N, 32):
        row_offsets = tl.arange(0, 32)
        rows = row_base + row_offsets
        logical_idx = n_offset + rows
        row_valid = logical_idx < k_len
        cache_idx = _paged_blockwise_cache_indices(
            n_offset + row_base,
            row_offsets,
            k_len,
            page_table_ptr_b,
            block_size,
            32,
            PAGED_GATHER_MODE,
            BOUNDARY_CHECK=True,
        )

        for col_base in tl.static_range(0, HEAD_DIM_PADDED, 32):
            cols = col_base + tl.arange(0, 32)
            src_ptrs = src_base + cache_idx[:, None] * row_stride + cols[None, :]
            load_mask = row_valid[:, None] & (cols[None, :] < d)
            # Keep paged gather loads on Triton's default cache policy. On this
            # TLE producer path, evict_last can lower to an illegal instruction.
            vals = tl.load(
                src_ptrs,
                mask=load_mask,
                other=0.0,
            )
            smem_rows = tl.broadcast_to(rows[:, None], (32, 32))
            smem_cols = tl.broadcast_to(cols[None, :], (32, 32))
            smem_ptrs = tle.gpu.local_ptr(smem_tile, (smem_rows, smem_cols))
            tl.store(smem_ptrs, vals)


# Debug-only variant. Do not use it in default WS paged paths: the extra
# arithmetic on loaded values has triggered runtime hangs in nonpersistent WS.
@triton.jit
def _copy_paged_kv_tile_to_smem_sync_safe(
    src_base,
    row_stride,
    page_table_ptr_b,
    smem_tile,
    n_offset,
    k_len,
    d: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    for row_base in tl.static_range(0, BLOCK_N, 32):
        row_offsets = tl.arange(0, 32)
        rows = row_base + row_offsets
        logical_idx = n_offset + rows
        row_valid = logical_idx < k_len
        cache_idx = _paged_blockwise_cache_indices(
            n_offset + row_base,
            row_offsets,
            k_len,
            page_table_ptr_b,
            block_size,
            32,
            PAGED_GATHER_MODE,
            BOUNDARY_CHECK=True,
        )

        for col_base in tl.static_range(0, HEAD_DIM_PADDED, 32):
            cols = col_base + tl.arange(0, 32)
            src_ptrs = src_base + cache_idx[:, None] * row_stride + cols[None, :]
            load_mask = row_valid[:, None] & (cols[None, :] < d)
            vals = tl.load(src_ptrs, mask=load_mask, other=0.0)
            vals = vals + 0.0
            smem_rows = tl.broadcast_to(rows[:, None], (32, 32))
            smem_cols = tl.broadcast_to(cols[None, :], (32, 32))
            smem_ptrs = tle.gpu.local_ptr(smem_tile, (smem_rows, smem_cols))
            tl.store(smem_ptrs, vals)


@triton.jit
def _copy_dense_tile_to_smem(
    src_base,
    row_stride,
    smem_tile,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    for row_base in tl.static_range(0, BLOCK_ROWS, 32):
        rows = row_base + tl.arange(0, 32)
        logical_rows = row_offset + rows
        row_valid = logical_rows < row_count
        for col_base in tl.static_range(0, HEAD_DIM_PADDED, 32):
            cols = col_base + tl.arange(0, 32)
            vals = tl.load(
                src_base + logical_rows[:, None] * row_stride + cols[None, :],
                mask=row_valid[:, None] & (cols[None, :] < d),
                other=0.0,
            )
            smem_rows = tl.broadcast_to(rows[:, None], (32, 32))
            smem_cols = tl.broadcast_to(cols[None, :], (32, 32))
            smem_ptrs = tle.gpu.local_ptr(smem_tile, (smem_rows, smem_cols))
            tl.store(smem_ptrs, vals)


@triton.jit
def _copy_dense_tile_to_smem_sync_safe(
    src_base,
    row_stride,
    smem_tile,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    for row_base in tl.static_range(0, BLOCK_ROWS, 32):
        rows = row_base + tl.arange(0, 32)
        logical_rows = row_offset + rows
        row_valid = logical_rows < row_count
        for col_base in tl.static_range(0, HEAD_DIM_PADDED, 32):
            cols = col_base + tl.arange(0, 32)
            vals = tl.load(
                src_base + logical_rows[:, None] * row_stride + cols[None, :],
                mask=row_valid[:, None] & (cols[None, :] < d),
                other=0.0,
            )
            vals = vals + 0.0
            smem_rows = tl.broadcast_to(rows[:, None], (32, 32))
            smem_cols = tl.broadcast_to(cols[None, :], (32, 32))
            smem_ptrs = tle.gpu.local_ptr(smem_tile, (smem_rows, smem_cols))
            tl.store(smem_ptrs, vals)


@triton.jit
def _store_dense_tile_from_regs(
    dst_base,
    row_stride,
    vals,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    rows = row_offset + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    ptrs = dst_base + rows[:, None] * row_stride + cols[None, :]
    mask = (rows[:, None] < row_count) & (cols[None, :] < d)
    tl.store(ptrs, vals, mask=mask)




__all__ = [name for name in globals() if not name.startswith("__")]
