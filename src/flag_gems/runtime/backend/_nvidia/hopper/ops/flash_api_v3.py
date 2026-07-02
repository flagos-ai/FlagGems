"""
Host-side launcher for the TLE-only Hopper FA3 varlen forward kernel.

The public signature and return tuple intentionally match
``flag_gems.ops.flash_api.mha_varlan_fwd`` so the Hopper attention override can
dispatch to this file for ``fa_version=3``.  Unsupported FA3 inputs fail fast
with a clear RuntimeError; the older non-TLE FA3 kernel is intentionally absent.
"""

import logging
import os

import torch
import triton

import flag_gems
from flag_gems.ops.flash_api import fwd_params
from flag_gems.runtime import torch_device_fn

from .flash_kernel_v3 import (
    TLE_FA3_AVAILABLE,
    fa3_tle_paged_gather_mode,
    fa3_tle_paged_gather_name,
    flash_varlen_fwd_v3_tle_direct_kernel,
    flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel,
    flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel,
    flash_varlen_fwd_v3_tle_kernel,
    flash_varlen_fwd_v3_tle_short_kernel,
    fa3_tle_metadata_dispatch,
)
from .fa3_ws.utils import (
    _FA3_TLE_BUCKET_DIRECT_DECODE,
    _FA3_TLE_BUCKET_DIRECT_PAGED_DECODE,
    _FA3_TLE_BUCKET_LONG,
    _FA3_TLE_BUCKET_PAGED_MEDIUM,
    _FA3_TLE_BUCKET_PAGED_SMALL,
    _FA3_TLE_BUCKET_SHORT,
    _FA3_TLE_FAMILY_AUTO,
)

logger = logging.getLogger(__name__)

_TMA_ALLOCATOR_REGISTERED = False


def _check_device(x):
    if x.device.type != flag_gems.device:
        raise RuntimeError(f"expected {flag_gems.device} tensor, got {x.device}")


def _ensure_tma_allocator():
    global _TMA_ALLOCATOR_REGISTERED
    if _TMA_ALLOCATOR_REGISTERED:
        return
    if not hasattr(triton, "set_allocator"):
        raise RuntimeError(
            "TLE FA3 requires Triton with on-device TMA descriptors "
            "(missing triton.set_allocator)."
        )

    def _alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_REGISTERED = True


def _auto_splitkv_from_metadata() -> bool:
    return os.getenv("FLAG_GEMS_FA3_TLE_AUTO_SPLITKV") == "1"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid {name}={value!r}; expected an integer") from exc


def _paged_prefill_route() -> str:
    value = os.getenv("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_ROUTE", "auto")
    value = value.strip().lower()
    allowed = {"auto", "direct", "long"}
    if value not in allowed:
        raise RuntimeError(
            "invalid FLAG_GEMS_FA3_TLE_PAGED_PREFILL_ROUTE="
            f"{value!r}; expected one of {', '.join(sorted(allowed))}"
        )
    return value


def is_fa3_supported() -> bool:
    if not TLE_FA3_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        import triton.language as tl

        return hasattr(tl, "make_tensor_descriptor") and hasattr(
            triton, "set_allocator"
        )
    except Exception:
        return False


def _round_multiple(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _tma_strides_are_aligned(tensor: torch.Tensor) -> bool:
    elem_bytes = tensor.element_size()
    return all((stride * elem_bytes) % 16 == 0 for stride in tensor.stride()[:-1])


def _ws_short_tma_strides_are_aligned(q, k, v, out, is_paged: bool) -> bool:
    if not _tma_strides_are_aligned(q) or not _tma_strides_are_aligned(out):
        return False
    if is_paged:
        return True
    return _tma_strides_are_aligned(k) and _tma_strides_are_aligned(v)


def _require_tle_supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    return_softmax,
    leftpad_k,
):
    if not is_fa3_supported():
        raise RuntimeError(
            "TLE FA3 requires CUDA Hopper, Triton TMA descriptors, and "
            "triton.experimental.tle."
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "TLE FA3 currently supports torch.float16 and torch.bfloat16 inputs only."
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("TLE FA3 requires q, k, and v to have the same dtype.")
    if p_dropout != 0:
        raise RuntimeError("TLE FA3 does not support dropout.")
    if return_softmax:
        raise RuntimeError("TLE FA3 does not support returning the softmax matrix.")
    if leftpad_k is not None:
        raise RuntimeError("TLE FA3 does not support leftpad_k.")
    if q.ndim != 3:
        raise RuntimeError(f"TLE FA3 expects q with shape (total_q, h, d), got {q.shape}.")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("TLE FA3 requires q/k/v to be contiguous in the head dimension.")
    if cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous():
        raise RuntimeError("TLE FA3 requires contiguous int32 cu_seqlens_q.")
    if cu_seqlens_k.dtype != torch.int32 or not cu_seqlens_k.is_contiguous():
        raise RuntimeError("TLE FA3 requires contiguous int32 cu_seqlens_k placeholder.")
    if seqused_k is not None and (
        seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous()
    ):
        raise RuntimeError("TLE FA3 requires contiguous int32 seqused_k when provided.")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise RuntimeError("TLE FA3 requires positive max_seqlen_q and max_seqlen_k.")

    head_size = q.shape[-1]
    if head_size < 32 or head_size > 256 or head_size % 8 != 0:
        raise RuntimeError("TLE FA3 requires 32 <= head_dim <= 256 and head_dim % 8 == 0.")

    is_paged = page_table is not None
    if is_paged:
        if page_table.dtype != torch.int32 or page_table.ndim != 2:
            raise RuntimeError("TLE FA3 paged mode requires an int32 2D block table.")
        if page_table.stride(-1) != 1:
            raise RuntimeError("TLE FA3 paged mode requires contiguous block-table rows.")
        if seqused_k is None:
            raise RuntimeError("TLE FA3 paged mode requires seqused_k.")
        if k.ndim != 4 or v.ndim != 4:
            raise RuntimeError("TLE FA3 paged mode expects k/v cache shape (pages, block, hk, d).")
    else:
        if k.ndim != 3 or v.ndim != 3:
            raise RuntimeError("TLE FA3 dense mode expects k/v shape (total_k, hk, d).")

    if out is not None and out.dtype != q.dtype:
        raise RuntimeError("TLE FA3 requires output dtype to match q.")
    if alibi_slopes is not None:
        if alibi_slopes.dtype != torch.float32 or alibi_slopes.stride(-1) != 1:
            raise RuntimeError("TLE FA3 requires fp32 ALiBi slopes with last stride 1.")


def mha_varlan_fwd_v3(
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
    scheduler_metadata=None,
    num_splits=0,
):
    _check_device(q)
    _check_device(k)
    _check_device(v)
    q_device = q.device
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    num_splits = int(num_splits or 0)
    if num_splits < 0:
        raise RuntimeError("TLE FA3 requires num_splits >= 0.")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    _require_tle_supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        page_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        return_softmax,
        leftpad_k,
    )

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)

    if k.size() != v.size():
        raise RuntimeError("TLE FA3 requires k and v to have the same shape.")
    if cu_seqlens_q.size() != (batch_size + 1,):
        raise RuntimeError("cu_seqlens_q must have shape (batch_size + 1,).")
    if cu_seqlens_k.size() != (batch_size + 1,):
        raise RuntimeError("cu_seqlens_k must have shape (batch_size + 1,).")
    if seqused_k is not None and seqused_k.size() != (batch_size,):
        raise RuntimeError("seqused_k must have shape (batch_size,).")
    if scheduler_metadata is not None:
        if scheduler_metadata.dtype != torch.int32 or not scheduler_metadata.is_contiguous():
            raise RuntimeError("scheduler_metadata must be contiguous int32 when provided.")
        if scheduler_metadata.device != q_device:
            raise RuntimeError("scheduler_metadata must be on the same device as q.")
    if num_heads % num_heads_k != 0:
        raise RuntimeError("TLE FA3 requires num_heads % num_heads_k == 0.")

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1
    is_local = window_size_left >= 0

    metadata_max_query_len = max_seqlen_q
    metadata_num_heads = num_heads
    metadata_cu_seqlens_q = cu_seqlens_q

    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
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
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)
    if q.shape != (total_q, num_heads, head_size):
        raise RuntimeError("internal TLE FA3 q shape mismatch after optional swap.")
    if is_paged:
        expected = (num_pages, block_size, num_heads_k, head_size)
        if k.shape != expected or v.shape != expected:
            raise RuntimeError(f"TLE FA3 expected paged k/v shape {expected}.")
    if k.stride() != v.stride():
        raise RuntimeError("TLE FA3 requires k and v to have matching strides.")

    if alibi_slopes is not None:
        if alibi_slopes.device != q_device:
            raise RuntimeError("ALiBi slopes must be on the same device as q.")
        if alibi_slopes.shape == (num_heads,):
            alibi_slopes_batch_stride = 0
        elif alibi_slopes.shape == (batch_size, num_heads):
            alibi_slopes_batch_stride = alibi_slopes.stride(0)
        else:
            raise RuntimeError(
                "ALiBi slopes must have shape (num_heads,) or (batch_size, num_heads)."
            )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    if softcap > 0.0:
        is_softcap = True
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax = softcap
        adjusted_scale_softmax_log2e = softcap * 1.4426950408889634
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * 1.4426950408889634

    def _metadata_max_num_splits(metadata) -> int:
        if metadata is None or metadata.numel() <= 1:
            return 1
        dynamic = metadata[1 : 1 + batch_size]
        if dynamic.numel() == 0:
            return 1
        return max(1, int(dynamic.max().item()))

    scheduler_metadata_source = "none"
    effective_num_splits = max(0, int(num_splits or 0))
    auto_splitkv = _auto_splitkv_from_metadata()
    if seqused_k is not None:
        if scheduler_metadata is not None:
            scheduler_metadata_source = "explicit_metadata"
            if effective_num_splits <= 0:
                metadata_num_splits = _metadata_max_num_splits(scheduler_metadata)
                effective_num_splits = metadata_num_splits if auto_splitkv else 1
                if not auto_splitkv:
                    scheduler_metadata_source = "explicit_metadata_no_auto_split"
        elif effective_num_splits <= 0:
            scheduler_metadata = flag_gems.get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=metadata_max_query_len,
                max_seqlen_k=max_seqlen_k,
                num_heads=metadata_num_heads,
                num_heads_k=num_heads_k,
                headdim=head_size,
                headdim_v=head_size,
                qkv_dtype=q.dtype,
                seqused_k=seqused_k,
                cu_seqlens_q=metadata_cu_seqlens_q,
                cu_seqlens_k=None,
                page_size=block_size if is_paged else None,
                is_causal=is_causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                has_softcap=is_softcap,
                num_splits=0,
            )
            metadata_num_splits = _metadata_max_num_splits(scheduler_metadata)
            effective_num_splits = metadata_num_splits if auto_splitkv else 1
            scheduler_metadata_source = (
                "auto_metadata_split"
                if auto_splitkv
                else "auto_metadata_no_auto_split"
            )
        else:
            scheduler_metadata_source = "explicit_num_splits"
    effective_num_splits = max(1, effective_num_splits)

    head_size_rounded = _round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = _round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = _round_multiple(max_seqlen_k, 32)

    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q)
        else:
            out_ = None
            out = torch.empty_like(q)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q_device)
        p = torch.empty((), device=q_device)
        philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        params = fwd_params(
            q,
            k,
            v,
            out,
            p,
            lse,
            q.stride(-3),
            k.stride(-3),
            v.stride(-3),
            q.stride(-2),
            k.stride(-2),
            v.stride(-2),
            out.stride(-3),
            out.stride(-2),
            q_batch_stride,
            k_batch_stride,
            v_batch_stride,
            o_batch_stride,
            cu_seqlens_q is not None,
            cu_seqlens_q,
            seqused_k is None,
            cu_seqlens_k,
            seqused_k is not None,
            seqused_k,
            batch_size,
            k_batch_size,
            num_heads,
            num_heads_k,
            num_heads // num_heads_k,
            max_seqlen_q,
            max_seqlen_k,
            seqlen_q_rounded,
            seqlen_k_rounded,
            head_size,
            head_size_rounded,
            is_softcap,
            adjusted_softcap,
            adjusted_scale_softmax,
            adjusted_scale_softmax_log2e,
            False,
            0.0,
            1.0,
            255,
            philox_args,
            False,
            is_causal,
            is_local,
            window_size_left,
            window_size_right,
            seqlenq_ngroups_swapped,
            is_paged,
            is_alibi,
            alibi_slopes,
            alibi_slopes_batch_stride,
            total_q,
            page_table,
            page_table_batch_stride,
            block_size,
        )

        _ensure_tma_allocator()
        num_sms = torch.cuda.get_device_properties(q_device).multi_processor_count
        args = tuple(getattr(params, k) for k in params.__slots__)
        paged_gather_mode = fa3_tle_paged_gather_mode()
        dispatch = fa3_tle_metadata_dispatch(
            max_query_len=metadata_max_query_len,
            is_paged=is_paged,
            has_cache_kv=seqused_k is not None,
            num_splits=effective_num_splits,
            has_scheduler_metadata=scheduler_metadata is not None,
            metadata_source=scheduler_metadata_source,
        )
        avg_query_len = total_q / max(batch_size, 1)
        paged_prefill_route = _paged_prefill_route()
        paged_prefill_candidate = (
            dispatch.has_cache_kv
            and is_paged
            and not dispatch.split_kv
            and metadata_max_query_len
            >= _env_int("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q", 1024)
            and avg_query_len
            >= _env_int("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_AVG_Q", 128)
        )
        if paged_prefill_route == "direct":
            paged_prefill_long = False
        elif paged_prefill_route == "long":
            paged_prefill_long = (
                dispatch.has_cache_kv and is_paged and not dispatch.split_kv
            )
        else:
            paged_prefill_long = paged_prefill_candidate

        def _require_tma_aligned():
            if (
                not _tma_strides_are_aligned(q)
                or not _tma_strides_are_aligned(out)
                or (not is_paged and not _tma_strides_are_aligned(k))
                or (not is_paged and not _tma_strides_are_aligned(v))
            ):
                raise RuntimeError(
                    "TLE FA3 TMA-backed path requires 16-byte aligned Q/K/V/O strides."
                )

        def _log_dispatch(kernel_name: str, splits: int = 1):
            logger.debug(
                "kernel: flash_varlen_fwd_v3_tle kernel=%s layout=%s mode=%s splits=%s",
                kernel_name,
                dispatch.layout,
                dispatch.mode,
                splits,
            )
            if os.getenv("FLAG_GEMS_FA3_TLE_LOG_PLAN") == "1":
                print(
                    "FLAG_GEMS_FA3_TLE_PLAN "
                    f"layout={dispatch.layout} mode={dispatch.mode} "
                    f"kernel={kernel_name} splits={splits} scheduler_metadata="
                    f"{dispatch.has_scheduler_metadata} metadata_source="
                    f"{dispatch.metadata_source} paged_gather="
                    f"{fa3_tle_paged_gather_name(paged_gather_mode)}"
                )

        if dispatch.split_kv and dispatch.has_cache_kv:
            _log_dispatch("flashdecoding", effective_num_splits)
            partial_out = torch.empty(
                (effective_num_splits, num_heads, total_q, head_size),
                dtype=torch.float32,
                device=q_device,
            )
            partial_m = torch.empty(
                (effective_num_splits, num_heads, total_q),
                dtype=torch.float32,
                device=q_device,
            )
            partial_l = torch.empty_like(partial_m)
            split_grid = (max_seqlen_q, batch_size, num_heads * effective_num_splits)
            flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel[split_grid](
                *args,
                partial_out,
                partial_m,
                partial_l,
                scheduler_metadata if scheduler_metadata is not None else page_table,
                NUM_SPLITS=effective_num_splits,
                SPLIT_POLICY=0,
                HAS_SCHEDULER_METADATA=scheduler_metadata is not None,
                PAGED_GATHER_MODE=paged_gather_mode,
            )
            combine_block_m = 8
            combine_grid = (
                triton.cdiv(max_seqlen_q, combine_block_m),
                batch_size,
                num_heads,
            )
            flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel[combine_grid](
                out,
                lse,
                partial_out,
                partial_m,
                partial_l,
                out.stride(-3),
                out.stride(-2),
                o_batch_stride,
                cu_seqlens_q is not None,
                cu_seqlens_q,
                batch_size,
                num_heads,
                max_seqlen_q,
                head_size,
                total_q,
                adjusted_scale_softmax,
                adjusted_scale_softmax_log2e,
                BLOCK_M=combine_block_m,
                BLOCK_K=1 << (head_size - 1).bit_length(),
                NUM_SPLITS=effective_num_splits,
            )
        elif dispatch.has_cache_kv and not paged_prefill_long:
            _log_dispatch("direct")
            grid = lambda meta: (
                triton.cdiv(max_seqlen_q, meta["BLOCK_M"]),
                batch_size,
                num_heads,
            )
            flash_varlen_fwd_v3_tle_direct_kernel[grid](
                *args,
                DIRECT_SHAPE_BUCKET=_FA3_TLE_BUCKET_DIRECT_PAGED_DECODE
                if is_paged
                else _FA3_TLE_BUCKET_DIRECT_DECODE,
                MIN_Q_LEN_TO_PROCESS=0,
                MAX_Q_LEN_TO_PROCESS=2**31 - 1,
                PAGED_GATHER_MODE=paged_gather_mode,
            )
        elif max_seqlen_q <= 128 or total_q / max(batch_size, 1) <= 64:
            _require_tma_aligned()
            _log_dispatch("short")
            if is_paged and max_seqlen_q <= 512 and max_seqlen_k <= 1024:
                short_bucket = (
                    _FA3_TLE_BUCKET_PAGED_SMALL
                    if max_seqlen_q <= 128
                    else _FA3_TLE_BUCKET_PAGED_MEDIUM
                )
            else:
                short_bucket = _FA3_TLE_BUCKET_SHORT
            grid = lambda meta: (
                triton.cdiv(max_seqlen_q, meta["BLOCK_M"]),
                batch_size,
                num_heads,
            )
            flash_varlen_fwd_v3_tle_short_kernel[grid](
                *args,
                SHORT_SHAPE_BUCKET=short_bucket,
                MIN_Q_LEN_TO_PROCESS=0,
                MAX_Q_LEN_TO_PROCESS=2**31 - 1,
                PAGED_GATHER_MODE=paged_gather_mode,
            )
        else:
            _require_tma_aligned()
            _log_dispatch("long_paged_prefill" if paged_prefill_long else "long")
            grid = lambda meta: (
                min(
                    num_sms,
                    triton.cdiv(max_seqlen_q, meta["BLOCK_M"])
                    * batch_size
                    * num_heads,
                ),
            )
            flash_varlen_fwd_v3_tle_kernel[grid](
                *args,
                SHAPE_BUCKET=_FA3_TLE_BUCKET_LONG,
                FORCE_FAMILY_ID=_FA3_TLE_FAMILY_AUTO,
                MIN_Q_LEN_TO_PROCESS=0,
                MAX_Q_LEN_TO_PROCESS=2**31 - 1,
                PAGED_GATHER_MODE=paged_gather_mode,
            )

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
