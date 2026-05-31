"""
Host-side launcher for the FA3-style varlen forward kernel.

Drop-in replacement for `mha_varlan_fwd` in flash_api.py. Same arguments,
same return tuple -- the existing `flash_attn_varlen_func` wrapper in
attention.py only needs to dispatch on `fa_version` to choose between this
launcher and the v2 one.

We intentionally do *not* re-implement the seqlenq_ngroups_swapped logic
or the LSE bookkeeping: those are pure shape/index tricks and are
identical to the v2 launcher. Where we can, we just call into the same
`fwd_params` struct.
"""

import logging
import math

import torch
import triton

import flag_gems
from flag_gems.ops.flash_api import fwd_params  # reuse the slot struct
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .flash_kernel_v3 import flash_varlen_fwd_v3_kernel

logger = logging.getLogger(__name__)


def _check_device(x):
    assert x.device.type == flag_gems.device


# ---------------------------------------------------------------------------
# TMA descriptors require a global memory allocator. We register one once on
# import; this matches the pattern used in Triton's 06-fused-attention.py.
# Idempotent: re-registering with the same callback is a no-op.
# ---------------------------------------------------------------------------
_TMA_ALLOCATOR_REGISTERED = False


def _bucket_avg_rows(avg_rows):
    if avg_rows <= 16:
        return 16
    if avg_rows <= 32:
        return 32
    if avg_rows <= 64:
        return 64
    return 128


def _bucket_avg_rows_per_cta(total_q, batch_size, num_heads):
    total_rows = total_q * num_heads
    num_sms = torch_device_fn.get_device_properties(
        flag_gems.device
    ).multi_processor_count
    avg_rows_per_sm = total_rows / max(num_sms, 1)
    avg_rows_per_batch = total_q / max(batch_size, 1)
    return _bucket_avg_rows(min(avg_rows_per_batch, avg_rows_per_sm))


def _ensure_tma_allocator():
    global _TMA_ALLOCATOR_REGISTERED
    if _TMA_ALLOCATOR_REGISTERED:
        return
    if not hasattr(triton, "set_allocator"):
        # Old Triton (< 3.2): TMA descriptors aren't supported. Caller should
        # have dispatched to v2 already, but guard anyway.
        raise RuntimeError(
            "FA3 path requires Triton with on-device TMA descriptors "
            "(triton.set_allocator). Found older Triton; please use fa_version=2."
        )

    def _alloc_fn(size: int, alignment: int, stream):
        # Allocator must return a CUDA tensor large enough for the descriptor.
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_REGISTERED = True


# ---------------------------------------------------------------------------
# Capability gate.
# ---------------------------------------------------------------------------
def is_fa3_supported() -> bool:
    """Return True iff we should attempt the FA3 path on the current GPU."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:  # need Hopper or newer
        return False
    # Triton needs to expose make_tensor_descriptor.
    try:
        import triton.language as tl

        return hasattr(tl, "make_tensor_descriptor") and hasattr(
            triton, "set_allocator"
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The actual launcher. Signature is identical to flash_api.mha_varlan_fwd
# so attention.py can swap one for the other based on fa_version.
# ---------------------------------------------------------------------------
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
):
    _check_device(q)
    _check_device(k)
    _check_device(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FA3 currently only supports fp16 and bf16 (FP8 path intentionally omitted)."
    assert q_dtype == k.dtype == v.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_q.is_contiguous()
    assert cu_seqlens_k.dtype == torch.int32 and cu_seqlens_k.is_contiguous()
    assert leftpad_k is None, "leftpad_k is not supported in FA3 path."

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # ------------------------------------------------------------------
    # Shape unpacking (identical to v2 launcher).
    # ------------------------------------------------------------------
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)
    if seqused_k is not None:
        assert seqused_k.is_contiguous() and seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0

    # MQA/GQA seqlenq-ngroups swap: same trick as v2 (single-token Q
    # decode is faster if we put the kv-replicated heads on the seq-axis).
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

    assert head_size <= 256
    assert head_size % 8 == 0
    assert num_heads % num_heads_k == 0
    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported with softcap."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    # log2-scale baking (identical to v2).
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

    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype == torch.float
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

    # ------------------------------------------------------------------
    # Allocate outputs.
    # ------------------------------------------------------------------
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

        # LSE layout: (num_heads, total_q) -- matches v2 contract.
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
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        # In FA3 we don't return the dropout-applied softmax matrix
        # (the v2 path does, but it's only used for debugging). We allocate
        # an empty placeholder to keep the return tuple shape stable.
        p = torch.empty((), device=q_device)
        return_softmax_v3 = False  # forced off; FA3 kernel doesn't store P

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        p_dropout_inv = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout_inv * 255.0)
        rp_dropout = 1.0 / p_dropout_inv if p_dropout > 0 else 1.0

        # --------------------------------------------------------------
        # Pack params (mirrors v2 launcher exactly).
        # --------------------------------------------------------------
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
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax_v3,
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

        # --------------------------------------------------------------
        # Register the TMA allocator and launch.
        # --------------------------------------------------------------
        _ensure_tma_allocator()
        logger.debug("kernel: flash_varlen_fwd_v3")

        grid = lambda meta: (
            triton.cdiv(max_seqlen_q, meta["BLOCK_M"]),
            batch_size,
            num_heads,
        )
        avg_rows_per_cta_bucket = _bucket_avg_rows_per_cta(
            total_q, batch_size, num_heads
        )
        args = tuple(getattr(params, k) for k in params.__slots__) + (
            avg_rows_per_cta_bucket,
        )
        flash_varlen_fwd_v3_kernel[grid](*args)

        # --------------------------------------------------------------
        # Undo the seqlenq-ngroups swap if we did it.
        # --------------------------------------------------------------
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

    # Same return shape as v2 launcher: (out, q, k, v, lse, philox, unused, p)
    return out, q, k, v, lse, philox_args, unused, p
