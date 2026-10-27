import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _beam_search_score_kernel(
    log_probs,
    beam_scores,
    output,
    N,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    UPCAST_F32: tl.constexpr,
):
    """Flat 1D beam search score kernel: out[i] = log_probs[i] + beam_scores[i // V]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < N
        row = offs // V
        v = tl.load(log_probs + offs, mask=mask, other=0.0)
        b = tl.load(beam_scores + row, mask=mask, other=0.0)
    else:
        row = offs // V
        v = tl.load(log_probs + offs)
        b = tl.load(beam_scores + row)
    if UPCAST_F32:
        # Floating dtypes: accumulate in fp32 (the upcast/downcast round trip is
        # what the tuned block sizes below were measured with).
        acc = v.to(tl.float32) + b.to(tl.float32)
    else:
        # Integer dtypes: an fp32 accumulator silently rounds every value above
        # 2**24, while ATen's broadcast add is exact.  Add in the element type.
        acc = v + b
    if NEED_MASK:
        tl.store(output + offs, acc, mask=mask)
    else:
        tl.store(output + offs, acc)


def _block_and_warps(numel, dtype):
    """Tile/launch choice per element count and dtype.

    Retuned on the XPU3 (P800) card together with
    :data:`_XPU_LAUNCH_OPTIONS`: once the per-core staging buffer is big enough
    to keep the block DMA fed, the kernel stops being DMA-issue bound and the
    optimum moves to much larger 1D tiles than the stock table used.
    """
    if dtype == torch.float32:
        if numel <= 16384:
            return 8192, 8
        if numel <= 65536:
            return 16384, 4
        if numel <= 262144:
            return 32768, 4
        if numel <= 1048576:
            return 65536, 4
        return 262144, 4
    if dtype == torch.float16:
        if numel <= 16384:
            return 8192, 8
        if numel <= 65536:
            return 16384, 4
        if numel <= 262144:
            return 65536, 4
        if numel <= 1048576:
            return 131072, 4
        return 262144, 4
    if numel <= 65536:
        return 8192, 8
    if numel <= 262144:
        return 16384, 4
    return 65536, 4


_XPU_LAUNCH_OPTIONS = {"buffer_size_limit": 4096}


def _launch_beam_search_score(log_probs, beam_scores, outputs):
    if log_probs.dim() != 2:
        raise ValueError("beam_search_score expects 2D log_probs on Kunlunxin")
    batch_size, vocab_size = log_probs.shape
    if beam_scores.numel() != batch_size:
        raise ValueError(
            "beam_scores must contain one score per batch entry on Kunlunxin"
        )
    numel = log_probs.numel()
    if numel == 0 or batch_size == 0:
        return outputs
    if not log_probs.is_contiguous():
        log_probs = log_probs.contiguous()
    beam_flat = beam_scores
    if not beam_flat.is_contiguous():
        beam_flat = beam_flat.contiguous()
    beam_flat = beam_flat.reshape(-1)
    block, num_warps = _block_and_warps(numel, log_probs.dtype)
    need_mask = 1 if numel % block else 0
    grid = (triton.cdiv(numel, block),)
    _beam_search_score_kernel[grid](
        log_probs,
        beam_flat,
        outputs,
        numel,
        V=vocab_size,
        BLOCK=block,
        NEED_MASK=need_mask,
        UPCAST_F32=_upcast_f32(log_probs.dtype),
        num_warps=num_warps,
        **_XPU_LAUNCH_OPTIONS,
    )
    return outputs


def _flat_beam_scores(beam_scores, batch_size):
    """Normalize beam_scores to [B] flat. Accepts 1D [B] or 2D [B, 1]."""
    if beam_scores.dim() > 2 or (beam_scores.dim() == 2 and beam_scores.shape[-1] != 1):
        raise ValueError(
            "beam_scores must have shape [batch] or [batch, 1] on Kunlunxin"
        )
    return beam_scores.reshape(batch_size)


_EXACT_ACC_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _upcast_f32(dtype):
    """Whether the kernel should accumulate in fp32 for this element type."""
    return dtype not in _EXACT_ACC_DTYPES


def beam_search_score(log_probs, beam_scores):
    """Out-of-place beam search score: log_probs [B, V] + beam_scores [B]."""
    logger.debug("GEMS_KUNLUNXIN BEAM_SEARCH_SCORE")
    batch_size = log_probs.shape[0]
    beam_flat = _flat_beam_scores(beam_scores, batch_size)
    # The kernel is flat 1-D and assumes the output is contiguous.  Plain
    # `empty_like` would propagate a transposed / permuted input's strides
    # (native `add` does preserve them), and the flat store would then write
    # every value to the wrong logical position.  Allocate contiguous so the
    # values are always right; the returned layout is contiguous instead of
    # `preserve_format`.
    outputs = torch.empty_like(log_probs, memory_format=torch.contiguous_format)
    return _launch_beam_search_score(log_probs, beam_flat, outputs)


def beam_search_score_(log_probs, beam_scores):
    """In-place variant writing back into log_probs."""
    logger.debug("GEMS_KUNLUNXIN BEAM_SEARCH_SCORE_")
    batch_size = log_probs.shape[0]
    beam_flat = _flat_beam_scores(beam_scores, batch_size)
    if not log_probs.is_contiguous():
        staged = log_probs.contiguous()
        _launch_beam_search_score(staged, beam_flat, staged)
        log_probs.copy_(staged)
        return log_probs
    return _launch_beam_search_score(log_probs, beam_flat, log_probs)
