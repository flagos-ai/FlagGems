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
):
    """Flat 1D beam search score kernel: out[i] = log_probs[i] + beam_scores[i // V]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < N
        row = offs // V
        v = tl.load(log_probs + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(beam_scores + row, mask=mask, other=0.0).to(tl.float32)
    else:
        row = offs // V
        v = tl.load(log_probs + offs).to(tl.float32)
        b = tl.load(beam_scores + row).to(tl.float32)
    acc = v + b
    if NEED_MASK:
        tl.store(output + offs, acc, mask=mask)
    else:
        tl.store(output + offs, acc)


def _block_and_warps(numel, dtype):
    if dtype == torch.float32:
        if numel <= 131072:
            return 8192, 4
        if numel <= 1048576:
            return 65536, 4
        return 262144, 8
    if dtype == torch.float16:
        if numel <= 32768:
            return 8192, 8
        if numel <= 131072:
            return 16384, 8
        if numel <= 524288:
            return 16384, 4
        if numel <= 1048576:
            return 65536, 8
        return 262144, 8
    if numel <= 32768:
        return 8192, 8
    if numel <= 131072:
        return 16384, 8
    if numel <= 524288:
        return 16384, 2
    return 262144, 8


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
        num_warps=num_warps,
    )
    return outputs


def _flat_beam_scores(beam_scores, batch_size):
    """Normalize beam_scores to [B] flat. Accepts 1D [B] or 2D [B, 1]."""
    if beam_scores.dim() > 2 or (beam_scores.dim() == 2 and beam_scores.shape[-1] != 1):
        raise ValueError(
            "beam_scores must have shape [batch] or [batch, 1] on Kunlunxin"
        )
    return beam_scores.reshape(batch_size)


def beam_search_score(log_probs, beam_scores):
    """Out-of-place beam search score: log_probs [B, V] + beam_scores [B]."""
    logger.debug("GEMS_KUNLUNXIN BEAM_SEARCH_SCORE")
    batch_size = log_probs.shape[0]
    beam_flat = _flat_beam_scores(beam_scores, batch_size)
    outputs = torch.empty_like(log_probs)
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
