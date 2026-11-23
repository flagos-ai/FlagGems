import torch
import triton
import triton.language as tl


@triton.jit
def _beam_search_score_2d(
    log_probs,
    beam_scores,
    out,
    vocab_size,
    row_stride,
    EVEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    base = log_probs + pid_b * row_stride
    obase = out + pid_b * row_stride
    if EVEN:
        lp = tl.load(base + offs)
        bs = tl.load(beam_scores + pid_b)
        tl.store(obase + offs, lp + bs)
    else:
        mask = offs < vocab_size
        lp = tl.load(base + offs, mask=mask, other=0.0)
        bs = tl.load(beam_scores + pid_b)
        tl.store(obase + offs, lp + bs, mask=mask)


@triton.jit
def _beam_search_score_flat(
    log_probs,
    beam_scores,
    out,
    vocab_size,
    num_rows,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        b = offs // vocab_size
        lp = tl.load(log_probs + offs)
        bs = tl.load(beam_scores + b)
        tl.store(out + offs, lp + bs)
    else:
        mask = offs < vocab_size * num_rows
        b = offs // vocab_size
        lp = tl.load(log_probs + offs, mask=mask, other=0.0)
        bs = tl.load(beam_scores + b, mask=mask, other=0.0)
        tl.store(out + offs, lp + bs, mask=mask)


_BLOCK_2D_SMALL = 1024
_NUM_WARPS_2D_SMALL = 4
_BLOCK_2D_MED = 2048
_NUM_WARPS_2D_MED = 4
_BLOCK_FLAT = 2048
_NUM_WARPS_FLAT = 4
_MED_NUMEL = 1 << 17
_BIG_NUMEL = 1 << 20


def run(log_probs, beam_scores):
    numel = log_probs.numel()
    if numel == 0:
        return torch.empty_like(log_probs)
    last_dim = log_probs.shape[-1]
    num_rows = numel // last_dim
    out = torch.empty_like(log_probs)
    bs = beam_scores.reshape(-1)
    if numel >= _BIG_NUMEL:
        even = numel % _BLOCK_FLAT == 0
        grid = ((numel + _BLOCK_FLAT - 1) // _BLOCK_FLAT,)
        _beam_search_score_flat[grid](
            log_probs,
            bs,
            out,
            last_dim,
            num_rows,
            EVEN=even,
            BLOCK=_BLOCK_FLAT,
            num_warps=_NUM_WARPS_FLAT,
        )
    else:
        # For a contiguous tensor of any rank, the flattened row index b
        # starts at b * last_dim (leading dims flattened). Non-contiguous
        # 2D tensors fall back to stride(0) row offsets.
        if log_probs.is_contiguous():
            row_stride = last_dim
        elif log_probs.dim() > 1:
            row_stride = log_probs.stride(0)
        else:
            row_stride = last_dim
        if numel >= _MED_NUMEL:
            even = last_dim % _BLOCK_2D_MED == 0
            grid = (num_rows, triton.cdiv(last_dim, _BLOCK_2D_MED))
            _beam_search_score_2d[grid](
                log_probs,
                bs,
                out,
                last_dim,
                row_stride,
                EVEN=even,
                BLOCK_V=_BLOCK_2D_MED,
                num_warps=_NUM_WARPS_2D_MED,
            )
        else:
            even = last_dim % _BLOCK_2D_SMALL == 0
            grid = (num_rows, triton.cdiv(last_dim, _BLOCK_2D_SMALL))
            _beam_search_score_2d[grid](
                log_probs,
                bs,
                out,
                last_dim,
                row_stride,
                EVEN=even,
                BLOCK_V=_BLOCK_2D_SMALL,
                num_warps=_NUM_WARPS_2D_SMALL,
            )
    return out


# Alias for FlagGems import convention
beam_search_score = run
