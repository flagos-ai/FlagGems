import torch
import triton
import triton.language as tl


@triton.jit
def _repetition_penalty_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    penalties_ptr,
    num_seqs,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    vocab_offset = tl.program_id(1) * BLOCK_SIZE

    if seq_idx >= num_seqs:
        return

    penalty = tl.load(penalties_ptr + seq_idx)

    vocab_idx = vocab_offset + tl.arange(0, BLOCK_SIZE)

    valid_vocab = vocab_idx < vocab_size

    logits_idx = seq_idx * vocab_size + vocab_idx
    mask_idx = logits_idx

    prompt_mask = tl.load(prompt_mask_ptr + mask_idx, mask=valid_vocab, other=False)
    output_mask = tl.load(output_mask_ptr + mask_idx, mask=valid_vocab, other=False)
    logits = tl.load(logits_ptr + logits_idx, mask=valid_vocab, other=0.0)

    is_repeated = prompt_mask | output_mask

    logits = tl.where(is_repeated & (logits > 0), logits / penalty, logits)
    logits = tl.where(is_repeated & (logits <= 0), logits * penalty, logits)

    tl.store(logits_ptr + logits_idx, logits, mask=valid_vocab)


def apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties):
    assert logits.is_contiguous(), "logits must be contiguous"
    assert (
        prompt_mask.is_contiguous() and prompt_mask.dtype == torch.bool
    ), "prompt_mask must be contiguous bool tensor"
    assert (
        output_mask.is_contiguous() and output_mask.dtype == torch.bool
    ), "output_mask must be contiguous bool tensor"
    assert (
        repetition_penalties.is_contiguous()
    ), "repetition_penalties must be contiguous"
    assert logits.dim() == 2, f"logits must be 2D, got {logits.dim()}D"
    assert (
        logits.shape == prompt_mask.shape == output_mask.shape
    ), "shape mismatch between logits and masks"
    assert (
        repetition_penalties.dim() == 1
        and repetition_penalties.numel() == logits.shape[0]
    ), "repetition_penalties must be 1D with length equal to num_seqs"

    num_seqs, vocab_size = logits.shape

    BLOCK_SIZE = 1024

    grid = (
        num_seqs,
        triton.cdiv(vocab_size, BLOCK_SIZE),
    )

    _repetition_penalty_kernel[grid](
        logits,
        prompt_mask,
        output_mask,
        repetition_penalties,
        num_seqs,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return None
