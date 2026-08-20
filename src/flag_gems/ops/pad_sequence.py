import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# ============================================================
# Kernels
# ============================================================


@triton.jit
def _pad_sequence_small_kernel(
    out_ptr,
    seq_ptr,
    seq_len,
    max_len,
    feature,
    batch_index,
    batch,
    padding_value,
    BLOCK: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
):
    """Kernel for small batches (batch <= 2)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    total = max_len * feature
    mask = offs < total

    t = offs // feature
    f = offs % feature
    valid = t < seq_len

    value = tl.load(
        seq_ptr + t * feature + f,
        mask=mask & valid,
        other=padding_value,
    )

    if BATCH_FIRST:
        dst = out_ptr + batch_index * max_len * feature + offs
    else:
        dst = out_ptr + t * batch * feature + batch_index * feature + f

    tl.store(dst, value, mask=mask)


@triton.jit
def _pad_sequence_batch_copy_kernel(
    out_ptr,
    src_ptr,
    total,
    BLOCK: tl.constexpr,
):
    """Linear copy kernel for medium workloads (batch_first=False)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    value = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, value, mask=mask)


@triton.jit
def _pad_sequence_flat_kernel(
    out_ptr,
    seq_base_ptr,
    seq_offsets,
    seq_lens,
    batch,
    max_len,
    feature,
    padding_value,
    BLOCK: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
):
    """Kernel for large batches (batch > 8)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    total = batch * max_len * feature
    mask = offs < total

    if BATCH_FIRST:
        batch_id = offs // (max_len * feature)
        remain = offs % (max_len * feature)
        t = remain // feature
        f = remain % feature
    else:
        t = offs // (batch * feature)
        remain = offs % (batch * feature)
        batch_id = remain // feature
        f = remain % feature

    seq_len = tl.load(seq_lens + batch_id, mask=mask, other=0)
    valid = t < seq_len

    offset = tl.load(seq_offsets + batch_id, mask=mask, other=0)

    value = tl.load(
        seq_base_ptr + offset + t * feature + f,
        mask=mask & valid,
        other=padding_value,
    )

    tl.store(out_ptr + offs, value, mask=mask)


# ============================================================
# Helper functions
# ============================================================


def _select_block(feature):
    """Select block size based on feature dimension."""
    if feature <= 32:
        return 64
    elif feature <= 128:
        return 128
    elif feature <= 512:
        return 256
    else:
        return 512


def _select_warps(feature):
    """Select number of warps based on feature dimension."""
    if feature <= 128:
        return 2
    elif feature <= 512:
        return 4
    else:
        return 8


def _build_metadata(sequences, device):
    """Build flat offsets and lengths for each sequence."""
    offsets = []
    lengths = []
    offset = 0

    for seq in sequences:
        offsets.append(offset)
        lengths.append(seq.shape[0])
        offset += seq.numel()

    seq_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)

    return seq_offsets, seq_lens


# ============================================================
# Strategy wrappers
# ============================================================

_DIRECT_COPY_THRESHOLD = 8192


def _pad_sequence_small(
    sequences, out, batch, max_len, feature, batch_first, padding_value
):
    """Strategy for batch <= 2."""
    BLOCK = _select_block(feature)
    WARPS = _select_warps(feature)
    grid = (triton.cdiv(max_len * feature, BLOCK),)

    with torch_device_fn.device(out.device):
        for batch_id, seq in enumerate(sequences):
            _pad_sequence_small_kernel[grid](
                out,
                seq,
                seq.shape[0],
                max_len,
                feature,
                batch_id,
                batch,
                padding_value,
                BLOCK=BLOCK,
                BATCH_FIRST=batch_first,
                num_warps=WARPS,
            )

    return out


def _pad_sequence_direct_copy(
    sequences, out, batch, max_len, feature, batch_first, padding_value
):
    """Direct copy strategy for small workloads or batch_first=True."""
    out.fill_(padding_value)

    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        if batch_first:
            out[i, :length].copy_(seq)
        else:
            out[:length, i].copy_(seq)

    return out


def _pad_sequence_batch_copy(sequences, out, batch, max_len, feature, padding_value):
    """Linear copy strategy for medium workloads with batch_first=False."""
    temp = torch.full(
        (max_len, batch, feature),
        padding_value,
        dtype=out.dtype,
        device=out.device,
    )

    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        temp[:length, i].copy_(seq)

    numel = temp.numel()
    BLOCK = _select_block(feature)
    WARPS = _select_warps(feature)
    grid = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(out.device):
        _pad_sequence_batch_copy_kernel[grid](
            out, temp, numel, BLOCK=BLOCK, num_warps=WARPS
        )

    return out


def _pad_sequence_flat(
    sequences, out, batch, max_len, feature, batch_first, padding_value
):
    """Flatten strategy for batch > 8."""
    flat = [seq.reshape(-1) for seq in sequences]
    seq_base = torch.cat(flat)

    seq_offsets, seq_lens = _build_metadata(sequences, out.device)

    numel = batch * max_len * feature
    BLOCK = _select_block(feature)
    WARPS = _select_warps(feature)
    grid = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(out.device):
        _pad_sequence_flat_kernel[grid](
            out,
            seq_base,
            seq_offsets,
            seq_lens,
            batch,
            max_len,
            feature,
            padding_value,
            BLOCK=BLOCK,
            BATCH_FIRST=batch_first,
            num_warps=WARPS,
        )

    return out


# ============================================================
# Public API
# ============================================================


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """Pad variable length tensors into a single batch tensor."""
    logger.debug("pad_sequence batch=%d batch_first=%s", len(sequences), batch_first)

    if len(sequences) == 0:
        raise RuntimeError("pad_sequence empty input")

    batch = len(sequences)
    device = sequences[0].device
    dtype = sequences[0].dtype

    # Ensure all sequences are contiguous
    seqs = [seq if seq.is_contiguous() else seq.contiguous() for seq in sequences]

    max_len = max(seq.shape[0] for seq in seqs)

    feature = 1
    for d in seqs[0].shape[1:]:
        feature *= d

    if batch_first:
        out_shape = (batch, max_len, *seqs[0].shape[1:])
    else:
        out_shape = (max_len, batch, *seqs[0].shape[1:])

    out = torch.empty(out_shape, dtype=dtype, device=device)

    total_elements = batch * max_len * feature

    # Dispatch based on batch size and workload
    if batch <= 2:
        return _pad_sequence_small(
            seqs, out, batch, max_len, feature, batch_first, padding_value
        )
    elif batch <= 8:
        if total_elements <= _DIRECT_COPY_THRESHOLD or batch_first:
            # Direct copy is simpler and faster for small workloads,
            # and also handles batch_first=True layout correctly.
            return _pad_sequence_direct_copy(
                seqs, out, batch, max_len, feature, batch_first, padding_value
            )
        else:
            return _pad_sequence_batch_copy(
                seqs, out, batch, max_len, feature, padding_value
            )
    else:
        return _pad_sequence_flat(
            seqs, out, batch, max_len, feature, batch_first, padding_value
        )


def pad_sequence_out(
    sequences,
    batch_first=False,
    padding_value=0.0,
    *,
    out,
):
    """Out variant of pad_sequence."""
    result = pad_sequence(sequences, batch_first, padding_value)

    if out.shape != result.shape:
        out.resize_(result.shape)

    out.copy_(result)
    return out
