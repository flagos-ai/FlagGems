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
    # Preserve the trailing dimensions for copy_; the kernel still copies linearly.
    temp = torch.full(
        out.shape,
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
    logger.debug("GEMS PAD_SEQUENCE")

    batch = len(sequences)
    if batch == 0:
        raise RuntimeError("pad_sequence empty input")

    first = sequences[0]
    first_shape = first.shape
    if len(first_shape) == 0:
        raise RuntimeError("pad_sequence requires at least one dimension")
    device = first.device
    dtype = first.dtype
    trailing_shape = first_shape[1:]
    max_len = first_shape[0]
    seqs = [first if first.is_contiguous() else first.contiguous()]

    for sequence in sequences[1:]:
        shape = sequence.shape
        if len(shape) == 0:
            raise RuntimeError("pad_sequence requires at least one dimension")
        if shape[1:] != trailing_shape:
            raise RuntimeError("pad_sequence expects matching trailing dimensions")
        if sequence.device != device:
            raise RuntimeError(
                "pad_sequence expects all input tensors to be on the same device"
            )
        # Match native output dtype and the Hygon backend. Homogeneous inputs
        # do not need an additional conversion kernel.
        if sequence.dtype != dtype:
            sequence = sequence.to(dtype=dtype)
        if not sequence.is_contiguous():
            sequence = sequence.contiguous()
        seqs.append(sequence)
        max_len = max(max_len, shape[0])

    feature = 1
    for d in trailing_shape:
        feature *= d

    if batch_first:
        out_shape = (batch, max_len, *trailing_shape)
    else:
        out_shape = (max_len, batch, *trailing_shape)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    total_elements = batch * max_len * feature
    if total_elements == 0:
        return out

    # Keep the existing dispatch thresholds and kernel implementations.
    if batch <= 2:
        return _pad_sequence_small(
            seqs, out, batch, max_len, feature, batch_first, padding_value
        )
    elif batch <= 8:
        if total_elements <= _DIRECT_COPY_THRESHOLD or batch_first:
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
