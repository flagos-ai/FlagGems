import logging

import torch

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


def _copy_into(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy ``src`` into the (possibly strided) view ``dst``.

    Prefer the tle DMA path; fall back to the native strided-copy engine
    (`aten::_copy_from`, which flag_gems never overrides) when tle cannot
    express the layout / dtype (e.g. a dtype-converting copy tle rejects).
    """
    if src.dtype == dst.dtype and tle_copy(src, dst):
        return
    torch.ops.aten._copy_from(src, dst, False)


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """Pad variable length tensors into a single batch tensor.

    XPU rewrite of the generic implementation: the vendor triton kernels leave
    the padding region uninitialised (the `other=padding_value` masked load and
    the strided/scattered padding store are not honoured on this backend, so a
    `torch.empty` output keeps garbage -> maxdiff ~6e4). Here the output is
    pre-filled with `padding_value` (which covers every padding element) and
    each sequence is copied into its slice through the DMA copy engine, so no
    element is ever left unwritten.
    """
    logger.debug("GEMS_KUNLUNXIN PAD_SEQUENCE")

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

    out = torch.full(out_shape, padding_value, dtype=dtype, device=device)
    total_elements = batch * max_len * feature
    if total_elements == 0:
        return out

    for i, seq in enumerate(seqs):
        length = seq.shape[0]
        if length == 0:
            continue
        if batch_first:
            dst = out[i, :length]
        else:
            dst = out[:length, i]
        _copy_into(seq, dst)

    return out
