import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_

logger = logging.getLogger(__name__)

# Bounded, backend-proven flat tile (matches expand_copy's _BCAST_BLOCK); a
# plain contiguous load/store block that stays clear of the tile==32768 /
# tile==1024-pointwise hazards recorded for this backend.
_PAD_BLOCK = 4096


@triton.jit
def _pad_fill_kernel(out_ptr, n, value, BLOCK: tl.constexpr):
    """Write ``value`` into every element of the contiguous ``n``-element
    buffer ``out_ptr`` (gems Triton replacement for the torch.full fill)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.full((BLOCK,), value, dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _pad_copy_contig_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Flat block-DMA copy of a contiguous ``n``-element source into a
    contiguous destination slice (batch_first layout)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def _pad_copy_strided_kernel(
    src_ptr, dst_ptr, feature, row_stride, BLOCK: tl.constexpr
):
    """Copy a contiguous ``[length, feature]`` source into a row-strided
    destination slice (batch_first=False: consecutive time steps are
    ``row_stride`` elements apart, the ``feature`` block is contiguous).

    Program ``(t, c)`` moves the ``feature`` block of time step ``t``; no
    per-element div/mod (2-D grid), so it avoids the slow device-array
    index-arithmetic path recorded for this backend.
    """
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < feature
    vals = tl.load(src_ptr + pid_t * feature + offs, mask=mask)
    tl.store(dst_ptr + pid_t * row_stride + offs, vals, mask=mask)


def _fill_padding(out: torch.Tensor, padding_value) -> None:
    """Fill the whole contiguous output buffer with ``padding_value``."""
    n = out.numel()
    grid = (triton.cdiv(n, _PAD_BLOCK),)
    _pad_fill_kernel[grid](out, n, padding_value, BLOCK=_PAD_BLOCK, num_warps=4)


def _copy_into(src: torch.Tensor, dst: torch.Tensor, feature: int) -> None:
    """Copy the contiguous sequence ``src`` into its (possibly row-strided)
    output slot ``dst`` via a gems Triton kernel (no torch data movement).

    ``dst`` is a view into the contiguous output: for batch_first it is a
    fully contiguous block (``dst.stride(0) == feature``) handled by the flat
    copy; for batch_first=False it is row-strided in the length dimension with
    a contiguous ``feature`` block.
    """
    length = dst.shape[0]
    if dst.is_contiguous():
        n = length * feature
        grid = (triton.cdiv(n, _PAD_BLOCK),)
        _pad_copy_contig_kernel[grid](src, dst, n, BLOCK=_PAD_BLOCK, num_warps=4)
        return
    row_stride = dst.stride(0)
    grid = (length, triton.cdiv(feature, _PAD_BLOCK))
    _pad_copy_strided_kernel[grid](
        src, dst, feature, row_stride, BLOCK=_PAD_BLOCK, num_warps=4
    )


def _gems_copy(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy ``src`` into ``dst`` via the generic gems Triton pointwise copy
    (``flag_gems.ops.copy.copy_``), called by reference so it never redispatches
    to the vendor ``copy_``. Reads an arbitrary-stride ``src`` into a contiguous
    ``dst`` in a single pass and casts on store when the dtypes differ.

    gems ``copy_`` redispatches to ``aten.copy_`` on a 0-numel dst, so guard it
    (there is nothing to move and the fresh ``dst`` already has the right shape).
    """
    if dst.numel() == 0:
        return dst
    _gems_copy_(dst, src)
    return dst


def _materialize(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return a contiguous, ``dtype``-typed tensor holding ``t``'s values.

    Replaces the input-regularization ``t.contiguous()`` / ``t.to(dtype=dtype)``
    torch fallbacks with a gems Triton copy: allocate a fresh contiguous buffer
    (pure ``torch.empty``) and fill it via ``_gems_copy`` (which also performs
    the dtype cast on store). No-op when ``t`` is already contiguous and typed.
    """
    if t.dtype == dtype and t.is_contiguous():
        return t
    return _gems_copy(torch.empty(t.shape, dtype=dtype, device=t.device), t)


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """Pad variable length tensors into a single batch tensor.

    XPU rewrite of the generic implementation: the vendor triton kernels leave
    the padding region uninitialised (the `other=padding_value` masked load and
    the strided/scattered padding store are not honoured on this backend, so a
    `torch.empty` output keeps garbage -> maxdiff ~6e4). Here the output is
    allocated with `torch.empty` and pre-filled with `padding_value` via a gems
    Triton fill kernel (which covers every padding element); each sequence is
    then copied into its slice through a gems Triton copy kernel, so no element
    is ever left unwritten and no torch data-movement/fill op is used.
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
    seqs = [_materialize(first, dtype)]

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
        seqs.append(_materialize(sequence, dtype))
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

    _fill_padding(out, padding_value)

    for i, seq in enumerate(seqs):
        length = seq.shape[0]
        if length == 0:
            continue
        if batch_first:
            dst = out[i, :length]
        else:
            dst = out[:length, i]
        _copy_into(seq, dst, feature)

    return out
