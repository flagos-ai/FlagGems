# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlunxin override for ``aten::int_repr`` / ``aten::int_repr.out``.

``int_repr`` is a bit-exact typed memory copy of an affine-quantized tensor's
storage (``quint8 -> uint8``, ``qint8 -> int8``, ``qint32 -> int32``); the
quantization metadata never enters the result, so the device work is pure DMA.

The generic implementation copies native-dtype lanes with a fixed
``BLOCK=4096`` and an always-true mask.  On XPU3 that is far from the DMA floor
(measured: 1-byte 16 MiB copy 218 us vs ``copy_`` 21 us; a 4-byte 64 MiB copy
229 us vs ~85 us) and its always-masked strided kernel even fails to compile
for some large int8 ``[..., ::2]`` layouts.  This override keeps the same
host-side contract and routes the copy through three device paths:

1. **linear** (the input is storage-dense, e.g. contiguous / transposed /
   channels_last / offset-slice): element offsets from ``data_ptr()`` are a
   single ``[0, numel)`` span, so the pair is copied as the widest *integer
   word* (int64, else int32, else byte) with a 16-byte-aligned bulk, a short
   unaligned head and a <=word tail.  When the output is freshly allocated the
   destination storage offset is made equal to the source's so both bases share
   one alignment and the bulk reaches the copy floor.
2. **packed run** (a non-dense input with a maximal trailing unit-stride run
   >= 64, e.g. a broadcast ``expand``): one program per (run, chunk), source
   base computed once per program from the outer dims/strides.
3. **constexpr gather** (everything else, e.g. ``[..., ::2]`` or a caller
   ``out`` with different strides): one lane per logical element, with the
   shape/stride digits as ``tl.constexpr`` so the divisors fold into
   multiply-shift sequences.

All launches use a constant tuple grid (a per-call ``grid=lambda`` stalls the
host ~120 ms on this backend) and the flat/run kernels take a ``NEED_MASK``
constexpr so an exact tile avoids the slow masked lowering.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["int_repr"]

_Q_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

# Rank covered by the constexpr gather (matches the generic implementation and
# the operator's tested ranks).
_MAX_RANK = 5
# Byte span below which a single element-typed launch beats a word split.
_WORD_MIN_BYTES = 1 << 16
# Flat word tile ladder (XPU3 throughput is tile-size driven, not grid driven).
_FLAT_MAX_BLOCK = 32768
# Small element-typed tile for the tiny-tensor fallback.
_SMALL_BLOCK = 4096
# Packed-run routing thresholds.
_RUN_MIN_LEN = 64
_RUN_MAX_RUNS = 65536
_RUN_BLOCK = 8192
# Gather tile caps: int8 vector gathers miscompile above 2048 lanes on XPU3,
# 4-byte lanes are fine up to 8192.
_GATHER_BLOCK_BYTE = 2048
_GATHER_BLOCK_WIDE = 8192


@triton.jit
def _int_repr_flat_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Linear copy of ``n_words`` word-typed elements."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < n_words
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs))


@triton.jit
def _int_repr_run_kernel(
    src_ptr,
    dst_ptr,
    run_len,
    chunks,
    s0,
    s1,
    s2,
    s3,
    s4,
    d0,
    d1,
    d2,
    d3,
    d4,
    N_OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy packed trailing runs: one contiguous ``run_len`` chunk per program.

    The destination is contiguous, so a run occupies ``run_idx * run_len``.
    The source base is a scalar decomposition over the *outer* dims; the mask
    stays on because a scalar-base 1-D access without a mask is silently
    miscompiled on XPU3 for non-contiguous runs.
    """
    pid = tl.program_id(0)
    run_idx = pid // chunks
    chunk = pid % chunks

    ss = [s0, s1, s2, s3, s4]
    dd = [d0, d1, d2, d3, d4]
    rem = run_idx
    src_base = tl.full([], 0, tl.int64)
    for j in tl.static_range(N_OUTER - 1, -1, -1):
        src_base += (rem % dd[j]).to(tl.int64) * ss[j]
        rem = rem // dd[j]

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < run_len
    vals = tl.load(src_ptr + src_base + offs, mask=mask)
    dst_off = run_idx.to(tl.int64) * run_len + offs
    tl.store(dst_ptr + dst_off, vals, mask=mask)


@triton.jit
def _int_repr_gather_kernel(
    src_ptr,
    dst_ptr,
    n,
    s0: tl.constexpr,
    s1: tl.constexpr,
    s2: tl.constexpr,
    s3: tl.constexpr,
    s4: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    d4: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    o2: tl.constexpr,
    o3: tl.constexpr,
    o4: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Affine gather with constexpr shape / stride digits.

    ``offs`` enumerates the logical flat index row-major; the digits map it to
    a source offset (input strides) and a destination offset (output strides).
    Constexpr divisors avoid the per-lane 64-bit integer division that a
    runtime divisor would cost.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ds = [d0, d1, d2, d3, d4]
    ss = [s0, s1, s2, s3, s4]
    os_ = [o0, o1, o2, o3, o4]
    rem = offs
    src_off = offs * 0
    dst_off = offs * 0
    for j in tl.static_range(RANK - 1, -1, -1):
        c = rem % ds[j]
        src_off += c * ss[j]
        dst_off += c * os_[j]
        rem = rem // ds[j]
    if NEED_MASK:
        mask = offs < n
        tl.store(dst_ptr + dst_off, tl.load(src_ptr + src_off, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + dst_off, tl.load(src_ptr + src_off))


def _int_view(tensor, dtype):
    """Zero-copy integer alias of a quantized tensor's storage."""
    try:
        return tensor.view(dtype)
    except (RuntimeError, TypeError, NotImplementedError):
        alias = torch.empty(0, dtype=dtype, device=tensor.device)
        alias.set_(
            tensor.untyped_storage(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
        return alias


def _byte_alias(tensor, nbytes, byte_shift=0):
    """Rank-1 int8 alias of ``tensor``'s storage ``byte_shift`` bytes in."""
    off_bytes = tensor.storage_offset() * tensor.element_size() + byte_shift
    return torch.empty(0, dtype=torch.int8, device=tensor.device).set_(
        tensor.untyped_storage(), off_bytes, (nbytes,), (1,)
    )


def _word_alias(tensor, wdtype, words, byte_shift=0):
    """Rank-1 integer-word alias of ``tensor``'s storage."""
    wsize = wdtype.itemsize
    off_bytes = tensor.storage_offset() * tensor.element_size() + byte_shift
    return torch.empty(0, dtype=wdtype, device=tensor.device).set_(
        tensor.untyped_storage(), off_bytes // wsize, (words,), (1,)
    )


def _is_storage_dense(numel, shape, stride):
    """True when the strides enumerate one dense span of ``numel`` elements."""
    if numel <= 1:
        return True
    if len(shape) <= 1:
        return stride[-1] == 1
    expect = 1
    for n, s in sorted(zip(shape, stride), key=lambda p: p[1]):
        if n <= 1:
            continue
        if s != expect:
            return False
        expect *= n
    return True


def _flat_block(n_words):
    if n_words >= 1 << 19:
        return _FLAT_MAX_BLOCK
    if n_words >= 1 << 16:
        return 16384
    if n_words >= 1 << 13:
        return 8192
    if n_words >= 1 << 10:
        return 4096
    return 1024


def _launch_flat(src, dst, n_words, block):
    grid = (triton.cdiv(n_words, block),)
    _int_repr_flat_kernel[grid](
        src,
        dst,
        n_words,
        BLOCK=block,
        NEED_MASK=(n_words % block != 0),
        num_warps=16 if block >= 16384 else 8,
    )


def _element_flat(src, dst, numel):
    """Single element-typed flat launch (tiny tensors / unalignable bases)."""
    nbytes = numel * src.element_size()
    block = _flat_block(nbytes) if nbytes >= _WORD_MIN_BYTES else _SMALL_BLOCK
    block = min(block, triton.next_power_of_2(numel))
    _launch_flat(src, dst, numel, max(block, 64))


def _align_words(src_base, dst_base, nbytes, esize):
    """Widest integer word whose *base alignment* both pointers satisfy.

    Measured on XPU3: a destination tensor with a non-zero storage offset runs
    the flat copy at ~half the DMA rate even when ``data_ptr()`` is 16-byte
    aligned, while a non-zero *source* offset is free.  So the destination is
    always left at offset 0 and no head shift is applied; the word is chosen as
    the widest {int64, int32} that divides both base addresses.
    """
    for wsize in (8, 4):
        if nbytes >= 2 * wsize and src_base % wsize == 0 and dst_base % wsize == 0:
            return wsize
    return 1


def _copy_linear(in_view, out, numel):
    """Bit-exact copy of a single dense span.

    The destination always keeps its own (zero) storage offset; a non-zero
    source offset is handled by the word alias itself, not by a shifted
    destination view (see ``_align_words``).
    """
    esize = out.element_size()
    nbytes = numel * esize
    if nbytes < _WORD_MIN_BYTES:
        _element_flat(in_view, out, numel)
        return

    src_base = in_view.data_ptr()
    dst_base = out.data_ptr()
    wsize = _align_words(src_base, dst_base, nbytes, esize)
    if wsize == 1:
        _element_flat(in_view, out, numel)
        return

    words = nbytes // wsize
    tail = nbytes - words * wsize

    wdtype = torch.int64 if wsize == 8 else torch.int32
    src_w = _word_alias(in_view, wdtype, words)
    dst_w = _word_alias(out, wdtype, words)
    _launch_flat(src_w, dst_w, words, _flat_block(words))

    if tail:
        tb = _byte_alias(in_view, tail, words * wsize)
        tb_d = _byte_alias(out, tail, words * wsize)
        _launch_flat(tb, tb_d, tail, 32)


def _packed_run(shape, stride):
    """Return ``(run_len, n_outer)`` for the maximal packed trailing run."""
    run = 1
    j = len(shape) - 1
    while j >= 0:
        if shape[j] == 1:
            j -= 1
            continue
        if stride[j] == run:
            run *= shape[j]
            j -= 1
        else:
            break
    return run, j + 1


def _pad(seq, fill, n=_MAX_RANK):
    seq = list(seq)[:n]
    return seq + [fill] * (n - len(seq))


def _collapse(shape, src_stride, dst_stride):
    """Drop size-1 dims and merge adjacent dims that stay packed in both
    layouts, so the gather/grid rank fits ``_MAX_RANK`` for tall tensors."""
    dims, ss, dd = [], [], []
    for n, s, d in zip(shape, src_stride, dst_stride):
        if n == 1:
            continue
        if dims and s == ss[-1] * dims[-1] and d == dd[-1] * dims[-1]:
            dims[-1] *= n
            ss[-1] = s
            dd[-1] = d
        else:
            dims.append(n)
            ss.append(s)
            dd.append(d)
    return dims, ss, dd


def _copy_run(in_view, out, shape, stride, run_len, n_outer):
    numel = 1
    for d in shape:
        numel *= d
    n_runs = numel // run_len
    esize = in_view.element_size()
    run_bytes = run_len * esize

    # Word-alias the run when every base and outer stride preserves the word
    # granularity (true for a broadcast expand: the outer strides are 0).
    wsize = 0
    for cand in (8, 4):
        if (
            run_bytes % cand == 0
            and in_view.data_ptr() % cand == 0
            and out.data_ptr() % cand == 0
            and all((s * esize) % cand == 0 for s in stride[:n_outer])
        ):
            wsize = cand
            break
    if wsize:
        wdtype = torch.int64 if wsize == 8 else torch.int32
        total_words = numel * esize // wsize
        src = _word_alias(in_view, wdtype, total_words)
        dst = _word_alias(out, wdtype, total_words)
        run_units = run_bytes // wsize
        s_in = [int(s * esize // wsize) for s in stride[:n_outer]]
        block = min(triton.next_power_of_2(run_units), _RUN_BLOCK)
        chunks = triton.cdiv(run_units, block)
        _int_repr_run_kernel[(n_runs * chunks,)](
            src,
            dst,
            run_units,
            chunks,
            *_pad(s_in, 0),
            *_pad([int(d) for d in shape[:n_outer]], 1),
            N_OUTER=n_outer,
            BLOCK=block,
            num_warps=8,
        )
        return

    block = min(triton.next_power_of_2(run_len), _RUN_BLOCK)
    chunks = triton.cdiv(run_len, block)
    _int_repr_run_kernel[(n_runs * chunks,)](
        in_view,
        out,
        run_len,
        chunks,
        *_pad([int(s) for s in stride[:n_outer]], 0),
        *_pad([int(d) for d in shape[:n_outer]], 1),
        N_OUTER=n_outer,
        BLOCK=block,
        num_warps=8,
    )


def _copy_gather(in_view, out, shape, stride, dst_stride):
    numel = 1
    for d in shape:
        numel *= d
    rank = len(shape)
    esize = out.element_size()
    cap = _GATHER_BLOCK_BYTE if esize == 1 else _GATHER_BLOCK_WIDE
    if esize == 1:
        # The 1-byte gather path is fastest at its 2048-lane cap and gets
        # slower at 1024 (measured), so keep the cap for every layout.
        block = min(cap, triton.next_power_of_2(numel))
    else:
        # 4-byte layouts: a small layout wants enough programs to fill the
        # cores, a large one wants the widest legal tile.
        want = triton.next_power_of_2(max(triton.cdiv(numel, 64), 1))
        block = min(cap, want)
    block = max(block, 64)
    grid = (triton.cdiv(numel, block),)
    _int_repr_gather_kernel[grid](
        in_view,
        out,
        numel,
        *_pad([int(s) for s in stride], 0),
        *_pad([int(d) for d in shape], 1),
        *_pad([int(s) for s in dst_stride], 0),
        RANK=rank,
        BLOCK=block,
        NEED_MASK=(numel % block != 0),
        num_warps=8,
    )


def _alloc_dense_like(self, shape, stride, out_dtype):
    """Dense output preserving the input strides (destination offset 0).

    The destination storage offset is kept at 0: an offset destination view
    halves the flat-copy DMA rate on XPU3 even when 16-byte aligned.  A
    non-zero source offset is harmless and is absorbed by the word alias.
    """
    return torch.empty_strided(shape, stride, dtype=out_dtype, device=self.device)


def int_repr(self, out=None):
    """Return the raw integer codes of the affine-quantized tensor ``self``."""
    logger.debug("GEMS_KUNLUNXIN INT_REPR")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "int_repr(): argument 'self' must be a Tensor, not "
            f"{type(self).__name__}"
        )
    if not self.is_quantized:
        raise RuntimeError(
            f"int_repr(): expected a quantized tensor, got dtype {self.dtype}"
        )
    out_dtype = _Q_TO_INT.get(self.dtype)
    if out_dtype is None:
        raise RuntimeError(f"int_repr(): unsupported quantized dtype {self.dtype}")

    shape = self.shape
    numel = self.numel()
    stride = self.stride()
    dense = _is_storage_dense(numel, shape, stride)
    in_view = _int_view(self, out_dtype)

    if out is None:
        if dense:
            out = _alloc_dense_like(self, shape, stride, out_dtype)
        else:
            out = torch.empty(shape, dtype=out_dtype, device=self.device)
    else:
        if not isinstance(out, torch.Tensor) or out.dtype != out_dtype:
            raise RuntimeError(
                "int_repr(): expected out tensor to have dtype "
                f"{out_dtype}, but got "
                f"{getattr(out, 'dtype', type(out).__name__)} instead"
            )
        if out.shape != shape:
            out.resize_(shape)

    if numel == 0:
        # The freshly allocated / resized empty result is already correct;
        # never launch a zero grid.
        return out

    if stride == out.stride():
        _copy_linear(in_view, out, numel)
        return out

    dims, sstride, dstride = _collapse(shape, stride, out.stride())
    if not dims:
        # Every dim was size-1: the tensors are 1-element / empty and would
        # have been handled by the linear path above only if numel > 1.
        _copy_gather(in_view, out, [1], [0], [0])
        return out
    run_len, n_outer = _packed_run(dims, sstride)
    if (
        out.is_contiguous()
        and run_len >= _RUN_MIN_LEN
        and n_outer <= _MAX_RANK
        and (numel // run_len) <= _RUN_MAX_RUNS
    ):
        _copy_run(in_view, out, dims, sstride, run_len, n_outer)
    else:
        _copy_gather(in_view, out, dims, sstride, dstride)
    return out
