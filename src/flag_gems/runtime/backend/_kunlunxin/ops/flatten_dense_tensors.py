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
"""Kunlunxin (XPU3) override for ``aten::flatten_dense_tensors``.

Two XPU3-specific problems in the generic implementation are addressed here:

1. **Compile-time wall (host stall).**  The generic metadata path builds one
   ``_flatten_general_kernel`` whose body runs an 8-deep ``tl.static_range`` of
   vector int64 ``%``/``//`` plus a masked load/store.  On this stack the
   ``make_ttxir`` stage of that single kernel takes >35 minutes (measured on
   the very first non-contiguous case of ``tests/test_flatten_dense_tensors.py``
   -- ``count=1, layout=transpose``), so the functional suite never finishes.
   Here the dimensions are collapsed on the host and every non-contiguous input
   is copied by a rank-bounded affine gather kernel (same lowering as
   ``select_copy``), which compiles in seconds.

2. **Copy throughput / launch shape.**  The generic contiguous path copies with
   small per-dtype tiles (1024 lanes for 16-bit dtypes) over a 2-D
   ``(tensor, chunk)`` grid whose chunk count follows the *per-tensor* size, so
   medium shapes end up with far fewer than the 64 available programs.  This
   file (a) bit-casts same-dtype copies to the widest 32/64-bit integer word
   view (4/2 elements per lane for fp16/fp32), (b) sizes the flat tile from the
   *total* word count so the grid fills the machine, (c) uses a plain tuple grid
   and (d) an ``EVEN`` maskless branch.

Semantics preserved: list order, mixed-dtype promotion (``torch.cat`` rules),
the zero-copy ``view(-1)`` fast path for a single contiguous input, zero-size
inputs, non-contiguous views, empty-list ``RuntimeError`` and non-tensor
``TypeError``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float64: tl.float64,
    torch.int64: tl.int64,
    torch.int32: tl.int32,
    torch.int16: tl.int16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.bool: tl.int8,
}

_MAX_SLOTS = 8
_MAX_RANK = 8

# Flat word-copy tile ladder: (minimum total word count, block, num_warps).
# 64 K int64 words = 512 KB per program is the measured bandwidth sweet spot;
# smaller shapes shrink the tile so the grid still fills all 64 cores.
_FLAT_LADDER = (
    (2**19, 65536, 32),
    (2**15, 8192, 8),
    (0, 2048, 4),
)

# Bounded tile for one-element-per-lane strided gathers.
_GATHER_BLOCK = 1024


def _pick_flat_block(total_words):
    for lo, block, warps in _FLAT_LADDER:
        if total_words >= lo:
            return block, warps
    return 2048, 4


def _word_view(out, tensors, numels, elem_offsets, esize):
    """Widest same-dtype integer word view shared by out and every input.

    Copying the same dtype through a wider integer word is bit-identical and
    removes the ~3x DMA penalty of 2-byte lanes on XPU3.  Conditions: every
    input and the output must be divisible by the word, every cumulative
    element offset of the output must be word-aligned, and all data pointers
    (and the output pointer) must be word-aligned.
    """
    out_ptr = out.data_ptr()
    for wdt, wsize in ((torch.int64, 8), (torch.int32, 4)):
        if esize >= wsize:
            continue
        per_word = wsize // esize
        if out_ptr % wsize != 0:
            continue
        if any(n % per_word != 0 for n in numels):
            continue
        if any((o * esize) % wsize != 0 for o in elem_offsets):
            continue
        if any(t.data_ptr() % wsize != 0 for t in tensors):
            continue
        return wdt, per_word
    return None, 1


@triton.jit
def _flatten_words_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    en0,
    en1,
    en2,
    en3,
    en4,
    en5,
    en6,
    en7,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    MASKED: tl.constexpr,
):
    """One program copies one BLOCK-sized word chunk fully inside one input.

    ``st``/``en`` are the input's word range inside the packed output, so the
    store offset is the global word offset and the load offset is relative to
    the input pointer.  Slot metadata travels as scalar kernel arguments, so no
    device metadata tensor (and no H2D copy) is needed per call.
    """
    t = tl.program_id(0)
    k = tl.program_id(1)

    ptr = tl.where(
        t == 0,
        p0,
        tl.where(
            t == 1,
            p1,
            tl.where(
                t == 2,
                p2,
                tl.where(
                    t == 3,
                    p3,
                    tl.where(
                        t == 4, p4, tl.where(t == 5, p5, tl.where(t == 6, p6, p7))
                    ),
                ),
            ),
        ),
    )
    st = tl.where(
        t == 0,
        st0,
        tl.where(
            t == 1,
            st1,
            tl.where(
                t == 2,
                st2,
                tl.where(
                    t == 3,
                    st3,
                    tl.where(
                        t == 4, st4, tl.where(t == 5, st5, tl.where(t == 6, st6, st7))
                    ),
                ),
            ),
        ),
    )
    en = tl.where(
        t == 0,
        en0,
        tl.where(
            t == 1,
            en1,
            tl.where(
                t == 2,
                en2,
                tl.where(
                    t == 3,
                    en3,
                    tl.where(
                        t == 4, en4, tl.where(t == 5, en5, tl.where(t == 6, en6, en7))
                    ),
                ),
            ),
        ),
    )

    offs = st + k.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pp = ptr.to(tl.pointer_type(DT))
    if MASKED:
        mask = offs < en
        x = tl.load(pp + (offs - st), mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)
    else:
        # EVEN branch: every input word range is a whole multiple of BLOCK, so
        # an always-true mask (the slow masked-memory path) is avoided.
        x = tl.load(pp + (offs - st))
        tl.store(out_ptr + offs, x)


@triton.jit
def _flat_elem_kernel(
    src_ptr,
    dst_ptr,
    dst_base,
    n_elements,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Flat one-element-per-lane copy of a contiguous input into ``dst_base``."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pp = src_ptr.to(tl.pointer_type(DT))
    if NEED_MASK:
        mask = offs < n_elements
        x = tl.load(pp + offs, mask=mask)
        tl.store(dst_ptr + dst_base + offs, x, mask=mask)
    else:
        x = tl.load(pp + offs)
        tl.store(dst_ptr + dst_base + offs, x)


@triton.jit
def _gather_kernel(
    src_ptr,
    dst_ptr,
    dst_base,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    n_elements,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    """Affine gather of one strided input into the packed output.

    The logical flat index is decomposed along the (collapsed, row-major)
    shape and mapped onto the input strides.  ``dims``/``strides`` travel as
    scalar arguments, so no per-call device metadata is built.
    """
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    ss = [s0, s1, s2, s3, s4, s5, s6, s7]
    dd = [d0, d1, d2, d3, d4, d5, d6, d7]
    rem = offs
    phys = tl.zeros([BLOCK], tl.int64)
    for j in tl.static_range(RANK - 1, -1, -1):
        phys += (rem % dd[j]).to(tl.int64) * ss[j]
        rem = rem // dd[j]

    pp = src_ptr.to(tl.pointer_type(DT))
    x = tl.load(pp + phys, mask=mask)
    tl.store(dst_ptr + dst_base + offs, x, mask=mask)


def _pad8(values, fill):
    values = list(values)[:_MAX_SLOTS]
    return values + [fill] * (_MAX_SLOTS - len(values))


def _collapse_dims(shape, strides):
    """Drop size-1 dims and merge adjacent dims that are already packed."""
    dims = []
    strs = []
    for d, s in zip(reversed(shape), reversed(strides)):
        if d == 1:
            continue
        if dims and s == strs[-1] * dims[-1]:
            # Packed: the merged dim keeps the *inner* stride (the outer one is
            # exactly inner_extent * inner_stride, so a flat index maps to it).
            dims[-1] *= d
        else:
            dims.append(d)
            strs.append(s)
    dims.reverse()
    strs.reverse()
    return dims, strs


def _launch_words(out, tensors, numels, elem_offsets, per_word, wdt):
    """Single 2-D-grid word copy of a uniform-dtype contiguous group."""
    src_words = [t.view(-1).view(wdt) for t in tensors]
    out_words = out.view(wdt)
    # The kernel must load/store the *word* type (the aliased view), not the
    # element type of the inputs.
    tl_dt = tl.int64 if wdt is torch.int64 else tl.int32
    word_numels = [n // per_word for n in numels]
    word_offsets = [o // per_word for o in elem_offsets]
    word_ends = [o + n for o, n in zip(word_offsets, word_numels)]
    total_words = word_ends[-1]
    block, warps = _pick_flat_block(total_words)
    chunks = [(n + block - 1) // block for n in word_numels]
    max_chunks = max(chunks)
    masked = not (
        all(n == word_numels[0] for n in word_numels)
        and word_numels[0] % block == 0
    )

    n = len(tensors)
    ptrs = _pad8([t.data_ptr() for t in src_words], 0)
    sts = _pad8(word_offsets, 0)
    ens = _pad8(word_ends, 0)
    # Plain tuple grid: a per-call grid callable defeats the launch cache on
    # this stack (~120 ms per launch).
    _flatten_words_kernel[(n, max_chunks)](
        out_words,
        *ptrs,
        *sts,
        *ens,
        BLOCK=block,
        DT=tl_dt,
        MASKED=masked,
        num_warps=warps,
    )


def _copy_flat(out, tensor, dst_base, tl_dt):
    """Element-wise flat copy of a contiguous input (cast or unaligned case)."""
    n = tensor.numel()
    total = n
    block = min(triton.next_power_of_2(max(total, 1)), 8192)
    block = max(block, 1024)
    grid = (triton.cdiv(total, block),)
    _flat_elem_kernel[grid](
        tensor,
        out,
        dst_base,
        total,
        BLOCK=block,
        DT=tl_dt,
        NEED_MASK=(total % block != 0),
        num_warps=4,
    )


def _copy_strided(out, tensor, dst_base, tl_dt):
    """Affine gather of a non-contiguous input into the packed output."""
    n = tensor.numel()
    dims, strs = _collapse_dims(list(tensor.shape), list(tensor.stride()))
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError(
            "flatten_dense_tensors: input rank after collapsing is "
            f"{rank}, at most {_MAX_RANK} is supported"
        )
    grid = (triton.cdiv(n, _GATHER_BLOCK),)
    _gather_kernel[grid](
        tensor,
        out,
        dst_base,
        *_pad8(strs, 0),
        *_pad8(dims, 1),
        n,
        RANK=rank,
        BLOCK=_GATHER_BLOCK,
        DT=tl_dt,
    )


def flatten_dense_tensors(tensors):
    """``aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor``."""
    logger.debug("GEMS_KUNLUNXIN FLATTEN_DENSE_TENSORS")
    if isinstance(tensors, torch.Tensor):
        raise TypeError(
            "flatten_dense_tensors(): argument 'tensors' (position 1) must be "
            "a list of Tensors, not Tensor"
        )
    for item in tensors:
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                "flatten_dense_tensors(): expected Tensor as element, but got "
                f"{type(item).__name__}"
            )
    ts = list(tensors)
    if ts:
        device = ts[0].device
        if any(t.device != device for t in ts):
            raise RuntimeError("all tensors must be on the same device")
        if any(t.layout != torch.strided for t in ts):
            raise RuntimeError("flatten_dense_tensors requires dense tensors")
        if any(t.is_conj() or t.is_neg() for t in ts):
            raise NotImplementedError(
                "lazy conjugate/negative tensors are not supported"
            )
    n = len(ts)
    if n == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    # A single contiguous input is a zero-copy flattened view, exactly like the
    # reference; everything else is copied.
    if n == 1 and ts[0].is_contiguous():
        return ts[0].view(-1)

    numels = [t.numel() for t in ts]
    elem_offsets = [0] * n
    for i in range(1, n):
        elem_offsets[i] = elem_offsets[i - 1] + numels[i - 1]
    total = elem_offsets[-1] + numels[-1] if n else 0

    out_dtype = ts[0].dtype
    uniform = True
    for t in ts[1:]:
        if t.dtype != out_dtype:
            uniform = False
    if not uniform:
        out_dtype = ts[0].dtype
        for t in ts[1:]:
            out_dtype = torch.promote_types(out_dtype, t.dtype)

    out = torch.empty(total, dtype=out_dtype, device=ts[0].device)
    if total == 0:
        return out

    if _TL_DTYPE.get(ts[0].dtype) is None or _TL_DTYPE.get(out_dtype) is None:
        raise RuntimeError(
            "flatten_dense_tensors: unsupported dtype %s" % out_dtype
        )

    # Walk the list in consecutive same-dtype runs so mixed dtype lists are
    # copied group by group (the store casts to the promoted output dtype).
    cur = 0
    while cur < n:
        j = cur + 1
        while j < n and ts[j].dtype == ts[cur].dtype:
            j += 1
        group = ts[cur:j]
        group_numels = numels[cur:j]
        group_offsets = elem_offsets[cur:j]
        tl_dt = _TL_DTYPE[group[0].dtype]
        contig = all(t.is_contiguous() for t in group)
        if contig and group[0].dtype == out_dtype and len(group) <= _MAX_SLOTS:
            wdt, per_word = _word_view(
                out, group, group_numels, group_offsets, group[0].element_size()
            )
            if wdt is not None:
                _launch_words(
                    out, group, group_numels, group_offsets, per_word, wdt
                )
            else:
                for k, t in enumerate(group):
                    _copy_flat(out, t, group_offsets[k], tl_dt)
        elif contig:
            for k, t in enumerate(group):
                _copy_flat(out, t, group_offsets[k], tl_dt)
        else:
            for k, t in enumerate(group):
                if group_numels[k] == 0:
                    continue
                _copy_strided(out, t, group_offsets[k], tl_dt)
        cur = j
    return out
