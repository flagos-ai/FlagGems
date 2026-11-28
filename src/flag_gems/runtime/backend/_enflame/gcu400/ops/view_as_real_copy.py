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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Flat fast path (contiguous inputs): a persistent grid-stride over dtype-sized
# tiles.  Tile shape, multi-buffer depth, warp count and the CTA cap all live in
# ``_flat_launch_config``; the CTA count moved from 48 to 24 together with the
# multi-buffered tile because 24 wider residents beat 48 narrower ones on every
# bench size (268M reals: 1249us vs 1260us) while halving the dispatch tax that
# the 4M-real shapes pay.
_FLAT_STAGES = 1
# The row path keeps the 48-CTA grid: its work items are already long
# contiguous runs, so extra residents hide more of the row-base scalar work.
_ROW_GRID = 48
# Row fast path (last complex dim contiguous, leading dims strided): one work
# item copies a contiguous run of ``L`` reals (a whole last-dim row slice).
# Rows shorter than this keep the generic kernel; the scalar row-base
# decomposition only pays off when the row carries real payload.
_ROW_MIN_ROW_REALS = 512
_ROW_BLOCK_CAP = 16384
# Gather path (generic strides): one program covers BLOCK complex elements
# (2 * BLOCK reals) per grid-stride step; the grid stays at the persistent
# cap so huge strided inputs never pay the >48-CTA dispatch tax.
_BLOCK_SIZE = 1024
_GENERIC_GRID = 48
# Leading (batch) dimensions are decomposed with a fixed number of slots so
# the kernel always receives its full argument sets; higher-rank inputs are
# rejected by the wrapper.
_MAX_LEAD_RANK = 5
# The flat and row kernels index in int32 reals; anything at or beyond this
# bound (output numel or max touched source word) falls back to the int64
# gather kernel.
_INT32_LIMIT = 2**31

_REAL_DTYPE = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@triton.jit
def _flip_lanes(values, offsets, FLIP: tl.constexpr):
    """Resolve the lazy conjugate/negative bits on a streamed tile.

    ``FLIP`` is 1 = conjugate (negate the imag lane), 2 = negative (negate
    both lanes), 3 = conjugate+negative (negate the real lane): the
    composition is commutative and every flip is an exact IEEE sign flip.
    """
    if FLIP == 1:
        values = tl.where((offsets & 1) != 0, -values, values)
    elif FLIP == 2:
        values = -values
    elif FLIP == 3:
        values = tl.where((offsets & 1) == 0, -values, values)
    return values


@libentry()
@triton.jit
def _view_as_real_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_reals,
    FLIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EVEN: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Stream a contiguous real view into the contiguous ``(..., 2)`` output.

    ``src_ptr`` points at the real view of the complex input and ``dst_ptr``
    at the freshly allocated real output; both cover the same number of
    reals, so one linear pass over the real stream is exact and fully
    coalesced.  ``EVEN`` drops the tail predicate when the stream is an
    exact multiple of the *super-tile* (``BLOCK_SIZE * UNROLL``); otherwise
    every one of the ``UNROLL`` sub-tiles is masked individually.

    ``UNROLL`` is a manual multi-buffer: one grid-stride step issues
    ``UNROLL`` independent loads before the matching stores, so several
    memory transactions are in flight per program.  On GCU400 the plain
    single-tile loop tops out around 1.6-1.7 TB/s on a 1 GiB-per-side copy
    while the ``UNROLL`` twin reaches ~1.85 TB/s (1288us -> 1249us at 268M
    reals).  ``UNROLL == 1`` keeps the original single-stream shape for the
    small sizes where the extra registers buy nothing.
    """
    pid = tl.program_id(axis=0)
    span = BLOCK_SIZE * UNROLL
    num_super = (n_reals + span - 1) // span
    lane = tl.arange(0, BLOCK_SIZE)
    for super_id in tl.range(pid, num_super, GRID_SIZE, num_stages=NUM_STAGES):
        base = super_id * span
        offsets = base + lane
        if UNROLL == 1:
            if EVEN:
                values = tl.load(src_ptr + offsets)
                tl.store(dst_ptr + offsets, _flip_lanes(values, offsets, FLIP))
            else:
                mask = offsets < n_reals
                values = tl.load(src_ptr + offsets, mask=mask)
                tl.store(
                    dst_ptr + offsets,
                    _flip_lanes(values, offsets, FLIP),
                    mask=mask,
                )
        elif UNROLL == 2:
            offsets1 = offsets + BLOCK_SIZE
            if EVEN:
                values0 = tl.load(src_ptr + offsets)
                values1 = tl.load(src_ptr + offsets1)
                tl.store(dst_ptr + offsets, _flip_lanes(values0, offsets, FLIP))
                tl.store(dst_ptr + offsets1, _flip_lanes(values1, offsets1, FLIP))
            else:
                mask0 = offsets < n_reals
                mask1 = offsets1 < n_reals
                values0 = tl.load(src_ptr + offsets, mask=mask0)
                values1 = tl.load(src_ptr + offsets1, mask=mask1)
                tl.store(
                    dst_ptr + offsets, _flip_lanes(values0, offsets, FLIP), mask=mask0
                )
                tl.store(
                    dst_ptr + offsets1,
                    _flip_lanes(values1, offsets1, FLIP),
                    mask=mask1,
                )
        else:
            offsets1 = offsets + BLOCK_SIZE
            offsets2 = offsets1 + BLOCK_SIZE
            offsets3 = offsets2 + BLOCK_SIZE
            if EVEN:
                values0 = tl.load(src_ptr + offsets)
                values1 = tl.load(src_ptr + offsets1)
                values2 = tl.load(src_ptr + offsets2)
                values3 = tl.load(src_ptr + offsets3)
                tl.store(dst_ptr + offsets, _flip_lanes(values0, offsets, FLIP))
                tl.store(dst_ptr + offsets1, _flip_lanes(values1, offsets1, FLIP))
                tl.store(dst_ptr + offsets2, _flip_lanes(values2, offsets2, FLIP))
                tl.store(dst_ptr + offsets3, _flip_lanes(values3, offsets3, FLIP))
            else:
                mask0 = offsets < n_reals
                mask1 = offsets1 < n_reals
                mask2 = offsets2 < n_reals
                mask3 = offsets3 < n_reals
                values0 = tl.load(src_ptr + offsets, mask=mask0)
                values1 = tl.load(src_ptr + offsets1, mask=mask1)
                values2 = tl.load(src_ptr + offsets2, mask=mask2)
                values3 = tl.load(src_ptr + offsets3, mask=mask3)
                tl.store(
                    dst_ptr + offsets, _flip_lanes(values0, offsets, FLIP), mask=mask0
                )
                tl.store(
                    dst_ptr + offsets1,
                    _flip_lanes(values1, offsets1, FLIP),
                    mask=mask1,
                )
                tl.store(
                    dst_ptr + offsets2,
                    _flip_lanes(values2, offsets2, FLIP),
                    mask=mask2,
                )
                tl.store(
                    dst_ptr + offsets3,
                    _flip_lanes(values3, offsets3, FLIP),
                    mask=mask3,
                )


@libentry()
@triton.jit
def _view_as_real_copy_row_kernel(
    src_ptr,
    dst_ptr,
    ld0,
    ld1,
    ld2,
    ld3,
    ls0,
    ls1,
    ls2,
    ls3,
    n_rows,
    row_reals,
    n_col_blocks,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    FLIP: tl.constexpr,
    NCB1: tl.constexpr,
    FULL: tl.constexpr,
):
    """Copy rows of a last-dim-contiguous complex view into the flat output.

    Eligible inputs have complex ``stride[-1] == 1``: every leading-index
    ("row") slice is then one contiguous run of ``row_reals`` source words,
    while the output run for row ``o`` sits at the flat offset ``o *
    row_reals``.  Each work item is one (row, column-block) pair; the row
    base is a scalar decomposition over the leading dims, so the vector
    loads/stores stay pure ``base + arange`` and never degrade to a gather.

    ``ld*``/``ls*`` carry the leading dims and their strides in real
    elements (slots padded with 1/0 up to rank 4); ``RANK`` is the number
    of leading dims.  ``NCB1`` specializes the common single-block-per-row
    case so the chunk ``div`` disappears, and ``FULL`` drops the row-tail
    predicate when the tile covers the row exactly.  ``FLIP`` resolves the
    lazy conjugate/negative bits in-register (see the flat kernel
    docstring).
    """
    pid = tl.program_id(axis=0)
    total = n_rows * n_col_blocks
    for chunk in tl.range(pid, total, GRID_SIZE):
        if NCB1:
            row = chunk
            col_block = 0
        else:
            row = chunk // n_col_blocks
            col_block = chunk - row * n_col_blocks
        lead_dims = [ld0, ld1, ld2, ld3]
        lead_strides = [ls0, ls1, ls2, ls3]
        rem = row
        base = tl.zeros([], dtype=tl.int32)
        for j in tl.static_range(RANK - 1, 0, -1):
            idx = rem % lead_dims[j]
            rem = rem // lead_dims[j]
            base += idx * lead_strides[j]
        base += rem * lead_strides[0]
        offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if FULL:
            values = tl.load(src_ptr + base + offsets)
        else:
            mask = offsets < row_reals
            values = tl.load(src_ptr + base + offsets, mask=mask)
        if FLIP == 1:
            values = tl.where((offsets & 1) != 0, -values, values)
        elif FLIP == 2:
            values = -values
        elif FLIP == 3:
            values = tl.where((offsets & 1) == 0, -values, values)
        if FULL:
            tl.store(dst_ptr + row * row_reals + offsets, values)
        else:
            tl.store(dst_ptr + row * row_reals + offsets, values, mask=mask)


@libentry()
@triton.jit
def _view_as_real_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    n_complex,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    FLIP: tl.constexpr,
):
    """Copy a non-contiguous complex buffer into a contiguous ``(..., 2)`` real one.

    The output is contiguous, so the flat complex index decomposes row-major
    over the (complex) tensor dimensions; each recovered coordinate is scaled
    by the matching stride of the input's real view, which is expressed in real
    elements. The trailing lane keeps stride 1, so the (real, imag) pair of a
    complex element is always adjacent and is loaded as a two-wide vector.

    The slots carry the dimensions in input order: ``d0``/``s0`` is the
    outermost dimension and ``d[RANK-1]``/``s[RANK-1]`` the innermost one (the
    dimension whose stride separates the complex elements). The generic path
    peels digits from the innermost slot outward, so both paths share the same
    convention.

    A ``RANK == 2`` specialization is kept because it is the overwhelmingly
    common non-contiguous case (a row/column slice of an otherwise flat
    buffer); the tight two-step ``div``/``sub`` decomposition beats the generic
    static loop by a wide margin on that layout.

    ``FLIP`` resolves the lazy conjugate/negative bits in-register (see the
    flat kernel docstring); the programs grid-stride under the persistent
    grid cap.
    """
    pid = tl.program_id(axis=0)
    num_blocks = (n_complex + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, GRID_SIZE):
        i = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = i < n_complex
        i = i.to(tl.int64)

        if RANK == 2:
            # d1/s1 is the innermost dim (fastest digit), d0/s0 the outer one.
            outer = i // d1
            src_base = (i - outer * d1) * s1 + outer * s0
        else:
            lead_dims = [d0, d1, d2, d3, d4]
            lead_strides = [s0, s1, s2, s3, s4]
            rem = i
            src_base = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
            for j in tl.static_range(RANK - 1, -1, -1):
                idx = rem % lead_dims[j]
                rem = rem // lead_dims[j]
                src_base += idx * lead_strides[j]

        lane = tl.arange(0, 2)
        src_offsets = src_base[:, None] + lane[None, :]
        dst_offsets = i[:, None] * 2 + lane[None, :]
        mask2 = mask[:, None]
        values = tl.load(src_ptr + src_offsets, mask=mask2)
        if FLIP == 1:
            values = tl.where(lane[None, :] != 0, -values, values)
        elif FLIP == 2:
            values = -values
        elif FLIP == 3:
            values = tl.where(lane[None, :] == 0, -values, values)
        tl.store(dst_ptr + dst_offsets, values, mask=mask2)


def _raw_real_view(t, dtype=None):
    """Build the ``(..., 2)`` real view straight from the complex storage.

    This is exactly the view ``torch.view_as_real`` would produce -- the
    storage offset and every stride doubled, plus a trailing size-2 stride-1
    dimension -- except it deliberately carries no lazy conjugate/negative
    bits: torch_gcu's ``resolve_conj``/``resolve_neg`` never strip the bits
    (P21), so ``torch.view_as_real`` keeps refusing such inputs with
    "unresolved conjugated tensors". The kernels' ``FLIP`` lanes resolve the
    bits in-register instead, keeping the op on a single kernel launch.

    ``dtype`` overrides the element type of the view, which is how the
    complex128 fast path reads the ComplexFloat-replaced payload as plain
    fp32 words.
    """
    if dtype is None:
        dtype = _REAL_DTYPE[t.dtype]
    view = torch.empty(0, dtype=dtype, device=t.device)
    view.set_(
        t.untyped_storage(),
        t.storage_offset() * 2,
        tuple(t.shape) + (2,),
        tuple(s * 2 for s in t.stride()) + (1,),
    )
    return view


def _fp32_word_ghost(out_f64):
    """Alias a freshly allocated float64 output as same-shape fp32 words.

    On this device a float64 tensor physically stores one compact fp32 word
    per element ("Double use Float replace"), so writing the fp32 word view
    is bit-identical to writing the widened float64 values the payload can
    represent.  The ghost therefore lets the complex128 fast path reuse the
    fp32 copy kernels without any f64 arithmetic (P02/P04) or cast kernels.
    """
    ghost = torch.empty(0, dtype=torch.float32, device=out_f64.device)
    ghost.set_(
        out_f64.untyped_storage(),
        out_f64.storage_offset(),
        out_f64.shape,
        out_f64.stride(),
    )
    return ghost


_C128_COMPACT_PAYLOAD = {}


def _c128_payload_is_compact(device):
    """One-time sentinel probe that complex128 storage is fp32-word compact.

    The complex128 fast path relies on the "ComplexDouble use ComplexFloat
    replace" layout (verified once per device): a 1-element float64 tensor
    whose fp32 word alias reads back 1.0 proves the compact layout.  Any
    other layout (or a failing probe) keeps the legacy narrow-widen chain,
    which stays correct regardless of the physical representation.
    """
    key = device.index if device.type == "gcu" else str(device)
    if key not in _C128_COMPACT_PAYLOAD:
        try:
            sentinel = torch.tensor([1.0], dtype=torch.float64, device=device)
            word = torch.empty(0, dtype=torch.float32, device=device)
            word.set_(
                sentinel.untyped_storage(),
                sentinel.storage_offset(),
                sentinel.shape,
                sentinel.stride(),
            )
            _C128_COMPACT_PAYLOAD[key] = bool(word.cpu()[0].item() == 1.0)
        except Exception:
            _C128_COMPACT_PAYLOAD[key] = False
    return _C128_COMPACT_PAYLOAD[key]


def _flat_launch_config(dtype, n_reals):
    """``(block, unroll, warps, grid)`` ladder for the contiguous stream.

    Measured on GCU400 with the benchmark harness timing (warmup 1s / rep
    100ms / median) over the bench sizes, kernel alone:

    =========  ===========  ===============================  ================
    element    n_reals       winner                           torch copy_
    =========  ===========  ===============================  ================
    fp16       4M           24 CTA / B8192/U4/nw2  13.8us      13.8us
    fp16       67M          24 CTA / B65536/U2/nw2 153.0us     132.6us
    fp16       268M         48 CTA / B65536/U1/nw2 649us       504.5us
    fp32       4M           24 CTA / B32768/U2/nw2 22.2us      21.5us
    fp32       33M          24 CTA / B65536/U2/nw2 135.7us     133.7us
    fp32       67M          24 CTA / B65536/U2/nw2 280.9us     257.9us
    fp32       268M         24 CTA / B16384/U2/nw2 1249.4us    1010.0us
    =========  ===========  ===============================  ================

    The 16-bit stream wants the widest tile everywhere but the small band;
    the 32-bit stream must shrink its tile once each side is ~1 GiB (a 64 KiB
    tile there costs ~5%).  The CTA count only has to grow back for the
    16-bit 268M shape, where 24 CTAs leave ~9% on the table (709us -> 649us):
    its 0.5 GiB-per-side stream is short enough per tile that the extra
    residents still hide the latency.
    """
    if dtype == torch.float16:
        if n_reals <= 2**23:
            return 8192, 4, 2, 24
        if n_reals >= 2**28:
            # 16-bit 268M: the multi-buffer is a pessimization here (700us with
            # U2 vs 649us with plain U1) because the tile is already large
            # enough to saturate the stream; the 48-CTA cap comes back with it.
            return 65536, 1, 2, 48
        return 65536, 2, 2, 24
    if n_reals <= 2**23:
        return 32768, 2, 2, 24
    if n_reals >= 2**28:
        return 16384, 2, 2, 24
    return 65536, 2, 2, 24


def _launch_real_copy(src, dst, flip, n_complex):
    """Dispatch the real-word copy ``src -> dst`` on a single kernel launch.

    ``src``/``dst`` are ``(..., 2)`` real views (element type fp16/fp32, or
    the fp32-word alias pair for complex128) covering ``2 * n_complex``
    words.  Fast paths: contiguous streams take the flat kernel; inputs whose
    last complex dim is contiguous take the row kernel (contiguous per-row
    runs, scalar row bases).  Everything else falls back to the int64 gather
    kernel, which handles arbitrary strides and huge spans.
    """
    n_reals = dst.numel()

    if src.is_contiguous() and n_reals < _INT32_LIMIT:
        block, unroll, num_warps, max_grid = _flat_launch_config(src.dtype, n_reals)
        span = block * unroll
        num_super = (n_reals + span - 1) // span
        grid = min(max_grid, num_super)
        _view_as_real_copy_flat_kernel[(grid,)](
            src,
            dst,
            n_reals,
            FLIP=flip,
            BLOCK_SIZE=block,
            GRID_SIZE=grid,
            NUM_STAGES=_FLAT_STAGES,
            EVEN=(n_reals % span == 0),
            UNROLL=unroll,
            num_warps=num_warps,
        )
        return

    # Row fast path: complex stride[-1] == 1 makes every leading-index slice
    # one contiguous source run; require a meaningful row payload, a lead
    # rank that fits the slot budget and int32 address safety on both sides.
    if (
        src.dim() >= 3
        and src.dim() - 2 <= _MAX_LEAD_RANK - 1
        and src.stride(-2) == 2
        and 2 * src.shape[-2] >= _ROW_MIN_ROW_REALS
        and min(src.stride()) >= 0
    ):
        row_reals = 2 * src.shape[-2]
        max_word = src.storage_offset() + sum(
            (d - 1) * abs(s) for d, s in zip(src.shape, src.stride())
        )
        if max_word < _INT32_LIMIT and n_reals < _INT32_LIMIT:
            block = min(_ROW_BLOCK_CAP, triton.next_power_of_2(row_reals))
            n_col_blocks = (row_reals + block - 1) // block
            n_rows = n_reals // row_reals
            total = n_rows * n_col_blocks
            grid = min(_ROW_GRID, total)
            rank = src.dim() - 2  # leading complex dims
            lead_dims = list(src.shape[:-2]) + [1] * (4 - rank)
            lead_strides = list(src.stride()[:-2]) + [0] * (4 - rank)
            num_warps = 4 if src.dtype == torch.float16 else 2
            _view_as_real_copy_row_kernel[(grid,)](
                src,
                dst,
                *lead_dims,
                *lead_strides,
                n_rows,
                row_reals,
                n_col_blocks,
                RANK=rank,
                BLOCK_SIZE=block,
                GRID_SIZE=grid,
                FLIP=flip,
                NCB1=(n_col_blocks == 1),
                FULL=(n_col_blocks == 1 and block == row_reals),
                num_warps=num_warps,
            )
            return

    # Generic gather fallback (int64 addressing, arbitrary strides).
    rank = src.dim() - 1
    if rank > _MAX_LEAD_RANK:
        raise RuntimeError(
            f"view_as_real_copy supports at most {_MAX_LEAD_RANK} "
            f"dimensions, got {src.dim() - 1}"
        )
    num_blocks = (n_complex + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    grid = min(_GENERIC_GRID, num_blocks)
    dims = list(src.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
    strides = list(src.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
    _view_as_real_copy_strided_kernel[(grid,)](
        src,
        dst,
        *dims,
        *strides,
        n_complex,
        RANK=rank,
        BLOCK_SIZE=_BLOCK_SIZE,
        GRID_SIZE=grid,
        FLIP=flip,
    )


def _host_view_as_real_copy(t):
    """aten CPU round trip for the fp64 domain and other exotic fallbacks.

    ``.cpu()`` materializes the lazy conj/neg bits into the payload (verified
    value-exact), the CPU kernel produces the reference pair, and the result
    moves back to the original device. CPU dispatch never re-enters the
    FlagGems gcu registration, so there is no use_gems loop.
    """
    host = t.cpu()
    out = torch.ops.aten.view_as_real_copy(host)
    return out.to(t.device)


def view_as_real_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a complex tensor into a real tensor whose last dimension has size 2.

    Mirrors ``aten::view_as_real_copy``: the out-of-place counterpart of
    ``torch.view_as_real``. ``complex64`` maps to ``float32``, ``complex128``
    to ``float64`` and ``complex32`` to ``float16``. The result is a
    contiguous, freshly allocated tensor that never aliases the input; the
    batch dimensions may be arbitrarily strided and no particular layout is
    required. Every non-complex dtype is rejected with the same error PyTorch
    raises.

    Dispatch (single kernel launch on every device path):

    * contiguous input -> flat streaming kernel (family-tuned block ladder,
      ``EVEN`` specialization for divisible streams);
    * last complex dim contiguous, leading dims strided -> row kernel that
      turns each leading-index slice into one contiguous run copy (this is
      the common sliced/transposed-lead layout, e.g. ``pool[::2]``);
    * anything else -> int64 gather kernel (correct for arbitrary strides);
    * complex128 -> the payload is ComplexFloat-replaced on this device, so
      a sentinel-guarded fp32-word alias pair feeds the same fp32 kernels
      (zero casts, zero extra launches); non-compact layouts keep the
      legacy narrow-widen chain.

    Lazy conjugate/negative bits are resolved in-register by the ``FLIP``
    lanes on every fast path (see ``_raw_real_view``).
    """
    logger.debug("GEMS_ENFLAME VIEW_AS_REAL_COPY")

    real_dtype = _REAL_DTYPE.get(self.dtype)
    if real_dtype is None:
        raise RuntimeError(
            f"view_as_real is only supported for complex tensors, not for "
            f"{str(self.dtype).replace('torch.', '')} tensors"
        )

    out = torch.empty(tuple(self.shape) + (2,), dtype=real_dtype, device=self.device)
    n_complex = self.numel()
    if n_complex == 0:
        return out

    # The lazy conj/neg bits ride along as FLIP lanes (see _raw_real_view).
    flip = (1 if self.is_conj() else 0) | (2 if self.is_neg() else 0)

    if real_dtype == torch.float64:
        # complex128 fast path: the physical payload is one compact fp32 word
        # per f64 slot (ComplexFloat replace), so an fp32 word view of the
        # source plus an fp32 word ghost of the float64 output reproduce the
        # exact bit pattern through the plain fp32 copy kernels -- no f64
        # arithmetic (P02/P04), no cast kernels, conj/neg included.  The
        # sentinel probe guards the layout assumption once per device.
        if _c128_payload_is_compact(self.device):
            src = _raw_real_view(self, dtype=torch.float32)
            with torch_device_fn.device(self.device):
                _launch_real_copy(src, _fp32_word_ghost(out), flip, n_complex)
            return out
        # Legacy chain for exotic layouts: narrow the payload losslessly to
        # complex64 on device, reuse the complex64 path, widen the finished
        # pair back to float64 on device. Lazy conj/neg bits -- and any cast
        # that fails -- take the aten CPU round trip instead.
        if flip:
            return _host_view_as_real_copy(self)
        try:
            narrow = self.to(torch.complex64)
        except Exception:
            return _host_view_as_real_copy(self)
        return view_as_real_copy(narrow).to(torch.float64)

    src_real = _raw_real_view(self)
    with torch_device_fn.device(self.device):
        _launch_real_copy(src_real, out, flip, n_complex)
    return out
