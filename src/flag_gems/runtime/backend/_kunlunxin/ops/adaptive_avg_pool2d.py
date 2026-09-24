import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# tle.gpu (cluster path) coalesced-DMA reduction for the [1,1] global-pool case.
# The per-plane plane_kernel below reads GM with a per-lane VEC loop (strided,
# ~10-20 GB/s) and is grid=#planes. For many-plane shapes tle.gpu.copy does a
# coalesced TensorDescriptor block DMA into an LM tile and reduces core-locally
# (XBLOCK a multiple of core_num => no cross-core tt.reduce), which is far faster
# (e.g. (16,128,28,28)->[1,1]: 0.039x -> ~0.96x). It only wins when there are
# enough planes to fill the grid; for planes ~64 the plane_kernel single-wave
# stream is better, so tle is gated on plane count. Import is optional: any
# import/compile/launch failure falls back to plane_kernel.
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except Exception:  # pragma: no cover - environment without tle
    _HAS_TLE = False

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}
# Only engage tle for many-plane shapes (grid must be filled to beat plane_kernel).
TLE_MIN_PLANES = 256

INT_KERNEL_MAX_UNROLL = 65536
# Exact-division (avg_pool) fast path. A flat kernel that assigns each output
# element to one lane and reduces its KH*KW window with a fully-unrolled scalar
# loop is far faster on XPU than the per-(nc,oh)-row int kernel, because it
# collapses the launch grid from N*C*OH tiny programs to cdiv(numel, BLOCK)
# fat programs. The static unroll only stays cheap for small windows, so it is
# gated on KH*KW; larger windows fall through to the general kernel.
FLAT_MAX_WINDOW = 64
FLAT_BLOCK = 1024
# bf16 has no native XPU3 ALU and its strided/gather float load mis-maps lanes,
# so the flat kernels load the raw int16 bits and reconstruct fp32. That extra
# per-element arithmetic makes the static unroll heavier, so bf16 gets lower
# window caps and cannot use the runtime-KH flat_r path (it spills to uni_sram).
FLAT_MAX_WINDOW_BF16 = 49
FLAT_GEN_MAX_WINDOW_BF16 = 36
# Non-exact (true adaptive) fast path. Same flat-grid idea, but each lane computes
# its own adaptive window bounds. Fully static unroll over MAX_KH*MAX_KW keeps the
# accumulator in registers (a runtime loop spills to uni_sram and fails to compile),
# so it is gated on MAX_KH*MAX_KW. Larger windows fall through to the general kernel.
FLAT_GEN_MAX_WINDOW = 64
# Large exact windows are too big to fully unroll; a flat kernel with a runtime KH
# loop (static KW) still collapses the launch grid and beats the per-row int kernel.
# For very large windows with few outputs the runtime-KH loop serializes badly, so
# it is capped; above the cap the per-row int kernel is the least-bad fallback.
FLAT_R_BLOCK = 2048
FLAT_R_MAX_WINDOW = 256

# Large exact windows via the tle.gpu FOLD mechanism. The flat_r per-lane strided
# gather runs at ~23 GB/s; a two-pass coalesced-DMA fold is ~7x faster and, unlike
# flat_r, works for bf16 (block DMA avoids the strided bf16 de-interleave bug).
# Pass 1 folds the KH (outer, strided-by-IW) axis by keeping a per-column [XBLOCK,
# KBLOCK] fp32 accumulator and doing pure elementwise adds across a static_range(KH)
# loop (no cross-core reduce) -> partial[B, IW]. Pass 2 reduces the contiguous KW
# axis with a plain triton kernel (torch .sum would be re-dispatched through
# use_gems into a slow gems reduction). XBLOCK is pinned to the core count
# (one output row per core); IW/window/KH must be large enough to be worth the
# two-pass launch, else the small-window flat kernel is faster.
FOLD_CORE_NUM = 64
FOLD_MIN_WINDOW = 65
# IW>=56 (KBLOCK>=32) is required: IW=28 (KBLOCK=16) intermittently wedges the
# NOC and crashes the process/container on this backend, so those shapes stay on
# the flat kernel. IW 56/112/224 fold cleanly for all dtypes.
FOLD_MIN_IW = 56
FOLD_MAX_KH = 64
FOLD_KBLOCK_CAP = 256
# Multi-plane exact shapes reduce a strided KH window; the per-lane flat kernel
# gathers at ~20 GB/s and scores ~0.03-0.09x on these, whereas the coalesced-DMA
# fold measures 0.17-0.58x (fp16/fp32) and 3.6-9.3x (bf16). So for many-plane
# shapes the fold is tried before the small-window flat kernel. Few-plane shapes
# (grid too small to fill) keep the flat kernel.
FOLD_MIN_PLANES = 256


@triton.jit
def _load_f32(ptr, mask, BF16: tl.constexpr):
    # bf16 strided/gather loads on XPU3 go through a vmerge float de-interleave
    # that mis-maps lanes; loading the raw int16 bits and reconstructing the
    # fp32 value via (bits << 16) avoids the buggy path and is bit-exact.
    if BF16:
        raw = tl.load(ptr, mask=mask, other=0)
        return ((raw.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)
    return tl.load(ptr, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _store_f32(ptr, val, mask, BF16: tl.constexpr):
    # Mixing an int16-view load with a bf16-typed store in one kernel trips the
    # XPU3 vectorizer (misaligned address), so the bf16 result is rounded to
    # bf16 bits (round-to-nearest-even) and stored through an int16 view too.
    if BF16:
        rb = val.to(tl.int32, bitcast=True)
        rounded = (rb + 0x7FFF + ((rb >> 16) & 1)) >> 16
        tl.store(ptr, rounded.to(tl.int16), mask=mask)
    else:
        tl.store(ptr, val, mask=mask)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_flat_kernel(
    input,
    output,
    total,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    AREA: tl.constexpr,
    IHIW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    pid = ext.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    ow = q % OW
    t = q // OW
    oh = t % OH
    nc = t // OH
    in_base = nc * IHIW + (oh * KH) * IW + ow * KW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            acc += _load_f32(input + in_base + kh * IW + kw, mask, BF16)
    _store_f32(output + q, acc / AREA, mask, BF16)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_flat_r_kernel(
    input,
    output,
    total,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    AREA: tl.constexpr,
    IHIW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    pid = ext.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    ow = q % OW
    t = q // OW
    oh = t % OH
    nc = t // OH
    in_base = nc * IHIW + (oh * KH) * IW + ow * KW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in range(0, KH):
        for kw in tl.static_range(KW):
            acc += _load_f32(input + in_base + kh * IW + kw, mask, BF16)
    _store_f32(output + q, acc / AREA, mask, BF16)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_flat_general_kernel(
    input,
    output,
    total,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAX_KH: tl.constexpr,
    MAX_KW: tl.constexpr,
    IHIW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    pid = ext.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    ow = q % OW
    t = q // OW
    oh = t % OH
    nc = t // OH
    ih_start = (oh * IH) // OH
    ih_end = ((oh + 1) * IH + OH - 1) // OH
    iw_start = (ow * IW) // OW
    iw_end = ((ow + 1) * IW + OW - 1) // OW
    plane = nc * IHIW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in tl.static_range(MAX_KH):
        ih = ih_start + kh
        ih_ok = ih < ih_end
        safe_ih = tl.minimum(ih, ih_end - 1)
        for kw in tl.static_range(MAX_KW):
            iw = iw_start + kw
            active = mask & ih_ok & (iw < iw_end)
            safe_iw = tl.minimum(iw, iw_end - 1)
            off = plane + safe_ih * IW + safe_iw
            v = _load_f32(input + off, mask, BF16)
            acc += tl.where(active, v, 0.0)
    area = (ih_end - ih_start) * (iw_end - iw_start)
    _store_f32(output + q, acc / area, mask, BF16)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_plane_kernel(input, output, HW, VEC: tl.constexpr):
    program_id = ext.program_id(0)
    base = input + program_id * HW
    acc = tl.zeros((VEC,), dtype=tl.float32)
    for off in range(0, HW, VEC):
        idx = off + tl.arange(0, VEC)
        acc += tl.load(base + idx, mask=idx < HW, other=0.0).to(tl.float32)
    tl.store(output + program_id, tl.sum(acc) / HW)


@triton.jit
def _adaptive_avg_pool2d_tle_plane_kernel(
    a_desc,
    c_desc,
    HW,
    INV_AREA,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # [1,1] pool == mean over each contiguous plane == a row-reduce of
    # [planes, HW] along axis=1. Each program owns XBLOCK rows (XBLOCK a multiple
    # of core_num => rows are core-local, so tl.sum(axis=1) needs no cross-core
    # reduce), streams the HW axis in YBLOCK-wide coalesced block DMAs into LM,
    # masks the out-of-bounds tail columns to 0, and reduces once.
    pid = tl.program_id(0)
    row_off = pid * XBLOCK
    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))
    acc = tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    for coff in tl.range(0, HW, YBLOCK):
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        v = tl.load(a_ptrs).to(tl.float32)
        acc += tl.where(coff + col_ids < HW, v, 0.0)
    tl.store(c_ptrs, (tl.sum(acc, axis=1) * INV_AREA).to(DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_pool11_config(planes, HW, dtype):
    # (XBLOCK, YBLOCK), both powers of 2. XBLOCK is a multiple of core_num(=64)
    # so the axis=1 reduce is core-local. The per-core LM stack bounds
    # XBLOCK*YBLOCK (~32768 elems @4B / ~65536 @2B; wider => compile fail
    # "TLE kernel stack over local-memory budget").
    #
    # planes>=1024: grid already filled at XB=256; these shapes have short HW,
    #   so the narrow YB=128 stream tile is fine.
    # 256<=planes<1024: grid-starved (only 2-4 programs at XB=256), AND these
    #   shapes have long HW (>=3136) => the YB=128 stream needs 25-98 serial DMA
    #   iterations. Widen YBLOCK (fewer iterations) with a smaller XBLOCK (more
    #   programs) to raise both occupancy and bandwidth. Measured fp32
    #   (8,64,56,56)->[1,1]: XB=256/YB=128 0.657 -> XB=128/YB=256 0.852.
    #   bf16 is kept at YB=128: a wider YB silently miscomputes its axis=1 reduce.
    if planes < TLE_MIN_PLANES:
        return None
    if planes >= 1024:
        return 256, 128
    if dtype == torch.bfloat16:
        return 128, 128
    if HW <= 4096:
        return 128, 256
    # very long HW: shrink XBLOCK further to afford a wider YBLOCK within the LM
    # budget (fp16 2B can hold YB=1024; fp32 4B caps at YB=512).
    if dtype == torch.float16:
        return 64, 1024
    return 64, 512


def _adaptive_avg_pool2d_tle_plane(input_contiguous, output, planes, HW):
    # Returns True if the tle.gpu coalesced-DMA reduction handled the [1,1] case,
    # False (falls back to plane_kernel) if tle is unavailable or the launch fails.
    if not _HAS_TLE:
        return False
    dtype = input_contiguous.dtype
    tl_dtype = _TLE_TL_DTYPE.get(dtype)
    if tl_dtype is None:
        return False
    cfg = _tle_pool11_config(planes, HW, dtype)
    if cfg is None:
        return False
    xblock, yblock = cfg
    try:
        a2d = input_contiguous.view(planes, HW)
        out1d = output.view(planes)
        a_desc = TensorDescriptor.from_tensor(a2d, block_shape=[xblock, yblock])
        c_desc = TensorDescriptor.from_tensor(out1d, block_shape=[xblock])
        grid = (triton.cdiv(planes, xblock),)
        _adaptive_avg_pool2d_tle_plane_kernel[grid](
            a_desc,
            c_desc,
            HW,
            1.0 / HW,
            XBLOCK=xblock,
            YBLOCK=yblock,
            DTYPE=tl_dtype,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("tle.gpu [1,1] path failed, falling back: %s", e)
        return False


@triton.jit
def _adaptive_avg_pool2d_fold_kh_kernel(
    a_desc,
    c_desc,
    KH: tl.constexpr,
    IW: tl.constexpr,
    KBLOCK: tl.constexpr,
    KBLOCKS: tl.constexpr,
    XBLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Fold the KH (outer, strided-by-IW) reduction axis. Each program owns XBLOCK
    # contiguous output rows and one pow2 KBLOCK column tile of IW. It keeps an
    # fp32 [XBLOCK, KBLOCK] accumulator and adds the KH strided slices with pure
    # elementwise adds across a static_range loop -> no cross-core reduce, no
    # tt.reduce-in-loop (both previously walled). Coalesced block DMA also avoids
    # the strided bf16 de-interleave bug, so bf16 needs no int16 reconstruction.
    pid = tl.program_id(0)
    kb = pid % KBLOCKS
    rblk = pid // KBLOCKS
    row_off = rblk * XBLOCK
    # Shift the last column tile back so it stays in-bounds; re-reading columns is
    # harmless because each program sums the whole KH axis for its own columns.
    k_off = tl.minimum(kb * KBLOCK, IW - KBLOCK)
    a_lmem = tle.gpu.alloc(
        [XBLOCK, KBLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc(
        [XBLOCK, KBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    r_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, KBLOCK))
    c_ids = tl.broadcast_to(tl.arange(0, KBLOCK)[None, :], (XBLOCK, KBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (r_ids, c_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (r_ids, c_ids))
    acc = tl.zeros([XBLOCK, KBLOCK], tl.float32)
    for kh in tl.static_range(KH):
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, KBLOCK], [row_off, kh * IW + k_off])
        acc = acc + tl.load(a_ptrs).to(tl.float32)
    # bf16 stores the KH-folded partial in fp32 (rounding to bf16 here would fail
    # the accuracy tolerance before the KW reduce); fp16/fp32 keep their native
    # dtype so the partial DMA and pass-2 read stay narrow.
    tl.store(c_ptrs, acc.to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK, KBLOCK], [row_off, k_off])


@triton.jit
def _adaptive_avg_pool2d_fold_kw_kernel(
    partial,
    output,
    n_out,
    KW: tl.constexpr,
    INV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Pass 2: reduce the contiguous KW axis of the KH-folded partial[B, OW*KW].
    # One output element per lane; each lane sums KW *contiguous* partials (inner
    # dim, so this is a coalesced load, not a strided gather). Done as a plain
    # triton kernel so it is NOT re-dispatched through use_gems (a torch
    # partial.sum(KW) becomes a slow gems reduction under use_gems, which was
    # erasing the fold's pass-1 win for fp16/fp32).
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_out
    base = idx * KW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kw in tl.static_range(KW):
        acc += tl.load(partial + base + kw, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + idx, (acc * INV).to(output.dtype.element_ty), mask=mask)


def _lpow2_le(v):
    p = 1
    while p * 2 <= v:
        p *= 2
    return p


def _adaptive_avg_pool2d_fold(input_contiguous, output, OH, OW, KH, KW, IW):
    # tle.gpu FOLD path for large exact windows: pass 1 folds KH (tle), pass 2
    # reduces KW (plain triton). Returns True on success, False to fall back.
    # Correct for all dtypes.
    if not _HAS_TLE:
        return False
    dtype = input_contiguous.dtype
    tl_dtype = _TLE_TL_DTYPE.get(dtype)
    if tl_dtype is None:
        return False
    B = output.numel() // OW
    if B < FOLD_CORE_NUM or IW < FOLD_MIN_IW or KH > FOLD_MAX_KH:
        return False
    kblock = min(_lpow2_le(IW), FOLD_KBLOCK_CAP)
    kblocks = triton.cdiv(IW, kblock)
    xblock = FOLD_CORE_NUM
    # bf16 needs an fp32 partial (accuracy); fp16/fp32 keep their native dtype so
    # the partial stays narrow.
    is_bf16 = dtype == torch.bfloat16
    partial_dtype = torch.float32 if is_bf16 else dtype
    out_tl_dtype = tl.float32 if is_bf16 else tl_dtype
    try:
        inp2d = input_contiguous.view(B, KH * IW)
        partial = torch.empty(
            B, IW, dtype=partial_dtype, device=input_contiguous.device
        )
        a_desc = TensorDescriptor.from_tensor(inp2d, block_shape=[xblock, kblock])
        c_desc = TensorDescriptor.from_tensor(partial, block_shape=[xblock, kblock])
        grid = (triton.cdiv(B, xblock) * kblocks,)
        _adaptive_avg_pool2d_fold_kh_kernel[grid](
            a_desc,
            c_desc,
            KH,
            IW,
            kblock,
            kblocks,
            XBLOCK=xblock,
            DTYPE=tl_dtype,
            OUT_DTYPE=out_tl_dtype,
        )
        # Pass 2: reduce the contiguous KW axis with a plain triton kernel. Doing
        # this in torch (partial.sum(KW)) would be re-dispatched through use_gems
        # into a slow gems reduction, which erased the pass-1 win for fp16/fp32.
        n_out = B * OW
        fold_block = 1024
        _adaptive_avg_pool2d_fold_kw_kernel[(triton.cdiv(n_out, fold_block),)](
            partial.view(-1),
            output.view(-1),
            n_out,
            KW=KW,
            INV=1.0 / (KH * KW),
            BLOCK=fold_block,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("tle.gpu fold path failed, falling back: %s", e)
        return False


@libentry()
@triton.jit
def _adaptive_avg_pool2d_general_kernel(
    input,
    output,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAX_KH,
    MAX_KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = ext.program_id(0)
    ow = tl.arange(0, BLOCK_SIZE)
    valid = ow < OW
    oh = program_id % OH
    nc = program_id // OH

    ih_start = (oh * IH) // OH
    ih_end = ((oh + 1) * IH + OH - 1) // OH
    iw_start = (ow * IW) // OW
    iw_end = ((ow + 1) * IW + OW - 1) // OW

    value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kh in range(0, MAX_KH):
        ih = ih_start + kh
        for kw in tl.static_range(MAX_KW):
            iw = iw_start + kw
            active = valid & (ih < ih_end) & (iw < iw_end)
            safe_ih = tl.minimum(ih, ih_end - 1)
            safe_iw = tl.minimum(iw, tl.minimum(iw_end - 1, IW - 1))
            input_offset = (nc * IH + safe_ih) * IW + safe_iw
            loaded = tl.load(input + input_offset).to(tl.float32)
            value += tl.where(active, loaded, 0.0)

    area = (ih_end - ih_start) * (iw_end - iw_start)
    output_offset = (nc * OH + oh) * OW + ow
    tl.store(output + output_offset, value / area, mask=valid)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_int_kernel(
    input,
    output,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BKW: tl.constexpr,
    AREA: tl.constexpr,
):
    program_id = ext.program_id(0)
    oh = program_id % OH
    nc = program_id // OH
    r = tl.arange(0, BG)
    c = tl.arange(0, BKW)
    base = ((nc * OH + oh) * KH) * IW

    value = tl.zeros((BG,), dtype=tl.float32)
    for kh in tl.static_range(KH):
        tile = tl.load(
            input + base + kh * IW + r[:, None] * KW + c[None, :],
            mask=(r[:, None] < G) & (c[None, :] < KW),
            other=0.0,
        ).to(tl.float32)
        value += tl.sum(tile, axis=1)

    ow = tl.arange(0, BG)
    tl.store(output + (nc * OH + oh) * OW + ow, value / AREA, mask=ow < OW)


def adaptive_avg_pool2d(input, output_size):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL2D")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    output_height, output_width = output_size
    input_contiguous = input.contiguous()
    input_height, input_width = input_contiguous.shape[-2:]
    output_shape = (*input_contiguous.shape[:-2], output_height, output_width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if output.numel() == 0:
        return output

    output_rows = output.numel() // output_width
    # bf16 strided/gather loads mis-map lanes on XPU3; the flat kernels load the
    # raw int16 bits instead and reconstruct fp32 (see _load_f32).
    is_bf16 = input_contiguous.dtype == torch.bfloat16
    flat_input = input_contiguous.view(torch.int16) if is_bf16 else input_contiguous
    flat_output = output.view(torch.int16) if is_bf16 else output
    with torch_device_fn.device(input.device):
        if output_height == 1 and output_width == 1 and input_contiguous.size(-1) > 0:
            planes = output.numel()
            hw = input_height * input_width
            # Many-plane shapes: coalesced tle.gpu block DMA + core-local reduce
            # (far faster than the per-lane strided plane_kernel). Falls back to
            # plane_kernel for few-plane shapes or if the tle launch fails.
            if _adaptive_avg_pool2d_tle_plane(input_contiguous, output, planes, hw):
                return output
            _adaptive_avg_pool2d_plane_kernel[(planes,)](
                input_contiguous,
                output,
                hw,
                VEC=min(triton.next_power_of_2(hw), 8192),
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
            return output
        if (
            output_height > 0
            and output_width > 0
            and input_height % output_height == 0
            and input_width % output_width == 0
        ):
            kernel_height = input_height // output_height
            kernel_width = input_width // output_width
            window = kernel_height * kernel_width
            planes = output_rows // output_height
            # Many-plane shapes: the per-lane flat gather is bandwidth-starved
            # (~0.03-0.09x). Try the coalesced-DMA fold first; it wins 3-9x here
            # and covers the small windows the flat kernel would otherwise take.
            # Falls back to flat/flat_r/int if tle is unavailable or IW is too
            # narrow (the helper's IW>=FOLD_MIN_IW guard skips the NOC-unstable
            # IW=28 shapes without launching a kernel).
            if planes >= FOLD_MIN_PLANES and _adaptive_avg_pool2d_fold(
                input_contiguous,
                output,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                input_width,
            ):
                return output
            flat_cap = FLAT_MAX_WINDOW_BF16 if is_bf16 else FLAT_MAX_WINDOW
            if window <= flat_cap:
                _adaptive_avg_pool2d_flat_kernel[
                    (triton.cdiv(output.numel(), FLAT_BLOCK),)
                ](
                    flat_input,
                    flat_output,
                    output.numel(),
                    IW=input_width,
                    OH=output_height,
                    OW=output_width,
                    KH=kernel_height,
                    KW=kernel_width,
                    AREA=kernel_height * kernel_width,
                    IHIW=input_height * input_width,
                    BLOCK=FLAT_BLOCK,
                    BF16=is_bf16,
                )
                return output
            # Large exact window: the tle.gpu fold (coalesced block DMA, KH folded
            # with elementwise adds, KW reduced with a plain triton kernel) is ~7x
            # faster than the flat_r strided gather and works for all dtypes incl
            # bf16. Falls back to flat_r / int_kernel if tle is unavailable or fails.
            if window >= FOLD_MIN_WINDOW and _adaptive_avg_pool2d_fold(
                input_contiguous,
                output,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                input_width,
            ):
                return output
            # Large exact window: too big to fully unroll. A flat grid with a
            # runtime KH loop (static KW) still collapses the launch grid and
            # beats the per-row int kernel, but only up to a window cap. bf16
            # cannot use it (runtime-KH int16 body spills to uni_sram).
            if not is_bf16 and window <= FLAT_R_MAX_WINDOW:
                _adaptive_avg_pool2d_flat_r_kernel[
                    (triton.cdiv(output.numel(), FLAT_R_BLOCK),)
                ](
                    flat_input,
                    flat_output,
                    output.numel(),
                    IW=input_width,
                    OH=output_height,
                    OW=output_width,
                    KH=kernel_height,
                    KW=kernel_width,
                    AREA=kernel_height * kernel_width,
                    IHIW=input_height * input_width,
                    BLOCK=FLAT_R_BLOCK,
                    BF16=is_bf16,
                )
                return output
            if kernel_height * kernel_width <= INT_KERNEL_MAX_UNROLL:
                groups = input_width // kernel_width
                _adaptive_avg_pool2d_int_kernel[(output_rows,)](
                    input_contiguous,
                    output,
                    IW=input_width,
                    OH=output_height,
                    OW=output_width,
                    KH=kernel_height,
                    KW=kernel_width,
                    G=groups,
                    BG=triton.next_power_of_2(groups),
                    BKW=triton.next_power_of_2(kernel_width),
                    AREA=kernel_height * kernel_width,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
                return output

        max_kernel_height = triton.cdiv(input_height, output_height) + 1
        max_kernel_width = triton.cdiv(input_width, output_width) + 1
        gen_cap = FLAT_GEN_MAX_WINDOW_BF16 if is_bf16 else FLAT_GEN_MAX_WINDOW
        if max_kernel_height * max_kernel_width <= gen_cap:
            _adaptive_avg_pool2d_flat_general_kernel[
                (triton.cdiv(output.numel(), FLAT_BLOCK),)
            ](
                flat_input,
                flat_output,
                output.numel(),
                IH=input_height,
                IW=input_width,
                OH=output_height,
                OW=output_width,
                MAX_KH=max_kernel_height,
                MAX_KW=max_kernel_width,
                IHIW=input_height * input_width,
                BLOCK=FLAT_BLOCK,
                BF16=is_bf16,
            )
            return output
        block_size = triton.next_power_of_2(output_width)
        _adaptive_avg_pool2d_general_kernel[(output_rows,)](
            input_contiguous,
            output,
            IH=input_height,
            IW=input_width,
            OH=output_height,
            OW=output_width,
            MAX_KH=max_kernel_height,
            MAX_KW=max_kernel_width,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return output
