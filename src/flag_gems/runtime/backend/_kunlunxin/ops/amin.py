import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

from ..utils.block_size_utils import get_block_size_1d
from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# tle.gpu (cluster path) coalesced-DMA row-reduce for the small-N row-reduce
# case.  For narrow reductions (small N) with many rows, the per-lane strided
# amin_rows_kernel is launch/overhead bound (torch does the whole reduce in
# ~6-9us); a tle.gpu kernel that packs XBLOCK=256 rows/program (4 rows/core)
# and streams the N axis in coalesced block DMAs into LM has far fewer programs
# and higher bandwidth.  Optional import: any failure leaves _HAS_TLE False and
# the code falls back to the existing kernels.
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

# Engage the tle small-N row path only for narrow reductions with many rows.
_TLE_ROW_MAX_N = 256
_TLE_ROW_MIN_M = 256


# ---------------------------------------------------------------------------
# tle.raw hand-written XTDK (xpu3) payloads.
#
# The amin benchmark is dominated by memory-bound reductions where the Triton
# path leaves large gaps:
#   * 3D middle-dim reduce (out[o,w]=min_m inp[o,m,w]) -- the previous biggest
#     laggard.  The Triton path served this via permute(0,2,1) -> non-contiguous
#     -> a slow transpose copy -> row-reduce.  The raw kernel reads the native
#     [outer][mid][inner] layout and vvmin-accumulates a width-`inner` column-min
#     with NO transpose (1-pass, or a 2-pass mid-split for high-mid shapes).
#   * 2D last-dim row-min for moderate N (SIMD vvmin + horizontal fold).
#   * 1D full-reduce for very large numel (two-pass streaming SIMD).
# All payloads compute the exact float min (no accumulation error) and were
# validated bit-exact (torch.equal) against torch.amin on P800.  Any import or
# launch failure leaves _HAS_RAW False and the existing Triton kernels run.
# ---------------------------------------------------------------------------
_HAS_RAW = False
if _HAS_TLE:
    try:
        import os as _os

        import triton.experimental.tle as _tle_ext

        # Precompiled, secrecy-hardened device objects (packed from the .xpu
        # sources via docs/xpu3/how_to_pack_payload/pack_payload.py --obj-dir).
        # We ship the .o (not the .xpu source): stub name must equal the entry
        # symbol and the signature must match the C++ ABI.  In object= mode the
        # merge-time signature check is skipped (payload isn't in IR), so a wrong
        # signature fails at runtime, not compile -- keep the stub in lockstep
        # with the C++.  Re-pack after editing any .xpu -- the compile-cache key
        # is the .o content digest, so a stale .o silently keeps the old code.
        _RAW_DIR = _os.path.dirname(_os.path.abspath(__file__))
        _PAY_OBJ = _os.path.join(_os.path.dirname(_RAW_DIR), "payload", "obj")
        # All amin device kernels are shipped in a single merged object
        # (amin.o = relocatable link of the per-kernel .o).  In object= mode the
        # loader resolves each stub by its entry-symbol name, so one .o holding
        # every symbol works exactly like the old per-kernel files.  Re-pack with
        #   ld.lld -r amin_*.o -o amin.o
        # after editing/repacking any individual kernel .o.
        _AMIN_OBJ = _os.path.join(_PAY_OBJ, "amin.o")
        _FLAT_XPU = _AMIN_OBJ
        _FLAT1C_XPU = _AMIN_OBJ
        _ROW_XPU = {
            torch.float16: _AMIN_OBJ,
            torch.float32: _AMIN_OBJ,
            torch.bfloat16: _AMIN_OBJ,
        }
        _MID_XPU = {
            torch.float16: _AMIN_OBJ,
            torch.float32: _AMIN_OBJ,
            torch.bfloat16: _AMIN_OBJ,
        }
        _MIDPART_XPU = {
            torch.float16: _AMIN_OBJ,
            torch.float32: _AMIN_OBJ,
            torch.bfloat16: _AMIN_OBJ,
        }
        # -- 2D last-dim FUSED row-min + in-place broadcast writeback --
        _ROWFILL_XPU = {
            torch.float16: _AMIN_OBJ,
            torch.float32: _AMIN_OBJ,
            torch.bfloat16: _AMIN_OBJ,
        }
        # -- 3D middle-dim FUSED column-min + in-place broadcast writeback --
        _MIDFILL_XPU = {
            torch.float16: _AMIN_OBJ,
            torch.float32: _AMIN_OBJ,
            torch.bfloat16: _AMIN_OBJ,
        }

        # -- 1D flat: part (each core min-reduces a slice) + final (core0 folds) --
        @_tle_ext.raw.dialect("xpu3", object=_FLAT_XPU, arch=3)
        def amin_flat_part_f16(inp, part, total, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT_XPU, arch=3)
        def amin_flat_final_f16(part, out, np): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT_XPU, arch=3)
        def amin_flat_part_f32(inp, part, total, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT_XPU, arch=3)
        def amin_flat_final_f32(part, out, np): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT_XPU, arch=3)
        def amin_flat_part_bf16(inp, part, total, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT_XPU, arch=3)
        def amin_flat_final_bf16(part, out, np): ...

        # -- 1D flat single-cluster single-launch (64 cores -> shared partials
        #    -> core0 folds -> out).  No 2nd launch; wins for latency/launch-
        #    bound totals (~<=2M elems); loses to 2-pass for bandwidth-bound. --
        @_tle_ext.raw.dialect("xpu3", object=_FLAT1C_XPU, arch=3)
        def amin_flat1c_f16(inp, out, total): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT1C_XPU, arch=3)
        def amin_flat1c_f32(inp, out, total): ...
        @_tle_ext.raw.dialect("xpu3", object=_FLAT1C_XPU, arch=3)
        def amin_flat1c_bf16(inp, out, total): ...

        # -- 2D last-dim row-min --
        @_tle_ext.raw.dialect("xpu3", object=_ROW_XPU[torch.float16], arch=3)
        def amin_rows_f16_t(inp, out, rows, N, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_ROW_XPU[torch.float32], arch=3)
        def amin_rows_f32_t(inp, out, rows, N, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_ROW_XPU[torch.bfloat16], arch=3)
        def amin_rows_bf16_t(inp, out, rows, N, ncores): ...

        # -- 3D middle-dim column-min (1-pass) --
        @_tle_ext.raw.dialect("xpu3", object=_MID_XPU[torch.float16], arch=3)
        def amin_mid_f16(inp, out, outer, mid, inner, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_MID_XPU[torch.float32], arch=3)
        def amin_mid_f32(inp, out, outer, mid, inner, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_MID_XPU[torch.bfloat16], arch=3)
        def amin_mid_bf16(inp, out, outer, mid, inner, ncores): ...

        # -- 3D middle-dim column-min (2-pass part: split mid into nseg) --
        @_tle_ext.raw.dialect("xpu3", object=_MIDPART_XPU[torch.float16], arch=3)
        def amin_mid_part_f16(inp, part, outer, mid, inner, nseg, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_MIDPART_XPU[torch.float32], arch=3)
        def amin_mid_part_f32(inp, part, outer, mid, inner, nseg, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_MIDPART_XPU[torch.bfloat16], arch=3)
        def amin_mid_part_bf16(inp, part, outer, mid, inner, nseg, ncores): ...

        # -- 2D last-dim FUSED row-min + in-place broadcast fill (one launch,
        #    no separate copy pass; inp is both source and destination) --
        @_tle_ext.raw.dialect("xpu3", object=_ROWFILL_XPU[torch.float16], arch=3)
        def amin_rowfill_f16_t(inp, rows, N, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_ROWFILL_XPU[torch.float32], arch=3)
        def amin_rowfill_f32_t(inp, rows, N, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_ROWFILL_XPU[torch.bfloat16], arch=3)
        def amin_rowfill_bf16_t(inp, rows, N, ncores): ...

        # -- 3D middle-dim FUSED column-min + in-place broadcast fill (one launch,
        #    no separate copy pass; inp is both source and destination) --
        @_tle_ext.raw.dialect("xpu3", object=_MIDFILL_XPU[torch.float16], arch=3)
        def amin_midfill_f16(inp, outer, mid, inner, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_MIDFILL_XPU[torch.float32], arch=3)
        def amin_midfill_f32(inp, outer, mid, inner, ncores): ...
        @_tle_ext.raw.dialect("xpu3", object=_MIDFILL_XPU[torch.bfloat16], arch=3)
        def amin_midfill_bf16(inp, outer, mid, inner, ncores): ...

        @triton.jit(do_not_specialize=["total", "ncores"])
        def _raw_flat_part_f16(Inp, Part, total, ncores):
            tle.raw.call(amin_flat_part_f16, (Inp, Part, total, ncores))

        @triton.jit(do_not_specialize=["np"])
        def _raw_flat_final_f16(Part, Out, np):
            tle.raw.call(amin_flat_final_f16, (Part, Out, np))

        @triton.jit(do_not_specialize=["total", "ncores"])
        def _raw_flat_part_f32(Inp, Part, total, ncores):
            tle.raw.call(amin_flat_part_f32, (Inp, Part, total, ncores))

        @triton.jit(do_not_specialize=["np"])
        def _raw_flat_final_f32(Part, Out, np):
            tle.raw.call(amin_flat_final_f32, (Part, Out, np))

        @triton.jit(do_not_specialize=["total", "ncores"])
        def _raw_flat_part_bf16(Inp, Part, total, ncores):
            tle.raw.call(amin_flat_part_bf16, (Inp, Part, total, ncores))

        @triton.jit(do_not_specialize=["np"])
        def _raw_flat_final_bf16(Part, Out, np):
            tle.raw.call(amin_flat_final_bf16, (Part, Out, np))

        @triton.jit(do_not_specialize=["total"])
        def _raw_flat1c_f16(Inp, Out, total):
            tle.raw.call(amin_flat1c_f16, (Inp, Out, total))

        @triton.jit(do_not_specialize=["total"])
        def _raw_flat1c_f32(Inp, Out, total):
            tle.raw.call(amin_flat1c_f32, (Inp, Out, total))

        @triton.jit(do_not_specialize=["total"])
        def _raw_flat1c_bf16(Inp, Out, total):
            tle.raw.call(amin_flat1c_bf16, (Inp, Out, total))

        @triton.jit(do_not_specialize=["rows", "N", "ncores"])
        def _raw_rows_f16(Inp, Out, rows, N, ncores):
            tle.raw.call(amin_rows_f16_t, (Inp, Out, rows, N, ncores))

        @triton.jit(do_not_specialize=["rows", "N", "ncores"])
        def _raw_rows_f32(Inp, Out, rows, N, ncores):
            tle.raw.call(amin_rows_f32_t, (Inp, Out, rows, N, ncores))

        @triton.jit(do_not_specialize=["rows", "N", "ncores"])
        def _raw_rows_bf16(Inp, Out, rows, N, ncores):
            tle.raw.call(amin_rows_bf16_t, (Inp, Out, rows, N, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "ncores"])
        def _raw_mid_f16(Inp, Out, outer, mid, inner, ncores):
            tle.raw.call(amin_mid_f16, (Inp, Out, outer, mid, inner, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "ncores"])
        def _raw_mid_f32(Inp, Out, outer, mid, inner, ncores):
            tle.raw.call(amin_mid_f32, (Inp, Out, outer, mid, inner, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "ncores"])
        def _raw_mid_bf16(Inp, Out, outer, mid, inner, ncores):
            tle.raw.call(amin_mid_bf16, (Inp, Out, outer, mid, inner, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "nseg", "ncores"])
        def _raw_midpart_f16(Inp, Part, outer, mid, inner, nseg, ncores):
            tle.raw.call(
                amin_mid_part_f16, (Inp, Part, outer, mid, inner, nseg, ncores)
            )

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "nseg", "ncores"])
        def _raw_midpart_f32(Inp, Part, outer, mid, inner, nseg, ncores):
            tle.raw.call(
                amin_mid_part_f32, (Inp, Part, outer, mid, inner, nseg, ncores)
            )

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "nseg", "ncores"])
        def _raw_midpart_bf16(Inp, Part, outer, mid, inner, nseg, ncores):
            tle.raw.call(
                amin_mid_part_bf16, (Inp, Part, outer, mid, inner, nseg, ncores)
            )

        @triton.jit(do_not_specialize=["rows", "N", "ncores"])
        def _raw_rowfill_f16(Inp, rows, N, ncores):
            tle.raw.call(amin_rowfill_f16_t, (Inp, rows, N, ncores))

        @triton.jit(do_not_specialize=["rows", "N", "ncores"])
        def _raw_rowfill_f32(Inp, rows, N, ncores):
            tle.raw.call(amin_rowfill_f32_t, (Inp, rows, N, ncores))

        @triton.jit(do_not_specialize=["rows", "N", "ncores"])
        def _raw_rowfill_bf16(Inp, rows, N, ncores):
            tle.raw.call(amin_rowfill_bf16_t, (Inp, rows, N, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "ncores"])
        def _raw_midfill_f16(Inp, outer, mid, inner, ncores):
            tle.raw.call(amin_midfill_f16, (Inp, outer, mid, inner, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "ncores"])
        def _raw_midfill_f32(Inp, outer, mid, inner, ncores):
            tle.raw.call(amin_midfill_f32, (Inp, outer, mid, inner, ncores))

        @triton.jit(do_not_specialize=["outer", "mid", "inner", "ncores"])
        def _raw_midfill_bf16(Inp, outer, mid, inner, ncores):
            tle.raw.call(amin_midfill_bf16, (Inp, outer, mid, inner, ncores))

        _RAW_FLAT = {
            torch.float16: (_raw_flat_part_f16, _raw_flat_final_f16),
            torch.float32: (_raw_flat_part_f32, _raw_flat_final_f32),
            torch.bfloat16: (_raw_flat_part_bf16, _raw_flat_final_bf16),
        }
        _RAW_FLAT1C = {
            torch.float16: _raw_flat1c_f16,
            torch.float32: _raw_flat1c_f32,
            torch.bfloat16: _raw_flat1c_bf16,
        }
        _RAW_ROWS = {
            torch.float16: _raw_rows_f16,
            torch.float32: _raw_rows_f32,
            torch.bfloat16: _raw_rows_bf16,
        }
        _RAW_MID = {
            torch.float16: _raw_mid_f16,
            torch.float32: _raw_mid_f32,
            torch.bfloat16: _raw_mid_bf16,
        }
        _RAW_MIDPART = {
            torch.float16: _raw_midpart_f16,
            torch.float32: _raw_midpart_f32,
            torch.bfloat16: _raw_midpart_bf16,
        }
        _RAW_ROWFILL = {
            torch.float16: _raw_rowfill_f16,
            torch.float32: _raw_rowfill_f32,
            torch.bfloat16: _raw_rowfill_bf16,
        }
        _RAW_MIDFILL = {
            torch.float16: _raw_midfill_f16,
            torch.float32: _raw_midfill_f32,
            torch.bfloat16: _raw_midfill_bf16,
        }
        _HAS_RAW = True
    except Exception:  # pragma: no cover - environment without tle.raw
        _HAS_RAW = False


# Raw-path routing thresholds (P800, do_bench median vs torch.amin):
#   * 3D middle-dim: raw wins hugely for mid>1 (0.86-1.19 vs 0.11-0.25 baseline).
#   * 2D last-dim: raw wins for moderate N; loses for huge N (>=~1<<20 the native
#     torch reduce out-bandwidths the streaming kernel) -> cap N.
#   * 1D flat: raw wins for very large numel; near-parity at ~1M.
_RAW_ROW_MAX_N = 1 << 16  # 65536: raw row-min beats baseline up to here
_RAW_FLAT_MIN_NUMEL = 1 << 20  # engage raw 1D at/above ~1M elements
_RAW_FLAT_1CLU_MAX = 1 << 21  # <=2M: single-cluster single-launch wins (no 2nd
#   launch floor); above this the 2-pass multi-cluster is faster (bandwidth).
_RAW_MID_2PASS_MIN_MID = 128  # split mid into segments above this
_ROWFILL_MAX_N = 1 << 20  # fused in-place row-min+fill validated up to 1M cols
# Above these per-dtype numels the fused single-launch rowfill loses to the
# SPLIT path: our amin() reduce + native IMPLICIT broadcast copy_(result_keepdim).
# The benchmark's torch ref writes back with implicit broadcast copy_ (fill-like,
# ~free); matching it (instead of the fused self-written LM2GM fill or an
# expand_as strided gather) makes the writeback ~free so gems latency approaches
# reduce-only.  fp32's per-byte reduce is slower so it crosses over earlier.
_ROWFILL_SPLIT_NUMEL = {
    torch.float16: 1 << 24,
    torch.float32: 1 << 22,
    torch.bfloat16: 1 << 24,
}
# Fused in-place 3D middle-dim column-min+fill wins only for SMALL mid (outer=64
# caps the outer-split kernel at a single cluster; large mid then serializes the
# whole mid*inner block per core and loses to the amin()+copy split which spreads
# mid across all 12 clusters).  Per-dtype crossover from amin_midfill_probe3:
# fp16/fp32 win through mid=64 (lose at 128); bf16's widen path only wins at the
# degenerate mid<=1, and regresses hard by mid=16, so keep its threshold tiny.
_MIDFILL_MAX_MID = {torch.float16: 64, torch.float32: 64, torch.bfloat16: 8}


def _raw_grid_for_flat(numel):
    """Cluster count (g in [1,12]) for the 1D flat part kernel: small totals
    prefer few clusters (launch-bound), 1G-scale prefers all 12."""
    g = numel >> 20
    if g < 1:
        g = 1
    if g > 12:
        g = 12
    return g


def _amin_raw_mid(inp, out, outer, mid, inner):
    """3D middle-dim column-min via the raw kernel.  `out` is a contiguous
    [outer, inner] buffer.  Returns True on success (bit-exact), else False."""
    dt = inp.dtype
    if not _HAS_RAW or dt not in _RAW_MID:
        return False
    try:
        mid_fn = _RAW_MID[dt]
        if mid >= _RAW_MID_2PASS_MIN_MID:
            # 2-pass: split mid into nseg segments (full 12-cluster launch each).
            nseg = 12
            part_fn = _RAW_MIDPART[dt]
            part = torch.empty((outer, nseg, inner), dtype=dt, device=inp.device)
            g = 12
            part_fn[(g,)](inp, part, outer, mid, inner, nseg, g * 64)
            mid_fn[(g,)](part, out, outer, nseg, inner, g * 64)
        else:
            g = 12 if outer >= 12 * 64 else max(1, (outer + 63) // 64)
            mid_fn[(g,)](inp, out, outer, mid, inner, g * 64)
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("raw amin mid path failed, falling back: %s", e)
        return False


def _raw_grid_for_rows(M, N, elem_bytes):
    """Pick the cluster count for a 2D row-min.  Small totals are latency /
    launch bound: spinning up all 12 clusters (768 cores) for a few-K-element
    reduction pays 12x cluster launch+sync overhead for no bandwidth gain, so
    scale the grid down.  Large totals are bandwidth bound and want all 12.
    Keyed on BYTE volume (not element count) so fp32 -- which moves 2x the
    bytes of fp16/bf16 at the same shape -- correctly gets more clusters.
    Measured wins: (1024,16) fp16/bf16 g12->g2 0.40->0.66, fp32 g12->g4
    0.72->0.90; (1024,256) fp32 g12->g6 0.70->0.77; >=4MB unchanged at g12."""
    nbytes = M * N * elem_bytes
    if nbytes < (48 << 10):  # <48KB
        g = 2
    elif nbytes < (768 << 10):  # <768KB
        g = 4
    elif nbytes < (4 << 20):  # <4MB
        g = 6
    else:
        g = 12
    # The kernel stages per-core output rows in a 512-slot LM buffer; ensure
    # ceil(M / (g*64)) <= 512 by bumping the grid up when needed.
    g_min = (M + 512 * 64 - 1) // (512 * 64)
    if g < g_min:
        g = g_min
    if g > 12:
        g = 12
    return g


def _amin_raw_rows(src, out, M, N):
    """2D last-dim row-min via the raw kernel.  Returns True on success."""
    dt = src.dtype
    if not _HAS_RAW or dt not in _RAW_ROWS:
        return False
    if N <= 1 or N > _RAW_ROW_MAX_N:
        return False
    # The row kernel stages per-core output rows in a 512-slot LM buffer;
    # require ceil(M / ncores) <= 512 so no core overruns it.
    if (M + 12 * 64 - 1) // (12 * 64) > 512:
        return False
    try:
        fn = _RAW_ROWS[dt]
        g = _raw_grid_for_rows(M, N, src.element_size())
        nc = g * 64
        # Over-parallelization guard.  When the launched core count (g*64) is
        # >= the row count and each row is small (N<=512), the kernel degrades
        # to <=1 row per core: 64-128 cores each issue a tiny GM2LM plus a
        # 1-element scattered LM2GM, and those tiny transfers serialize on the
        # cluster DMA descriptor queue -- device time balloons.  Batching ~4
        # rows/core amortizes the per-core fixed cost.  Profiled (64,64):
        # device kernel 10.7us(nc128) -> 3.6us(nc16), Gems speedup 0.25 -> 0.75
        # across all 3 dtypes (torch reduce_mt is 2.7us).
        if nc >= M and N <= 512:
            nc = max(16, (M + 3) // 4)
            g = (nc + 63) // 64
        fn[(g,)](src, out, M, N, nc)
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("raw amin rows path failed, falling back: %s", e)
        return False


def _amin_inplace_rowfill(inp, M, N):
    """FUSED 2D last-dim in-place amin_: one raw kernel reads each row, reduces
    to its min, and broadcast-fills the min back over all N positions -- no
    separate reduction buffer and no second broadcast-copy pass.  Measured 1.3-
    2x faster than the (raw reduce + native copy) split for moderate/large N;
    for huge N (>=~1<<20) native torch reduce out-bandwidths the streaming fill,
    so the caller keeps that shape on the split path.  Returns True on success."""
    dt = inp.dtype
    if not _HAS_RAW or dt not in _RAW_ROWFILL:
        return False
    if N <= 1:
        return False
    try:
        fn = _RAW_ROWFILL[dt]
        g = _raw_grid_for_rows(M, N, inp.element_size())
        nc = g * 64
        # Same over-parallelization guard as _amin_raw_rows: when g*64 >= rows
        # and N is small, batch ~4 rows/core to avoid tiny-transfer DMA
        # descriptor serialization.
        if nc >= M and N <= 512:
            nc = max(16, (M + 3) // 4)
            g = (nc + 63) // 64
        fn[(g,)](inp, M, N, nc)
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("raw amin rowfill path failed, falling back: %s", e)
        return False


def _amin_inplace_midfill(inp, outer, mid, inner):
    """FUSED 3D middle-dim in-place amin_: one raw kernel column-min-reduces the
    (mid,inner) block for each outer slice and broadcast-fills the result back
    over all mid rows -- no separate reduction buffer, no second copy pass.  Only
    profitable for small mid (see _MIDFILL_MAX_MID); the caller keeps large mid on
    the amin()+copy split (which spreads mid across all clusters).  Returns True
    on success."""
    dt = inp.dtype
    if not _HAS_RAW or dt not in _RAW_MIDFILL:
        return False
    try:
        fn = _RAW_MIDFILL[dt]
        g = min(12, max(1, (outer + 63) // 64))
        nc = g * 64
        fn[(g,)](inp, outer, mid, inner, nc)
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("raw amin midfill path failed, falling back: %s", e)
        return False


def _flat_via_rows_R(numel, elem_size):
    """Pick a row count R to fold a 1D full-reduce into a 2D (R, numel/R)
    row-min (multi-cluster pass1) + a tiny flat pass2.  A single-cluster flat
    reduce only gets 1/12 of the chip's read bandwidth; reshaping to a wide 2D
    lets all 12 clusters stream in parallel.  We search by the SIMD-aligned
    inner width N (=numel/R, aligned 32 for 2-byte / 16 for 4-byte); N=256 is
    the profiled sweet spot (coalescing vs. row count) and, crucially, divides
    both 2^20 and non-power-of-two totals like 1049600=2^10*5^2*41 (N=256 ->
    R=4100) that a fixed R list misses.  Requires R>=768 (fill all 12 clusters)
    and ceil(R/768)<=512 (per-core row buffer).  Returns R or None.  Profiled
    1049600: fp32 0.419->0.721, fp16 0.576->0.654, bf16 0.533->0.658."""
    align = 32 if elem_size == 2 else 16
    for N in (256, 512, 128, 1024):
        if N % align or numel % N:
            continue
        R = numel // N
        if R < 768:
            continue
        if (R + 767) // 768 > 512:
            continue
        return R
    return None


def _amin_raw_flat(inp, out, numel):
    """1D full-reduce via the raw two-pass kernel.  Returns True on success."""
    dt = inp.dtype
    if not _HAS_RAW or dt not in _RAW_FLAT:
        return False
    if numel < _RAW_FLAT_MIN_NUMEL or numel >= (1 << 31):
        return False
    try:
        # For latency/launch-bound totals (<=2M) prefer a 2-pass via the tuned
        # multi-cluster row kernel when the total factors into a SIMD-aligned
        # 2D shape (all 12 clusters stream in parallel).  Otherwise fall back to
        # a single-cluster single-launch reduce: it folds all 64 cores' partials
        # in cluster-shared SM and writes the scalar directly, avoiding the
        # 2nd-launch floor.  Above 2M the flat two-pass wins (bandwidth).
        if numel <= _RAW_FLAT_1CLU_MAX and dt in _RAW_FLAT1C:
            if dt in _RAW_ROWS:
                R = _flat_via_rows_R(numel, inp.element_size())
                if R is not None:
                    N = numel // R
                    part = torch.empty((R,), dtype=dt, device=inp.device)
                    _RAW_ROWS[dt][(12,)](inp.view(R, N), part, R, N, 12 * 64)
                    _RAW_FLAT1C[dt][(1,)](part, out, R)
                    return True
            _RAW_FLAT1C[dt][(1,)](inp, out, numel)
            return True
        part_fn, final_fn = _RAW_FLAT[dt]
        g = _raw_grid_for_flat(numel)
        nc = g * 64
        part = torch.empty((nc,), dtype=dt, device=inp.device)
        part_fn[(g,)](inp, part, numel, nc)
        final_fn[(1,)](part, out, nc)
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("raw amin flat path failed, falling back: %s", e)
        return False


_FULL_REDUCTION_BLOCK_SIZE = 8192

_FLAT_CHUNK = 32768
_FLAT_ROW_WIDTH = 8192
_FLAT_CHUNK_MAX_NUMEL = 1 << 26
# Above this numel the scalar broadcast writeback is bandwidth-bound, so the
# fast fill_ path (~2TB/s) beats the stride-0 expand copy_ (~700GBPS).  Below
# it fill_'s fixed per-call overhead (~30us) dominates and regresses the small
# flat shapes (e.g. 1M dropped 0.64->0.24), so keep copy_ there.  Crossover
# measured ~16M on P800.
_FLAT_FILL_MIN_NUMEL = 1 << 25
_BLOCK_N_MAX = 8192


@libentry()
@triton.jit
def amin_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    if NEED_MASK:
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    else:
        inp_val = tl.load(inp_ptrs)
    amin_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amin_val)


@libentry()
@triton.jit
def amin_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    tl.store(out, amin_val)


@libentry()
@triton.jit
def amin_rows_kernel(inp, out, M, NW, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Row-reduce over cols [0, NW) with NW % BLOCK_N == 0.  Fully unmasked
    loads (rows clamped to [0, M-1]), reduce-OUTSIDE accumulation into a
    [BLOCK_M, BLOCK_N] tile with a single final `tl.min(axis=1)`, masked row
    stores.  Exact (same validated pattern as _sum_row_full_kernel)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_c = tl.where(rows < M, rows, M - 1)
    inp = inp + rows_c * NW
    out = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        acc = tl.minimum(acc, a)
    tl.store(out, tl.min(acc, axis=1)[:, None], row_mask)


@triton.jit
def amin_tle_row_kernel(
    a_desc, c_desc, N, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr, DTYPE: tl.constexpr
):
    """tle.gpu coalesced-DMA row-MIN of a [M, N] contiguous matrix along axis=1.
    Each program owns XBLOCK rows (XBLOCK a multiple of core_num=64 => rows are
    core-local, so tl.min(axis=1) needs no cross-core reduce), streams the N
    axis in YBLOCK-wide coalesced block DMAs into LM, masks OOB tail columns to
    +inf, and reduces once.  Mirrors the validated adaptive_avg_pool2d tle
    plane kernel (sum -> min)."""
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
    acc = tl.full([XBLOCK, YBLOCK], float("inf"), tl.float32)
    for coff in tl.range(0, N, YBLOCK):
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        v = tl.load(a_ptrs).to(tl.float32)
        v = tl.where(coff + col_ids < N, v, float("inf"))
        acc = tl.minimum(acc, v)
    tl.store(c_ptrs, tl.min(acc, axis=1).to(DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_row_config(M, N, dtype):
    """(XBLOCK, YBLOCK) for the tle small-N row path, or None if ineligible.
    XBLOCK is a multiple of core_num(=64) so the axis=1 reduce is core-local;
    YBLOCK is a power of 2 sized to the (small) reduction width.

    XBLOCK MUST be chosen per-dtype from the empirically compile-safe set.  The
    tle axis=1 reduce codegen fails ("size mismatch when packing elements for
    LLVM struct") for some (dtype, XBLOCK) combos, and Triton does NOT cache a
    failed compile -- so an XBLOCK that fails to compile would force a full
    recompile on every single call (catastrophic).  Config-lock sweep on P800
    (harness/perf_ir/amin_tle_lock_cfg.py):
      * fp16: XB256/XB128 fail to compile for narrow YB; only XB64 compiles and
        its speedup (0.21/0.36) does not beat the existing fp16 path -> disable.
      * fp32: XB128 compiles for both N=16 and N=256 (best, 0.63/0.53); XB256
        fails at YB256.
      * bf16: XB256 compiles for both (best, 0.56/0.41); YB kept <=128."""
    if not _HAS_TLE or dtype not in _TLE_TL_DTYPE:
        return None
    if N > _TLE_ROW_MAX_N or M < _TLE_ROW_MIN_M or M % 64 != 0:
        return None
    if dtype == torch.float16:
        # fp16 is compile-safe on the tle row path only when YBLOCK >= 64.
        # With YBLOCK == 16 (N <= 32) an XBLOCK >= 128 kernel fails to compile
        # ("v" PassManager) and XBLOCK == 64 loses to the fallback, so tiny-N
        # fp16 stays on the existing path.  For YBLOCK >= 64 (e.g. N == 256)
        # XBLOCK == 256 compiles and matches bf16: measured (1024,256) fp16
        # 0.251 fallback -> 0.433 tle.  (harness/perf_ir/amin_fp16_tle_scan.py)
        ypow = triton.next_power_of_2(N)
        if ypow < 64:
            return None
        xblock = 256 if M % 256 == 0 else (128 if M % 128 == 0 else 64)
        if xblock == 64:
            return None
        cap = 128
        yblock = min(ypow, cap)
        return xblock, yblock
    if dtype == torch.bfloat16:
        xblock = 256 if M % 256 == 0 else (128 if M % 128 == 0 else 64)
        cap = 128
    else:  # torch.float32
        xblock = 128 if M % 128 == 0 else 64
        cap = 256
    yblock = min(triton.next_power_of_2(N), cap)
    yblock = max(yblock, 1)
    return xblock, yblock


def _amin_tle_row(src, out, M, N):
    """Try the tle.gpu coalesced-DMA row-MIN.  Returns True on success, False
    (caller falls back to the existing kernels) if tle is unavailable or the
    launch fails."""
    cfg = _tle_row_config(M, N, src.dtype)
    if cfg is None:
        return False
    xblock, yblock = cfg
    try:
        a2d = src.view(M, N)
        out1d = out.view(M)
        a_desc = TensorDescriptor.from_tensor(a2d, block_shape=[xblock, yblock])
        c_desc = TensorDescriptor.from_tensor(out1d, block_shape=[xblock])
        grid = (triton.cdiv(M, xblock),)
        amin_tle_row_kernel[grid](
            a_desc, c_desc, N, xblock, yblock, _TLE_TL_DTYPE[src.dtype]
        )
        return True
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.debug("tle.gpu amin row path failed, falling back: %s", e)
        return False


@libentry()
@triton.jit
def amin_flat_chunk_kernel(inp, out, CHUNK: tl.constexpr):
    """Unmasked exact-size 1-D chunk reduce.  Grid = number of full chunks."""
    pid = ext.program_id(0)
    off = pid * CHUNK + tl.arange(0, CHUNK)
    a = tl.load(inp + off).to(tl.float32)
    tl.store(out + pid, tl.min(a))


@libentry()
@triton.jit
def amin_flat_tail_kernel(inp, out, start, NTAIL, TL: tl.constexpr):
    """Single-shot masked tail (NTAIL <= 8192 lanes; validated 2026-08-22).
    start is a scalar flat offset."""
    off = tl.arange(0, TL)
    a = tl.load(inp + start + off, mask=off < NTAIL, other=float("inf")).to(tl.float32)
    tl.store(out, tl.min(a))


@libentry()
@triton.jit
def amin_flat_group_kernel(mid, gsum, GCHUNK: tl.constexpr):
    """Unmasked group-reduce of a zero-padded partial buffer (compresses
    8192 partials per program)."""
    pid = ext.program_id(0)
    off = pid * GCHUNK + tl.arange(0, GCHUNK)
    a = tl.load(mid + off).to(tl.float32)
    tl.store(gsum + pid, tl.min(a))


@libentry()
@triton.jit
def amin_flat_merge_kernel(mid, out, np, NLANES: tl.constexpr):
    """Single-shot masked merge of np partials (np <= 8192)."""
    off = tl.arange(0, NLANES)
    a = tl.load(mid + off, mask=off < np, other=float("inf")).to(tl.float32)
    tl.store(out, tl.min(a))


_FAST_BN_FP16 = (1024, 512, 256, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 1024, 256, 128, 64, 16, 32)
_FAST_BM_FP16 = (128, 64, 32, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 16, 8, 4, 2)


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, or
    None when no mask-free tile covers this shape."""
    if M < 2:
        return None
    if N >= (1 << 20):
        if N % 8192 == 0:
            bm = next(
                (
                    m
                    for m in ((8, 4, 16, 32, 2) if not is_fp32 else (4, 8, 16, 32, 2))
                    if M % m == 0
                ),
                None,
            )
            if bm is not None:
                return bm, 8192
    if N >= (1 << 16):
        if N % 1024 == 0:
            bm = next(
                (
                    m
                    for m in (
                        (128, 64, 32, 16, 8, 4) if not is_fp32 else (32, 16, 64, 8, 4)
                    )
                    if M % m == 0
                ),
                None,
            )
            if bm is not None:
                return bm, 1024
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((b for b in bms if M % b == 0), None)
    if bm is None:
        return None
    return bm, bn


@libentry()
@triton.jit
def amin_rows_masked_kernel(inp, out, M, N, BLOCK: tl.constexpr):
    """Masked fallback (shapes no unmasked tile covers): one program per row,
    [BLOCK]-lane 1-D loads, single final `tl.min(axis=0)`.

    The previous [BLOCK_M, BLOCK_N] 2-D masked form miscompiles on this XPU
    for non-divisible shapes (probed 2026-09-10):
      * `tl.where(mask, a, inf)` re-masking a masked bf16 load returns wrong
        lanes on column-masked configs (M=40999, N=600, BM=64/4096, BN=1024:
        291/40999 output columns wrong; the form without the re-mask is
        exact there);
      * 2-D loads whose row stride is not 16B-aligned (N * elem_size % 16
        != 0, e.g. N=40999, BN=8192) return garbage even without the
        re-mask (370/600 rows wrong; N=40960 with the same blocks: exact).
    1-D masked loads per row (the amin_kernel_1 / amin_flat_tail_kernel
    family) avoid both; verified exact for fp16/fp32/bf16 on the functional
    matrix, both (600, 40999) orientations, and odd-N / multi-chunk / N<BLOCK
    edge cases."""
    row = ext.program_id(0)
    start = row * N
    off = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], value=float("inf"), dtype=tl.float32)
    for s in range(0, N, BLOCK):
        cols = s + off
        m = cols < N
        a = tl.load(inp + start + cols, mask=m, other=float("inf")).to(tl.float32)
        acc = tl.minimum(acc, a)
    tl.store(out + row, tl.min(acc, axis=0))


def _amin_flat(inp, out, device):
    """Full (dim=None) reduction over `inp` (any numel)."""
    numel = inp.numel()
    with torch_device_fn.device(device):
        if numel <= _FULL_REDUCTION_BLOCK_SIZE:
            amin_flat_merge_kernel[(1, 1, 1)](
                inp, out, numel, _FULL_REDUCTION_BLOCK_SIZE
            )
            return
        if numel < _FLAT_CHUNK_MAX_NUMEL:
            nfull = numel // _FLAT_CHUNK
            tail = numel - nfull * _FLAT_CHUNK
            nb = nfull + (1 if tail else 0)
            mid = torch.empty((nb,), dtype=inp.dtype, device=device)
            if nfull:
                amin_flat_chunk_kernel[(nfull, 1, 1)](
                    inp, mid, _FLAT_CHUNK, buffer_size_limit=2048
                )
            if tail:
                if tail <= 8192:
                    amin_flat_tail_kernel[(1, 1, 1)](
                        inp,
                        mid[nfull : nfull + 1],
                        nfull * _FLAT_CHUNK,
                        tail,
                        triton.next_power_of_2(tail),
                    )
                else:
                    TL = triton.next_power_of_2(tail)
                    staged = torch.zeros((TL,), dtype=inp.dtype, device=device)
                    src_tail = inp[nfull * _FLAT_CHUNK :]
                    if not tle_copy(src_tail, staged[:tail]):
                        torch.ops.aten._copy_from(src_tail, staged[:tail], False)
                    amin_flat_chunk_kernel[(1, 1, 1)](
                        staged, mid[nfull : nfull + 1], TL, buffer_size_limit=2048
                    )
            amin_flat_merge_kernel[(1, 1, 1)](mid, out, nb, triton.next_power_of_2(nb))
        else:
            rows = numel // _FLAT_ROW_WIDTH
            res = numel - rows * _FLAT_ROW_WIDTH
            bm = next((m for m in _FAST_BM_FP16 if rows % m == 0), _FAST_BM_FP16[0])
            nb = rows + (1 if res else 0)
            mid = torch.empty((nb,), dtype=inp.dtype, device=device)
            amin_rows_kernel[(rows // bm, 1)](
                inp,
                mid,
                rows,
                _FLAT_ROW_WIDTH,
                bm,
                1024,
                buffer_size_limit=2048,
            )
            if res:
                amin_flat_tail_kernel[(1, 1, 1)](
                    inp,
                    mid[rows:],
                    rows * _FLAT_ROW_WIDTH,
                    res,
                    triton.next_power_of_2(res),
                )
            if nb <= 8192:
                amin_flat_merge_kernel[(1, 1, 1)](
                    mid, out, nb, triton.next_power_of_2(nb)
                )
            else:
                g = (nb + 8191) // 8192
                padded = torch.zeros((g * 8192,), dtype=inp.dtype, device=device)
                if not tle_copy(mid, padded[:nb]):
                    torch.ops.aten._copy_from(mid, padded[:nb], False)
                gsum = torch.empty((g,), dtype=inp.dtype, device=device)
                amin_flat_group_kernel[(g, 1, 1)](padded, gsum, 8192)
                amin_flat_merge_kernel[(1, 1, 1)](
                    gsum, out, g, triton.next_power_of_2(g)
                )


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN")
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dtype = inp.dtype
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            flat = inp.reshape(-1)
            if not _amin_raw_flat(flat, out, flat.numel()):
                _amin_flat(flat, out, inp.device)
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if N == 1:
            out = torch.empty(shape, dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                if not tle_copy(inp, out):
                    torch.ops.aten._copy_from(inp, out, False)
            if not keepdim:
                out = out.squeeze(dim=dim)
            return out

        # ---- raw XTDK fast paths for the single-dim contiguous case ----
        # For a single contiguous reduction dim d, out[o, w] = min_m inp[o, m, w]
        # with outer = prod(shape[:d]), mid = shape[d] (== N), inner =
        # prod(shape[d+1:]).  inner > 1 is a middle-dim column-min (the previous
        # biggest laggard); inner == 1 is a last-dim row-min.  Both read the
        # native layout and skip the permute/transpose copy the Triton path uses.
        if (
            _HAS_RAW
            and len(dim) == 1
            and dtype in _TLE_TL_DTYPE
            and inp.is_contiguous()
        ):
            d = dim[0]
            inner = 1
            for i in range(d + 1, inp.dim()):
                inner *= inp.shape[i]
            outer = inp.numel() // (N * inner)
            if 1 < inner <= 64:
                out = torch.empty(shape, dtype=dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    if _amin_raw_mid(inp, out.reshape(outer, inner), outer, N, inner):
                        if not keepdim:
                            out = out.squeeze(dim=dim)
                        return out
            elif inner == 1 and 1 < N <= _RAW_ROW_MAX_N:
                out = torch.empty(shape, dtype=dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    if _amin_raw_rows(
                        inp.reshape(outer, N), out.reshape(outer), outer, N
                    ):
                        if not keepdim:
                            out = out.squeeze(dim=dim)
                        return out

        dim_i = inp.dim()
        stride = inp.stride()
        batch_dim = [i for i in range(dim_i) if i not in dim]
        sorted_reduction_dim = sorted(dim, key=lambda x: stride[x], reverse=True)
        order = batch_dim + sorted_reduction_dim
        view = inp.permute(order)
        if view.is_contiguous():
            src = view
        else:
            src = torch.empty(list(view.shape), dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                if not tle_copy(view, src):
                    torch.ops.aten._copy_from(view, src, False)

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        is_fp32 = dtype == torch.float32
        tile = _pick_fast_tile(M, N, is_fp32)
        with torch_device_fn.device(inp.device):
            if _amin_tle_row(src, out, M, N):
                pass
            elif tile is not None:
                block_m, block_n = tile
                grid = (triton.cdiv(M, block_m),)
                amin_rows_kernel[grid](
                    src,
                    out,
                    M,
                    N,
                    block_m,
                    block_n,
                    buffer_size_limit=2048,
                )
            else:
                block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
                amin_rows_masked_kernel[(M, 1)](
                    src,
                    out,
                    M,
                    N,
                    block_n,
                    buffer_size_limit=2048,
                )
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def amin_(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN_")
    if isinstance(dim, int):
        dim = [dim]
    if dim is None or len(dim) == 0:
        M = inp.numel()
        dtype = inp.dtype
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        # Prefer the raw XTDK flat reduction (single-cluster single-launch for
        # <=2M, tuned two-pass above) -- same fast path amin() uses for dim=None.
        # Falls back to the staged / 2-kernel Triton reductions if unavailable.
        flat = inp.reshape(-1)
        with torch_device_fn.device(inp.device):
            if _amin_raw_flat(flat, out.reshape([]), M):
                if M >= _FLAT_FILL_MIN_NUMEL:
                    inp.fill_(out.reshape(()))
                else:
                    inp.copy_(out if out.shape == inp.shape else out.expand_as(inp))
                return inp
        # For very large numel the amin_kernel_1/2 pair collapses the whole
        # partial buffer in a single program (BLOCK_MID = next_pow2(mid_size)),
        # which is catastrophic (e.g. numel=1<<30 measured ~0.08x).  Reuse the
        # staged _amin_flat reduction there; keep the light 2-kernel path for
        # small/medium numel where it has less launch overhead.
        if M >= _FLAT_CHUNK_MAX_NUMEL:
            with torch_device_fn.device(inp.device):
                _amin_flat(inp.reshape(-1), out.reshape([]), inp.device)
            inp.fill_(out.reshape(()))
            return inp
        block_size = get_block_size_1d(M, inp.element_size())
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amin_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
                M % block_size != 0,
                buffer_size_limit=2048,
            )
            amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
        inp.copy_(out if out.shape == inp.shape else out.expand_as(inp))
        return inp
    else:
        # FUSED fast path: reducing the last dim of a contiguous tensor is a
        # per-row min broadcast back over that row -- do it in one raw kernel
        # (reduce + in-place fill) instead of amin()+copy_.  Measured 1.3-2x
        # over the split for moderate N; validated bit-exact.  Only when a single
        # trailing dim is reduced, the tensor is contiguous, and N is within the
        # tested range; otherwise fall through to the generic reduce+copy.
        ndim = inp.dim()
        norm = sorted((d % ndim) for d in dim)
        if (
            len(norm) == 1
            and norm[0] == ndim - 1
            and inp.dtype in _RAW_ROWFILL
            and inp.is_contiguous()
        ):
            N = inp.shape[-1]
            M = inp.numel() // N if N > 0 else 0
            if (
                1 < N <= _ROWFILL_MAX_N
                and M > 0
                and (inp.numel() < _ROWFILL_SPLIT_NUMEL[inp.dtype])
            ):
                with torch_device_fn.device(inp.device):
                    if _amin_inplace_rowfill(inp, M, N):
                        return inp
        # FUSED fast path: reducing a genuine interior (middle) dim of a
        # contiguous tensor is a per-column min broadcast back over that dim --
        # fuse reduce + in-place fill in one raw kernel.  Only for small mid and
        # inner<=64 (kernel LM/accumulator budget); large mid loses the
        # single-cluster occupancy race and stays on the reduce+copy split.
        if (
            len(norm) == 1
            and 0 < norm[0] < ndim - 1
            and inp.dtype in _RAW_MIDFILL
            and inp.is_contiguous()
        ):
            d = norm[0]
            shp = inp.shape
            mid = shp[d]
            inner = 1
            for s in shp[d + 1 :]:
                inner *= s
            outer = inp.numel() // (mid * inner) if mid * inner > 0 else 0
            if 0 < inner <= 64 and 0 < mid <= _MIDFILL_MAX_MID[inp.dtype] and outer > 0:
                with torch_device_fn.device(inp.device):
                    if _amin_inplace_midfill(inp, outer, mid, inner):
                        return inp
        result = amin(inp, dim=dim, keepdim=True)
        # IMPLICIT broadcast writeback: copy_ from the [.,1,.] keepdim result hits
        # the native fill-like fast path (~free), matching the benchmark torch ref.
        # An explicit expand_as view would force the slow stride-0 gather path.
        inp.copy_(result)
        return inp
