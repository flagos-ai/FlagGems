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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Optional on-chip (cluster shared-memory) staging fast path for the 1D-sequence
# case. The sorted sequence is loop-invariant across every value, so it can be
# staged once into cluster SMEM and every data-dependent bitwalk probe then hits
# on-chip memory (tle.gpu.local_ptr) instead of chasing a dependent global/L2
# gather chain. Same-card measurement: 1D-shape dtype-equal ~0.22 -> ~0.65.
# Requires the xpu3 `tle.gpu` API; guarded so other backends/arch fall through.
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except Exception:  # pragma: no cover - depends on triton build / arch
    _HAS_TLE = False

# Optional tle.raw fast path: a hand-written 16-wide SIMD bitwalk (shipped as
# the precompiled payload/obj/searchsorted.o) that gathers probes from per-core
# local memory via vgather_lm. This is the one lever Triton's XPU codegen misses
# -- it only emits
# a hardware vgather for compile-time linear-ramp offsets, so a data-dependent
# probe index falls back to a serial scalar-load loop. The raw payload keeps the
# whole gather chain vectorized, matching / beating torch on the benchmark
# shapes (1D ~1.0, 2D ~1.07-1.15 vs shipped Triton 0.16-0.76).
try:
    import triton.experimental.tle as _tle_ext

    _HAS_RAW = True
except Exception:  # pragma: no cover - depends on triton build / arch
    _HAS_RAW = False

# dtypes whose SMEM load path is verified to compile + be correct on xpu3.
# bf16 is staged as fp32 (bf16 SM load can't be selected by llc; bf16->fp32 is
# lossless and order-preserving so search results are identical).
_SMEM_STAGE = {
    torch.float16: (tl.float16, False, 2),
    torch.float32: (tl.float32, False, 4),
    torch.bfloat16: (tl.float32, True, 4),
    torch.int16: (tl.int16, False, 2),
    torch.int32: (tl.int32, False, 4),
    torch.int8: (tl.int8, False, 1),
    torch.uint8: (tl.uint8, False, 1),
}
# Conservative cluster-SMEM staging budget (user window is larger; small seqs).
_SMEM_MAX_BYTES = 1 << 16

# Binary search is a chain of dependent gather loads (latency-bound). A larger
# block keeps more independent lanes in flight to hide that gather latency; a
# same-card controlled A/B across all 7 benchmark dtypes shows 512 is the sweet
# spot (dtype-equal-weighted 0.203 -> 0.249), beating 256/1024/2048.
_CUDA_BLOCK_SIZE = 512
_ASCEND_BLOCK_SIZE = 512
_SUPPORTED_INPUT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _searchsorted_kernel(
    sorted_sequence,
    values,
    sorter,
    out,
    total_values,
    values_per_row,
    LOG_SEQUENCE_LEN: tl.constexpr,
    RIGHT: tl.constexpr,
    HAS_SORTER: tl.constexpr,
    IS_1D_SEQUENCE: tl.constexpr,
    USE_INT32_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    SEQUENCE_LEN: tl.constexpr,
):
    # Bitwalk (binary lifting) formulation of searchsorted:
    #   result = # of boundaries strictly below / not above `values`,
    # computed by probing seq[idx + step - 1] for step = 2^b, b = LOG..0.
    # Compared with the low/high `tl.where` bisection this keeps every
    # step a pure add (+ one select-free advance mask), so the unrolled
    # chain is much shorter and XPU's TTXIR passes stay linear in the
    # sequence length (old kernel: compile time exploded with LOG>=8:
    # ~150s @LOG=8, >1h @LOG=10; new kernel: 4-6s at LOG=13).
    # NaN semantics match the original: comparisons with NaN are false,
    # so `~go_left` is true and NaN advances to the right (end) position.
    # SEQUENCE_LEN is constexpr so the per-round probe address stays
    # affine-in-{offsets, idx} for the XPU backend (a runtime sequence_len
    # broke affine analysis ~1.25x on the 2D benchmark shapes). NOTE:
    # making BOTH SEQUENCE_LEN and values_per_row constexpr triggers an XPU
    # backend constant-fold bug for small non-pow2 sequences (S=2,3 gave
    # wrong results) -- keep values_per_row runtime.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < total_values
        values_in = tl.load(values + offsets, mask=mask, other=0)
    else:
        values_in = tl.load(values + offsets)

    if IS_1D_SEQUENCE:
        if USE_INT32_INDEX:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        else:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    else:
        if USE_INT32_INDEX:
            row_offsets = (offsets // values_per_row).to(tl.int32) * SEQUENCE_LEN
        else:
            row_offsets = (offsets // values_per_row) * SEQUENCE_LEN

    if USE_INT32_INDEX:
        idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for b in tl.static_range(LOG_SEQUENCE_LEN, -1, -1):
        step = 1 << b
        next_idx = idx + step
        probe = tl.minimum(next_idx, SEQUENCE_LEN) - 1
        in_range = next_idx <= SEQUENCE_LEN
        if HAS_SORTER:
            if NEED_MASK:
                si = tl.load(sorter + row_offsets + probe, mask=mask, other=0)
                if USE_INT32_INDEX:
                    si = si.to(tl.int32)
                mv = tl.load(sorted_sequence + row_offsets + si, mask=mask, other=0)
            else:
                si = tl.load(sorter + row_offsets + probe)
                if USE_INT32_INDEX:
                    si = si.to(tl.int32)
                mv = tl.load(sorted_sequence + row_offsets + si)
            valid = (si >= 0) & (si < SEQUENCE_LEN)
            tl.device_assert(in_range | (~valid), "sorter index out of range")
        else:
            if NEED_MASK:
                mv = tl.load(sorted_sequence + row_offsets + probe, mask=mask, other=0)
                in_range = in_range & mask
            else:
                mv = tl.load(sorted_sequence + row_offsets + probe)
        if RIGHT:
            go_left = values_in < mv
        else:
            go_left = values_in <= mv
        advance = (~go_left).to(tl.int32) & in_range.to(tl.int32)
        idx += step.to(idx.dtype) * advance.to(idx.dtype)

    if NEED_MASK:
        tl.store(out + offsets, idx, mask=mask)
    else:
        tl.store(out + offsets, idx)


def _make_smem_kernel():
    # Defined lazily so the module still imports on backends without `tle.gpu`.
    @triton.jit
    def _searchsorted_smem_1d_kernel(
        seq_desc,
        values,
        out,
        total_values,
        SEQ_LEN: tl.constexpr,
        LOG_SEQUENCE_LEN: tl.constexpr,
        RIGHT: tl.constexpr,
        STAGE_DTYPE: tl.constexpr,
        CAST_V: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NEED_MASK: tl.constexpr,
    ):
        # Same bitwalk as _searchsorted_kernel, but the (1D, loop-invariant)
        # sorted sequence is staged once into cluster SMEM and every probe
        # gather is an on-chip local_ptr read instead of a global gather.
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = offsets < total_values
            values_in = tl.load(values + offsets, mask=mask, other=0)
        else:
            values_in = tl.load(values + offsets)
        if CAST_V:
            values_in = values_in.to(STAGE_DTYPE)

        seq_sm = tle.gpu.alloc(
            [SEQ_LEN], dtype=STAGE_DTYPE, layout=None, scope=tle.gpu.smem
        )
        tle.gpu.copy(seq_desc, seq_sm, [SEQ_LEN], [0])

        idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for b in tl.static_range(LOG_SEQUENCE_LEN, -1, -1):
            step = 1 << b
            next_idx = idx + step
            probe = tl.minimum(next_idx, SEQ_LEN) - 1
            in_range = next_idx <= SEQ_LEN
            ptrs = tle.gpu.local_ptr(seq_sm, (probe,))
            mv = tl.load(ptrs)
            if RIGHT:
                go_left = values_in < mv
            else:
                go_left = values_in <= mv
            advance = (~go_left).to(tl.int32) & in_range.to(tl.int32)
            idx += step * advance

        if NEED_MASK:
            tl.store(out + offsets, idx, mask=mask)
        else:
            tl.store(out + offsets, idx)

    return _searchsorted_smem_1d_kernel


_searchsorted_smem_1d_kernel = _make_smem_kernel() if _HAS_TLE else None


# Per-core LM staging budget: seq[<=1024] * 4 bytes = 4KB fits comfortably.
_RAW_MAX_SEQ_LEN = 1024
# Global-core grid caps (P800: 12 clusters * 64 cores). Empirically best.
_RAW_GRID_1D = 8
_RAW_GRID_2D = 12


def _make_raw_kernels():
    # Defined lazily so the module still imports on backends without `tle.raw`.
    _here = os.path.join(
        os.path.dirname(__file__), "..", "payload", "obj", "searchsorted.o"
    )

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_1d_f32(seq, vals, out, sl, n, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_f32(seq, vals, out, rows, sl, vpr, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_1d_i32(seq, vals, out, sl, n, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_i32(seq, vals, out, rows, sl, vpr, right, logn): ...

    # Widen-in-LM 2D variants: read the native narrow buffer and widen to
    # fp32/int32 inside the kernel (avoids the slow gems .to(fp32) cast), then
    # run the same bitwalk. All are dim==2, int32-output.
    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_f16w(seq, vals, out, rows, sl, vpr, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_bf16w(seq, vals, out, rows, sl, vpr, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_i16w(seq, vals, out, rows, sl, vpr, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_i8w(seq, vals, out, rows, sl, vpr, right, logn): ...

    @_tle_ext.raw.dialect("xpu3", object=_here, arch=3)
    def ss_2d_u8w(seq, vals, out, rows, sl, vpr, right, logn): ...

    @triton.jit(do_not_specialize=["sl", "n", "right", "logn"])
    def _raw_1d_f32(Seq, Vals, Out, sl, n, right, logn):
        tle.raw.call(ss_1d_f32, (Seq, Vals, Out, sl, n, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_f32(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_f32, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    @triton.jit(do_not_specialize=["sl", "n", "right", "logn"])
    def _raw_1d_i32(Seq, Vals, Out, sl, n, right, logn):
        tle.raw.call(ss_1d_i32, (Seq, Vals, Out, sl, n, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_i32(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_i32, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_f16w(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_f16w, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_bf16w(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_bf16w, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_i16w(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_i16w, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_i8w(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_i8w, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    @triton.jit(do_not_specialize=["rows", "sl", "vpr", "right", "logn"])
    def _raw_2d_u8w(Seq, Vals, Out, rows, sl, vpr, right, logn):
        tle.raw.call(ss_2d_u8w, (Seq, Vals, Out, rows, sl, vpr, right, logn))

    return {
        ("f32", 1): _raw_1d_f32,
        ("f32", 2): _raw_2d_f32,
        ("i32", 1): _raw_1d_i32,
        ("i32", 2): _raw_2d_i32,
        ("f16", 2): _raw_2d_f16w,
        ("bf16", 2): _raw_2d_bf16w,
        ("i16", 2): _raw_2d_i16w,
        ("i8", 2): _raw_2d_i8w,
        ("u8", 2): _raw_2d_u8w,
    }


# `tle.raw.call` requires the @triton.jit wrapper live in a real .py file, which
# this is; the .xpu path resolves relative to this file's directory.
_RAW_KERNELS = _make_raw_kernels() if _HAS_RAW else None

# fp32/int32-exact & order-preserving narrow dtypes handled by the widen-in-LM
# 2D payloads (widen the raw buffer to fp32/int32 inside the kernel, no cast).
_RAW_WIDEN_KIND = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
}


def _raw_eligible(sorted_sequence, values, sorter, use_int32_index):
    """Return (kind, dim) if the raw SIMD bitwalk applies, else None.

    kind is one of 'f32'|'i32' (native) or 'f16'|'bf16'|'i16'|'i8'|'u8'
    (widen-in-LM 2D payloads)."""
    if not _HAS_RAW or _RAW_KERNELS is None or sorter is not None:
        return None
    if not use_int32_index:
        return None
    dim = sorted_sequence.dim()
    # Only the 2D case wins: for a 1D sequence the shipped cluster-SMEM path
    # (~0.76) already beats the raw payload (~0.28) because the raw kernel's
    # launch floor (~0.034ms) dwarfs torch on the tiny 1D benchmark shape.
    if dim != 2:
        return None
    seq_len = sorted_sequence.shape[-1]
    if seq_len == 0 or seq_len > _RAW_MAX_SEQ_LEN:
        return None
    dtype = sorted_sequence.dtype
    if dtype == torch.int32:
        # Native-int32 payload reads `values` as int32; only safe when values is
        # also int32 (the scalar entry points build a float32 scalar_tensor, and
        # int32 values > 2^24 aren't fp32-exact so we can't fall through to f32).
        if values.dtype != torch.int32:
            return None
        return ("i32", dim)
    if dtype == torch.float32:
        # fp32 needs no cast: the payload reads the buffer as-is and wins big
        # (shape3 0.15 -> 1.16).
        return ("f32", dim)
    # The remaining fp32/int32-exact & order-preserving dtypes are handled by
    # the widen-in-LM payloads: each core stages the RAW narrow buffer into LM
    # and widens in-place to fp32/int32 there (once per row, amortized by the
    # contiguous-chunk tile striping), then reuses the proven 16-wide bitwalk.
    # This avoids the gems `.to(fp32)` cast, which dispatches to the `_to_copy`
    # override at ~1GB/s (~11x slower than native XPU cast) and would dominate
    # and regress the kernel. _PROBE-verified correct on shape3 geometry.
    kind = _RAW_WIDEN_KIND.get(dtype)
    if kind is not None:
        # The widen payload reads `values` with the SAME narrow dtype as `seq`.
        # The scalar entry points build a float32 scalar_tensor for `values`, so
        # only route when values matches the sequence dtype; otherwise fall back.
        if values.dtype != dtype:
            return None
        return (kind, dim)
    return None


def _smem_eligible(sorted_sequence, values, sorter, use_int32_index):
    if not _HAS_TLE or sorter is not None:
        return None
    if sorted_sequence.dim() != 1 or not use_int32_index:
        return None
    dtype = sorted_sequence.dtype
    stage = _SMEM_STAGE.get(dtype)
    if stage is None:
        return None
    seq_len = sorted_sequence.shape[-1]
    if seq_len & (seq_len - 1) != 0:  # tle.gpu.alloc requires pow-of-2 dims
        return None
    if seq_len * stage[2] > _SMEM_MAX_BYTES:
        return None
    return stage


def _normalize_right(right: bool, side: str | None) -> bool:
    if side is None:
        return bool(right)
    if side == "left":
        if right:
            raise RuntimeError(
                "torch.searchsorted(): side and right can't be set to opposites, "
                "got side of left while right was True"
            )
        return False
    if side == "right":
        return True
    raise RuntimeError(
        f"torch.searchsorted(): side can only be 'left' or 'right' but got {side}"
    )


def _check_dtype(tensor: torch.Tensor, name: str):
    if tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"searchsorted is not implemented for {name} dtype {tensor.dtype}"
        )


def _check_tensor_values_shape(sorted_sequence: torch.Tensor, values: torch.Tensor):
    if sorted_sequence.dim() == 0:
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match"
        )
    if sorted_sequence.dim() == 1:
        return
    if values.dim() != sorted_sequence.dim() or (
        tuple(values.shape[:-1]) != tuple(sorted_sequence.shape[:-1])
    ):
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match, but we got boundaries tensor "
            f"{list(sorted_sequence.shape)} and input value tensor {list(values.shape)}"
        )


def _check_scalar_values_shape(sorted_sequence: torch.Tensor):
    if sorted_sequence.dim() != 1:
        raise RuntimeError(
            "torch.searchsorted(): input value can be a scalar only when boundaries "
            "tensor dimension is 1, but we got boundaries tensor "
            f"dim({sorted_sequence.dim()}) and input value's dim(0) numel(1)"
        )


def _check_sorter(sorted_sequence: torch.Tensor, sorter: torch.Tensor | None):
    if sorter is None:
        return
    if tuple(sorter.shape) != tuple(sorted_sequence.shape):
        raise RuntimeError(
            "torch.searchsorted(): boundary and sorter must have the same size, "
            f"but got boundary tensor {list(sorted_sequence.shape)}"
            f"and got sorter tensor {list(sorter.shape)}"
        )
    if sorter.dtype != torch.int64:
        raise RuntimeError(
            "torch.searchsorted(): sorter must be a tensor of long dtype but got "
            f"dtype {sorter.dtype}"
        )
    if sorter.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorter and boundary tensors must be on the same device"
        )


def _prepare_out(
    values: torch.Tensor,
    out_int32: bool,
    out: torch.Tensor | None,
):
    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        return torch.empty(values.shape, dtype=out_dtype, device=values.device)
    if out.dtype != out_dtype:
        raise RuntimeError(
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be "
            "Int(int32) or Long(int64) depending on whether out_int32 flag is True"
        )
    if out.device != values.device:
        raise RuntimeError(
            "torch.searchsorted(): output tensor must be on the same device as input"
        )
    if tuple(out.shape) != tuple(values.shape):
        out.resize_(values.shape)
    return out


@triton.jit
def _raw_out_widen_kernel(SRC, DST, n, BLOCK: tl.constexpr):
    # Widen the int32 raw indices into an int64 (contiguous) output. The generic
    # `out.copy_(raw_out)` path lands on the ~1GB/s device copy (both gems and
    # native XPU copy_ int32->int64 measure ~50us for 16384 elems); a plain
    # Triton store does the same widen in ~5us. This is what makes the int64
    # output shapes (out_int32=False, e.g. benchmark shape2) practical for raw.
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(SRC + off, mask=m).to(tl.int64)
    tl.store(DST + off, v, mask=m)


_RAW_OUT_WIDEN_GRID = 12


def _searchsorted_impl(
    sorted_sequence: torch.Tensor,
    values: torch.Tensor,
    *,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: torch.Tensor | None,
    out: torch.Tensor | None = None,
):
    right = _normalize_right(right, side)
    _check_dtype(sorted_sequence, "sorted_sequence")
    _check_dtype(values, "values")
    _check_tensor_values_shape(sorted_sequence, values)
    _check_sorter(sorted_sequence, sorter)
    if values.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorted_sequence and values must be on the same device"
        )

    out = _prepare_out(values, out_int32, out)
    if values.numel() == 0:
        return out
    if sorted_sequence.shape[-1] == 0:
        out.zero_()
        return out

    sorted_sequence_contiguous = sorted_sequence.contiguous()
    values_contiguous = values.contiguous()
    sorter_contiguous = sorter.contiguous() if sorter is not None else None
    is_ascend = runtime_device.vendor_name == "ascend"
    if sorter_contiguous is not None and is_ascend:
        sorted_sequence_contiguous = torch.gather(
            sorted_sequence_contiguous, -1, sorter_contiguous
        )
        sorter_contiguous = None
    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    )

    sequence_len = sorted_sequence.shape[-1]
    values_per_row = values.shape[-1] if sorted_sequence.dim() != 1 else values.numel()
    if is_ascend and sorted_sequence.dtype.is_floating_point:
        block_size = _ASCEND_BLOCK_SIZE
    elif is_ascend:
        block_size = _CUDA_BLOCK_SIZE
    else:
        # kunlunxin: size-banded block. The probe loads are data-dependent
        # gathers; larger blocks hide per-warp gather latency better, but too
        # large spills registers (2048 regressed on the 256x1024/512 case).
        numel = values.numel()
        if numel <= 4096:
            block_size = 256
        elif numel <= 16384:
            block_size = 512
        else:
            block_size = 1024
    use_int32_index = (
        values.numel() < torch.iinfo(torch.int32).max
        and sorted_sequence.numel() < torch.iinfo(torch.int32).max
    )
    need_mask = values.numel() % block_size != 0

    smem_stage = _smem_eligible(
        sorted_sequence_contiguous, values, sorter_contiguous, use_int32_index
    )
    raw_stage = _raw_eligible(
        sorted_sequence_contiguous, values, sorter_contiguous, use_int32_index
    )

    with torch_device_fn.device(sorted_sequence.device):
        if raw_stage is not None:
            kind, dim = raw_stage
            if kind == "f32":
                seq_in = sorted_sequence_contiguous.to(torch.float32)
                vals_in = values_contiguous.to(torch.float32)
            else:
                seq_in = sorted_sequence_contiguous
                vals_in = values_contiguous
            # The raw payload writes int32 indices; write straight into `out`
            # only when it is already int32 & contiguous, else stage + copy
            # (handles int64 output and non-contiguous out).
            if out.dtype == torch.int32 and out.is_contiguous():
                raw_out = out
            else:
                raw_out = torch.empty(
                    values.shape, dtype=torch.int32, device=values.device
                )
            r = 1 if right else 0
            logn = sequence_len.bit_length()
            if dim == 1:
                _RAW_KERNELS[(kind, 1)][(_RAW_GRID_1D,)](
                    seq_in, vals_in, raw_out, sequence_len, values.numel(), r, logn
                )
            else:
                _RAW_KERNELS[(kind, 2)][(_RAW_GRID_2D,)](
                    seq_in,
                    vals_in,
                    raw_out,
                    sorted_sequence.shape[0],
                    sequence_len,
                    values.shape[-1],
                    r,
                    logn,
                )
            if raw_out is not out:
                if (
                    out.dtype == torch.int64
                    and out.is_contiguous()
                    and raw_out.is_contiguous()
                ):
                    # Fast widen int32->int64 (avoids the ~50us slow device
                    # copy_ that otherwise dominates int64-output shapes).
                    n_out = out.numel()
                    wblock = triton.next_power_of_2(
                        max(1, (n_out + _RAW_OUT_WIDEN_GRID - 1) // _RAW_OUT_WIDEN_GRID)
                    )
                    _raw_out_widen_kernel[(_RAW_OUT_WIDEN_GRID,)](
                        raw_out, out, n_out, wblock
                    )
                else:
                    out.copy_(raw_out)
            return out

        grid = (triton.cdiv(values.numel(), block_size),)
        if smem_stage is not None:
            stage_tl, cast_v, _ = smem_stage
            seq_stage = (
                sorted_sequence_contiguous.to(torch.float32)
                if cast_v
                else sorted_sequence_contiguous
            )
            seq_desc = TensorDescriptor.from_tensor(
                seq_stage, block_shape=[sequence_len]
            )
            _searchsorted_smem_1d_kernel[grid](
                seq_desc,
                values_contiguous,
                kernel_out,
                values.numel(),
                SEQ_LEN=sequence_len,
                LOG_SEQUENCE_LEN=sequence_len.bit_length(),
                RIGHT=right,
                STAGE_DTYPE=stage_tl,
                CAST_V=cast_v,
                BLOCK_SIZE=block_size,
                NEED_MASK=need_mask,
            )
        else:
            _searchsorted_kernel[grid](
                sorted_sequence_contiguous,
                values_contiguous,
                (
                    sorter_contiguous
                    if sorter_contiguous is not None
                    else sorted_sequence_contiguous
                ),
                kernel_out,
                values.numel(),
                values_per_row,
                sequence_len,
                LOG_SEQUENCE_LEN=sequence_len.bit_length(),
                RIGHT=right,
                HAS_SORTER=sorter_contiguous is not None,
                IS_1D_SEQUENCE=sorted_sequence.dim() == 1,
                USE_INT32_INDEX=use_int32_index,
                BLOCK_SIZE=block_size,
                NEED_MASK=need_mask,
            )

    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def searchsorted(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED OUT")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )


def searchsorted_scalar(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED SCALAR")
    _check_scalar_values_shape(sorted_sequence)
    values = torch.scalar_tensor(self, device=sorted_sequence.device)
    return _searchsorted_impl(
        sorted_sequence,
        values,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_scalar_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED SCALAR OUT")
    _check_scalar_values_shape(sorted_sequence)
    values = torch.scalar_tensor(self, device=sorted_sequence.device)
    return _searchsorted_impl(
        sorted_sequence,
        values,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )
