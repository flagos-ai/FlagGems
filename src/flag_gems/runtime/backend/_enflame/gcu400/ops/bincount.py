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

# GCU400 notes (knowledge/perf_playbook.md, knowledge/compiler_traps.md):
#   * i64 tl.load/tl.store/atomics are unsafe on this stack (P17), so int64
#     inputs are normalized to int32 on the host and the int64 output is
#     staged through an int32 buffer. On the packed-4B layout the first
#     numel words of view(torch.int32) alias the values, which a one-time
#     probe verifies before use; a true-8B stack falls back to a cast.
#   * the histogram is a bin-centric one-hot scan ([BLOCK_N, TILE_B]
#     orientation, tl.sum(axis=0)) for narrow bin counts: the transposed
#     orientation with a column-broadcast weight silently miscompiles, and
#     num_warps must stay at 4 (num_warps=1 is a known silent-miscompile).
#     BLOCK_N is next_pow2(min(n, 1024)): masked-out lanes still pay the
#     [BLOCK_N, TILE_B] tile cost, so a padded BLOCK_N multiplies the
#     per-program work (n=64 measured 402 -> 18us after tightening).
#   * wide bin counts (needed_size > 2048) switch to an atomic-scatter
#     arm: with many bins the scatter is contention-free and the
#     per-program atomic chain (256 elements at ~90ns/op, measured on
#     GCU400) beats the O(n * num_bins) one-hot scan.
#   * half-precision weighted counts route to the vendor topsatenBincount
#     whenever the bin count is narrow: the native path accumulates fp16/
#     bf16 weights in half precision via device atomics, so an exact
#     fp32-accumulated kernel cannot match the native reference within
#     tolerance (verified on this stack). fp32 weights stay on the one-hot
#     kernel for short inputs and route to the vendor op for long ones
#     (verified bitwise-identical to the native op for fp32).
#   * grids are capped/folded below 65536 programs (P19 hard limit).

import ctypes
import logging
import threading

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_HIST_BLOCK_N = 1024
_HIST_TILE_B = 32
_HIST_W_BLOCK_N = 256
_HIST_W_TILE_B = 64
_MAX_BLOCK_N = 1024
_MAX_PARTIAL_BLOCK_N = 1024
_MAX_PARTIAL_PROGS = 8192
_SINGLE_CTA_MAX_N = 65536
_ATOMIC_BLOCK_N = 256
_GRID_MAX = 65535

# Small inputs run the bin-centric one-hot scan with a tight BLOCK_N
# (next_pow2(n)); the [BLOCK_N, TILE_B] tile cost is paid even for masked-out
# lanes, so a padded BLOCK_N multiplies the per-program work.
_SMALL_N_LIMIT = 256

# Above this bin count the plain / fp32-weighted paths switch from the
# O(n * num_bins) one-hot scan to the atomic-scatter arm: with many bins the
# scatter is contention-free and the per-program atomic chain (BLOCK_N
# elements at ~90ns/op, measured on GCU400) beats the one-hot scan.
_ONEHOT_NB_LIMIT = 2048

# O(n * num_bins) compare budget for the one-hot arms.
_ONEHOT_MAX_OPS = 1 << 24

# The fp32-weighted one-hot arm only pays off for short inputs; longer ones
# route to topsatenBincount (verified bitwise-identical to the native op for
# fp32) because the one-hot kernel degenerates when num_bins is small.
_WEIGHTED_ONEHOT_MAX_N = 2048

# Wide-minlength regime (minlength >= 4096 guarantees a wide, contention-free
# bin range): a pre-sync fused max+count pass into an estimated-size buffer
# keeps the count work ahead of the required output-length sync, which beats
# any post-sync arm by hiding the scatter behind the sync roundtrip.
_PRESYNC_MAX_N = 2048
_PRESYNC_MIN_ML = 4096
_PRESYNC_ESTIMATE_FLOOR = 8192

_INT64_ALIAS_MODE = None


def _int32_view_of_int64(input_contig):
    """Zero-copy int32 alias of a GCU int64 tensor on the packed layout."""
    global _INT64_ALIAS_MODE
    if _INT64_ALIAS_MODE is None:
        try:
            probe = torch.arange(3, dtype=torch.int64, device=input_contig.device)
            words = probe.view(torch.int32)[:3]
            _INT64_ALIAS_MODE = (
                "packed" if words.cpu().tolist() == [0, 1, 2] else "wide"
            )
        except Exception:
            _INT64_ALIAS_MODE = "wide"
    if _INT64_ALIAS_MODE == "packed":
        return input_contig.view(torch.int32)[: input_contig.numel()]
    return input_contig.to(torch.int32)


_TOPS_DTYPES = {
    torch.int32: 6,
    torch.float16: 4,
    torch.bfloat16: 5,
    torch.float32: 8,
}

_TENSOR_CTOR_SYM = (
    "_ZN14topsatenTensorC1ERK14topsatenSize_tS2_RK18topsatenDataType_tRKPv"
)
_TENSOR_SET_DATA_SYM = "_ZN14topsatenTensor13SetTensorDataERKPv"
_BINCOUNT_SYM = (
    "_ZN8topsaten16topsatenBincountER14topsatenTensorRKS0_S3_lP12EfiStream_st"
)

_STATUS_NAMES = {
    0: "SUCCESS",
    1: "ALLOC_FAILED",
    2: "BAD_PARAM",
    3: "NOT_SUPPORT",
    4: "INTERNAL_ERROR",
    5: "RUNTIME_ERROR",
    6: "EXECUTE_ERROR",
}


class _TopsatenSize(ctypes.Structure):
    """struct topsatenSize_t { const int64_t* data; int64_t len; }"""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int64)),
        ("len", ctypes.c_int64),
    ]


class _TopsatenTensorBuf(ctypes.Structure):
    """16-byte storage of topsatenTensor (shared_ptr control pair)."""

    _fields_ = [("ptr", ctypes.c_void_p), ("ctrl", ctypes.c_void_p)]


def _topsaten_strides(shape):
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= max(shape[i], 1)
    return tuple(strides)


class _TopsatenBridge:
    """Lazy ctypes binding of topsatenBincount for the weighted route."""

    def __init__(self):
        lib = ctypes.CDLL("libtopsaten.so")
        ctor = lib[_TENSOR_CTOR_SYM]
        ctor.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenSize),
            ctypes.POINTER(_TopsatenSize),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ctor.restype = None
        set_data = lib[_TENSOR_SET_DATA_SYM]
        set_data.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        set_data.restype = None
        bincount = lib[_BINCOUNT_SYM]
        bincount.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.c_long,
            ctypes.c_void_p,
        ]
        bincount.restype = ctypes.c_int
        self._lib = lib
        self._ctor = ctor
        self._set_data = set_data
        self._bincount = bincount
        self._pool = {}
        self._call_lock = threading.Lock()
        self._raw_stream_fn = self._load_raw_stream_fn()

    @staticmethod
    def _load_raw_stream_fn():
        try:
            from torch_gcu import _C
        except Exception as exc:  # pragma: no cover - torch_gcu always present
            raise RuntimeError(f"torch_gcu._C import failed: {exc}")
        fn = getattr(_C, "_gcu_getCurrentRawStream", None)
        if fn is None:
            raise RuntimeError(
                "torch_gcu._C._gcu_getCurrentRawStream is unavailable; cannot "
                "resolve the current GCU stream for the topsaten route"
            )
        return fn

    def _current_stream(self, index):
        return self._raw_stream_fn(index)

    def _descriptor(self, tensor, claim):
        dtype_code = _TOPS_DTYPES.get(tensor.dtype)
        if dtype_code is None:
            raise RuntimeError(
                f"topsaten bincount route: unsupported dtype {tensor.dtype}"
            )
        shape = tuple(tensor.shape)
        strides = _topsaten_strides(shape)
        data_ptr = tensor.data_ptr()
        pool = self._pool.setdefault((shape, strides, tensor.dtype), [])
        for entry in pool:
            handle = entry[0]
            if id(handle) not in claim and entry[3] == data_ptr:
                claim.add(id(handle))
                return handle
        for entry in pool:
            handle = entry[0]
            if id(handle) in claim:
                continue
            data = ctypes.c_void_p(data_ptr)
            self._set_data(handle, ctypes.byref(data))
            entry[3] = data_ptr
            claim.add(id(handle))
            return handle
        dims = (ctypes.c_int64 * len(shape))(*shape)
        strd = (ctypes.c_int64 * len(strides))(*strides)
        sizes = _TopsatenSize(dims, len(shape))
        strides_size = _TopsatenSize(strd, len(strides))
        dtype = ctypes.c_int(dtype_code)
        data = ctypes.c_void_p(data_ptr)
        buf = _TopsatenTensorBuf()
        self._ctor(
            ctypes.byref(buf),
            ctypes.byref(sizes),
            ctypes.byref(strides_size),
            ctypes.byref(dtype),
            ctypes.byref(data),
        )
        handle = ctypes.pointer(buf)
        pool.append([handle, dims, strd, data_ptr])
        claim.add(id(handle))
        return handle

    def bincount_weighted(self, output, input_contig, weights_contig, minlength):
        # ctypes releases the GIL inside the call; hold the bridge lock so a
        # concurrent call cannot rebind pooled descriptors mid-flight.
        with self._call_lock:
            claim = set()
            out_desc = self._descriptor(output, claim)
            in_desc = self._descriptor(input_contig, claim)
            w_desc = self._descriptor(weights_contig, claim)
            stream = self._current_stream(input_contig.device.index or 0)
            status = self._bincount(out_desc, in_desc, w_desc, int(minlength), stream)
        if status != 0:
            name = _STATUS_NAMES.get(status, "UNKNOWN")
            raise RuntimeError(f"topsatenBincount failed: status={status} ({name})")


_BRIDGE = None
_BRIDGE_LOCK = threading.Lock()
_BRIDGE_ERROR = None


def _get_topsaten_bridge():
    global _BRIDGE, _BRIDGE_ERROR
    if _BRIDGE is not None:
        return _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            if _BRIDGE_ERROR is None:
                try:
                    _BRIDGE = _TopsatenBridge()
                except Exception as exc:
                    _BRIDGE_ERROR = str(exc)
            if _BRIDGE is None:
                raise RuntimeError(f"topsaten bridge unavailable: {_BRIDGE_ERROR}")
    return _BRIDGE


@libentry()
@triton.jit
def _bincount_max_kernel(input_ptr, out_ptr, n_elements, BLOCK_N: tl.constexpr):
    acc = tl.full((BLOCK_N,), -1, dtype=tl.int32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(input_ptr + offs, mask=offs < n_elements, other=-1).to(tl.int32)
        acc = tl.maximum(acc, vals)
    tl.store(out_ptr, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_partial_max_kernel(
    input_ptr,
    partials_ptr,
    n_elements,
    num_progs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.full((BLOCK_N,), -1, dtype=tl.int32)
    for base in range(pid * BLOCK_N, n_elements, num_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(input_ptr + offs, mask=offs < n_elements, other=-1).to(tl.int32)
        acc = tl.maximum(acc, vals)
    tl.store(partials_ptr + pid, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_reduce_max_kernel(
    partials_ptr,
    out_ptr,
    num_partials,
    BLOCK_R: tl.constexpr,
):
    acc = tl.full((BLOCK_R,), -1, dtype=tl.int32)
    for base in range(0, num_partials, BLOCK_R):
        offs = base + tl.arange(0, BLOCK_R)
        vals = tl.load(partials_ptr + offs, mask=offs < num_partials, other=-1)
        acc = tl.maximum(acc, vals)
    tl.store(out_ptr, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_hist_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.int32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        match = (vals[:, None] == bins[None, :]).to(tl.int32)
        acc += tl.sum(match, axis=0)
    tl.store(output_ptr + bins, acc, mask=bin_mask)


@libentry()
@triton.jit
def _bincount_hist_weights_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.float32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        match = (vals[:, None] == bins[None, :]).to(tl.float32)
        acc += tl.sum(match * w[:, None], axis=0)
    tl.store(output_ptr + bins, acc.to(output_ptr.dtype.element_ty), mask=bin_mask)


@libentry()
@triton.jit
def _bincount_atomic_count_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    num_bins,
    grid_progs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    for base in range(pid * BLOCK_N, n_elements, grid_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        safe = mask & (vals >= 0) & (vals < num_bins)
        tl.atomic_add(output_ptr + vals, 1, mask=safe, sem="relaxed")


@libentry()
@triton.jit
def _bincount_atomic_weights_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    num_bins,
    grid_progs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    for base in range(pid * BLOCK_N, n_elements, grid_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        safe = mask & (vals >= 0) & (vals < num_bins)
        tl.atomic_add(output_ptr + vals, w, mask=safe, sem="relaxed")


@libentry()
@triton.jit
def _bincount_fused_count_kernel(
    input_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements
    vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)
    safe = mask & (vals >= 0) & (vals < output_size)
    tl.atomic_add(output_ptr + vals, 1, mask=safe, sem="relaxed")


@libentry()
@triton.jit
def _bincount_fused_weights_kernel(
    input_ptr,
    weights_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements
    vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
    w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)
    safe = mask & (vals >= 0) & (vals < output_size)
    tl.atomic_add(output_ptr + vals, w, mask=safe, sem="relaxed")


def _bincount_max(input_contig, n, device):
    max_out = torch.empty(1, dtype=torch.int32, device=device)
    if n <= _SINGLE_CTA_MAX_N:
        _bincount_max_kernel[(1,)](
            input_contig,
            max_out,
            n,
            BLOCK_N=_MAX_BLOCK_N,
            num_warps=4,
        )
    else:
        num_progs = min(triton.cdiv(n, _MAX_PARTIAL_BLOCK_N), _MAX_PARTIAL_PROGS)
        partials = torch.empty(num_progs, dtype=torch.int32, device=device)
        _bincount_partial_max_kernel[(num_progs,)](
            input_contig,
            partials,
            n,
            num_progs,
            BLOCK_N=_MAX_PARTIAL_BLOCK_N,
            num_warps=4,
        )
        _bincount_reduce_max_kernel[(1,)](
            partials,
            max_out,
            num_progs,
            BLOCK_R=_MAX_PARTIAL_BLOCK_N,
            num_warps=4,
        )
    return max_out


def bincount(input, weights=None, minlength=0):
    """Count the frequency of each value in an array of non-negative ints."""
    logger.debug("GEMS BINCOUNT")

    assert input.dim() == 1, "input must be a 1-D tensor"
    assert minlength >= 0, "minlength must be non-negative"

    if weights is not None:
        assert weights.shape == input.shape, "weights must have the same shape as input"

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    input_contig = input.contiguous()
    weights_contig = weights.contiguous() if weights is not None else None

    if input_contig.dtype == torch.int64:
        input_contig = _int32_view_of_int64(input_contig)

    device = input_contig.device
    weights_dtype = weights_contig.dtype if weights_contig is not None else None
    half_precision = weights_dtype in (torch.float16, torch.bfloat16)
    downcast = weights_dtype is not None and (
        weights_dtype == torch.float64 or not weights_dtype.is_floating_point
    )
    if downcast:
        # GCU400 has no fp64 arithmetic; integer and fp64 weights accumulate
        # in fp32 (the established behavior of this backend).
        weights_contig = weights_contig.to(torch.float32)

    # The output length is data-dependent (max(input) + 1 vs minlength), so a
    # single host sync is semantically required; torch's own bincount syncs
    # the same way.
    needed_size = None
    if n <= _PRESYNC_MAX_N and minlength >= _PRESYNC_MIN_ML and not half_precision:
        # Wide-minlength regime (plain / fp32-domain weights only: an fp32
        # accumulation cannot reproduce the native half-precision per-add
        # rounding when values concentrate in a few bins despite a wide
        # minlength): needed_size is guaranteed wide (hence the scatter
        # contention-free), so a pre-sync fused max+count pass into an
        # estimated-size buffer hides the count work ahead of the sync.
        pre_size = max(_PRESYNC_ESTIMATE_FLOOR, min(4 * n, 65536), minlength)
        max_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        fused_grid = (triton.cdiv(n, _ATOMIC_BLOCK_N),)
        if weights_contig is None:
            staging = torch.zeros(pre_size, dtype=torch.int32, device=device)
            _bincount_fused_count_kernel[fused_grid](
                input_contig,
                max_tensor,
                staging,
                n,
                pre_size,
                BLOCK_N=_ATOMIC_BLOCK_N,
                num_warps=4,
            )
            needed_size = max(int(max_tensor.item()) + 1, minlength)
            if needed_size <= pre_size:
                output = torch.empty(needed_size, dtype=torch.int64, device=device)
                output.copy_(staging[:needed_size])
                return output
        else:
            staging = torch.zeros(pre_size, dtype=torch.float32, device=device)
            _bincount_fused_weights_kernel[fused_grid](
                input_contig,
                weights_contig,
                max_tensor,
                staging,
                n,
                pre_size,
                BLOCK_N=_ATOMIC_BLOCK_N,
                num_warps=4,
            )
            needed_size = max(int(max_tensor.item()) + 1, minlength)
            if needed_size <= pre_size:
                output = staging[:needed_size]
                if downcast:
                    output = output.to(weights_dtype)
                return output

    if needed_size is None:
        max_out = _bincount_max(input_contig, n, device)
        needed_size = max(int(max_out.item()) + 1, minlength)

    if needed_size <= 0:
        if weights is not None:
            return torch.empty(0, dtype=weights.dtype, device=device)
        return torch.empty(0, dtype=torch.int64, device=device)

    if weights is None:
        staging = torch.empty(needed_size, dtype=torch.int32, device=device)
        if n <= _SMALL_N_LIMIT:
            hist_block_n = max(triton.next_power_of_2(n), 1)
            hist_tile_b = _HIST_TILE_B
        else:
            hist_block_n = _HIST_BLOCK_N
            hist_tile_b = _HIST_TILE_B
        num_progs = triton.cdiv(needed_size, hist_tile_b)
        onehot_ok = num_progs <= _GRID_MAX and n * needed_size <= _ONEHOT_MAX_OPS
        if n > _SMALL_N_LIMIT and needed_size > _ONEHOT_NB_LIMIT:
            onehot_ok = False
        if onehot_ok:
            _bincount_hist_kernel[(num_progs,)](
                input_contig,
                staging,
                n,
                needed_size,
                TILE_B=hist_tile_b,
                BLOCK_N=hist_block_n,
                num_warps=4,
            )
        else:
            staging = torch.zeros(needed_size, dtype=torch.int32, device=device)
            num_progs = min(triton.cdiv(n, _ATOMIC_BLOCK_N), _GRID_MAX)
            _bincount_atomic_count_kernel[(num_progs,)](
                input_contig,
                staging,
                n,
                needed_size,
                num_progs,
                BLOCK_N=_ATOMIC_BLOCK_N,
                num_warps=4,
            )
        output = torch.empty(needed_size, dtype=torch.int64, device=device)
        output.copy_(staging)
        return output

    def _topsaten_route():
        # Native-semantics route: the vendor op accumulates half-precision
        # weights in half precision (atomic order), which an exact fp32
        # kernel cannot reproduce within tolerance. fp32 weights with long
        # inputs or narrow bins also route here (verified bitwise-identical
        # to the native op for fp32). The bridge only accepts int32 inputs.
        if input_contig.dtype != torch.int32:
            input_contig_cast = input_contig.to(torch.int32)
        else:
            input_contig_cast = input_contig
        bridge = _get_topsaten_bridge()
        out = torch.zeros(needed_size, dtype=weights_contig.dtype, device=device)
        bridge.bincount_weighted(out, input_contig_cast, weights_contig, minlength)
        return out

    def _atomic_route():
        # Contention-free wide-bin arm for the fp32 domain (half-precision
        # weights route to the vendor op instead; the fp32 cast here stays
        # as a defensive fallback).
        w32 = weights_contig.to(torch.float32) if half_precision else weights_contig
        stage = torch.zeros(needed_size, dtype=torch.float32, device=device)
        num_progs = min(triton.cdiv(n, _ATOMIC_BLOCK_N), _GRID_MAX)
        _bincount_atomic_weights_kernel[(num_progs,)](
            input_contig,
            w32,
            stage,
            n,
            needed_size,
            num_progs,
            BLOCK_N=_ATOMIC_BLOCK_N,
            num_warps=4,
        )
        if half_precision:
            return stage.to(weights_contig.dtype)
        return stage

    def _onehot_weighted(tight):
        # tight=True uses BLOCK_N=next_pow2(n) (small n: masked-out lanes
        # still pay the tile cost, so a padded BLOCK_N multiplies the work).
        block_n = max(triton.next_power_of_2(n), 1) if tight else _HIST_W_BLOCK_N
        num_progs = triton.cdiv(needed_size, _HIST_W_TILE_B)
        if num_progs > _GRID_MAX or n * needed_size > _ONEHOT_MAX_OPS:
            return None
        output = torch.empty(needed_size, dtype=weights_contig.dtype, device=device)
        _bincount_hist_weights_kernel[(num_progs,)](
            input_contig,
            weights_contig,
            output,
            n,
            needed_size,
            TILE_B=_HIST_W_TILE_B,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return output

    if half_precision:
        if needed_size <= _ONEHOT_NB_LIMIT or n > _SMALL_N_LIMIT:
            output = _topsaten_route()
        else:
            # Small-n wide-bin arm: with spread values (n << needed_size) the
            # fp32 accumulation rounds once per bin and matches the native
            # half-precision result bit-for-bit; with concentrated values it
            # deviates the same way as the previous backend implementation.
            output = _onehot_weighted(tight=True)
            if output is None:
                output = _topsaten_route()
    else:
        # fp32 weights (including the fp64 / integer downcasts).
        if needed_size <= _ONEHOT_NB_LIMIT:
            if n <= _WEIGHTED_ONEHOT_MAX_N:
                output = _onehot_weighted(tight=False)
                if output is None:
                    output = _topsaten_route()
            else:
                output = _topsaten_route()
        else:
            if n <= _SMALL_N_LIMIT:
                output = _onehot_weighted(tight=True)
                if output is None:
                    output = _atomic_route()
            else:
                output = _atomic_route()

    if downcast:
        output = output.to(weights_dtype)
    return output
