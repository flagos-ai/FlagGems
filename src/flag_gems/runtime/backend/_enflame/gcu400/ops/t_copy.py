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

"""gcu400 t_copy: materialized 2D transpose.

Large-shape primary path: ctypes direct call into ``libtopsaten``
``topsatenCopy`` with the input presented as its own transposed view
(shape (N, M), strides (x.stride(1), x.stride(0))) -- the same
materialization entry point torch_gcu dispatches ``aten::t_copy`` to, i.e.
native-kernel performance minus the aten dispatch overhead.  (``topsatenT``
and ``topsatenPermute`` are view ops and never write data.)  A Triton tile
transpose on this backend tops out around 0.78-0.80x of native at
(4096,4096): the transpose readout through DSM costs ~0.25ns/KB on top of
the DRAM stream, which the tile shape cannot avoid (square, 256x512 and
512x256 tiles all measured within noise of each other).

Small shapes (numel < 65536) and every case where libtopsaten is
unavailable keep the Triton path: launch overhead dominates there and the
Triton launcher is on par with native dispatch.

Triton fallback ladder (also used for non-float dtypes):
1. persistent unmasked tile transpose -- grid pinned to <=48 programs, each
   looping over tiles.  Masked 2D transpose tiles are silently wrong on
   this backend for remainder shapes (P15: bf16 (2,3) fails with the
   upstream 32x32 masked kernel), so the tile kernel runs unmasked and only
   for exact power-of-two divisions of both dims.
2. flat gather kernel (div/mod per output element) -- correct for any
   strides, int32-safe sizes only.
3. legacy masked 2D kernel with int64 offsets -- only for >=2^31 elements,
   preserving the pre-rewrite behavior for out-of-int32-range tensors.
"""

import ctypes
import logging
import threading

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# topsaten direct-call runtime (ctypes binding, no compiled extension)
# ---------------------------------------------------------------------------

# topsatenT / topsatenPermute are VIEW ops (they never write data); the
# materializing entry is topsatenCopy with the input presented as its own
# transposed view: shape (N, M), strides (x.stride(1), x.stride(0)).
_COPY_SYM = "_ZN8topsaten12topsatenCopyER14topsatenTensorRKS0_bP12EfiStream_st"
_TENSOR_CTOR_SYM = (
    "_ZN14topsatenTensorC1ERK14topsatenSize_tS2_RK18topsatenDataType_tRKPv"
)
_TENSOR_SET_DATA_SYM = "_ZN14topsatenTensor13SetTensorDataERKPv"

# topsatenDataType_t subset used by this op.
_DTYPES = {
    torch.float16: 4,
    torch.bfloat16: 5,
    torch.float32: 8,
}


class _TopsatenSize(ctypes.Structure):
    """``struct topsatenSize_t { const int64_t* data; int64_t len; }``"""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int64)),
        ("len", ctypes.c_int64),
    ]


class _TensorBuf(ctypes.Structure):
    """16-byte storage of ``topsatenTensor`` (``shared_ptr<Impl>``)."""

    _fields_ = [("ptr", ctypes.c_void_p), ("ctrl", ctypes.c_void_p)]


class _Runtime:
    """Lazy libtopsaten load + symbol binding; caches the failure."""

    def __init__(self):
        self.lib = None
        self.ctor = None
        self.set_data = None
        self.copy = None
        self.error = None

    def load(self):
        if self.lib is not None:
            return self
        if self.error is not None:
            raise RuntimeError(f"topsaten runtime unavailable: {self.error}")

        lib = None
        last_err = None
        for name in ("libtopsaten.so.3", "libtopsaten.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                last_err = exc
        if lib is None:
            self.error = f"cannot load libtopsaten ({last_err})"
            raise RuntimeError(self.error)

        try:
            self.ctor = lib[_TENSOR_CTOR_SYM]
            self.ctor.argtypes = [
                ctypes.POINTER(_TensorBuf),
                ctypes.POINTER(_TopsatenSize),
                ctypes.POINTER(_TopsatenSize),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.ctor.restype = None

            self.set_data = lib[_TENSOR_SET_DATA_SYM]
            self.set_data.argtypes = [
                ctypes.POINTER(_TensorBuf),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.set_data.restype = None

            self.copy = lib[_COPY_SYM]
            self.copy.argtypes = [
                ctypes.POINTER(_TensorBuf),
                ctypes.POINTER(_TensorBuf),
                ctypes.c_bool,
                ctypes.c_void_p,
            ]
            self.copy.restype = ctypes.c_int
        except KeyError as exc:
            self.error = f"libtopsaten symbol missing: {exc}"
            raise RuntimeError(self.error)

        self.lib = lib
        return self


_RUNTIME = _Runtime()
_RUNTIME_LOCK = threading.Lock()
_WARNED = False

# topsatenTensor descriptor pool: the destructor is not exported, so
# descriptors are pooled per (shape, dtype, strides) and rebind via
# SetTensorData.  ``claim`` (per call) keeps two same-signature tensors from
# aliasing.
_POOL = {}
# (shape, dtype, strides) -> list of [handle, dims_array, strides_array, data_ptr]

_GET_RAW_STREAM = None


def _current_stream(device_index):
    global _GET_RAW_STREAM
    if _GET_RAW_STREAM is None:
        from torch_gcu import _C

        _GET_RAW_STREAM = getattr(_C, "_gcu_getCurrentRawStream")
    return ctypes.c_void_p(int(_GET_RAW_STREAM(device_index)))


def _descriptor(tensor, shape, strides, claim):
    """Bind ``tensor``'s storage as a ``topsatenTensor`` with the given view."""
    data_ptr = tensor.data_ptr()
    key = (shape, tensor.dtype, strides)
    pool = _POOL.setdefault(key, [])
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
        _RUNTIME.set_data(handle, ctypes.byref(data))
        entry[3] = data_ptr
        claim.add(id(handle))
        return handle

    dims = (ctypes.c_int64 * len(shape))(*shape)
    strd_arr = (ctypes.c_int64 * len(strides))(*strides)
    sizes = _TopsatenSize(dims, len(shape))
    strides_size = _TopsatenSize(strd_arr, len(strides))
    dtype = ctypes.c_int(_DTYPES[tensor.dtype])
    data = ctypes.c_void_p(data_ptr)
    buf = _TensorBuf()
    _RUNTIME.ctor(
        ctypes.byref(buf),
        ctypes.byref(sizes),
        ctypes.byref(strides_size),
        ctypes.byref(dtype),
        ctypes.byref(data),
    )
    handle = ctypes.pointer(buf)
    pool.append([handle, dims, strd_arr, data_ptr])
    claim.add(id(handle))
    return handle


def _topsaten_t_copy(inp, out):
    """2D t_copy via ``topsatenCopy`` from a transposed view; True on success."""
    global _WARNED
    if inp.dtype not in _DTYPES:
        return False
    if not (inp.is_contiguous() and out.is_contiguous()):
        return False
    try:
        with _RUNTIME_LOCK:
            _RUNTIME.load()
    except RuntimeError as exc:
        if not _WARNED:
            _WARNED = True
            logger.warning("t_copy: %s; using Triton fallback", exc)
        return False

    device_index = inp.device.index if inp.device.index is not None else 0
    try:
        stream = _current_stream(device_index)
        M, N = inp.shape
        claim = set()
        in_handle = _descriptor(inp, (N, M), (inp.stride(1), inp.stride(0)), claim)
        out_handle = _descriptor(out, (N, M), tuple(out.stride()), claim)
        status = _RUNTIME.copy(out_handle, in_handle, False, stream)
    except Exception as exc:
        if not _WARNED:
            _WARNED = True
            logger.warning("t_copy: topsatenCopy call failed: %s", exc)
        return False
    if status != 0:
        if not _WARNED:
            _WARNED = True
            logger.warning("t_copy: topsatenCopy returned status=%d", status)
        return False
    return True


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def t_copy_2d_kernel(
    in_ptr,
    out_ptr,
    in_stride_0,
    in_stride_1,
    out_stride_0,
    out_stride_1,
    M,  # input dim0
    N,  # input dim1
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # corresponds to out rows [0..N)
    j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # corresponds to out cols [0..M)

    i64 = i.to(tl.int64)[None, :]  # shape [1, BM]
    j64 = j.to(tl.int64)[:, None]  # shape [BN, 1]

    # out shape = (N, M)
    mask = (i64 < N) & (j64 < M)

    # in index = (j, i) -> in_offset = j*in_stride_0 + i*in_stride_1
    in_offsets = j64 * in_stride_0 + i64 * in_stride_1
    # out index = (i, j) -> out_offset = i*out_stride_0 + j*out_stride_1
    out_offsets = i64 * out_stride_0 + j64 * out_stride_1

    x = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, x, mask=mask)


@triton.jit
def copy_1d_strided_kernel(
    in_ptr,
    out_ptr,
    in_stride,
    out_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    offs64 = offs.to(tl.int64)
    in_idx = offs64 * in_stride
    out_idx = offs64 * out_stride
    x = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + out_idx, x, mask=mask)


@triton.jit
def t_copy_flat_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def t_copy_gather_2d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    M,  # input dim0 == out dim1
    in_stride_0,
    in_stride_1,
    out_stride_0,
    out_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # out is (N, M): linear k -> (a = k // M row, b = k % M col);
    # out[a, b] = in[b, a]
    a = offsets // M
    b = offsets % M
    in_offs = b * in_stride_0 + a * in_stride_1
    out_offs = a * out_stride_0 + b * out_stride_1
    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + out_offs, vals, mask=mask)


@triton.jit
def t_copy_tile_persist_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    num_n,
    num_tiles,
    num_progs,
    in_row_stride,
    out_row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Persistent form: grid is pinned to num_progs programs (<= 48, so the
    # P19 grid limits never apply) and each program strides over tiles.
    pid = tl.program_id(0)
    for t in range(pid, num_tiles, num_progs):
        pid_m = t // num_n
        pid_n = t % num_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # x dim0 -> out dim1
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # x dim1 -> out dim0
        # loads run along the stride-1 dim of the input, stores along the
        # stride-1 dim of the output
        in_offs = rows[:, None] * in_row_stride + cols[None, :]
        vals = tl.load(in_ptr + in_offs)
        out_offs = cols[None, :] * out_row_stride + rows[:, None]
        tl.store(out_ptr + out_offs, vals)


def _launch_tile_persistent(inp, out, M, N, in_s0, out_s0):
    numel = M * N
    if numel >= 65536:
        cap = 256
        if inp.dtype == torch.float32:
            num_progs, num_warps = 24, 2
        else:
            num_progs, num_warps = 48, 4
    else:
        cap = 64
        num_progs, num_warps = 48, 2
    # largest power of two dividing each dim, capped: exact division is then
    # guaranteed, so the tile kernel can run unmasked (P15)
    block_m = min(cap, M & -M)
    block_n = min(cap, N & -N)
    if block_m * block_n < 64:
        return False
    num_m = M // block_m
    num_n = N // block_n
    num_tiles = num_m * num_n
    grid = (min(num_progs, num_tiles),)
    t_copy_tile_persist_kernel[grid](
        inp,
        out,
        M,
        N,
        num_n,
        num_tiles,
        grid[0],
        in_s0,
        out_s0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return True


def _launch_t_copy_kernel(inp: torch.Tensor, out: torch.Tensor):
    if inp.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"t_copy kernels require {flag_gems.device} tensors")
    assert inp.dtype == out.dtype, "dtype mismatch between input and output"

    dim = inp.dim()
    if dim == 0:
        # Scalar copy
        n = 1
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        copy_1d_strided_kernel[grid](
            inp,
            out,
            0,
            0,
            n,
            BLOCK_SIZE=1,
        )
    elif dim == 1:
        n = inp.numel()
        if n == 0:
            return
        in_stride = inp.stride(0)
        out_stride = out.stride(0)
        assert out.numel() == n, "Output size mismatch for 1D t_copy"
        if in_stride == 1 and out_stride == 1:
            grid = (triton.cdiv(n, 4096),)
            t_copy_flat_kernel[grid](
                inp,
                out,
                n,
                BLOCK_SIZE=4096,
                num_warps=8,
            )
        else:
            grid = (triton.cdiv(n, 1024),)
            copy_1d_strided_kernel[grid](
                inp,
                out,
                in_stride,
                out_stride,
                n,
                BLOCK_SIZE=1024,
            )
    elif dim == 2:
        M, N = inp.shape  # input dims
        # out should be (N, M)
        assert (
            out.dim() == 2 and out.shape[0] == N and out.shape[1] == M
        ), "Output shape must be (input.size(1), input.size(0)) for t_copy"
        numel = M * N
        if numel == 0:
            return

        # Large shapes: topsatenT direct call == native kernel.
        if numel >= 65536 and _topsaten_t_copy(inp, out):
            return

        in_s0, in_s1 = inp.stride()
        out_s0, out_s1 = out.stride()

        # A column-major input (a .t() view of contiguous storage) makes the
        # transpose a flat linear copy when the output is row-major.
        if in_s0 == 1 and in_s1 == M and out_s0 == M and out_s1 == 1:
            grid = (triton.cdiv(numel, 4096),)
            t_copy_flat_kernel[grid](
                inp,
                out,
                numel,
                BLOCK_SIZE=4096,
                num_warps=8,
            )
            return

        # Unmasked tile transpose: needs stride-1 innermost dims on both
        # sides and int32-safe offsets.
        if (
            in_s1 == 1
            and out_s1 == 1
            and (M - 1) * in_s0 + (N - 1) < 2**31
            and (N - 1) * out_s0 + (M - 1) < 2**31
        ):
            if _launch_tile_persistent(inp, out, M, N, in_s0, out_s0):
                return

        # Flat gather: correct for any strides, int32-safe sizes.
        if (
            numel < 2**31
            and (M - 1) * in_s0 + (N - 1) * in_s1 < 2**31
            and (N - 1) * out_s0 + (M - 1) * out_s1 < 2**31
        ):
            grid = (triton.cdiv(numel, 1024),)
            t_copy_gather_2d_kernel[grid](
                inp,
                out,
                numel,
                M,
                in_s0,
                in_s1,
                out_s0,
                out_s1,
                BLOCK_SIZE=1024,
            )
            return

        # Legacy masked kernel (int64 offsets) for out-of-int32-range sizes.
        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_M"]),
            triton.cdiv(M, meta["BLOCK_N"]),
        )
        t_copy_2d_kernel[grid](
            inp,
            out,
            in_s0,
            in_s1,
            out_s0,
            out_s1,
            M,
            N,
            BLOCK_M=32,
            BLOCK_N=32,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")


def t_copy_out(
    input: torch.Tensor,
    out: torch.Tensor,
    memory_format: torch.memory_format | None = None,
):
    logger.debug("GEMS_ENFLAME T_COPY_OUT")
    _launch_t_copy_kernel(input, out)
    return out


def t_copy(input: torch.Tensor, memory_format: torch.memory_format | None = None):
    logger.debug("GEMS_ENFLAME T_COPY")
    dim = input.dim()
    if dim == 0:
        out = torch.empty((), dtype=input.dtype, device=input.device)
    elif dim == 1:
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    elif dim == 2:
        M, N = input.shape
        out = torch.empty(
            (N, M),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.contiguous_format,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    _launch_t_copy_kernel(input, out)
    return out
