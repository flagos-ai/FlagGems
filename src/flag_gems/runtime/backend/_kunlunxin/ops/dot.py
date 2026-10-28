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

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.sum import _launch_sum_flat
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle.language as tle  # noqa: F401
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

# Small N uses the fp32-accumulating 512-lane tree (accurate for fp16 against the
# fp64 reference); large N uses a fused TLE row-reduce that accumulates x*y in fp32
# without materialising the product, landing near/above aten. The tree below is kept
# only as a fallback for backends without the TLE extension.
SMALL_BLOCK = 512
WIDE_BLOCK = 8192
SMALL_N = 8192
DOT_UNROLL = 8
# Fused TLE dot geometry: a [rows, cols] row-reduce; 256x128 is the measured optimum.
_TLE_DOT_COLS = 4096
_TLE_DOT_XBLOCK = 256
_TLE_DOT_YBLOCK = 128
_TLE_DOT_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


def _tle_available():
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()


@libentry()
@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(x * y))


@libentry()
@triton.jit
def dot_kernel_1(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


@libentry()
@triton.jit
def dot_sum_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(in_ptr + offsets, mask=offsets < N, other=0.0)
    tl.store(out_ptr + pid, tl.sum(values))


@libentry()
@triton.jit
def dot_kernel_2(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < M
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(mid_val))


# Unmasked bulk kernels: one 8192-lane load+reduce per program.
@libentry()
@triton.jit
def dot_kernel_wide(x_ptr, y_ptr, mid_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


# Batched bulk kernel: one program reduces DOT_UNROLL independent 8192-lane
# unmasked tiles, storing ONE mid slot per tile (mid layout identical to the
# unbatched kernel, so the reduction tree below is unchanged).
@libentry()
@triton.jit
def dot_kernel_wide_batch(
    x_ptr, y_ptr, mid_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr
):
    pid = ext.program_id(0)
    base_off = pid * (BLOCK * UNROLL)
    for j in tl.static_range(UNROLL):
        offs = base_off + j * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
        tl.store(mid_ptr + pid * UNROLL + j, tl.sum(x * y))


@libentry()
@triton.jit
def dot_kernel_512(x_ptr, y_ptr, mid_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


# Exact power-of-two unmasked tile: never reads past the end, never masks.
@libentry()
@triton.jit
def dot_kernel_tile(x_ptr, y_ptr, out_ptr, TILE: tl.constexpr):
    offs = tl.arange(0, TILE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(out_ptr, tl.sum(x * y))


@libentry()
@triton.jit
def sum_kernel_wide(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(v))


@libentry()
@triton.jit
def sum_kernel_wide_batch(in_ptr, out_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = ext.program_id(0)
    base_off = pid * (BLOCK * UNROLL)
    for j in tl.static_range(UNROLL):
        offs = base_off + j * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(in_ptr + offs).to(tl.float32)
        tl.store(out_ptr + pid * UNROLL + j, tl.sum(v))


@libentry()
@triton.jit
def sum_kernel_512(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(v))


@libentry()
@triton.jit
def sum_kernel_tile(in_ptr, out_ptr, TILE: tl.constexpr):
    offs = tl.arange(0, TILE)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr, tl.sum(v))


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["x_desc", "y_desc", "c_desc"],
)
def dot_kernel_tle(
    x_desc,
    y_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * XBLOCK
    x_lmem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=IN_DTYPE, scope=tle.gpu.lmem)
    y_lmem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=IN_DTYPE, scope=tle.gpu.lmem)
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, scope=tle.gpu.lmem)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    x_ptrs = tle.gpu.local_ptr(x_lmem, (row_ids, col_ids))
    y_ptrs = tle.gpu.local_ptr(y_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))
    acc = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(x_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
                tl.store(y_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(x_desc, x_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        tle.gpu.copy(y_desc, y_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        acc = acc + tl.sum(
            tl.load(x_ptrs).to(ACC_DTYPE) * tl.load(y_ptrs).to(ACC_DTYPE), axis=1
        )
    tl.store(c_ptrs, acc.to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _dot_tle(x, y, out, N):
    """Fused TLE dot: [rows, cols] row-reduce of x*y, then flat-sum the partials."""
    rows, tail = divmod(N, _TLE_DOT_COLS)
    xb, yb = _TLE_DOT_XBLOCK, _TLE_DOT_YBLOCK
    need_zero = _TLE_DOT_COLS % yb != 0
    count = rows + (1 if tail else 0)
    partials = torch.empty((count,), dtype=torch.float32, device=x.device)
    if rows:
        x_main = x[: rows * _TLE_DOT_COLS].view(rows, _TLE_DOT_COLS)
        y_main = y[: rows * _TLE_DOT_COLS].view(rows, _TLE_DOT_COLS)
        dot_kernel_tle[(-(-rows // xb),)](
            TensorDescriptor.from_tensor(x_main, block_shape=[xb, yb]),
            TensorDescriptor.from_tensor(y_main, block_shape=[xb, yb]),
            TensorDescriptor.from_tensor(partials[:rows], block_shape=[xb]),
            _TLE_DOT_COLS,
            xb,
            yb,
            _TLE_DOT_TL_DTYPE[x.dtype],
            tl.float32,
            tl.float32,
            need_zero,
        )
    if tail:
        t = torch.empty((), dtype=torch.float32, device=x.device)
        _dot_small(x[rows * _TLE_DOT_COLS :], y[rows * _TLE_DOT_COLS :], t, tail)
        partials[rows] = t
    _launch_sum_flat(partials, out, torch.float32)


def _pow2_decomp(v):
    """Exact power-of-two aligned slices covering [0, v) with no overlap."""
    out = []
    off = 0
    while v:
        size = 1 << (v.bit_length() - 1)
        out.append((off, size))
        off += size
        v -= size
    return out


def _dot_large(x, y, out, N):
    """Large-N dot: fused TLE row-reduce when available, else the masked tree."""
    if _TLE_AVAILABLE and x.dtype in _TLE_DOT_TL_DTYPE:
        _dot_tle(x, y, out, N)
        return
    full, rem = divmod(N, WIDE_BLOCK)
    t512, tiny = divmod(rem, SMALL_BLOCK)
    t_tiles = _pow2_decomp(tiny)
    count = full + t512 + len(t_tiles)
    mid = torch.empty((count,), dtype=torch.float32, device=x.device)
    if full:
        b, r = divmod(full, DOT_UNROLL)
        if b:
            dot_kernel_wide_batch[(b,)](x, y, mid, WIDE_BLOCK, DOT_UNROLL)
        if r:
            dot_kernel_wide[(r,)](
                x[b * WIDE_BLOCK * DOT_UNROLL :],
                y[b * WIDE_BLOCK * DOT_UNROLL :],
                mid[b * DOT_UNROLL :],
                WIDE_BLOCK,
            )
    if t512:
        dot_kernel_512[(t512,)](
            x[full * WIDE_BLOCK :], y[full * WIDE_BLOCK :], mid[full:], SMALL_BLOCK
        )
    base = full + t512
    for j, (off, size) in enumerate(t_tiles):
        dot_kernel_tile[(1,)](
            x[full * WIDE_BLOCK + t512 * SMALL_BLOCK + off :],
            y[full * WIDE_BLOCK + t512 * SMALL_BLOCK + off :],
            mid[base + j :],
            size,
        )
    mid_size = count
    while mid_size > SMALL_BLOCK:
        full2, rem2 = divmod(mid_size, WIDE_BLOCK)
        t512b, tiny2 = divmod(rem2, SMALL_BLOCK)
        t2 = _pow2_decomp(tiny2)
        cnt2 = full2 + t512b + len(t2)
        nm = torch.empty((cnt2,), dtype=torch.float32, device=x.device)
        if full2:
            b2, r2 = divmod(full2, DOT_UNROLL)
            if b2:
                sum_kernel_wide_batch[(b2,)](mid, nm, WIDE_BLOCK, DOT_UNROLL)
            if r2:
                sum_kernel_wide[(r2,)](
                    mid[b2 * WIDE_BLOCK * DOT_UNROLL :],
                    nm[b2 * DOT_UNROLL :],
                    WIDE_BLOCK,
                )
        if t512b:
            sum_kernel_512[(t512b,)](mid[full2 * WIDE_BLOCK :], nm[full2:], SMALL_BLOCK)
        b2 = full2 + t512b
        for j, (off, size) in enumerate(t2):
            sum_kernel_tile[(1,)](
                mid[full2 * WIDE_BLOCK + t512b * SMALL_BLOCK + off :],
                nm[b2 + j :],
                size,
            )
        mid = nm
        mid_size = cnt2
    acc = torch.zeros((), dtype=torch.float32, device=x.device)
    for off, size in _pow2_decomp(mid_size):
        t = torch.empty((), dtype=torch.float32, device=x.device)
        sum_kernel_tile[(1,)](mid[off:], t, size)
        acc += t
    out.copy_(acc)


def _dot_small(x, y, out, N):
    """Historical 512-lane masked-tree path (N <= SMALL_N)."""
    if N <= SMALL_BLOCK:
        block_size = triton.next_power_of_2(N) if N else 1
        dot_kernel[(1,)](x, y, out, N, block_size)
        return
    mid_size = triton.cdiv(N, SMALL_BLOCK)
    mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
    dot_kernel_1[(mid_size,)](x, y, mid, N, SMALL_BLOCK)
    while mid_size > SMALL_BLOCK:
        next_size = triton.cdiv(mid_size, SMALL_BLOCK)
        next_mid = torch.empty((next_size,), dtype=torch.float32, device=x.device)
        dot_sum_kernel[(next_size,)](mid, next_mid, mid_size, SMALL_BLOCK)
        mid = next_mid
        mid_size = next_size
    dot_kernel_2[(1,)](mid, out, mid_size, triton.next_power_of_2(mid_size))


def dot(x, y):
    logger.debug("GEMS_KUNLUNXIN DOT")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]
    out = torch.empty([], dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        if N == 0:
            out.zero_()
        elif N <= SMALL_N:
            _dot_small(x, y, out, N)
        else:
            _dot_large(x, y, out, N)

    return out
