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
"""Kunlunxin override for aten::trapezoid(Tensor y, Tensor x, int dim).

The generic implementation is a pointer kernel whose two shifts
``x[k+1]`` / ``x[k-1]`` defeat the XPU stride analysis, so it runs the row
reduction as a per-lane gather: benchmark shapes land at 0.002-0.05x of aten.
This override keeps the generic host-side validation/autograd/broadcasting
(rebinding only the compute entry point) and computes the common
``y.shape[dim] == x.shape[dim]`` case with

* a TLE (tensor-descriptor, cluster-DMA) weighted row reduce for fp16/fp32
  when the reduced length tiles exactly, and
* a fused ``[BLOCK_M, BLOCK_N]`` pointer kernel otherwise,

both of which read ``y`` once. Complex / per-row-spacing / mismatched-length /
int-dtype cases delegate to the untouched generic path.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems.ops.trapezoid as _base
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.sum import sum_dim as _vendor_sum_dim

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    import triton.experimental.tle.language as tle

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)

_BUFFER_SIZE_LIMIT = 2048
_FLAT_BLOCK = 32768


# ---------------------------------------------------------------------------
# generic-complex support: the vendor sum kernel is exact for real tensors,
# while the generic one is wrong for a middle-dim reduce on this backend.
# ---------------------------------------------------------------------------
def _sum_dim_any(grad, dim, keepdim=False):
    if not grad.is_complex():
        return _vendor_sum_dim(grad, dim=dim, keepdim=keepdim)
    real = torch.view_as_real(grad)
    summed = _vendor_sum_dim(real, dim=dim, keepdim=keepdim)
    return torch.view_as_complex(summed.contiguous())


_base.gems_sum_dim = _vendor_sum_dim
_base._sum_dim_any = _sum_dim_any


# ---------------------------------------------------------------------------
# TLE weighted row reduce (x is a 1-D spacing vector shared by every row)
# ---------------------------------------------------------------------------
if _HAS_TLE:

    @triton.jit
    def _tle_wsum_kernel(
        y_desc,
        x_desc,
        o_desc,
        N,
        XBLOCK: tl.constexpr,
        YBLOCK: tl.constexpr,
        IN_DTYPE: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_off = pid * XBLOCK
        a_lmem = tle.gpu.alloc(
            [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
        )
        w_lmem = tle.gpu.alloc([YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem)
        c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)
        row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
        col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
        a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
        w_ptrs = tle.gpu.local_ptr(w_lmem, (tl.arange(0, YBLOCK),))
        c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))
        acc = tl.zeros([XBLOCK], ACC_DTYPE)
        for coff in tl.range(0, N, YBLOCK):
            tle.gpu.copy(y_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
            tle.gpu.copy(x_desc, w_lmem, [YBLOCK], [coff])
            yv = tl.load(a_ptrs).to(ACC_DTYPE)
            wv = tl.load(w_ptrs).to(ACC_DTYPE)
            acc = acc + tl.sum(yv * wv[None, :], axis=1)
        tl.store(c_ptrs, acc.to(OUT_DTYPE))
        tle.gpu.copy(c_lmem, o_desc, [XBLOCK], [row_off])

    _TL_DTYPE = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    _TL_OUT = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }


def _tle_geometry(M, N, itemsize):
    """(XBLOCK, YBLOCK) with YBLOCK dividing N, or None to decline."""
    xblock = min(512, max(128, triton.next_power_of_2(-(-M // 8))))
    row_bytes = N * itemsize
    if row_bytes >= 32768:
        xblock = max(64, xblock // 4)
    elif row_bytes <= 2048:
        xblock = 512 if M > 64 else 256
    tile_elems = 4096 * 64 // itemsize
    yblock = max(1, min(triton.next_power_of_2(N), tile_elems // xblock))
    # Leave room in the local-memory budget for the weight/result buffers.
    while xblock * yblock * itemsize > 64 * 2048 and yblock > 1:
        yblock >>= 1
    while yblock > 1 and N % yblock != 0:
        yblock >>= 1
    if yblock < 32 or N % yblock != 0:
        return None
    return xblock, yblock


# ---------------------------------------------------------------------------
# pointer fallbacks
# ---------------------------------------------------------------------------
@triton.jit
def _weights_kernel(x_ptr, w_ptr, N, OUT_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    kp = tl.minimum(off + 1, N - 1)
    km = tl.maximum(off - 1, 0)
    xp = tl.load(x_ptr + kp, mask=mask, other=0.0).to(tl.float32)
    xm = tl.load(x_ptr + km, mask=mask, other=0.0).to(tl.float32)
    tl.store(w_ptr + off, (0.5 * (xp - xm)).to(OUT_DTYPE), mask=mask)


@triton.jit
def _row_x_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    row_ptrs = y_ptr + rows[:, None] * N
    for off in range(0, N, BLOCK_N):
        k = off + tl.arange(0, BLOCK_N)[None, :]
        col = k < N
        if EVEN_N and EVEN_M:
            yv = tl.load(row_ptrs + k).to(tl.float32)
        else:
            yv = tl.load(row_ptrs + k, mask=col & rmask[:, None], other=0.0).to(
                tl.float32
            )
        kp = tl.minimum(k + 1, N - 1)
        km = tl.maximum(k - 1, 0)
        xp = tl.load(x_ptr + kp, mask=col, other=0.0).to(tl.float32)
        xm = tl.load(x_ptr + km, mask=col, other=0.0).to(tl.float32)
        acc += yv * (0.5 * (xp - xm))
    s = tl.sum(acc, 1)
    tl.store(out_ptr + rows, s, mask=rmask)


@triton.jit
def _flat_x_kernel(
    y_ptr,
    x_ptr,
    mid_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    col = off < N
    yv = tl.load(y_ptr + off, mask=col, other=0.0).to(tl.float32)
    kp = tl.minimum(off + 1, N - 1)
    km = tl.maximum(off - 1, 0)
    xp = tl.load(x_ptr + kp, mask=col, other=0.0).to(tl.float32)
    xm = tl.load(x_ptr + km, mask=col, other=0.0).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(yv * (0.5 * (xp - xm))))


@triton.jit
def _mid_sum_kernel(mid_ptr, out_ptr, LS, BLOCK_MID: tl.constexpr):
    mid = tl.arange(0, BLOCK_MID)
    v = tl.load(mid_ptr + mid, mask=mid < LS, other=0.0)
    tl.store(out_ptr, tl.sum(v))


_orig_impl = _base._trapezoid_x_impl


def _trapezoid_x_impl(y, x, dim):
    # Keep the generic validation/error contract unchanged.
    dim = _base._normalize_dim(y.dim(), dim)
    _base._check_bool(y, x)
    _base._check_complex_dtype(y, x)
    if _base._is_complex(y, x):
        return _orig_impl(y, x, dim)

    out_dtype = _base._result_dtype(y, x)
    if y.shape[dim] == 0:
        return torch.zeros(
            _base._zeros_shape(y.shape, dim), dtype=out_dtype, device=y.device
        )
    if out_dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return _orig_impl(y, x, dim)

    x_viewed = _base._view_x(x, y, dim)
    N = y.shape[dim]
    Nx = x_viewed.shape[dim]
    if N != Nx or N <= 1 or x.dim() != 1:
        # Mismatched pair lengths, degenerate sample counts, and per-row
        # spacing keep the generic implementation (not on the benchmark path).
        return _orig_impl(y, x, dim)

    full_shape = torch.broadcast_shapes(y.shape, x_viewed.shape)
    out_shape = full_shape[:dim] + full_shape[dim + 1 :]
    y2 = y.broadcast_to(full_shape).movedim(dim, -1).contiguous().reshape(-1, N)
    M = y2.shape[0]
    if M == 0:
        return torch.zeros(out_shape, dtype=out_dtype, device=y.device)
    x2 = x.contiguous()

    if M == 1:
        # A single row: split the flat vector across programs, then fold.
        block = min(_FLAT_BLOCK, triton.next_power_of_2(N))
        lsize = triton.cdiv(N, block)
        mid = torch.empty((lsize,), dtype=torch.float32, device=y.device)
        out = torch.empty((), dtype=out_dtype, device=y.device)
        with torch_device_fn.device(y.device):
            _flat_x_kernel[(lsize,)](
                y2, x2, mid, N, BLOCK=block, buffer_size_limit=_BUFFER_SIZE_LIMIT
            )
            _mid_sum_kernel[(1,)](
                mid,
                out,
                lsize,
                BLOCK_MID=triton.next_power_of_2(lsize),
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        return out.reshape(out_shape)

    geom = None
    if _HAS_TLE and out_dtype != torch.bfloat16:
        geom = _tle_geometry(M, N, y2.element_size())
    if geom is not None:
        xblock, yblock = geom
        out = torch.empty((M,), dtype=out_dtype, device=y.device)
        it = _TL_DTYPE[out_dtype]
        # The kernel consumes 0.5*(x[k+1]-x[k-1]); build that weight vector once
        # (O(N) vs the O(M*N) reduction) so the DMA path reads it as a plain tile.
        wv = torch.empty((N,), dtype=x2.dtype, device=y.device)
        with torch_device_fn.device(y.device):
            _weights_kernel[(triton.cdiv(N, 1024),)](
                x2,
                wv,
                N,
                OUT_DTYPE=_TL_DTYPE[x2.dtype],
                BLOCK=1024,
                num_warps=4,
            )
        y_desc = TensorDescriptor.from_tensor(y2, [xblock, yblock])
        x_desc = TensorDescriptor.from_tensor(wv, [yblock])
        o_desc = TensorDescriptor.from_tensor(out, [xblock])
        with torch_device_fn.device(y.device):
            _tle_wsum_kernel[(triton.cdiv(M, xblock),)](
                y_desc,
                x_desc,
                o_desc,
                N,
                XBLOCK=xblock,
                YBLOCK=yblock,
                IN_DTYPE=it,
                ACC_DTYPE=tl.float32,
                OUT_DTYPE=it,
            )
        return out.reshape(out_shape)

    bm = 64 if M >= 64 else triton.next_power_of_2(M)
    bn = min(2048, triton.next_power_of_2(N))
    out = torch.empty((M,), dtype=out_dtype, device=y.device)
    with torch_device_fn.device(y.device):
        _row_x_kernel[(triton.cdiv(M, bm),)](
            y2,
            x2,
            out,
            M,
            N,
            BLOCK_M=bm,
            BLOCK_N=bn,
            EVEN_N=(N % bn == 0),
            EVEN_M=(M % bm == 0),
            num_warps=4,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
    return out.reshape(out_shape)


_base._trapezoid_x_impl = _trapezoid_x_impl


def trapezoid_x(y, x, dim=-1):
    # Reuse the generic autograd Function; only the forward compute changed.
    return _base.trapezoid_x(y, x, dim)
