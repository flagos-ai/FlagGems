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

import torch
import triton
import triton.language as tl

MAX_DIMS = tl.constexpr(8)
BLOCK = 1024
SCATTER_BLOCK = 512


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
def _to_int_list(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, int):
        x = [x]
    return [int(v) for v in x]


def _to_int(x):
    if isinstance(x, torch.Tensor):
        x = x.item()
    return int(x)


def _prod(seq):
    p = 1
    for s in seq:
        p *= s
    return p


def _is_dense(size, stride):
    D = len(size)
    if D == 0:
        return False
    w = 1
    for k in range(D - 1, -1, -1):
        if stride[k] != w:
            return False
        w *= size[k]
    return True


def _exact_overlap(size, stride):
    D = len(size)
    seen = {}
    V = 1 if D == 0 else _prod(size)
    for v in range(V):
        t = v
        o = 0
        for k in range(D - 1, -1, -1):
            o += (t % size[k]) * stride[k]
            t //= size[k]
        if o in seen:
            return True
        seen[o] = 1
    return False


def _view_may_overlap(size, stride):
    """Sufficient non-overlap test; exact check for small V when uncertain."""
    D = len(size)
    if D == 0:
        return False
    order = sorted(range(D), key=lambda k: abs(stride[k]))
    span = 0
    for k in order:
        st = abs(stride[k])
        if st <= span:
            V = _prod(size)
            if V <= 65536:
                return _exact_overlap(size, stride)
            return True  # conservative
        span += (size[k] - 1) * st
    return False


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _copy_storage_kernel(self_ptr, buf_ptr, self_so, P, BLOCK: tl.constexpr):
    """buf[p] = self_storage[p] for p in [0, P). self_ptr + (p - self_so)
    addresses the underlying base storage for every p (negative offsets land
    before self's data pointer but inside the same allocation)."""
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < P
    val = tl.load(self_ptr + (p - self_so), mask=mask)
    tl.store(buf_ptr + p, val, mask=mask)


@triton.jit
def _fused_dense_kernel(
    self_ptr,
    src_ptr,
    buf_ptr,
    self_so,
    N,
    so,
    V,
    P,
    FULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < P
    if FULL:
        # so == 0 and V == N and self_so == 0 and buf_size == N: pure copy
        src_val = tl.load(src_ptr + p, mask=mask)
        src_val = src_val.to(buf_ptr.dtype.element_ty)
        tl.store(buf_ptr + p, src_val, mask=mask)
    else:
        j = p - self_so
        in_log = (j >= 0) & (j < N)
        covered = in_log & (j >= so) & (j < so + V)
        self_val = tl.load(self_ptr + j, mask=mask)
        src_val = tl.load(src_ptr + (j - so), mask=mask & covered)
        src_val = src_val.to(buf_ptr.dtype.element_ty)
        val = tl.where(covered, src_val, self_val)
        tl.store(buf_ptr + p, val, mask=mask)


@triton.jit
def _fused_1d_kernel(
    self_ptr,
    src_ptr,
    buf_ptr,
    self_so,
    N,
    so,
    P,
    s: tl.constexpr,
    size0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < P
    j = p - self_so
    in_log = (j >= 0) & (j < N)
    q = j - so
    covered = in_log & (q >= 0) & (q < s * size0) & (q % s == 0)
    self_val = tl.load(self_ptr + j, mask=mask)
    src_val = tl.load(src_ptr + (q // s), mask=mask & covered)
    src_val = src_val.to(buf_ptr.dtype.element_ty)
    val = tl.where(covered, src_val, self_val)
    tl.store(buf_ptr + p, val, mask=mask)


@triton.jit
def _fused_2d_g_kernel(
    self_ptr,
    src_ptr,
    buf_ptr,
    self_so,
    N,
    so,
    P,
    st0: tl.constexpr,
    g: tl.constexpr,
    size0: tl.constexpr,
    size1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """2-D view with innermost stride g: offset = i0*st0 + i1*g.
    Requires st0 % g == 0 and st0 // g >= size1 (non-overlapping)."""
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < P
    j = p - self_so
    in_log = (j >= 0) & (j < N)
    q = j - so
    u = q // g
    unit = st0 // g
    i1 = u % unit
    i0 = u // unit
    covered = in_log & (q >= 0) & (q % g == 0) & (i0 < size0) & (i1 < size1)
    self_val = tl.load(self_ptr + j, mask=mask)
    src_val = tl.load(src_ptr + (i0 * size1 + i1), mask=mask & covered)
    src_val = src_val.to(buf_ptr.dtype.element_ty)
    val = tl.where(covered, src_val, self_val)
    tl.store(buf_ptr + p, val, mask=mask)


@triton.jit
def _fused_scalar_kernel(
    self_ptr, src_ptr, buf_ptr, self_so, N, so, P, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < P
    j = p - self_so
    in_log = (j >= 0) & (j < N)
    covered = in_log & (j == so)
    self_val = tl.load(self_ptr + j, mask=mask)
    src_val = tl.load(src_ptr + tl.zeros_like(j), mask=mask & covered)
    src_val = src_val.to(buf_ptr.dtype.element_ty)
    val = tl.where(covered, src_val, self_val)
    tl.store(buf_ptr + p, val, mask=mask)


@triton.jit
def _scatter_general_kernel(
    src_ptr,
    buf_ptr,
    src_so,
    V,
    base_p,
    P,
    view_size_ptr,
    view_stride_ptr,
    view_D,
    SRC_CONTIG: tl.constexpr,
    src_stride_ptr,
    BLOCK: tl.constexpr,
):
    """Scatter into storage-domain buffer: buf[base_p + view_offset(v)] = src[v]."""
    pid = tl.program_id(0)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    mask = v < V
    idx = v
    p = base_p + tl.zeros_like(v)
    for kk in tl.static_range(MAX_DIMS):
        k = MAX_DIMS - 1 - kk
        if k < view_D:
            sk = tl.load(view_size_ptr + k)
            stk = tl.load(view_stride_ptr + k)
            ik = idx % sk
            idx = idx // sk
            p += ik * stk
    if SRC_CONTIG:
        src_off = src_so + v
    else:
        src_off = src_so + tl.zeros_like(v)
        idx2 = v
        for kk in tl.static_range(MAX_DIMS):
            k = MAX_DIMS - 1 - kk
            if k < view_D:
                sk = tl.load(view_size_ptr + k)
                sst = tl.load(src_stride_ptr + k)
                ik = idx2 % sk
                idx2 = idx2 // sk
                src_off += ik * sst
    valid = mask & (p >= 0) & (p < P)
    val = tl.load(src_ptr + src_off, mask=valid)
    val = val.to(buf_ptr.dtype.element_ty)
    tl.store(buf_ptr + p, val, mask=valid)


@triton.jit
def _win_kernel(
    win_ptr, V, base_p, P, view_size_ptr, view_stride_ptr, view_D, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    mask = v < V
    idx = v
    p = base_p + tl.zeros_like(v)
    for kk in tl.static_range(MAX_DIMS):
        k = MAX_DIMS - 1 - kk
        if k < view_D:
            sk = tl.load(view_size_ptr + k)
            stk = tl.load(view_stride_ptr + k)
            ik = idx % sk
            idx = idx // sk
            p += ik * stk
    valid = mask & (p >= 0) & (p < P)
    tl.atomic_max(win_ptr + p, v, mask=valid)


@triton.jit
def _fill_kernel(
    win_ptr,
    src_ptr,
    buf_ptr,
    src_so,
    P,
    view_size_ptr,
    src_stride_ptr,
    view_D,
    SRC_CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < P
    w = tl.load(win_ptr + p, mask=mask, other=-1)
    active = mask & (w >= 0)
    if SRC_CONTIG:
        src_off = src_so + w
    else:
        src_off = src_so + tl.zeros_like(w)
        idx2 = w
        for kk in tl.static_range(MAX_DIMS):
            k = MAX_DIMS - 1 - kk
            if k < view_D:
                sk = tl.load(view_size_ptr + k)
                sst = tl.load(src_stride_ptr + k)
                ik = idx2 % sk
                idx2 = idx2 // sk
                src_off += ik * sst
    val = tl.load(src_ptr + src_off, mask=active)
    val = val.to(buf_ptr.dtype.element_ty)
    tl.store(buf_ptr + p, val, mask=active)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def as_strided_scatter(self, src, size, stride, storage_offset=None):
    size = _to_int_list(size)
    stride = _to_int_list(stride)
    so = 0 if storage_offset is None else _to_int(storage_offset)
    D = len(size)
    if len(stride) != D:
        raise ValueError("as_strided_scatter: stride must have the same length as size")
    N = self.numel()
    V = _prod(size) if D else 1
    dev = self.device
    self_so = self.storage_offset()

    # The output preserves self's storage geometry: same storage size, same
    # strides, same storage_offset. Buffer covers the full base storage (and
    # the view footprint, in case the view extends beyond self's footprint).
    storage_size = self.untyped_storage().nbytes() // self.element_size()
    view_extent = 1
    if D:
        view_extent = 1 + sum(max(sz - 1, 0) * st for sz, st in zip(size, stride))
    buf_size = max(storage_size, self_so + so + view_extent)
    buf = torch.empty(buf_size, dtype=self.dtype, device=dev)

    if N == 0:
        return torch.as_strided(buf, self.shape, self.stride(), self_so)

    self_contig = self.is_contiguous()
    src_contig = src.is_contiguous()

    # ---- fused single-pass paths (contiguous self & src only) ----
    if self_contig and src_contig and V > 0:
        grid = (triton.cdiv(buf_size, BLOCK),)
        if _is_dense(size, stride):
            full = (self_so == 0) and (so == 0) and (V == N) and (buf_size == N)
            _fused_dense_kernel[grid](
                self,
                src,
                buf,
                self_so,
                N,
                so,
                V,
                buf_size,
                FULL=full,
                BLOCK=BLOCK,
                num_warps=8,
            )
            return torch.as_strided(buf, self.shape, self.stride(), self_so)
        if D == 1 and stride[0] > 0:
            _fused_1d_kernel[grid](
                self,
                src,
                buf,
                self_so,
                N,
                so,
                buf_size,
                s=stride[0],
                size0=size[0],
                BLOCK=BLOCK,
                num_warps=8,
            )
            return torch.as_strided(buf, self.shape, self.stride(), self_so)
        if (
            D == 2
            and stride[1] > 0
            and stride[0] % stride[1] == 0
            and stride[0] // stride[1] >= size[1]
        ):
            _fused_2d_g_kernel[grid](
                self,
                src,
                buf,
                self_so,
                N,
                so,
                buf_size,
                st0=stride[0],
                g=stride[1],
                size0=size[0],
                size1=size[1],
                BLOCK=BLOCK,
                num_warps=8,
            )
            return torch.as_strided(buf, self.shape, self.stride(), self_so)
        if D == 0:
            _fused_scalar_kernel[grid](
                self, src, buf, self_so, N, so, buf_size, BLOCK=BLOCK, num_warps=8
            )
            return torch.as_strided(buf, self.shape, self.stride(), self_so)

    # ---- general path: storage-domain copy + scatter ----
    _copy_storage_kernel[(triton.cdiv(buf_size, BLOCK),)](
        self, buf, self_so, buf_size, BLOCK=BLOCK, num_warps=8
    )
    if V > 0:
        base_p = self_so + so
        view_size_t = torch.tensor(size, dtype=torch.int32, device=dev)
        view_stride_t = torch.tensor(stride, dtype=torch.int32, device=dev)
        src_stride_t = torch.tensor(list(src.stride()), dtype=torch.int32, device=dev)
        overlap = _view_may_overlap(size, stride)
        grid_v = (triton.cdiv(V, SCATTER_BLOCK),)
        if overlap:
            win = torch.full((buf_size,), -1, dtype=torch.int32, device=dev)
            _win_kernel[grid_v](
                win,
                V,
                base_p,
                buf_size,
                view_size_t,
                view_stride_t,
                D,
                BLOCK=SCATTER_BLOCK,
                num_warps=4,
            )
            _fill_kernel[(triton.cdiv(buf_size, BLOCK),)](
                win,
                src,
                buf,
                src.storage_offset(),
                buf_size,
                view_size_t,
                src_stride_t,
                D,
                SRC_CONTIG=src_contig,
                BLOCK=BLOCK,
                num_warps=8,
            )
        else:
            _scatter_general_kernel[grid_v](
                src,
                buf,
                src.storage_offset(),
                V,
                base_p,
                buf_size,
                view_size_t,
                view_stride_t,
                D,
                SRC_CONTIG=src_contig,
                src_stride_ptr=src_stride_t,
                BLOCK=SCATTER_BLOCK,
                num_warps=4,
            )
    return torch.as_strided(buf, self.shape, self.stride(), self_so)
