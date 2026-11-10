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

"""foreach_add_scalar as a Triton implementation.

Semantics: for every tensor t_i in the input list, produce a freshly allocated
out_i with the same shape/dtype such that out_i = t_i + scalar, where the scalar
is converted to the element type (fp16/bf16 add is performed in fp32 opmath and
rounded once on store, matching ATen's CUDA foreach add).

Each element is streamed exactly once: one load, one add, one store.
Dense tensors (contiguous, or transposed but still non-overlapping and dense)
are traversed in storage order, which is a bijection for such tensors, so both
the load and the store are fully coalesced with no address arithmetic.
Genuinely strided tensors fall back to a rank-generic index decomposition.

The whole tensor list is issued as a single Triton launch (2-D grid over
(tile, tensor) with per-tensor masking) so the measured path pays one launch
instead of one launch per tensor. When the tensors of a group are contiguous
and share one shape, their outputs are carved out of a single allocation with
torch.unbind, so the launch carries one output base pointer plus (n, seq)
instead of one pointer per output.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_WARPS = 4
_MAX_RANK = 8
_FUSE_SLOTS = 8
_UNIFORM_SLOTS = 4

_HALF_DTYPES = (torch.float16, torch.bfloat16)

# MODE: 0 = native float add, 1 = fp16/bf16 fp32 opmath, 2 = integer, 3 = fp64
_MODE_CACHE = {}


def _mode_for(dtype):
    mode = _MODE_CACHE.get(dtype)
    if mode is None:
        if dtype in _HALF_DTYPES:
            mode = 1
        elif dtype == torch.float64:
            mode = 3
        elif dtype.is_floating_point:
            mode = 0
        else:
            mode = 2
        _MODE_CACHE[dtype] = mode
    return mode


@triton.jit
def _add_vec(x_ptr, y_ptr, n, s, offs, MODE: tl.constexpr):
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0)
    if MODE == 1:
        r = v.to(tl.float32) + s
    elif MODE == 2:
        r = v + s.to(tl.int64)
    elif MODE == 3:
        r = v + s.to(tl.float64)
    else:
        r = v + s
    tl.store(y_ptr + offs, r, mask=mask)


@triton.jit
def _add_flat_kernel(x_ptr, y_ptr, n, s, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    _add_vec(x_ptr, y_ptr, n, s, offs, MODE)


@triton.jit
def _add_uniform4_kernel(
    x0,
    x1,
    x2,
    x3,
    y_ptr,
    n,
    s,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Four equal-length tensors, outputs laid out consecutively at y_ptr."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    t = tl.program_id(1)
    # Promote the segment stride so slot bases past 2^31 elements stay positive.
    step = n.to(tl.int64)
    if t == 0:
        _add_vec(x0, y_ptr, n, s, offs, MODE)
    elif t == 1:
        _add_vec(x1, y_ptr + step, n, s, offs, MODE)
    elif t == 2:
        _add_vec(x2, y_ptr + 2 * step, n, s, offs, MODE)
    else:
        _add_vec(x3, y_ptr + 3 * step, n, s, offs, MODE)


@triton.jit
def _add_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    t = tl.program_id(1)
    if t == 0:
        _add_vec(x0, y0, n0, s, offs, MODE)
    elif t == 1:
        _add_vec(x1, y1, n1, s, offs, MODE)
    elif t == 2:
        _add_vec(x2, y2, n2, s, offs, MODE)
    elif t == 3:
        _add_vec(x3, y3, n3, s, offs, MODE)
    elif t == 4:
        _add_vec(x4, y4, n4, s, offs, MODE)
    elif t == 5:
        _add_vec(x5, y5, n5, s, offs, MODE)
    elif t == 6:
        _add_vec(x6, y6, n6, s, offs, MODE)
    else:
        _add_vec(x7, y7, n7, s, offs, MODE)


@triton.jit
def _add_strided_kernel(
    x_ptr,
    y_ptr,
    n,
    s,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    MODE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK >= 1:
        idx = rem % d0
        rem = rem // d0
        off += idx.to(tl.int64) * i0
    if RANK >= 2:
        idx = rem % d1
        rem = rem // d1
        off += idx.to(tl.int64) * i1
    if RANK >= 3:
        idx = rem % d2
        rem = rem // d2
        off += idx.to(tl.int64) * i2
    if RANK >= 4:
        idx = rem % d3
        rem = rem // d3
        off += idx.to(tl.int64) * i3
    if RANK >= 5:
        idx = rem % d4
        rem = rem // d4
        off += idx.to(tl.int64) * i4
    if RANK >= 6:
        idx = rem % d5
        rem = rem // d5
        off += idx.to(tl.int64) * i5
    if RANK >= 7:
        idx = rem % d6
        rem = rem // d6
        off += idx.to(tl.int64) * i6
    if RANK >= 8:
        idx = rem % d7
        rem = rem // d7
        off += idx.to(tl.int64) * i7
    v = tl.load(x_ptr + off, mask=mask, other=0)
    if MODE == 1:
        r = v.to(tl.float32) + s
    elif MODE == 2:
        r = v + s.to(tl.int64)
    elif MODE == 3:
        r = v + s.to(tl.float64)
    else:
        r = v + s
    tl.store(y_ptr + offs, r, mask=mask)


def _cdiv(a, b):
    return (a + b - 1) // b


# Module-level aliases keep the hot path free of attribute lookups.
_EMPTY = torch.empty
_EMPTY_LIKE = torch.empty_like
_UNIFORM = _add_uniform4_kernel
_FLAT = _add_flat_kernel
_STRIDED = _add_strided_kernel
_FUSED = _add_fused_kernel


def _is_dense(shape, strides):
    """Replicate ATen's non-overlapping-and-dense test from metadata only."""
    order = sorted(range(len(shape)), key=lambda axis: strides[axis])
    expected = 1
    for axis in order:
        size = shape[axis]
        if size == 1:
            continue
        if strides[axis] != expected:
            return False
        expected *= size
    return True


def _emit_group(entries, mode, value, outs):
    """entries: [index, tensor, numel, shape] of contiguous tensors."""
    count = len(entries)
    for start in range(0, count, _UNIFORM_SLOTS):
        chunk = entries[start : start + _UNIFORM_SLOTS]
        width = len(chunk)
        first = chunk[0]
        if width == 1:
            out = torch.empty_like(first[1])
            _add_flat_kernel[(_cdiv(first[2], _BLOCK),)](
                first[1],
                out,
                first[2],
                value,
                MODE=mode,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
            outs[first[0]] = out
            continue

        numel = first[2]
        shape = first[3]
        uniform = True
        for entry in chunk:
            if entry[2] != numel or entry[3] != shape:
                uniform = False
                break

        if uniform:
            flat = torch.empty(
                (width,) + tuple(shape),
                dtype=first[1].dtype,
                device=first[1].device,
            )
            xs = [entry[1] for entry in chunk]
            if width < _UNIFORM_SLOTS:
                xs.extend([first[1]] * (_UNIFORM_SLOTS - width))
            _add_uniform4_kernel[(_cdiv(numel, _BLOCK), width)](
                *xs,
                flat,
                numel,
                value,
                MODE=mode,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
            views = flat.unbind(0)
            for offset, entry in enumerate(chunk):
                outs[entry[0]] = views[offset]
            continue

        xs = [entry[1] for entry in chunk]
        ys = []
        ns = []
        for entry in chunk:
            out = torch.empty_like(entry[1])
            ys.append(out)
            ns.append(entry[2])
            outs[entry[0]] = out
        if width < _FUSE_SLOTS:
            pad = _FUSE_SLOTS - width
            xs.extend([first[1]] * pad)
            ys.extend([ys[0]] * pad)
            ns.extend([0] * pad)
        _add_fused_kernel[(_cdiv(max(ns[:width]), _BLOCK), width)](
            *xs,
            *ys,
            *ns,
            value,
            MODE=mode,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )


def foreach_add_scalar(self, scalar):
    tensors = self if isinstance(self, (list, tuple)) else [self]
    n_tensors = len(tensors)
    if n_tensors == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Fast path: a short list of identically shaped contiguous tensors plus one
    # scalar. One allocation, one unbind, one 7-argument launch.
    if not isinstance(scalar, (list, tuple)) and 1 < n_tensors <= _UNIFORM_SLOTS:
        first = tensors[0]
        numel = first.numel()
        if numel and first.is_contiguous():
            dtype = first.dtype
            device = first.device
            shape = first.shape
            same = True
            for tensor in tensors:
                if (
                    tensor.dtype is not dtype
                    or tensor.shape != shape
                    or tensor.device != device
                    or not tensor.is_contiguous()
                ):
                    same = False
                    break
            if same:
                if isinstance(scalar, torch.Tensor):
                    scalar = scalar.item()
                value = float(scalar)
                mode = _mode_for(dtype)
                flat = _EMPTY((n_tensors,) + tuple(shape), dtype=dtype, device=device)
                grid = (_cdiv(numel, _BLOCK), n_tensors)
                if n_tensors == _UNIFORM_SLOTS:
                    _UNIFORM[grid](
                        tensors[0],
                        tensors[1],
                        tensors[2],
                        tensors[3],
                        flat,
                        numel,
                        value,
                        MODE=mode,
                        BLOCK=_BLOCK,
                        num_warps=_WARPS,
                    )
                elif n_tensors == 3:
                    _UNIFORM[grid](
                        tensors[0],
                        tensors[1],
                        tensors[2],
                        tensors[0],
                        flat,
                        numel,
                        value,
                        MODE=mode,
                        BLOCK=_BLOCK,
                        num_warps=_WARPS,
                    )
                else:
                    _UNIFORM[grid](
                        tensors[0],
                        tensors[1],
                        tensors[0],
                        tensors[0],
                        flat,
                        numel,
                        value,
                        MODE=mode,
                        BLOCK=_BLOCK,
                        num_warps=_WARPS,
                    )
                return list(flat.unbind(0))

    if isinstance(scalar, (list, tuple)):
        scalar_list = list(scalar)
        if len(scalar_list) != n_tensors:
            raise RuntimeError("Tensor list and scalar list must have the same length.")
    else:
        scalar_list = None

    outs = [None] * n_tensors
    groups = {}
    dense_singles = []
    strided_singles = []

    for index in range(n_tensors):
        tensor = tensors[index]
        value = scalar_list[index] if scalar_list is not None else scalar
        if isinstance(value, torch.Tensor):
            value = value.item()
        value = float(value)

        mode = _mode_for(tensor.dtype)
        numel = tensor.numel()
        if numel == 0:
            outs[index] = torch.empty_like(tensor)
            continue

        if tensor.is_contiguous():
            groups.setdefault((mode, tensor.dtype, tensor.device, value), []).append(
                [index, tensor, numel, tensor.shape]
            )
            continue

        shape = tensor.shape
        if _is_dense(shape, tensor.stride()):
            dense_singles.append([index, tensor, numel, mode, value])
            continue

        strided_singles.append([index, tensor, numel, mode, value])

    for (mode, _dtype, _device, value), entries in groups.items():
        _emit_group(entries, mode, value, outs)

    for index, tensor, numel, mode, value in dense_singles:
        out = torch.empty_like(tensor)
        if out.stride() == tensor.stride():
            _add_flat_kernel[(_cdiv(numel, _BLOCK),)](
                tensor,
                out,
                numel,
                value,
                MODE=mode,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        else:
            del out
            out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            rank = tensor.dim()
            shapes = list(tensor.shape) + [1] * (_MAX_RANK - rank)
            strides = list(tensor.stride()) + [0] * (_MAX_RANK - rank)
            _add_strided_kernel[(_cdiv(numel, _BLOCK),)](
                tensor,
                out,
                numel,
                value,
                *shapes,
                *strides,
                MODE=mode,
                RANK=rank,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        outs[index] = out

    for index, tensor, numel, mode, value in strided_singles:
        out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        rank = tensor.dim()
        shapes = list(tensor.shape) + [1] * (_MAX_RANK - rank)
        strides = list(tensor.stride()) + [0] * (_MAX_RANK - rank)
        _add_strided_kernel[(_cdiv(numel, _BLOCK),)](
            tensor,
            out,
            numel,
            value,
            *shapes,
            *strides,
            MODE=mode,
            RANK=rank,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
        outs[index] = out

    return outs
