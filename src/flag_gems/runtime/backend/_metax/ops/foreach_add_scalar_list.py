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

"""foreach_add_scalar_list (out-of-place) implemented with Triton kernels.

Semantics reproduced (aten::_foreach_add.ScalarList):
    out[i] = self[i] + scalars[i]     (elementwise, no broadcasting across the list)
Each output is a freshly allocated tensor (torch.empty_like, i.e. preserve layout)
and therefore never aliases its input.  A length mismatch raises RuntimeError
before any allocation or launch happens.

Architecture: runs of neighbouring tensors that share (dtype, numel, contiguity)
are batched into a single kernel launch of up to four tensors.  The grid is
(tiles, k): program_id(0) is the tile inside one tensor and program_id(1) the
tensor, so the tensor is selected with a uniform if/elif chain over typed
pointer arguments and the tile offset keeps the canonical
``program_id(0) * BLOCK + arange`` form that lowers to 128-bit vector accesses.
"""

import torch
import triton
import triton.language as tl

_GROUP = 4  # tensors per launch
_MAX_RANK = 8


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _add_scalar_list_kernel(
    i0,
    i1,
    i2,
    i3,
    o0,
    o1,
    o2,
    o3,
    s0,
    s1,
    s2,
    s3,
    M,
    ACC64: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Batched contiguous ``x + scalar``: one tile of one tensor per program."""
    tid = tl.program_id(1)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    if tid == 0:
        ip, op, sv = i0, o0, s0
    elif tid == 1:
        ip, op, sv = i1, o1, s1
    elif tid == 2:
        ip, op, sv = i2, o2, s2
    else:
        ip, op, sv = i3, o3, s3
    if IS_FLOAT:
        x = tl.load(ip + offs, mask=mask, other=0.0)
    else:
        x = tl.load(ip + offs, mask=mask, other=0)
    if IS_FLOAT:
        if ACC64:
            y = x + sv
        else:
            y = x.to(tl.float32) + sv.to(tl.float32)
    else:
        y = x + sv.to(tl.int64).to(op.dtype.element_ty)
    tl.store(op + offs, y.to(op.dtype.element_ty), mask=mask)


@triton.jit
def _add_scalar_strided_kernel(
    in_ptr,
    out_ptr,
    s,
    M,
    is0,
    is1,
    is2,
    is3,
    is4,
    is5,
    is6,
    is7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    ACC64: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row-major logical enumeration mapped through separate in/out strides.

    Dims are left-padded with (size 1, stride 0) so ranks <= 8 share one
    unrolled decomposition.
    """
    base = tl.program_id(0).to(tl.int64) * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < M

    rem = offs
    i7 = rem % d7
    rem = rem // d7
    i6 = rem % d6
    rem = rem // d6
    i5 = rem % d5
    rem = rem // d5
    i4 = rem % d4
    rem = rem // d4
    i3 = rem % d3
    rem = rem // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem // d1

    in_off = (
        i0 * is0
        + i1 * is1
        + i2 * is2
        + i3 * is3
        + i4 * is4
        + i5 * is5
        + i6 * is6
        + i7 * is7
    )
    out_off = (
        i0 * os0
        + i1 * os1
        + i2 * os2
        + i3 * os3
        + i4 * os4
        + i5 * os5
        + i6 * os6
        + i7 * os7
    )

    if IS_FLOAT:
        x = tl.load(in_ptr + in_off, mask=mask, other=0.0)
    else:
        x = tl.load(in_ptr + in_off, mask=mask, other=0)
    if IS_FLOAT:
        if ACC64:
            y = x + s
        else:
            y = x.to(tl.float32) + s.to(tl.float32)
    else:
        y = x + s.to(tl.int64).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Host wrapper
# --------------------------------------------------------------------------- #
def _scalar_values(scalars, n):
    """Normalize the ScalarList / Scalar call forms and validate list length."""
    if isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            return [float(scalars)] * n
        values = list(scalars)
    elif isinstance(scalars, (list, tuple)):
        values = list(scalars)
    else:
        return [float(scalars)] * n
    if len(values) != n:
        raise RuntimeError("Tensor list and scalar list must have the same length.")
    return [float(v) for v in values]


def _pick_block(m):
    """Tile size / warp count from the runtime element count."""
    if m <= 8192:
        return 256, 1
    if m <= 1 << 20:
        return 1024, 4
    return 4096, 4


def _pad_dims(shape, strides):
    """Left-pad (size, stride) pairs with (1, 0) up to _MAX_RANK."""
    pad = _MAX_RANK - len(shape)
    if pad < 0:
        raise NotImplementedError(
            "foreach_add_scalar_list supports tensors with rank <= 8"
        )
    dims = [1] * pad + [int(v) for v in shape]
    ins = [0] * pad + [int(v) for v in strides]
    return dims, ins


def _launch_strided(t, out, value, m):
    block, num_warps = _pick_block(m)
    dims, in_strides = _pad_dims(t.shape, t.stride())
    _, out_strides = _pad_dims(out.shape, out.stride())
    grid = (triton.cdiv(m, block),)
    _add_scalar_strided_kernel[grid](
        t,
        out,
        value,
        m,
        in_strides[0],
        in_strides[1],
        in_strides[2],
        in_strides[3],
        in_strides[4],
        in_strides[5],
        in_strides[6],
        in_strides[7],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        out_strides[4],
        out_strides[5],
        out_strides[6],
        out_strides[7],
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        dims[7],
        ACC64=t.dtype == torch.float64,
        IS_FLOAT=t.is_floating_point(),
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_group(ins, outs, values, m):
    """One launch covering 1.._GROUP contiguous tensors of equal numel/dtype."""
    k = len(ins)
    block, num_warps = _pick_block(m)
    tiles = triton.cdiv(m, block)
    if k < _GROUP:
        ins = ins + [ins[0]] * (_GROUP - k)
        outs = outs + [outs[0]] * (_GROUP - k)
        values = values + [values[0]] * (_GROUP - k)
    _add_scalar_list_kernel[(tiles, k)](
        ins[0],
        ins[1],
        ins[2],
        ins[3],
        outs[0],
        outs[1],
        outs[2],
        outs[3],
        values[0],
        values[1],
        values[2],
        values[3],
        m,
        ACC64=ins[0].dtype == torch.float64,
        IS_FLOAT=ins[0].is_floating_point(),
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )


def foreach_add_scalar_list(self, scalars):
    tensors = [self] if isinstance(self, torch.Tensor) else list(self)
    n = len(tensors)
    values = _scalar_values(scalars, n)
    if n == 0:
        return []

    outs = [torch.empty_like(t) for t in tensors]

    i = 0
    while i < n:
        t = tensors[i]
        m = t.numel()
        if m == 0:
            i += 1
            continue
        if not t.is_contiguous():
            _launch_strided(t, outs[i], values[i], m)
            i += 1
            continue
        dtype = t.dtype
        j = i + 1
        while (
            j < n
            and j - i < _GROUP
            and tensors[j].dtype is dtype
            and tensors[j].numel() == m
            and tensors[j].is_contiguous()
        ):
            j += 1
        _launch_group(tensors[i:j], outs[i:j], values[i:j], m)
        i = j
    return outs
