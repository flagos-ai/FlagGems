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

import importlib.util
import logging
from functools import lru_cache
from hashlib import sha256

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import write_atomic

logger = logging.getLogger(__name__)


@triton.jit
def _mask_positions(mask, positions, count, N: tl.constexpr, BLOCK: tl.constexpr):
    # One ordered scan preserves row-major boolean indexing order. The count
    # is output metadata; reading it requires synchronization (as native nonzero).
    total = tl.full((), 0, tl.int64)
    for start in range(triton.cdiv(N, BLOCK)):
        offsets = start * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(mask + offsets, offsets < N, other=0) != 0
        ranks = tl.cumsum(values.to(tl.int32), 0) - 1
        tl.store(positions + total + ranks, offsets, values & (offsets < N))
        total += tl.sum(values.to(tl.int64), 0)
    tl.store(count, total)


@triton.jit
def _mask_coordinates(
    flat,
    out,
    N,
    SHAPE: tl.constexpr,
    DIM: tl.constexpr,
    DIVISOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(flat + offsets, offsets < N, other=0)
        tl.store(out + offsets, (values // DIVISOR) % SHAPE[DIM], offsets < N)


def _boolean_coordinates(mask):
    mask = mask.contiguous()
    flat = torch.empty(mask.numel(), device=mask.device, dtype=torch.int64)
    count = torch.empty((), device=mask.device, dtype=torch.int64)
    _mask_positions[(1,)](mask, flat, count, mask.numel(), 1024)
    size = int(count.item())
    result = []
    divisor = mask.numel()
    for dim, extent in enumerate(mask.shape):
        divisor //= extent if extent else 1
        coordinate = torch.empty(size, device=mask.device, dtype=torch.int64)
        if size:
            _mask_coordinates[(min(48, triton.cdiv(size, 1024)),)](
                flat, coordinate, size, tuple(mask.shape), dim, divisor, 1024
            )
        result.append(coordinate)
    return result


@lru_cache(maxsize=256)
def _gather_function(
    shape,
    strides,
    index_dims,
    index_strides,
    broadcast_shape,
    output_shape,
    plain_axes,
    broadcast_axes,
):
    # Generate only shape/stride address arithmetic; payloads and index values
    # remain device inputs. No contiguous copies or native index redispatch.
    lines = [
        "import triton",
        "import triton.language as tl",
        "from flag_gems.utils import libentry",
        "@libentry()",
        "@triton.jit",
    ]
    args = (
        ["inp"]
        + [f"index{i}" for i in range(len(index_dims))]
        + ["out", "N", "BLOCK: tl.constexpr", "PROGRAMS: tl.constexpr"]
    )
    lines += [
        "def kernel(" + ", ".join(args) + "):",
        "    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), PROGRAMS):",
        "        linear = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "        active = linear < N",
        "        remaining = linear.to(tl.int64)",
        "        source = tl.full((BLOCK,), 0, tl.int64)",
    ]
    for axis in reversed(range(len(output_shape))):
        lines += [
            f"        c{axis} = remaining % {output_shape[axis]}",
            f"        remaining = remaining // {output_shape[axis]}",
        ]
    for dim, axis in plain_axes:
        lines += [f"        source += c{axis} * {strides[dim]}"]
    for i, dim in enumerate(index_dims):
        # Extent-one coordinates are always zero. Leaving their strides in
        # the expression creates a splatted pointer under a vector mask after
        # folding, which FlagTree cannot lower. Use the scalar load below.
        terms = [
            f"c{axis} * {stride}"
            for axis, stride in zip(broadcast_axes, index_strides[i])
            if stride and output_shape[axis] > 1
        ]
        offset = " + ".join(terms)
        load = (
            f"tl.load(index{i} + ({offset}), active, other=0)"
            if terms
            else f"tl.load(index{i})"
        )
        lines += [
            f"        raw{i} = {load}.to(tl.int64)",
            f"        normalized{i} = tl.where(raw{i} < 0, raw{i} + {shape[dim]}, raw{i})",
            f"        tl.device_assert((~active) | ((normalized{i} >= 0) "
            f'& (normalized{i} < {shape[dim]})), "index out of bounds")',
            f"        source += normalized{i} * {strides[dim]}",
            f"        active = active & (normalized{i} >= 0) & (normalized{i} < {shape[dim]})",
        ]
    lines += [
        "        value = tl.load(inp + source, active, other=0)",
        "        tl.store(out + linear, value, linear < N)",
    ]
    plain_shape = tuple(output_shape[axis] for _, axis in plain_axes)
    plain_numel = 1
    for extent in plain_shape:
        plain_numel *= extent
    broadcast_numel = 1
    for extent in broadcast_shape:
        broadcast_numel *= extent
    span = sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    # A program owns a broadcast index and one small plain-data tile. Index
    # loads and int64 normalization are scalar, amortized across that tile.
    # Wide payloads also benefit beyond 65535 indices; narrow payloads retain
    # their existing traversal. Only bounded payload and output offsets narrow;
    # the indexed source base stays int64 when the view spans a wide address range.
    scalar_indices = (
        1 < broadcast_numel
        and plain_numel <= 1024
        and (broadcast_numel <= 65535 or plain_numel >= 128)
        and sum((output_shape[axis] - 1) * strides[dim] for dim, axis in plain_axes)
        < 2**31
        and broadcast_numel * plain_numel < 2**31
    )
    if scalar_indices:
        out_strides = [1] * len(output_shape)
        for axis in range(len(output_shape) - 2, -1, -1):
            out_strides[axis] = out_strides[axis + 1] * output_shape[axis + 1]
        block = triton.next_power_of_2(plain_numel)
        lines = lines[:5] + [
            "def kernel(" + ", ".join(args) + "):",
            f"    for item in range(tl.program_id(0), {broadcast_numel}, PROGRAMS):",
            "        broadcast_remaining = item.to(tl.int64)",
            "        destination_base = 0",
        ]
        for dim in reversed(range(len(broadcast_shape))):
            axis = broadcast_axes[dim]
            lines += [
                f"        b{dim} = broadcast_remaining % {broadcast_shape[dim]}",
                f"        broadcast_remaining = broadcast_remaining // {broadcast_shape[dim]}",
                f"        destination_base += b{dim} * {out_strides[axis]}",
            ]
        lines += [
            "        source_base = tl.full((), 0, tl.int64)",
            "        valid_index = tl.full((), True, tl.int1)",
        ]
        for i, dim in enumerate(index_dims):
            terms = [
                f"b{j} * {stride}"
                for j, stride in enumerate(index_strides[i])
                if stride and broadcast_shape[j] > 1
            ]
            offset = " + ".join(terms) or "0"
            lines += [
                f"        raw{i} = tl.load(index{i} + ({offset})).to(tl.int64)",
                f"        normalized{i} = tl.where(raw{i} < 0, raw{i} + {shape[dim]}, raw{i})",
                f"        valid{i} = (normalized{i} >= 0) & (normalized{i} < {shape[dim]})",
                f'        tl.device_assert(valid{i}, "index out of bounds")',
                f"        source_base += tl.where(valid{i}, normalized{i}, 0) * {strides[dim]}",
                f"        valid_index = valid_index & valid{i}",
            ]
        if span < 2**31:
            lines += ["        source_base = source_base.to(tl.int32)"]
        lines += [
            "        destination_base = destination_base.to(tl.int32)",
            f"        plain = tl.arange(0, {block})",
            "        remaining = plain",
            f"        source = tl.full(({block},), 0, tl.int32)",
            "        destination = destination_base + tl.full(("
            + str(block)
            + ",), 0, tl.int32)",
        ]
        for j in reversed(range(len(plain_axes))):
            dim, axis = plain_axes[j]
            lines += [
                f"        c{j} = remaining % {output_shape[axis]}",
                f"        remaining = remaining // {output_shape[axis]}",
                f"        source += c{j} * {strides[dim]}",
                f"        destination += c{j} * {out_strides[axis]}",
            ]
        lines += [
            f"        active = (plain < {plain_numel}) & valid_index",
            "        value = tl.load(inp + source_base + source, active, other=0)",
            f"        tl.store(out + destination, value, plain < {plain_numel})",
        ]

    code = "\n".join(lines) + "\n"
    key = sha256(code.encode()).hexdigest()
    path = code_cache_dir() / f"ascend_strided_index_{key}.py"
    write_atomic(path, code)
    spec = importlib.util.spec_from_file_location(f"ascend_strided_index_{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel, broadcast_numel if scalar_indices else None


def _index_impl(inp, indices):
    logger.debug("GEMS_ASCEND INDEX")
    if not indices:
        raise ValueError("at least one index must be provided")
    if all(item is None for item in indices):
        raise IndexError("Index out of range")
    expanded = []
    for item in indices:
        if item is None:
            expanded.append(None)
        elif item.dtype in (torch.bool, torch.uint8):
            if item.ndim == 0:
                raise IndexError(
                    "zero-dimensional boolean indices are not supported by aten.index"
                )
            start = len(expanded)
            if start + item.ndim > inp.ndim or tuple(item.shape) != tuple(
                inp.shape[start : start + item.ndim]
            ):
                raise IndexError("boolean index shape does not match indexed tensor")
            expanded.extend(_boolean_coordinates(item.to(inp.device)))
        elif item.dtype in (torch.int32, torch.int64):
            expanded.append(item.to(inp.device))
        else:
            raise IndexError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )
    if len(expanded) > inp.ndim:
        raise IndexError(f"too many indices for tensor of dimension {inp.ndim}")
    expanded.extend([None] * (inp.ndim - len(expanded)))
    index_dims = tuple(dim for dim, item in enumerate(expanded) if item is not None)
    tensors = [expanded[dim] for dim in index_dims]
    broadcast_shape = torch.broadcast_shapes(*(item.shape for item in tensors))
    tensors = [item.expand(broadcast_shape) for item in tensors]
    adjacent = index_dims == tuple(range(index_dims[0], index_dims[-1] + 1))
    plain_dims = [dim for dim in range(inp.ndim) if dim not in index_dims]
    before = (
        [dim for dim in plain_dims if index_dims and dim < index_dims[0]]
        if adjacent
        else []
    )
    after = [dim for dim in plain_dims if dim not in before]
    output_shape = (
        tuple(inp.shape[dim] for dim in before)
        + tuple(broadcast_shape)
        + tuple(inp.shape[dim] for dim in after)
    )
    broadcast_axes = tuple(range(len(before), len(before) + len(broadcast_shape)))
    plain_axes = tuple((dim, axis) for axis, dim in enumerate(before)) + tuple(
        (dim, len(before) + len(broadcast_shape) + axis)
        for axis, dim in enumerate(after)
    )
    out = torch.empty(output_shape, device=inp.device, dtype=inp.dtype)
    if out.numel():
        kernel, index_tiles = _gather_function(
            tuple(inp.shape),
            tuple(inp.stride()),
            index_dims,
            tuple(tuple(item.stride()) for item in tensors),
            tuple(broadcast_shape),
            output_shape,
            plain_axes,
            broadcast_axes,
        )
        programs = min(48, index_tiles or triton.cdiv(out.numel(), 1024))
        # Match Triton's standard debug-controlled bounds assertions. The
        # load mask remains active regardless of debug to prevent invalid reads.
        kernel[(programs,)](inp, *tensors, out, out.numel(), 1024, programs)
    return out


def index(inp, indices):
    with torch_device_fn.device(inp.device):
        return _index_impl(inp, indices)
