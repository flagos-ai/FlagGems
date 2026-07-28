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
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any[:, None], row_mask)


@libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_any)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val)


@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


def any(inp):
    logger.debug("GEMS ANY")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        any_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
        any_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=torch.bool, device=inp.device)
        inp = inp.to(torch.bool)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            any_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


class _TensorReduceLayout:
    """Logical keepdim layout for strided reductions."""

    @staticmethod
    def _coalesce_dims(
        dims: Sequence[tuple[int, tuple[int, ...]]],
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        # Match TensorIterator's rule for adjacent dimensions in fastest-first
        # order: size-1 dims are free, otherwise every tracked stride must be
        # contiguous.
        coalesced: list[tuple[int, tuple[int, ...]]] = []
        for shape, strides in dims:
            if not coalesced:
                coalesced.append((shape, strides))
                continue

            prev_shape, prev_strides = coalesced[-1]
            can_merge = (
                prev_shape == 1
                or shape == 1
                or all(
                    prev_shape * prev_stride == stride
                    for prev_stride, stride in zip(prev_strides, strides)
                )
            )
            if not can_merge:
                coalesced.append((shape, strides))
                continue

            merged_strides = strides if prev_shape == 1 else prev_strides
            coalesced[-1] = (prev_shape * shape, merged_strides)
        return tuple(coalesced)

    def __init__(self, inp: torch.Tensor, reduce_dims: Sequence[int]):
        self.shape = tuple(inp.shape)
        self.input_strides = tuple(inp.stride())
        self.reduce_dims = tuple(reduce_dims)
        reduce_dim_set = set(self.reduce_dims)
        self.out_shape = tuple(
            1 if dim in reduce_dim_set else size for dim, size in enumerate(self.shape)
        )

    def finalize(self, out: torch.Tensor):
        output_strides_full = tuple(out.stride())
        output_dims = tuple(
            dim for dim in range(len(self.shape)) if dim not in self.reduce_dims
        )

        # Keep reduce and output spaces separate so the kernel can split
        # reduce_linear and output_linear independently.
        reduce_order = tuple(
            sorted(self.reduce_dims, key=lambda dim: abs(self.input_strides[dim]))
        )
        output_order = tuple(
            sorted(output_dims, key=lambda dim: abs(output_strides_full[dim]))
        )

        reduce_items = self._coalesce_dims(
            tuple((self.shape[dim], (self.input_strides[dim],)) for dim in reduce_order)
        )
        output_items = self._coalesce_dims(
            tuple(
                (
                    self.shape[dim],
                    (self.input_strides[dim], output_strides_full[dim]),
                )
                for dim in output_order
            )
        )

        self.reduce_shapes = tuple(shape for shape, _ in reduce_items)
        self.input_reduce_strides = tuple(strides[0] for _, strides in reduce_items)
        self.output_shapes = tuple(shape for shape, _ in output_items)
        self.input_output_strides = tuple(strides[0] for _, strides in output_items)
        self.output_strides = tuple(strides[1] for _, strides in output_items)
        self.inputs_per_output = math.prod(self.reduce_shapes)
        self.num_outputs = math.prod(self.output_shapes)

    def kernel_args(self) -> tuple[int, ...]:
        return (
            *self.output_shapes,
            *self.input_output_strides,
            *self.output_strides,
            *self.reduce_shapes,
            *self.input_reduce_strides,
        )


def _generate_any_dims_kernel_source(
    num_output_dims: int, num_reduce_dims: int
) -> tuple[str, str]:
    kernel_name = f"_any_dims_kernel_o{num_output_dims}_r{num_reduce_dims}"
    metadata_args = (
        [f"output_shape{i}" for i in range(num_output_dims)]
        + [f"input_output_stride{i}" for i in range(num_output_dims)]
        + [f"output_stride{i}" for i in range(num_output_dims)]
        + [f"reduce_shape{i}" for i in range(num_reduce_dims)]
        + [f"input_reduce_stride{i}" for i in range(num_reduce_dims)]
    )
    code = IndentedBuffer()
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("@triton.jit(do_not_specialize=[")
    with code.indent():
        for arg_name in metadata_args:
            code.writeline(f"{arg_name!r},")
    code.writeline("])")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("inp,")
        code.writeline("out,")
        for arg_name in metadata_args:
            code.writeline(f"{arg_name},")
        code.writeline("BLOCK_M: tl.constexpr,")
        code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")
    with code.indent():
        output = " * ".join(f"output_shape{i}" for i in range(num_output_dims))
        input_per_output = " * ".join(
            f"reduce_shape{i}" for i in range(num_reduce_dims)
        )
        code.writeline(f"output = {output or '1'}")
        code.writeline(f"input_per_output = {input_per_output or '1'}")
        code.writeline("output_tiles = tl.cdiv(output, BLOCK_M)")
        code.writeline("pid = tl.program_id(0)")
        code.writeline("pid_y = pid % output_tiles")
        code.writeline("pid_x = pid // output_tiles")
        code.newline()
        code.writeline(
            "output_offsets_linear = (pid_y * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)"
        )
        code.writeline(
            "reduce_offsets_linear = (pid_x * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)"
        )
        code.writeline("output_mask = output_offsets_linear < output")
        code.writeline("reduce_mask = reduce_offsets_linear < input_per_output")
        code.newline()
        code.writeline("input_output_offsets = tl.zeros((BLOCK_M,), dtype=tl.int64)")
        code.writeline("output_offsets = tl.zeros((BLOCK_M,), dtype=tl.int64)")
        code.writeline("output_linear = output_offsets_linear")
        code.writeline("# Expand output_linear into input/output base offsets.")
        for i in range(num_output_dims):
            code.writeline(f"out_idx{i} = output_linear % output_shape{i}")
            code.writeline(
                f"input_output_offsets += out_idx{i} * input_output_stride{i}"
            )
            code.writeline(f"output_offsets += out_idx{i} * output_stride{i}")
            if i != num_output_dims - 1:
                code.writeline(f"output_linear = output_linear // output_shape{i}")
        code.newline()
        code.writeline("input_reduce_offsets = tl.zeros((BLOCK_N,), dtype=tl.int64)")
        code.writeline("reduce_linear = reduce_offsets_linear")
        code.writeline("# Expand reduce_linear into the input reduce offset.")
        for i in range(num_reduce_dims):
            code.writeline(f"red_idx{i} = reduce_linear % reduce_shape{i}")
            code.writeline(
                f"input_reduce_offsets += red_idx{i} * input_reduce_stride{i}"
            )
            if i != num_reduce_dims - 1:
                code.writeline(f"reduce_linear = reduce_linear // reduce_shape{i}")
        code.newline()
        code.writeline(
            "input_offsets = input_output_offsets[:, None] + input_reduce_offsets[None, :]"
        )
        code.writeline(
            "active = tl.load(out + output_offsets, mask=output_mask, other=1) == 0"
        )
        code.writeline(
            "mask = output_mask[:, None] & reduce_mask[None, :] & active[:, None]"
        )
        code.writeline("vals = tl.load(inp + input_offsets, mask=mask, other=0.0)")
        code.writeline("local_any = tl.max(vals != 0, axis=1).to(tl.int32)")
        # int32 atomic avoids backend issues with bool or 8-bit atomics.
        code.writeline(
            "tl.atomic_max(out + output_offsets, local_any, mask=output_mask & active & (local_any != 0))"
        )
    return kernel_name, code.getvalue()


def _any_dims_kernel_for_rank(num_output_dims: int, num_reduce_dims: int):
    key = f"{num_output_dims}_{num_reduce_dims}"
    if not hasattr(_any_dims_kernel_for_rank, "overloads"):
        _any_dims_kernel_for_rank.overloads = {}

    overloads = _any_dims_kernel_for_rank.overloads
    if key in overloads:
        return overloads[key]

    kernel_name, source = _generate_any_dims_kernel_source(
        num_output_dims, num_reduce_dims
    )
    file_name = f"any_dims_rank_{key}.py"
    file_path = code_cache_dir() / file_name
    write_atomic(file_path, source)

    spec = importlib.util.spec_from_file_location(
        f"_any_dims_gen_module_rank_{key}",
        file_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)
    overloads[key] = kernel
    return kernel


def _select_reduction_config(
    num_outputs,
    inputs_per_output,
):
    block_width = min(256, triton.next_power_of_2(inputs_per_output))
    block_height = min(32, triton.next_power_of_2(num_outputs))
    num_warps = 8 if block_width >= 256 else 4

    return block_height, block_width, num_warps


def any_dims(
    inp,
    dim=None,
    keepdim=False,
):
    logger.debug("GEMS ANY DIMS")
    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)

    dims = list(dim)
    assert all(-inp.ndim <= d < inp.ndim for d in dims), "Invalid dim"
    reduce_dims = sorted(set(d % inp.ndim for d in dims))
    layout = _TensorReduceLayout(inp, reduce_dims)
    # Partial reductions from different reduce blocks are merged with atomic_max.
    # Use int32 for broad backend support, then cast to the public bool result.
    out_i32 = torch.zeros(layout.out_shape, dtype=torch.int32, device=inp.device)
    layout.finalize(out_i32)

    if layout.num_outputs == 0 or layout.inputs_per_output == 0:
        out = out_i32.to(torch.bool)
        if not keepdim:
            for d in reversed(reduce_dims):
                out = out.squeeze(dim=d)
        return out

    if layout.num_outputs == 1:
        out = any(inp).reshape(layout.out_shape)
        if not keepdim:
            for d in reversed(reduce_dims):
                out = out.squeeze(dim=d)
        return out

    block_height, block_width, num_warps = _select_reduction_config(
        layout.num_outputs,
        layout.inputs_per_output,
    )
    grid = (
        triton.cdiv(layout.num_outputs, block_height)
        * triton.cdiv(layout.inputs_per_output, block_width),
    )
    kernel = _any_dims_kernel_for_rank(
        len(layout.output_shapes), len(layout.reduce_shapes)
    )

    with torch_device_fn.device(inp.device):
        kernel[grid](
            inp,
            out_i32,
            *layout.kernel_args(),
            BLOCK_M=block_height,
            BLOCK_N=block_width,
            num_warps=num_warps,
            num_stages=2,
        )

    out = out_i32.to(torch.bool)
    if not keepdim:
        for d in reversed(reduce_dims):
            out = out.squeeze(dim=d)
    return out
