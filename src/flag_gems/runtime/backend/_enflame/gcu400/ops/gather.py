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

import importlib
import logging
import os
from typing import Callable, Mapping

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# ---------------------------------------------------------------------------
# gather_backward on gcu400.
#
# The accumulation is order-dependent, so every Triton formulation needs one
# tl.atomic_add per index element; on gcu400 that datapath costs ~5ns/element
# (measured 4.5ms at 1024x1024 and 309ms at 1024x65536 for the final config,
# i.e. 12x slower than the vendor scatter_add kernel, and the atomic pipeline
# not the tile shape sets that floor: BM/BN 1..4 x 256..8192 and sem="relaxed"
# were all measured, the 1024-wide vector with the default semantics won).
# The implementation therefore keeps exactly one atomic per element and strips
# everything around it:
# - the two 2D layouts that own every core benchmark shape (scatter along the
#   last dim and along dim 0) use row-parallel kernels that walk the rows
#   grid-strided and derive the atomic address with a single multiply from the
#   loaded index, so no rank-N offset vectors and no per-element stride
#   arithmetic are built;
# - any other rank keeps the generic scatter(reduce="add") formulation.
# Everything stays in Triton: no aten/native kernel is delegated to.
# ---------------------------------------------------------------------------
# Tunable knobs (one-variable-at-a-time sweeps, see the round log):
# - GB_BLOCK_N_CAP bounds one row's index vector per atomic step,
# - GB_GRID_PER_SIP sets the row-loop width (rows are walked grid-strided),
# - GB_WARPS is the CTA width for the row-parallel kernels.
GB_BLOCK_N_CAP = 1024
GB_GRID_PER_SIP = 2
GB_WARPS = 8


@libentry()
@triton.jit(
    do_not_specialize=["M", "N_idx", "inp_stride0", "idx_stride0", "out_stride0"]
)
def gather_2d_row_kernel(
    inp_ptr,
    index_ptr,
    out_ptr,
    M,
    N_idx,
    inp_stride0,
    idx_stride0,
    out_stride0,
    BLOCK_N: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    for row in tl.range(pid, M, GRID_DIM):
        inp_row = inp_ptr + row * inp_stride0
        idx_row = index_ptr + row * idx_stride0
        out_row = out_ptr + row * out_stride0

        for col_start in tl.range(0, N_idx, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            mask = cols < N_idx
            idx = tl.load(idx_row + cols, mask=mask, other=0)
            val = tl.load(inp_row + idx, mask=mask, other=0)
            tl.store(out_row + cols, val, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "M",
        "N_idx",
        "dim_stride",
        "inp_stride0",
        "idx_stride0",
        "out_stride0",
    ]
)
def gather_2d_dim0_kernel(
    inp_ptr,
    index_ptr,
    out_ptr,
    M,
    N_idx,
    dim_stride,
    inp_stride0,
    idx_stride0,
    out_stride0,
    BLOCK_N: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    for row in tl.range(pid, M, GRID_DIM):
        idx_row = index_ptr + row * idx_stride0
        out_row = out_ptr + row * out_stride0

        for col_start in tl.range(0, N_idx, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            mask = cols < N_idx
            idx = tl.load(idx_row + cols, mask=mask, other=0)
            val = tl.load(inp_ptr + idx * dim_stride + cols, mask=mask, other=0)
            tl.store(out_row + cols, val, mask=mask)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_gather_kernel(rank, kernel_name, code):
    code.newline()
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics({'BLOCK_SIZE_N': lambda args: 8192})")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["inp, ", "index, ", "out, "]
        args += [f"inp_shape{i}," for i in range(rank)]
        args += [f"index_shape{i}, " for i in range(rank)]
        args += [f"out_shape{i}, " for i in range(rank)]
        args += [f"inp_stride{i}, " for i in range(rank)]
        args += [f"index_stride{i}, " for i in range(rank)]
        args += [f"out_stride{i}, " for i in range(rank)]
        args += ["dim, ", "dim_stride, ", "N, ", "BLOCK_SIZE_N: tl.constexpr, "]
        code.writelines(args)
    code.writeline("):")
    with code.indent():
        code.writeline("pid = tle.program_id(0)")
        code.writeline("num_ctas = tl.num_programs(0)")
        code.writeline("num_blocks = tl.cdiv(N, BLOCK_SIZE_N)")
        code.writeline("for block_idx in range(pid, num_blocks, num_ctas):")
        with code.indent():
            code.writeline(
                "offset = block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)"
            )
            code.newline()
            code.writeline("cur_offset = offset")
            for i in range(rank - 1, -1, -1):
                code.writeline(f"index_idx{i} = cur_offset % index_shape{i}")
                code.writeline(f"cur_offset = cur_offset // index_shape{i}")
            code.newline()
            comp = [f"index_idx{i} * index_stride{i}" for i in range(rank)]
            code.writeline(f"index_offset = {' + '.join(comp)}")
            code.writeline("mask = offset < N")
            code.writeline(
                "cur_index = tl.load(index + index_offset, mask=mask, other=0)"
            )
            code.newline()
            comp = [f"index_idx{i} * inp_stride{i}" for i in range(rank)]
            code.writeline(f"inp_offset = {' + '.join(comp)}")
            code.writeline("inp_offset += cur_index * dim_stride")
            code.writeline("cur_inp = tl.load(inp + inp_offset, mask=mask, other=0)")
            code.newline()
            comp = [f"index_idx{i} * out_stride{i}" for i in range(rank)]
            code.writeline(f"out_offset = {' + '.join(comp)}")
            code.writeline("tl.store(out + out_offset, value=cur_inp, mask=mask)")
    code.newline()
    code.newline()
    return code


def generate_gather_wrapper(rank, wrapper_name, kernel_name, code):
    code.writeline(f"def {wrapper_name}(inp, dim, index, out, dim_stride, N):")
    with code.indent():
        code.writeline("inp_shape = inp.shape")
        code.writeline("inp_stride = inp.stride()")
        code.writeline("index_shape = index.shape")
        code.writeline("index_stride = index.stride()")
        code.writeline("out_shape = out.shape")
        code.writeline("out_stride = out.stride()")
        code.writeline(
            "grid = lambda meta: (min(triton.cdiv(N, meta['BLOCK_SIZE_N']), 48), )"
        )
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["inp, ", "index, ", "out, "]
            args += [f"inp_shape[{i}], " for i in range(rank)]
            args += [f"index_shape[{i}], " for i in range(rank)]
            args += [f"out_shape[{i}], " for i in range(rank)]
            args += [f"inp_stride[{i}], " for i in range(rank)]
            args += [f"index_stride[{i}], " for i in range(rank)]
            args += [f"out_stride[{i}], " for i in range(rank)]
            args += ["dim, ", "dim_stride, ", "N, "]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return out")
    code.newline()
    code.newline()
    return code


def generate_code(inputs, wrapper_name, kernel_name, code):
    rank = inputs[0].ndim
    code = generate_imports(code)
    code = generate_gather_kernel(rank, kernel_name, code)
    code = generate_gather_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args, "_gather_wrapper", "_gather_flaggems_jit_function", code
            )
            file_name = f"gather_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        return args[0].ndim


_gather_func = GatherFunction()


def _gather_2d_fast(inp, dim, index, out):
    """Optimized path for 2D gather: row-based kernel avoids modulo."""
    M = index.shape[0]
    N_idx = index.shape[1]
    # BLOCK_N capped at 2048: gather is data-dependent random access; a larger tile only lengthens a single
    # memory-access queue without increasing parallelism; on a private probe 8192→2048 is 5%~13% faster on
    # (1024,1024)/(4096,4096).
    BLOCK_N = min(triton.next_power_of_2(N_idx), 2048)
    grid_dim = min(M, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        if dim == -1 or dim == 1:
            gather_2d_row_kernel[(grid_dim,)](
                inp,
                index,
                out,
                M,
                N_idx,
                inp.stride(0),
                index.stride(0),
                out.stride(0),
                BLOCK_N=BLOCK_N,
                GRID_DIM=grid_dim,
                num_warps=8,
            )
        elif dim == 0:
            gather_2d_dim0_kernel[(grid_dim,)](
                inp,
                index,
                out,
                M,
                N_idx,
                inp.stride(0),
                inp.stride(0),
                index.stride(0),
                out.stride(0),
                BLOCK_N=BLOCK_N,
                GRID_DIM=grid_dim,
                num_warps=8,
            )
        else:
            return False
    return True


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_ENFLAME GATHER")
    if inp.ndim != index.ndim:
        raise IndexError(
            f"self and index must have the same number of dimensions, "
            f"got self.ndim = {inp.ndim} and index.ndim = {index.ndim}"
        )
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    if inp.ndim == 2:
        if _gather_2d_fast(inp, dim, index, out):
            return out

    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)
    N = index.numel()
    _gather_func(inp_strided, dim, index, out, dim_stride, N)
    return out


@libentry()
@triton.jit(
    do_not_specialize=[
        "rows",
        "n_idx",
        "grad_stride0",
        "grad_stride1",
        "idx_stride0",
        "idx_stride1",
        "out_stride0",
        "out_stride1",
    ]
)
def gather_backward_2d_kernel(
    grad_ptr,
    index_ptr,
    out_ptr,
    rows,
    n_idx,
    grad_stride0,
    grad_stride1,
    idx_stride0,
    idx_stride1,
    out_stride0,
    out_stride1,
    SCATTER_LAST_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    """Row-parallel scatter-add for 2D gather_backward.

    SCATTER_LAST_DIM selects the layout: True scatters the last dim (out row is
    indexed by the loaded index), False scatters dim 0 (the atomic address is
    index * out_stride0 while the column stays linear, i.e. lanes still write
    consecutive addresses and only cross-row collisions contend).
    """
    pid = tl.program_id(0)
    for row in tl.range(pid, rows, GRID_DIM):
        grad_row = grad_ptr + row * grad_stride0
        idx_row = index_ptr + row * idx_stride0
        out_row = out_ptr + row * out_stride0
        for col_start in tl.range(0, n_idx, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            mask = cols < n_idx
            idx = tl.load(idx_row + cols * idx_stride1, mask=mask, other=0)
            val = tl.load(grad_row + cols * grad_stride1, mask=mask, other=0.0)
            if SCATTER_LAST_DIM:
                tl.atomic_add(out_row + idx * out_stride1, val, mask=mask)
            else:
                tl.atomic_add(
                    out_ptr + idx * out_stride0 + cols * out_stride1, val, mask=mask
                )


def _gather_backward_2d(grad, index, out, dim):
    """Dispatch the row-parallel scatter kernel for a 2D index/grad pair."""
    rows = index.shape[0]
    n_idx = index.shape[1]
    block_n = min(triton.next_power_of_2(n_idx), GB_BLOCK_N_CAP)
    grid_dim = min(rows, NUM_SIPS * GB_GRID_PER_SIP)
    with torch_device_fn.device(grad.device):
        gather_backward_2d_kernel[(grid_dim,)](
            grad,
            index,
            out,
            rows,
            n_idx,
            grad.stride(0),
            grad.stride(1),
            index.stride(0),
            index.stride(1),
            out.stride(0),
            out.stride(1),
            SCATTER_LAST_DIM=(dim == 1),
            BLOCK_N=block_n,
            GRID_DIM=grid_dim,
            num_warps=GB_WARPS,
        )


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_ENFLAME GATHER_BACKWARD")
    result = grad.new_zeros(self.shape)
    if index.ndim == 2:
        ndim_dim = dim if dim >= 0 else dim + 2
        if ndim_dim in (0, 1):
            _gather_backward_2d(grad, index, result, ndim_dim)
            return result

    from .scatter import scatter_

    return scatter_(result, dim, index, grad, reduce="add")
