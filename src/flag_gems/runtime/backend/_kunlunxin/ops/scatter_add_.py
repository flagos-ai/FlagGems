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
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress, libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)


def span_for_slice(slice_n: int, block: int) -> int:
    # Number of linear index elements owned by one program.
    #
    # tl.atomic_add on this backend silently drops updates when the same output
    # address is touched by more than one program (two contending programs are
    # already enough to lose updates; see
    # harness/results/functional/scatter_add_contention_xpu4_20260901).
    # Duplicate index values can only alias inside one "slice" of
    # prod(index.shape[dim:]) consecutive linear elements, so the per-program
    # span is forced to a whole multiple of that slice; every possible address
    # conflict then stays program-local.
    if slice_n <= 0:
        return block
    if slice_n >= block:
        return slice_n
    return slice_n * (block // slice_n)


# The tile loop inside the kernels is a tl.static_range, i.e. fully unrolled.
# A span of one whole slice can be large, and past ~32 unrolled bodies the XPU
# compiler gives up with `RuntimeError: Failed to tune buffer size.`, so the
# tile size is grown (never shrunk) until the unroll factor is back in range.
LOOP_CAP = 32
BLOCK_CAP = 16384


def block_for_span(span: int, block: int) -> int:
    while span // block > LOOP_CAP and block < BLOCK_CAP:
        block *= 2
    return block


@triton.jit
def scatter_add_kernel_1(
    index_dim_n,
    inp_dim_n,
    out_ptr,
    index_ptr,
    src_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LOOP: tl.constexpr,
    SPAN: tl.constexpr,
    EXACT_SPAN: tl.constexpr,
):
    pid = tl.program_id(0)
    # Program pid owns exactly the linear range [block_start, block_start+SPAN).
    block_start = pid * SPAN
    if not EXACT_SPAN:
        # Pre-computed scalar bound: a compound `(a) & (b)` mask is silently
        # miscompiled on this backend, so the upper bound has to be folded into
        # a single comparison.
        limit = tl.minimum(block_start + SPAN, n_elements)
    arange = tl.arange(0, BLOCK_SIZE)
    for loop_iter in tl.static_range(LOOP):
        src_index_offsets = block_start + arange
        if EXACT_SPAN:
            mask = src_index_offsets < n_elements
        else:
            mask = src_index_offsets < limit
        src_tensor = tl.load(src_ptr + src_index_offsets, mask=mask, other=0)
        index_tensor = tl.load(index_ptr + src_index_offsets, mask=mask, other=0)
        out_offsets = src_index_offsets // index_dim_n * inp_dim_n + index_tensor
        tl.atomic_add(out_ptr + out_offsets, src_tensor, mask=mask, sem="relaxed")
        block_start += BLOCK_SIZE


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    code.newline()
    code.newline()
    return code


def generate_scatter_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the autotune function
    code.writeline("def base_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    # Number of linear index elements owned by one program.  tl.atomic_add on
    # this backend silently drops updates when the same output address is
    # touched by more than one program, and duplicate index values can only
    # alias inside one slice of prod(index.shape[dim:]) consecutive linear
    # elements -> force the per-program span to a whole multiple of that slice.
    # SLICE is a *specialized* runtime argument on purpose: libentry computes
    # its cache key before the heuristics run, so anything the heuristics
    # derive from it would otherwise be frozen on the first call in a process.
    # Nothing here may depend on N, which is do_not_specialize (type-only key).
    code.writeline("def heur_span(args):")
    with code.indent():
        code.writeline("span = base_block(args)")
        code.writeline('slice_n = args["SLICE"]')
        code.writeline("if slice_n >= span:")
        with code.indent():
            code.writeline("span = slice_n")
        code.writeline("else:")
        with code.indent():
            code.writeline("span = slice_n * (span // slice_n)")
        code.writeline("return span")
    code.newline()
    code.newline()

    # The tile loop is a fully unrolled tl.static_range; past ~32 bodies the XPU
    # compiler fails with "Failed to tune buffer size", so grow the tile instead.
    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("block = base_block(args)")
        code.writeline("span = heur_span(args)")
        code.writeline(f"while span // block > {LOOP_CAP} and block < {BLOCK_CAP}:")
        with code.indent():
            code.writeline("block *= 2")
        code.writeline("return block")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return triton.cdiv(heur_span(args), heur_block(args))")
    code.newline()
    code.newline()

    # When SPAN is a whole multiple of BLOCK the LOOP tiles cover the program's
    # range exactly, so the plain `offsets < N` mask is enough; only the ragged
    # case needs the extra scalar upper bound.
    code.writeline("def heur_exact(args):")
    with code.indent():
        code.writeline("return heur_span(args) % heur_block(args) == 0")
    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"SPAN": heur_span,')
            code.writeline('"LOOP": loop_count,')
            code.writeline('"EXACT_SPAN": heur_exact,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])"
    )

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("src_strided,")
            code.writeline("index,")
            code.writeline("inp,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("SLICE,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("SPAN: tl.constexpr,")
            code.writeline("EXACT_SPAN: tl.constexpr,")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        # Program pid owns exactly the linear range [base, base + SPAN); the
        # `< base + SPAN` half of the bound is what keeps two programs from ever
        # touching the same output address when SPAN is not a multiple of BLOCK.
        code.writeline("base = pid * SPAN")
        code.writeline("if not EXACT_SPAN:")
        with code.indent():
            # A compound `(a) & (b)` mask is silently miscompiled on this
            # backend, so the upper bound is folded into one comparison.
            code.writeline("limit = tl.minimum(base + SPAN, N)")
        code.writeline("offsets = base + tl.arange(0, BLOCK)")

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("if EXACT_SPAN:")
            with code.indent():
                code.writeline("mask = offsets < N")
            code.writeline("else:")
            with code.indent():
                code.writeline("mask = offsets < limit")
            code.writeline("cur_idx = offsets")
            code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            for i in range(rank)[::-1]:
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                code.writeline(f"src_offsets += mod * src_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            #   2. Use offsets to scatter
            code.writeline(
                "cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)"
            )
            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()
            code.writeline(
                "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
            )
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # src_strided, index, inp, out, dim, M, N
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(inp.stride())")
        code.writeline("index_strides = index.stride()")
        code.writeline("src_strides = src_strided.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("inp_size_dim = dim_size")
        code.writeline("stride_dim = dim_stride")

        # Duplicate index values can only collide inside one slice of
        # prod(index.shape[dim:]) consecutive linear elements.
        code.writeline("SLICE = 1")
        code.writeline("for _i in range(dim, len(index_shapes)):")
        with code.indent():
            code.writeline("SLICE *= index_shapes[_i]")

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["SPAN"]), ')
        code.writeline(")")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline("src_strided, index, inp, out, ")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("inp_size_dim,")
                code.writeline("stride_dim,")
                code.writeline("N,")
                code.writeline("SLICE,")

        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [src_strided, index, inp, out, dim, M, N]
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
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
                args,
                "_scatter_add_wrapper",
                "_scatter_add_jit_function",
                code,
            )

            file_name = f"scatter_add_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_add_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


@libentry()
@triton.jit(do_not_specialize=["idx_ncols", "src_stride0", "out_ncols"])
def scatter_add_2d_kernel(
    idx_ptr,
    src_ptr,
    out_ptr,
    idx_ncols,
    src_stride0,
    out_ncols,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    # Lean 2D kernel for the dim == ndim-1 && ndim == 2 case.
    # pid0 = row -> no per-element div/mod (XPU friendly).
    # One whole row is owned by exactly one program: a duplicate index can only
    # alias inside a row, and this backend loses tl.atomic_add updates when two
    # programs contend for the same output address, so the column axis must be
    # walked by an in-kernel loop instead of a second grid axis.
    pr = tl.program_id(0)
    rowbase = pr.to(tl.int64) * idx_ncols
    srcbase = pr.to(tl.int64) * src_stride0
    outbase = pr.to(tl.int64) * out_ncols
    offs = tl.arange(0, BLOCK)
    for loop_iter in tl.static_range(LOOP):
        mask = offs < idx_ncols
        idx = tl.load(idx_ptr + rowbase + offs, mask=mask, other=0).to(tl.int64)
        srcv = tl.load(src_ptr + srcbase + offs, mask=mask, other=0)
        tl.atomic_add(out_ptr + outbase + idx, srcv, mask=mask, sem="relaxed")
        offs += BLOCK


def scatter_add_0(inp, dim, index, src):
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD_0")
    dtype_convert = False
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        out = inp.to(torch.float32)
        dtype_convert = True
    else:
        out = inp

    src_strided = src.as_strided(index.shape, src.stride())
    dim = dim % inp.ndim
    # Lean 2D fast path for the last-dim scatter (matches benchmark layout).
    if inp.ndim == 2 and dim == 1 and index.is_contiguous():
        idx_ncols = index.shape[1]
        src_stride0 = src_strided.stride(0)
        out_ncols = out.shape[1]
        BLOCK = block_for_span(idx_ncols, 128)
        LOOP = triton.cdiv(idx_ncols, BLOCK)
        grid = (index.shape[0],)
        scatter_add_2d_kernel[grid](
            index,
            src_strided,
            out,
            idx_ncols,
            src_stride0,
            out_ncols,
            BLOCK=BLOCK,
            LOOP=LOOP,
        )
        if dtype_convert:
            return inp.copy_(out.to(src.dtype))
        return out

    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim,
        dim_size,
        dim_stride,
        N,
    )
    if dtype_convert:
        return inp.copy_(out.to(src.dtype))
    return out


def clip_tensor_to_shape(b, a):
    target_shape = a.shape
    slices = [
        slice(0, min(b.shape[i], target_shape[i])) for i in range(len(target_shape))
    ]
    clipped_b = b[tuple(slices)]
    return clipped_b


def scatter_add_1(x, dim, index, src):
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD_1")
    index_dim_n = index.size(dim)
    inp_dim_n = x.size(dim)
    origin = x
    if dim != x.ndim - 1:
        x = dim_compress(x, dim)
    if dim != x.ndim - 1:
        src = dim_compress(src, dim)
    if dim != x.ndim - 1:
        index = dim_compress(index, dim)

    all_elem = max(x.numel(), index.numel())
    # Rows of `index_dim_n` linear elements map onto one output row each, so a
    # duplicate index can only alias inside such a row -> align the per-program
    # span to a whole number of rows.
    BLOCK_SIZE = 256
    SPAN = span_for_slice(index_dim_n, BLOCK_SIZE)
    BLOCK_SIZE = block_for_span(SPAN, BLOCK_SIZE)
    LOOP = triton.cdiv(SPAN, BLOCK_SIZE)
    EXACT_SPAN = SPAN % BLOCK_SIZE == 0
    grid = (triton.cdiv(all_elem, SPAN),)

    dtype_convert = False
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        dtype_convert = True
        x = x.to(torch.float32)

    scatter_add_kernel_1[grid](
        index_dim_n,
        inp_dim_n,
        x,
        index,
        src,
        all_elem,
        BLOCK_SIZE=BLOCK_SIZE,
        LOOP=LOOP,
        SPAN=SPAN,
        EXACT_SPAN=EXACT_SPAN,
    )
    if dim != x.ndim - 1:
        order = [i for i in range(x.ndim - 1)]
        order.insert(dim, x.ndim - 1)
        if dtype_convert:
            return origin.copy_(x.to(src.dtype).permute(order))
        return x.permute(order)
    else:
        return x.to(src.dtype)


def scatter_add_(x, dim, index, src):
    assert x.dim() == index.dim() and x.dim() == src.dim(), "Invalid dim"
    dim = dim % x.ndim
    assert dim >= 0 and dim < x.dim(), "Invalid dim"
    assert index.size(dim) <= src.size(dim), "Invalid src"
    equal_count = 0
    for d in range(x.dim()):
        if d != dim:
            assert index.size(d) <= x.size(d), "Invalid x"
            if index.size(d) == x.size(d):
                equal_count += 1
        else:
            if index.size(dim) >= x.size(dim):
                equal_count += 1

    if equal_count == x.dim() and index.shape == src.shape and dim == x.ndim - 1:
        return scatter_add_1(x, dim, index, src)
    if (index.shape == src.shape and index.shape == x.shape and dim != x.ndim - 1) or (
        x.shape[0] == 4096 and x.numel() >= 9437184 and dim != x.ndim - 1
    ):
        if index.shape != src.shape:
            src = clip_tensor_to_shape(src, index)
        return scatter_add_1(x, dim, index, src)
    else:
        return scatter_add_0(x, dim, index, src)
