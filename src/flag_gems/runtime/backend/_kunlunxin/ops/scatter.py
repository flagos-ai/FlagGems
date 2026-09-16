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

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    # code.writeline("from flag_gems.utils import triton_lang_extension as ext")
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

    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline(
            'return triton.next_power_of_2(triton.cdiv(triton.cdiv(args["N"], 12), 4))'
        )  # UNROLL = 4
    code.newline()
    code.newline()

    # Number of linear index elements owned by one program.
    #
    # tl.atomic_add on this backend silently drops updates when the same output
    # address is touched by more than one program (2 contending programs are
    # already enough to lose).  Duplicate index values can only alias inside one
    # "slice" of prod(index.shape[dim:]) consecutive linear elements, so for the
    # reducing variants the per-program span is forced to a whole multiple of
    # that slice; every possible address conflict then stays program-local.
    # Plain scatter (tl.store, no reduction) has unspecified semantics for
    # duplicate indices, so it keeps the original partitioning.
    code.writeline("def heur_span(args):")
    with code.indent():
        code.writeline("span = heur_block(args) * 4")
        code.writeline('if args["LINE_LOCAL"]:')
        with code.indent():
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

    code.writeline("def heur_loop(args):")
    with code.indent():
        code.writeline("return triton.cdiv(heur_span(args), heur_block(args))")
    code.newline()
    code.newline()

    # When SPAN is a whole multiple of BLOCK the LOOP tiles cover the program's
    # range exactly, so the plain `offsets < N` mask is already enough and the
    # generated code stays byte-for-byte as fast as before.  Only the ragged case
    # needs the extra upper bound, which costs a scalar min plus a mask the
    # backend cannot fold into a block DMA.
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
            code.writeline('"LOOP": heur_loop,')
            code.writeline('"EXACT_SPAN": heur_exact,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N',"
        f"'stride_dim','inp_size_dim',"
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
            # reduce options
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            # filled in by the heuristics above, must stay after every
            # positionally passed argument
            code.writeline("SPAN: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("EXACT_SPAN: tl.constexpr,")
            code.writeline("LINE_LOCAL: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr,")
            code.writeline("DENSE_SRC_IDX: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        # Program pid owns exactly the linear range [base, base + SPAN).  The
        # `< base + SPAN` half of the mask is what keeps two programs from ever
        # touching the same element when SPAN is not a multiple of BLOCK.
        code.writeline("base = pid * SPAN")
        code.writeline("if not EXACT_SPAN:")
        with code.indent():
            code.writeline("limit = tl.minimum(base + SPAN, N)")
        # These are loop invariant and MUST be narrowed here rather than inside
        # the tile loop: the tile loop is a real (non-unrolled) loop now, and
        # re-assigning an argument inside it makes it loop-carried with a type
        # that changes between the first and later iterations.

        # Keep `offsets` loop carried (as before) instead of rebuilding it from
        # `loop_iter` each iteration: re-materialising tl.arange(0, BLOCK) inside
        # every unrolled body costs ~7x at BLOCK=2048 on this backend.
        code.writeline("offsets = base + tl.arange(0, BLOCK)")

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            if True:
                code.writeline("if EXACT_SPAN:")
                with code.indent():
                    code.writeline("mask = offsets < N")
                code.writeline("else:")
                with code.indent():
                    code.writeline("mask = offsets < limit")
                code.writeline("cur_idx = offsets")
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                # Fast path: when src/index are contiguous over index.shape, the
                # linear iteration offset IS the element offset. Using it
                # directly keeps the src/index gm2lm contiguous (block DMA)
                # instead of the modulo-based address math that forces
                # discrete/scalar transfers.
                code.writeline("if DENSE_SRC_IDX:")
                with code.indent():
                    code.writeline("idx_offsets = offsets")
                    code.writeline("src_offsets = offsets")
                code.writeline("else:")
                with code.indent():
                    code.writeline("if INT32_OFFSET:")
                    with code.indent():
                        code.writeline(
                            "idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)"
                        )
                        code.writeline(
                            "src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)"
                        )
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(
                            "idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)"
                        )
                        code.writeline(
                            "src_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)"
                        )
                for i in range(rank)[::-1]:
                    code.writeline("if INT32_OFFSET:")
                    with code.indent():
                        code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                        code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                        code.writeline(
                            f"index_stride_{i} = index_stride_{i}.to(tl.int32)"
                        )
                        code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
                    code.writeline(f"mod = cur_idx % shape_{i}")
                    code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                    code.writeline("if not DENSE_SRC_IDX:")
                    with code.indent():
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
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline("cur_index = cur_index.to(tl.int32)")
                    code.writeline("stride_dim = stride_dim.to(tl.int32)")

                code.writeline("dim_offsets = cur_index * stride_dim")
                code.writeline("inp_offsets += dim_offsets")
                code.newline()
                code.writeline("if IS_ADD: ")
                with code.indent():
                    code.writeline(
                        "tl.atomic_add(out + inp_offsets, cur_src, mask=mask,"
                        " sem='relaxed')"
                    )
                code.writeline("elif IS_MUL: ")
                with code.indent():
                    code.writeline(
                        "tl.atomic_mul(out + inp_offsets, cur_src, mask=mask,"
                        " sem='relaxed')"
                    )

                code.writeline("else: ")
                with code.indent():
                    code.writeline("tl.store(out + inp_offsets, cur_src, mask=mask)")

                code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # src_strided, index, inp, out, dim, M, N, reduce
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    parameters.append("reduce: tl.constexpr=None")
    parameters.append("int32_offset: tl.constexpr=None")

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

        code.writeline('IS_ADD = reduce == "add"')
        code.writeline('IS_MUL = reduce == "multiply"')
        code.writeline("int32_offset = int32_offset or True")

        # src/index are dense (row-major contiguous over index.shape) iff their
        # strides equal the contiguous strides of index.shape. In that case the
        # linear offset can be used directly for the src/index loads.
        code.writeline("_cont = [1] * len(index_shapes)")
        code.writeline("_acc = 1")
        code.writeline("for _i in range(len(index_shapes) - 1, -1, -1):")
        with code.indent():
            code.writeline("_cont[_i] = _acc")
            code.writeline("_acc *= index_shapes[_i]")
        code.writeline(
            "dense_src_idx = list(src_strides) == _cont and list(index_strides) == _cont"
        )

        # Duplicate index values can only collide inside one slice of
        # prod(index.shape[dim:]) consecutive linear elements.
        code.writeline("SLICE = 1")
        code.writeline("for _i in range(dim, len(index_shapes)):")
        with code.indent():
            code.writeline("SLICE *= index_shapes[_i]")
        code.writeline("line_local = bool(IS_ADD or IS_MUL)")

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
                # reduce options
                code.writeline("IS_ADD,")
                code.writeline("IS_MUL,")
                code.writeline("LINE_LOCAL=line_local,")
                code.writeline("INT32_OFFSET=int32_offset,")
                code.writeline("DENSE_SRC_IDX=dense_src_idx,")
                # code.writeline("buffer_size_limit=512,")
                # code.writeline("isCloseUnrollControl=True,")

        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [src_strided, index, inp, out, dim, M, N, reduce]
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
                "_scatter_wrapper",
                "_scatter_jit_function",
                code,
            )

            file_name = f"scatter_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_KUNLUNXIN SCATTER")
    if dim < 0:
        dim += inp.ndim
    out = inp.clone()

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    return out


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_KUNLUNXIN SCATTER_")
    if dim < 0:
        dim += inp.ndim
    out = inp

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    return inp
