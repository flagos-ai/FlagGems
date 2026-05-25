import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)


@triton.jit
def scatter_add_2d_kernel(
    out_ptr, src_ptr, index_ptr,
    M, N, out_stride0, out_stride1,
    src_stride0, src_stride1,
    idx_stride0, idx_stride1,
    scatter_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    total = M * N
    base = pid * LOOP * BLOCK
    for _i in tl.static_range(LOOP):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < total
        row = offs // N
        col = offs % N

        src_off = row * src_stride0 + col * src_stride1
        idx_off = row * idx_stride0 + col * idx_stride1

        cur_src = tl.load(src_ptr + src_off, mask=mask, other=0.0)
        cur_idx = tl.load(index_ptr + idx_off, mask=mask, other=0)

        if scatter_dim == 0:
            out_off = cur_idx * out_stride0 + col * out_stride1
        else:
            out_off = row * out_stride0 + cur_idx * out_stride1

        tl.atomic_add(out_ptr + out_off, cur_src, mask=mask, sem='relaxed')
        base += BLOCK


@triton.jit
def scatter_add_row_kernel(
    out_ptr, src_ptr, index_ptr,
    N,
    out_stride0,
    src_stride0,
    idx_stride0,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    out_base = row * out_stride0
    src_base = row * src_stride0
    idx_base = row * idx_stride0

    for col_start in range(0, N, BLOCK_N):
        offs = col_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        s = tl.load(src_ptr + src_base + offs, mask=mask, other=0.0)
        i = tl.load(index_ptr + idx_base + offs, mask=mask, other=0)
        tl.atomic_add(out_ptr + out_base + i, s, mask=mask, sem='relaxed')


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
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return 4")
    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
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
            # reduce options
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                    code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
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
                    "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
                )
            code.writeline("elif IS_MUL: ")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, cur_inp * cur_src)")
                    code.writeline(
                        "cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res, sem='relaxed')"
                    )
                    code.writeline("stop |= cur_inp == cas_res")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

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

        code.writeline("import math")
        code.writeline("BLOCK = 128")
        code.writeline("LOOP = 4")
        code.writeline("while math.ceil(N / (BLOCK * LOOP)) > 65535:")
        code.writeline("    if BLOCK < 1024:")
        code.writeline("        BLOCK *= 2")
        code.writeline("    else:")
        code.writeline("        LOOP *= 2")

        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]), ')
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
                code.writeline("IS_ADD,")
                code.writeline("IS_MUL,")
                code.writeline("BLOCK=BLOCK,")
                code.writeline("LOOP=LOOP,")
                code.writeline("INT32_OFFSET=int32_offset,")
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


def _reduce_name_to_scatter_reduce(reduce):
    if reduce == "add":
        return "sum"
    elif reduce == "multiply":
        return "prod"
    return reduce


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS SCATTER")

    orig_dtype = inp.dtype
    needs_upcast = reduce == "multiply" and orig_dtype == torch.float16
    if needs_upcast:
        inp = inp.to(torch.float32)
        src = src.to(torch.float32)

    if reduce is not None:
        out = inp.clone()
        scatter_(out, dim, index, src, reduce=reduce)
        if needs_upcast:
            out = out.to(orig_dtype)
        return out

    out = inp.clone()

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    if index.dtype == torch.int64:
        index = index.to(torch.int32)
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
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    if needs_upcast:
        out = out.to(orig_dtype)
    return out


def _adaptive_scatter_config(total):
    if total <= 8192:
        return 256, 4
    elif total <= 131072:
        return 256, 8
    else:
        return 1024, 4


@triton.jit(do_not_specialize=['total_elements'])
def scatter_src_2d_kernel(
    out_ptr, src_ptr, index_ptr,
    total_elements,
    N,
    out_stride0, out_stride1,
    src_stride0, src_stride1,
    idx_stride0, idx_stride1,
    scatter_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elements
    row = offs // N
    col = offs % N

    src_off = row * src_stride0 + col * src_stride1
    idx_off = row * idx_stride0 + col * idx_stride1

    cur_src = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    cur_idx = tl.load(index_ptr + idx_off, mask=mask, other=0)

    if scatter_dim == 0:
        out_off = cur_idx * out_stride0 + col * out_stride1
    else:
        out_off = row * out_stride0 + cur_idx * out_stride1

    tl.store(out_ptr + out_off, cur_src, mask=mask)


@triton.jit(do_not_specialize=['N', 'M', 'out_stride0', 'src_stride0', 'idx_stride0'])
def scatter_src_row_kernel(
    out_ptr, src_ptr, index_ptr,
    N, M,
    out_stride0,
    src_stride0,
    idx_stride0,
    scatter_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for row in tl.range(pid, M, num_progs):
        out_base = row * out_stride0
        src_base = row * src_stride0
        idx_base = row * idx_stride0

        for col_start in range(0, N, BLOCK_N):
            offs = col_start + tl.arange(0, BLOCK_N)
            mask = offs < N
            s = tl.load(src_ptr + src_base + offs, mask=mask, other=0.0)
            i = tl.load(index_ptr + idx_base + offs, mask=mask, other=0)
            if scatter_dim == 0:
                tl.store(out_ptr + i * out_stride0 + offs, s, mask=mask)
            else:
                tl.store(out_ptr + out_base + i, s, mask=mask)


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS SCATTER_")

    orig_dtype = inp.dtype
    needs_upcast = reduce == "multiply" and orig_dtype == torch.float16
    if needs_upcast:
        out = inp.to(torch.float32)
        src = src.to(torch.float32)
    else:
        out = inp

    if reduce is not None:
        assert orig_dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    if reduce is None and inp.ndim == 2:
        dim_actual = dim % inp.ndim
        if index.dtype == torch.int64:
            index = index.to(torch.int32)
        M, N_col = index.shape
        total = M * N_col
        src_strided = src.as_strided(index.shape, src.stride())
        use_row = (dim_actual == 1 and inp.stride(1) == 1
                   and src_strided.stride(1) == 1 and index.stride(1) == 1
                   and N_col >= 64 and total > 4096)
        MAX_GRID = 48
        if use_row:
            BLOCK_N = min(N_col, 8192)
            if BLOCK_N < 32:
                BLOCK_N = 32
            nw = 1 if N_col <= 512 else 4
            grid_size = min(M, MAX_GRID)
            scatter_src_row_kernel[(grid_size,)](
                inp, src_strided, index,
                N_col, M,
                inp.stride(0),
                src_strided.stride(0),
                index.stride(0),
                dim_actual,
                BLOCK_N=BLOCK_N, num_warps=nw,
            )
        else:
            if total <= 8192:
                BLOCK = 1024
                nw = 1
            elif total <= 131072:
                BLOCK = 4096
                nw = 1
            else:
                BLOCK = 8192
                nw = 4
            grid = (triton.cdiv(total, BLOCK),)
            scatter_src_2d_kernel[grid](
                inp, src_strided, index,
                total, N_col,
                inp.stride(0), inp.stride(1),
                src.stride(0), src.stride(1),
                index.stride(0), index.stride(1),
                dim_actual,
                BLOCK=BLOCK, num_warps=nw,
            )
        return inp

    if reduce == "add" and inp.ndim == 2:
        dim_actual = dim % inp.ndim
        total = index.numel()
        M, N_col = index.shape
        mem_bytes = total * 4 * 3
        if mem_bytes > 5_000_000_000:
            CHUNK = max(1, 4_000_000_000 // (N_col * 4 * 3))
            for row_start in range(0, M, CHUNK):
                row_end = min(row_start + CHUNK, M)
                chunk_M = row_end - row_start
                idx_c = index[row_start:row_end]
                if idx_c.dtype == torch.int64:
                    idx_c = idx_c.to(torch.int32)
                if dim_actual == 1:
                    scatter_add_row_kernel[(chunk_M,)](
                        inp[row_start:row_end],
                        src[row_start:row_end],
                        idx_c,
                        N_col,
                        inp.stride(0), src.stride(0), idx_c.stride(0),
                        BLOCK_N=1024, num_warps=4,
                    )
                else:
                    ct = chunk_M * N_col
                    BLOCK, num_warps = _adaptive_scatter_config(ct)
                    LOOP = 4
                    while triton.cdiv(ct, BLOCK * LOOP) > 65535:
                        LOOP *= 2
                    grid = (triton.cdiv(ct, BLOCK * LOOP),)
                    scatter_add_2d_kernel[grid](
                        inp, src[row_start:row_end], idx_c,
                        chunk_M, N_col,
                        inp.stride(0), inp.stride(1),
                        src.stride(0), src.stride(1),
                        idx_c.stride(0), idx_c.stride(1),
                        dim_actual,
                        BLOCK=BLOCK, LOOP=LOOP, num_warps=num_warps,
                    )
            return inp
        idx = index.to(torch.int32) if index.dtype == torch.int64 else index
        BLOCK, num_warps = _adaptive_scatter_config(total)
        LOOP = 4
        while triton.cdiv(total, BLOCK * LOOP) > 65535:
            LOOP *= 2
        grid = (triton.cdiv(total, BLOCK * LOOP),)
        scatter_add_2d_kernel[grid](
            inp, src, idx,
            M, N_col,
            inp.stride(0), inp.stride(1),
            src.stride(0), src.stride(1),
            idx.stride(0), idx.stride(1),
            dim_actual,
            BLOCK=BLOCK,
            LOOP=LOOP,
            num_warps=num_warps,
        )
        return inp

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(out, dim, index.shape)
    dim_size = out.size(dim)
    dim_stride = out.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (out, index, src)))
    _scatter_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    if needs_upcast:
        inp.copy_(out.to(orig_dtype))
    return inp
