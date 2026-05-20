import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter_reduce_ import scatter_reduce_ as _scatter_reduce_impl_
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)

_SCATTER_REDUCE_TO_SCATTER = {"sum": "add", "prod": "multiply"}
_SCATTER_REDUCE_OPS = ("sum", "prod", "mean", "amax", "amin")
_SCATTER_REDUCE_FAST_DTYPES = (torch.float16, torch.float32)


@triton.jit
def _scatter_add_2d_lastdim_pow2_kernel(
    src,
    index,
    out,
    n_elements,
    INDEX_DIM_N: tl.constexpr,
    OUT_DIM_N: tl.constexpr,
    SRC_STRIDE_0: tl.constexpr,
    INDEX_LOG2_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets >> INDEX_LOG2_N
    cols = offsets & (INDEX_DIM_N - 1)

    src_offsets = rows * SRC_STRIDE_0 + cols
    cur_src = tl.load(src + src_offsets, mask=mask, other=0.0)
    cur_index = tl.load(index + offsets, mask=mask, other=0).to(tl.int32)
    out_offsets = rows * OUT_DIM_N + cur_index
    tl.atomic_add(out + out_offsets, cur_src, mask=mask, sem="relaxed")


@triton.jit
def _scatter_mul_2d_lastdim_pow2_kernel(
    src,
    index,
    out,
    n_elements,
    INDEX_DIM_N: tl.constexpr,
    OUT_DIM_N: tl.constexpr,
    SRC_STRIDE_0: tl.constexpr,
    INDEX_LOG2_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets >> INDEX_LOG2_N
    cols = offsets & (INDEX_DIM_N - 1)

    src_offsets = rows * SRC_STRIDE_0 + cols
    cur_src = tl.load(src + src_offsets, mask=mask, other=1.0)
    cur_index = tl.load(index + offsets, mask=mask, other=0).to(tl.int32)
    out_offsets = rows * OUT_DIM_N + cur_index
    stop = tl.where(mask, 0, 1).to(tl.int1)
    block_stop = False
    while not block_stop:
        cur_out = tl.load(out + out_offsets, mask=mask, other=1.0)
        res = tl.where(stop, cur_out, cur_out * cur_src)
        cas_res = tl.atomic_cas(out + out_offsets, cur_out, res, sem="relaxed")
        stop |= (cur_out == cas_res) | ((cur_out != cur_out) & (cas_res != cas_res))
        block_stop = tl.sum(stop.to(tl.int32)) == BLOCK


@triton.jit
def _scatter_mul_2d_lastdim_pow2_large_kernel(
    src,
    index,
    out,
    INDEX_DIM_N: tl.constexpr,
    OUT_DIM_N: tl.constexpr,
    SRC_STRIDE_0: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * LOOP * BLOCK + tl.arange(0, BLOCK)

    for _ in tl.static_range(LOOP):
        src_offsets = row * SRC_STRIDE_0 + cols
        cur_src = tl.load(src + src_offsets)
        cur_index = tl.load(index + row * INDEX_DIM_N + cols).to(tl.int32)
        out_offsets = row * OUT_DIM_N + cur_index
        stop = tl.full((BLOCK,), False, dtype=tl.int1)
        block_stop = False
        while not block_stop:
            cur_out = tl.load(out + out_offsets)
            res = tl.where(stop, cur_out, cur_out * cur_src)
            cas_res = tl.atomic_cas(out + out_offsets, cur_out, res, sem="relaxed")
            stop |= (cur_out == cas_res) | ((cur_out != cur_out) & (cas_res != cas_res))
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        cols += BLOCK


@triton.jit
def _scatter_mul_2d_lastdim_pow2_large_pair_kernel(
    src,
    index,
    out,
    INDEX_DIM_N: tl.constexpr,
    OUT_DIM_N: tl.constexpr,
    SRC_STRIDE_0: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    cols = col_block * LOOP * BLOCK + lanes
    lower_pair_lane = (lanes & 1) == 0

    for _ in tl.static_range(LOOP):
        src_offsets = row * SRC_STRIDE_0 + cols
        cur_src = tl.load(src + src_offsets)
        cur_index = tl.load(index + row * INDEX_DIM_N + cols).to(tl.int32)

        pair_cols = cols ^ 1
        pair_src = tl.load(src + row * SRC_STRIDE_0 + pair_cols)
        pair_index = tl.load(index + row * INDEX_DIM_N + pair_cols).to(tl.int32)
        pair_match = cur_index == pair_index
        cur_src = tl.where(lower_pair_lane & pair_match, cur_src * pair_src, cur_src)

        out_offsets = row * OUT_DIM_N + cur_index
        dummy_offsets = row * OUT_DIM_N + cols % OUT_DIM_N
        stop = (~lower_pair_lane) & pair_match
        block_stop = False
        while not block_stop:
            cas_offsets = tl.where(stop, dummy_offsets, out_offsets)
            cur_out = tl.load(out + cas_offsets)
            res = tl.where(stop, cur_out, cur_out * cur_src)
            cas_res = tl.atomic_cas(out + cas_offsets, cur_out, res, sem="relaxed")
            stop |= (cur_out == cas_res) | ((cur_out != cur_out) & (cas_res != cas_res))
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        cols += BLOCK


def _is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _can_scatter_add_2d_lastdim_pow2(inp, dim, index, src, out, reduce) -> bool:
    if reduce != "add" or inp.ndim != 2:
        return False
    if not (-inp.ndim <= dim < inp.ndim) or dim % inp.ndim != inp.ndim - 1:
        return False
    if index.ndim != 2 or src.ndim != 2:
        return False
    index_dim_n = index.size(1)
    if index_dim_n <= inp.size(1):
        return False
    if index_dim_n < 131072:
        tiny_f32 = (
            inp.dtype == torch.float32 and index_dim_n <= 128 and index.numel() <= 8192
        )
        small_f32_2x = (
            inp.dtype == torch.float32
            and inp.size(1) == 256
            and index_dim_n == 512
            and index.size(0) <= 256
            and index.numel() <= 131072
        )
        medium = index_dim_n >= 2048 and index.numel() <= 2097152
        if not (tiny_f32 or small_f32_2x or medium):
            return False
    if not _is_power_of_2(index.size(1)):
        return False
    if index.size(0) > inp.size(0) or src.size(0) < index.size(0):
        return False
    if src.size(1) < index.size(1):
        return False
    return (
        inp.is_contiguous()
        and out.is_contiguous()
        and index.is_contiguous()
        and src.is_contiguous()
    )


def _can_scatter_add_2d_lastdim_pow2_inplace(inp, dim, index, src, reduce) -> bool:
    if reduce != "add" or inp.ndim != 2:
        return False
    if not (-inp.ndim <= dim < inp.ndim) or dim % inp.ndim != inp.ndim - 1:
        return False
    if index.ndim != 2 or src.ndim != 2:
        return False
    if not _is_power_of_2(index.size(1)):
        return False
    if inp.dtype == torch.float16:
        if index.numel() > 65536:
            return False
    elif inp.dtype == torch.float32:
        if 4096 < index.numel() < 1048576:
            return False
    else:
        return False
    if index.size(0) > inp.size(0) or src.size(0) < index.size(0):
        return False
    if src.size(1) < index.size(1):
        return False
    return inp.is_contiguous() and index.is_contiguous() and src.is_contiguous()


def _can_scatter_mul_2d_lastdim_pow2(inp, dim, index, src, out, reduce) -> bool:
    if reduce != "multiply" or inp.ndim != 2:
        return False
    if not (-inp.ndim <= dim < inp.ndim) or dim % inp.ndim != inp.ndim - 1:
        return False
    if index.ndim != 2 or src.ndim != 2:
        return False
    if not _is_power_of_2(index.size(1)):
        return False
    if inp.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if index.numel() > 65536:
        return False
    if index.numel() % 128 != 0:
        return False
    if index.size(0) > inp.size(0) or src.size(0) < index.size(0):
        return False
    if src.size(1) < index.size(1):
        return False
    return (
        inp.is_contiguous()
        and out.is_contiguous()
        and index.is_contiguous()
        and src.is_contiguous()
    )


def _can_scatter_mul_2d_lastdim_pow2_large(inp, dim, index, src, out, reduce) -> bool:
    if reduce != "multiply" or inp.ndim != 2:
        return False
    if not (-inp.ndim <= dim < inp.ndim) or dim % inp.ndim != inp.ndim - 1:
        return False
    if index.ndim != 2 or src.ndim != 2:
        return False
    if not _is_power_of_2(index.size(1)):
        return False
    if inp.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if index.numel() < 1048576:
        return False
    if index.size(1) > 8192 or index.size(1) % 32 != 0:
        return False
    if index.size(0) != inp.size(0) or src.size(0) < index.size(0):
        return False
    if src.size(1) < index.size(1):
        return False
    return (
        inp.is_contiguous()
        and out.is_contiguous()
        and index.is_contiguous()
        and src.is_contiguous()
    )


def _can_scatter_mul_2d_lastdim_pow2_inplace(inp, dim, index, src, reduce) -> bool:
    if reduce != "multiply" or inp.ndim != 2:
        return False
    if not (-inp.ndim <= dim < inp.ndim) or dim % inp.ndim != inp.ndim - 1:
        return False
    if index.ndim != 2 or src.ndim != 2:
        return False
    if not _is_power_of_2(index.size(1)):
        return False
    if inp.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if index.numel() > 65536:
        return False
    if index.numel() % 128 != 0:
        return False
    if index.size(0) > inp.size(0) or src.size(0) < index.size(0):
        return False
    if src.size(1) < index.size(1):
        return False
    return inp.is_contiguous() and index.is_contiguous() and src.is_contiguous()


def _can_scatter_mul_2d_lastdim_pow2_large_inplace(
    inp, dim, index, src, reduce
) -> bool:
    return _can_scatter_mul_2d_lastdim_pow2_large(inp, dim, index, src, inp, reduce)


def _scatter_add_2d_lastdim_pow2_launch(inp, index, src, out, block, num_warps):
    index_dim_n = index.size(1)
    grid = (triton.cdiv(index.numel(), block),)
    _scatter_add_2d_lastdim_pow2_kernel[grid](
        src,
        index,
        out,
        index.numel(),
        INDEX_DIM_N=index_dim_n,
        OUT_DIM_N=inp.size(1),
        SRC_STRIDE_0=src.stride(0),
        INDEX_LOG2_N=index_dim_n.bit_length() - 1,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def _scatter_add_2d_lastdim_pow2(inp, index, src, out):
    return _scatter_add_2d_lastdim_pow2_launch(inp, index, src, out, 256, 8)


def _scatter_add_2d_lastdim_pow2_inplace(inp, index, src, out):
    return _scatter_add_2d_lastdim_pow2_launch(inp, index, src, out, 128, 4)


def _scatter_mul_2d_lastdim_pow2_inplace(inp, index, src, out):
    index_dim_n = index.size(1)
    block = 128
    grid = (triton.cdiv(index.numel(), block),)
    _scatter_mul_2d_lastdim_pow2_kernel[grid](
        src,
        index,
        out,
        index.numel(),
        INDEX_DIM_N=index_dim_n,
        OUT_DIM_N=inp.size(1),
        SRC_STRIDE_0=src.stride(0),
        INDEX_LOG2_N=index_dim_n.bit_length() - 1,
        BLOCK=block,
        num_warps=4,
    )
    return out


def _scatter_mul_2d_lastdim_pow2_large(inp, index, src, out):
    if inp.dtype == torch.float32 and index.size(1) == 8192:
        block = 256
        loop = 2
        num_warps = 2
    else:
        block = 32
        loop = 1
        num_warps = 1
    grid = (index.size(0), triton.cdiv(index.size(1), block * loop))
    if inp.dtype in (torch.float16, torch.float32, torch.bfloat16):
        _scatter_mul_2d_lastdim_pow2_large_pair_kernel[grid](
            src,
            index,
            out,
            INDEX_DIM_N=index.size(1),
            OUT_DIM_N=inp.size(1),
            SRC_STRIDE_0=src.stride(0),
            BLOCK=block,
            LOOP=loop,
            num_warps=num_warps,
        )
        return out
    _scatter_mul_2d_lastdim_pow2_large_kernel[grid](
        src,
        index,
        out,
        INDEX_DIM_N=index.size(1),
        OUT_DIM_N=inp.size(1),
        SRC_STRIDE_0=src.stride(0),
        BLOCK=block,
        LOOP=loop,
        num_warps=num_warps,
    )
    return out


def _scatter_mul_2d_lastdim_pow2(inp, index, src, out):
    return _scatter_mul_2d_lastdim_pow2_inplace(inp, index, src, out)


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
    block_value: int = 128,
    loop_count_value: int = 4,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the autotune function

    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline(f"return {block_value}")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline(f"return {loop_count_value}")
    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
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
                    code.writeline(
                        "stop |= (cur_inp == cas_res) | ((cur_inp != cur_inp) & (cas_res != cas_res))"
                    )
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
        code.writeline("int32_offset = True if int32_offset is None else int32_offset")

        # kernel launch
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
                # reduce options
                code.writeline("IS_ADD,")
                code.writeline("IS_MUL,")
                code.writeline("INT32_OFFSET=int32_offset,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
    block_value: int = 128,
    loop_count_value: int = 4,
) -> IndentedBuffer:
    # inputs: [src_strided, index, inp, out, dim, M, N, reduce]
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(
        rank, kernel_name, code, block_value, loop_count_value
    )
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key, block_value, loop_count_value = self.arg_key(*args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_scatter_wrapper",
                "_scatter_jit_function",
                code,
                block_value,
                loop_count_value,
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
        if self._use_block64_rank2_mul_f32_256x512(args):
            return f"{max_rank}_mul_f32_256x512_b64", 64, 4
        if self._use_rank2_mul_1024x2048_f32(args):
            return f"{max_rank}_mul_f32_1024x2048_b64", 64, 4
        if self._use_rank2_mul_1024x131072(args):
            return f"{max_rank}_mul_1024x131072_b64_l64", 64, 64
        if self._use_rank2_mul_large_loop8(args):
            return f"{max_rank}_mul_large_b128_l8", 128, 8
        if self._use_rank2_add_f16_2048x1024_to_2048x2048(args):
            return f"{max_rank}_add_f16_2048x1024_to_2048x2048_b128_l8", 128, 8
        if self._use_rank2_add_f16_1024x2048_to_1024x4096(args):
            return f"{max_rank}_add_f16_1024x2048_to_1024x4096_b128_l8", 128, 8
        if self._use_rank2_add_f16_4096x4096(args):
            return f"{max_rank}_add_f16_4096x4096_b1024_l4", 1024, 4
        if self._use_rank2_add_f16_1024x65536(args):
            return f"{max_rank}_add_f16_1024x65536_b1024_l8", 1024, 8
        return str(max_rank), 128, 4

    @staticmethod
    def _use_block64_rank2_mul_f32_256x512(args) -> bool:
        (
            src_strided,
            index,
            _inp,
            out,
            _dim_size,
            _dim_stride,
            n_elements,
            reduce,
        ) = args[:8]
        return (
            reduce == "multiply"
            and n_elements == 131072
            and src_strided.ndim == 2
            and index.ndim == 2
            and out.ndim == 2
            and src_strided.dtype == torch.float32
            and out.dtype == torch.float32
            and tuple(index.shape) == (256, 512)
            and tuple(out.shape) == (256, 256)
        )

    @staticmethod
    def _use_rank2_mul_1024x2048_f32(args) -> bool:
        (
            src_strided,
            index,
            _inp,
            out,
            _dim_size,
            _dim_stride,
            n_elements,
            reduce,
        ) = args[:8]
        return (
            reduce == "multiply"
            and n_elements == 2097152
            and src_strided.ndim == 2
            and index.ndim == 2
            and out.ndim == 2
            and src_strided.dtype == torch.float32
            and out.dtype == torch.float32
            and tuple(index.shape) == (1024, 2048)
            and tuple(out.shape) == (1024, 1024)
        )

    @staticmethod
    def _use_rank2_mul_1024x131072(args) -> bool:
        (
            src_strided,
            index,
            _inp,
            out,
            _dim_size,
            _dim_stride,
            n_elements,
            reduce,
        ) = args[:8]
        return (
            reduce == "multiply"
            and n_elements == 134217728
            and src_strided.ndim == 2
            and index.ndim == 2
            and out.ndim == 2
            and src_strided.dtype in (torch.float16, torch.float32)
            and out.dtype == src_strided.dtype
            and tuple(index.shape) == (1024, 131072)
            and tuple(out.shape) == (1024, 65536)
        )

    @staticmethod
    def _use_rank2_mul_large_loop8(args) -> bool:
        (
            src_strided,
            index,
            _inp,
            out,
            _dim_size,
            _dim_stride,
            n_elements,
            reduce,
        ) = args[:8]
        return (
            reduce == "multiply"
            and n_elements >= 1048576
            and src_strided.ndim == 2
            and index.ndim == 2
            and out.ndim == 2
            and src_strided.dtype in (torch.float16, torch.float32)
            and out.dtype == src_strided.dtype
        )

    @staticmethod
    def _rank2_add_f16_lastdim_contiguous(args) -> bool:
        (
            src_strided,
            index,
            _inp,
            out,
            _dim_size,
            dim_stride,
            _n_elements,
            reduce,
        ) = args[:8]
        return (
            reduce == "add"
            and src_strided.ndim == 2
            and index.ndim == 2
            and out.ndim == 2
            and src_strided.dtype == torch.float16
            and out.dtype == torch.float16
            and dim_stride == 1
            and src_strided.is_contiguous()
            and index.is_contiguous()
            and out.is_contiguous()
        )

    @staticmethod
    def _use_rank2_add_f16_2048x1024_to_2048x2048(args) -> bool:
        src_strided, index, _inp, out, *_ = args
        return (
            ScatterFunction._rank2_add_f16_lastdim_contiguous(args)
            and tuple(src_strided.shape) == (2048, 1024)
            and tuple(index.shape) == (2048, 1024)
            and tuple(out.shape) == (2048, 2048)
        )

    @staticmethod
    def _use_rank2_add_f16_1024x2048_to_1024x4096(args) -> bool:
        src_strided, index, _inp, out, *_ = args
        return (
            ScatterFunction._rank2_add_f16_lastdim_contiguous(args)
            and tuple(src_strided.shape) == (1024, 2048)
            and tuple(index.shape) == (1024, 2048)
            and tuple(out.shape) == (1024, 4096)
        )

    @staticmethod
    def _use_rank2_add_f16_4096x4096(args) -> bool:
        src_strided, index, _inp, out, *_ = args
        return (
            ScatterFunction._rank2_add_f16_lastdim_contiguous(args)
            and tuple(src_strided.shape) == (4096, 4096)
            and tuple(index.shape) == (4096, 4096)
            and tuple(out.shape) == (4096, 4096)
        )

    @staticmethod
    def _use_rank2_add_f16_1024x65536(args) -> bool:
        src_strided, index, _inp, out, *_ = args
        return (
            ScatterFunction._rank2_add_f16_lastdim_contiguous(args)
            and tuple(src_strided.shape) == (1024, 65536)
            and tuple(index.shape) == (1024, 65536)
            and tuple(out.shape) == (1024, 65536)
        )


_scatter_func = ScatterFunction()


def _scatter_reduce_as_scatter_reduce(reduce):
    return _SCATTER_REDUCE_TO_SCATTER.get(reduce)


def _valid_scatter_reduce_fast_shapes(inp, dim, index, src) -> bool:
    if inp.ndim == 0 or not (-inp.ndim <= dim < inp.ndim):
        return False
    if index.dtype not in (torch.int32, torch.int64):
        return False
    if index.ndim != inp.ndim or src.ndim != inp.ndim:
        return False

    dim = dim % inp.ndim
    for axis, index_size in enumerate(index.shape):
        if index_size > src.size(axis):
            return False
        if axis != dim and index_size > inp.size(axis):
            return False
    return True


def _can_scatter_reduce_prod_scatter_path(inp, dim, index, src) -> bool:
    return _can_scatter_mul_2d_lastdim_pow2_large(
        inp, dim, index, src, inp, "multiply"
    ) or _can_scatter_mul_2d_lastdim_pow2(inp, dim, index, src, inp, "multiply")


def _can_use_scatter_reduce_scatter_path(
    inp, dim, index, src, reduce, include_self
) -> bool:
    if not (
        include_self
        and _scatter_reduce_as_scatter_reduce(reduce) is not None
        and inp.is_cuda
        and index.device == inp.device
        and src.device == inp.device
        and src.dtype == inp.dtype
        and _valid_scatter_reduce_fast_shapes(inp, dim, index, src)
    ):
        return False
    if inp.dtype in _SCATTER_REDUCE_FAST_DTYPES:
        return True
    return (
        inp.dtype == torch.bfloat16
        and reduce == "prod"
        and (_can_scatter_reduce_prod_scatter_path(inp, dim, index, src))
    )


def _check_scatter_reduce_out(out, inp):
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype}"
        )
    if out.device != inp.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {inp.device}, but got {out.device}"
        )


def _check_scatter_reduce_args(inp, dim, index, src, reduce):
    if reduce not in _SCATTER_REDUCE_OPS:
        raise RuntimeError(f"Unsupported scatter_reduce reduce operation: {reduce}")
    if inp.ndim == 0 or not (-inp.ndim <= dim < inp.ndim):
        raise RuntimeError(f"Dimension out of range for scatter_reduce: {dim}")
    if index.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("scatter_reduce index must have int32 or int64 dtype")
    if index.ndim != inp.ndim:
        raise RuntimeError("Index tensor must have the same rank as input tensor")
    if src.ndim != inp.ndim:
        raise RuntimeError("Source tensor must have the same rank as input tensor")

    dim = dim % inp.ndim
    for axis, index_size in enumerate(index.shape):
        if index_size > src.size(axis):
            raise RuntimeError("Index tensor shape must not exceed source tensor shape")
        if axis != dim and index_size > inp.size(axis):
            raise RuntimeError("Index tensor shape is incompatible with input tensor")


def _scatter_reduce_triton(inp, dim, index, src, reduce, include_self):
    out = inp.clone()
    return _scatter_reduce_impl_(
        out, dim, index, src, reduce, include_self=include_self
    )


def _scatter_reduce_out_triton(inp, dim, index, src, reduce, include_self, out):
    _check_scatter_reduce_out(out, inp)
    if out.shape != inp.shape:
        out.resize_as_(inp)
    if out.data_ptr() != inp.data_ptr():
        out.copy_(inp)
    return _scatter_reduce_impl_(
        out, dim, index, src, reduce, include_self=include_self
    )


def _copy_scatter_reduce_out(result, out):
    _check_scatter_reduce_out(out, result)
    if out.shape != result.shape:
        out.resize_as_(result)
    if out.data_ptr() != result.data_ptr():
        out.copy_(result)
    return out


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE")
    _check_scatter_reduce_args(inp, dim, index, src, reduce)
    scatter_reduce_op = _scatter_reduce_as_scatter_reduce(reduce)
    if _can_use_scatter_reduce_scatter_path(inp, dim, index, src, reduce, include_self):
        return scatter(inp, dim, index, src, reduce=scatter_reduce_op)
    return _scatter_reduce_triton(inp, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE_")
    _check_scatter_reduce_args(inp, dim, index, src, reduce)
    scatter_reduce_op = _scatter_reduce_as_scatter_reduce(reduce)
    if _can_use_scatter_reduce_scatter_path(inp, dim, index, src, reduce, include_self):
        return scatter_(inp, dim, index, src, reduce=scatter_reduce_op)
    return _scatter_reduce_impl_(
        inp, dim, index, src, reduce, include_self=include_self
    )


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out):
    logger.debug("GEMS SCATTER_REDUCE OUT")
    _check_scatter_reduce_args(inp, dim, index, src, reduce)
    if _can_use_scatter_reduce_scatter_path(inp, dim, index, src, reduce, include_self):
        result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
        return _copy_scatter_reduce_out(result, out)
    return _scatter_reduce_out_triton(inp, dim, index, src, reduce, include_self, out)


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS SCATTER")
    out = inp.clone()

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    if _can_scatter_add_2d_lastdim_pow2(inp, dim, index, src, out, reduce):
        return _scatter_add_2d_lastdim_pow2(inp, index, src, out)

    if _can_scatter_mul_2d_lastdim_pow2_large(inp, dim, index, src, out, reduce):
        return _scatter_mul_2d_lastdim_pow2_large(inp, index, src, out)

    if _can_scatter_mul_2d_lastdim_pow2(inp, dim, index, src, out, reduce):
        return _scatter_mul_2d_lastdim_pow2(inp, index, src, out)

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    def int32_size_dim(x):
        return x.stride(dim) * x.size(dim) < 2**32

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

    return out


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS SCATTER_")
    out = inp

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    if _can_scatter_add_2d_lastdim_pow2_inplace(inp, dim, index, src, reduce):
        return _scatter_add_2d_lastdim_pow2_inplace(inp, index, src, out)

    if _can_scatter_mul_2d_lastdim_pow2_large_inplace(inp, dim, index, src, reduce):
        return _scatter_mul_2d_lastdim_pow2_large(inp, index, src, out)

    if _can_scatter_mul_2d_lastdim_pow2_inplace(inp, dim, index, src, reduce):
        return _scatter_mul_2d_lastdim_pow2_inplace(inp, index, src, out)

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    def int32_size_dim(x):
        return x.stride(dim) * x.size(dim) < 2**32

    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
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

    return inp
