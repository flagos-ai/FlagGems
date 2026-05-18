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
    code.newline()
    code.newline()
    return code


def generate_scatter_reduce_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
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
            code.writeline("IS_SUM: tl.constexpr,")
            code.writeline("IS_PROD: tl.constexpr,")
            code.writeline("IS_AMAX: tl.constexpr,")
            code.writeline("IS_AMIN: tl.constexpr,")
            code.writeline("IS_MEAN: tl.constexpr,")
            code.writeline("IS_FLOAT32: tl.constexpr,")
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

            # Sum/mean reduction using native atomic_add, including fp16/bf16.
            code.writeline("if IS_SUM or IS_MEAN:")
            with code.indent():
                code.writeline(
                    "tl.atomic_add("
                    "out + inp_offsets, cur_src, mask=mask, sem='relaxed'"
                    ")"
                )

            # Product reduction using CAS loop
            code.writeline("elif IS_PROD:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, cur_inp * cur_src)")
                    code.writeline(
                        "cas_res = tl.atomic_cas("
                        "out + inp_offsets, cur_inp, res, sem='relaxed'"
                        ")"
                    )
                    code.writeline(
                        "stop |= (cur_inp == cas_res) | "
                        "((cur_inp != cur_inp) & (cas_res != cas_res))"
                    )
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            # Max reduction using CAS loop (atomic_max doesn't support float)
            code.writeline("elif IS_AMAX:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where((cur_src != cur_src) | "
                        "(cur_inp != cur_inp), cur_src + cur_inp, "
                        "tl.where(cur_src > cur_inp, cur_src, cur_inp))"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, new_val)")
                    code.writeline(
                        "cas_res = tl.atomic_cas("
                        "out + inp_offsets, cur_inp, res, sem='relaxed'"
                        ")"
                    )
                    code.writeline(
                        "stop |= (cur_inp == cas_res) | "
                        "((cur_inp != cur_inp) & (cas_res != cas_res))"
                    )
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            # Min reduction using CAS loop (atomic_min doesn't support float)
            code.writeline("elif IS_AMIN:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where((cur_src != cur_src) | "
                        "(cur_inp != cur_inp), cur_src + cur_inp, "
                        "tl.where(cur_src < cur_inp, cur_src, cur_inp))"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, new_val)")
                    code.writeline(
                        "cas_res = tl.atomic_cas("
                        "out + inp_offsets, cur_inp, res, sem='relaxed'"
                        ")"
                    )
                    code.writeline(
                        "stop |= (cur_inp == cas_res) | "
                        "((cur_inp != cur_inp) & (cas_res != cas_res))"
                    )
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_scatter_reduce_2d_lastdim_kernel(
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate a 2D fast path for contiguous last-dimension scatter_reduce."""
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    code.writeline(
        "@triton.jit(do_not_specialize=["
        "'N','K','index_stride_0','src_stride_0','out_stride_0'])"
    )

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("src_strided,")
        code.writeline("index,")
        code.writeline("out,")
        code.writeline("count,")
        code.writeline("N,")
        code.writeline("K,")
        code.writeline("index_stride_0: int,")
        code.writeline("src_stride_0: int,")
        code.writeline("out_stride_0: int,")
        code.writeline("IS_SUM: tl.constexpr,")
        code.writeline("IS_PROD: tl.constexpr,")
        code.writeline("IS_AMAX: tl.constexpr,")
        code.writeline("IS_AMIN: tl.constexpr,")
        code.writeline("IS_MEAN: tl.constexpr,")
        code.writeline("IS_FLOAT32: tl.constexpr,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("LOOP: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("row = offsets // K")
            code.writeline("col = offsets - row * K")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("row = row.to(tl.int32)")
                code.writeline("col = col.to(tl.int32)")
                code.writeline("index_stride_0 = index_stride_0.to(tl.int32)")
                code.writeline("src_stride_0 = src_stride_0.to(tl.int32)")
                code.writeline("out_stride_0 = out_stride_0.to(tl.int32)")
            code.writeline("index_offsets = row * index_stride_0 + col")
            code.writeline("src_offsets = row * src_stride_0 + col")
            code.writeline("cur_index = tl.load(index + index_offsets, mask=mask, other=0)")
            code.writeline("cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
            code.writeline("out_offsets = row * out_stride_0 + cur_index")
            code.newline()

            code.writeline("if IS_SUM or IS_MEAN:")
            with code.indent():
                code.writeline(
                    "tl.atomic_add("
                    "out + out_offsets, cur_src, mask=mask, sem='relaxed'"
                    ")"
                )
                code.writeline("if IS_MEAN:")
                with code.indent():
                    code.writeline("one = tl.full((BLOCK,), 1, dtype=tl.int32)")
                    code.writeline(
                        "tl.atomic_add("
                        "count + out_offsets, one, mask=mask, sem='relaxed'"
                        ")"
                    )

            code.writeline("elif IS_PROD:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_out = tl.load(out + out_offsets, mask=mask, other=0)"
                    )
                    code.writeline("res = tl.where(stop, cur_out, cur_out * cur_src)")
                    code.writeline(
                        "cas_res = tl.atomic_cas("
                        "out + out_offsets, cur_out, res, sem='relaxed'"
                        ")"
                    )
                    code.writeline(
                        "stop |= (cur_out == cas_res) | "
                        "((cur_out != cur_out) & (cas_res != cas_res))"
                    )
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("elif IS_AMAX:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_out = tl.load(out + out_offsets, mask=mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where((cur_src != cur_src) | "
                        "(cur_out != cur_out), cur_src + cur_out, "
                        "tl.where(cur_src > cur_out, cur_src, cur_out))"
                    )
                    code.writeline("res = tl.where(stop, cur_out, new_val)")
                    code.writeline(
                        "cas_res = tl.atomic_cas("
                        "out + out_offsets, cur_out, res, sem='relaxed'"
                        ")"
                    )
                    code.writeline(
                        "stop |= (cur_out == cas_res) | "
                        "((cur_out != cur_out) & (cas_res != cas_res))"
                    )
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("elif IS_AMIN:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_out = tl.load(out + out_offsets, mask=mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where((cur_src != cur_src) | "
                        "(cur_out != cur_out), cur_src + cur_out, "
                        "tl.where(cur_src < cur_out, cur_src, cur_out))"
                    )
                    code.writeline("res = tl.where(stop, cur_out, new_val)")
                    code.writeline(
                        "cas_res = tl.atomic_cas("
                        "out + out_offsets, cur_out, res, sem='relaxed'"
                        ")"
                    )
                    code.writeline(
                        "stop |= (cur_out == cas_res) | "
                        "((cur_out != cur_out) & (cas_res != cas_res))"
                    )
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_scatter_reduce_2d_lastdim_key_kernels(
    init_kernel_name: str,
    scatter_kernel_name: str,
    finalize_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate ordered-int key amax/amin kernels for the 2D last-dim fast path."""
    code.newline()
    code.writeline("@triton.jit")
    code.writeline("def _scatter_reduce_float_to_ordered_i32(x):")
    with code.indent():
        code.writeline("xf = x.to(tl.float32)")
        code.writeline("bits = xf.to(tl.int32, bitcast=True)")
        code.writeline("sign = bits >> 31")
        code.writeline("return bits ^ (sign & 0x7FFFFFFF)")
    code.newline()
    code.writeline("@triton.jit")
    code.writeline("def _scatter_reduce_ordered_i32_to_float(k):")
    with code.indent():
        code.writeline("neg = k < 0")
        code.writeline("bits = tl.where(neg, k ^ 0x7FFFFFFF, k)")
        code.writeline("return bits.to(tl.float32, bitcast=True)")
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    code.writeline("@triton.jit(do_not_specialize=['N_OUT'])")
    code.writeline(f"def {init_kernel_name}(")
    with code.indent():
        code.writeline("out,")
        code.writeline("keys,")
        code.writeline("nan_flags,")
        code.writeline("N_OUT,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("LOOP: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")
    code.writeline("):")
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N_OUT")
            code.writeline("cur = tl.load(out + offsets, mask=mask, other=0.0)")
            code.writeline("is_nan = cur != cur")
            code.writeline("key = _scatter_reduce_float_to_ordered_i32(cur)")
            code.writeline("tl.store(keys + offsets, key, mask=mask)")
            code.writeline(
                "tl.store(nan_flags + offsets, is_nan.to(tl.int32), mask=mask)"
            )
            code.writeline("offsets += BLOCK")
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    code.writeline(
        "@triton.jit(do_not_specialize=["
        "'N','K','index_stride_0','src_stride_0','out_stride_0'])"
    )
    code.writeline(f"def {scatter_kernel_name}(")
    with code.indent():
        code.writeline("src_strided,")
        code.writeline("index,")
        code.writeline("keys,")
        code.writeline("nan_flags,")
        code.writeline("N,")
        code.writeline("K,")
        code.writeline("index_stride_0: int,")
        code.writeline("src_stride_0: int,")
        code.writeline("out_stride_0: int,")
        code.writeline("IS_AMAX: tl.constexpr,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("LOOP: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")
    code.writeline("):")
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("row = offsets // K")
            code.writeline("col = offsets - row * K")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("row = row.to(tl.int32)")
                code.writeline("col = col.to(tl.int32)")
                code.writeline("index_stride_0 = index_stride_0.to(tl.int32)")
                code.writeline("src_stride_0 = src_stride_0.to(tl.int32)")
                code.writeline("out_stride_0 = out_stride_0.to(tl.int32)")
            code.writeline("index_offsets = row * index_stride_0 + col")
            code.writeline("src_offsets = row * src_stride_0 + col")
            code.writeline("cur_index = tl.load(index + index_offsets, mask=mask, other=0)")
            code.writeline("cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0.0)")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
            code.writeline("out_offsets = row * out_stride_0 + cur_index")
            code.writeline("is_nan = cur_src != cur_src")
            code.writeline("key = _scatter_reduce_float_to_ordered_i32(cur_src)")
            code.writeline(
                "tl.atomic_xchg("
                "nan_flags + out_offsets, 1, mask=mask & is_nan, sem='relaxed'"
                ")"
            )
            code.writeline("if IS_AMAX:")
            with code.indent():
                code.writeline(
                    "tl.atomic_max("
                    "keys + out_offsets, key, mask=mask & ~is_nan, sem='relaxed'"
                    ")"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline(
                    "tl.atomic_min("
                    "keys + out_offsets, key, mask=mask & ~is_nan, sem='relaxed'"
                    ")"
                )
            code.writeline("offsets += BLOCK")
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    code.writeline("@triton.jit(do_not_specialize=['N_OUT'])")
    code.writeline(f"def {finalize_kernel_name}(")
    with code.indent():
        code.writeline("keys,")
        code.writeline("nan_flags,")
        code.writeline("out,")
        code.writeline("N_OUT,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("LOOP: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")
    code.writeline("):")
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N_OUT")
            code.writeline("key = tl.load(keys + offsets, mask=mask, other=0)")
            code.writeline(
                "has_nan = tl.load(nan_flags + offsets, mask=mask, other=0) != 0"
            )
            code.writeline("val = _scatter_reduce_ordered_i32_to_float(key)")
            code.writeline("nan_val = val + float('nan')")
            code.writeline("tl.store(out + offsets, tl.where(has_nan, nan_val, val), mask=mask)")
            code.writeline("offsets += BLOCK")
    code.newline()
    code.newline()
    return code


def generate_count_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate kernel to count elements for mean reduction."""
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
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{shape_vars}])"
    )

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("count,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
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

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

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

            # Add 1 for each element
            code.writeline("one = tl.full((BLOCK,), 1, dtype=tl.int32)")
            code.writeline(
                "tl.atomic_add(count + inp_offsets, one, mask=mask, sem='relaxed')"
            )
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_init_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate kernel to set only indexed destinations to reduction identity."""
    code.newline()

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
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{shape_vars}])"
    )

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("IS_PROD: tl.constexpr,")
            code.writeline("IS_AMAX: tl.constexpr,")
            code.writeline("IS_AMIN: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
                code.writeline("stride_dim = stride_dim.to(tl.int32)")

            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.writeline("identity = tl.zeros((BLOCK,), dtype=tl.float32)")
            code.writeline("if IS_PROD:")
            with code.indent():
                code.writeline("identity = identity + 1.0")
            code.writeline("elif IS_AMAX:")
            with code.indent():
                code.writeline("identity = identity - float('inf')")
            code.writeline("elif IS_AMIN:")
            with code.indent():
                code.writeline("identity = identity + float('inf')")
            code.writeline("tl.store(out + inp_offsets, identity, mask=mask)")
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_init_2d_lastdim_kernel(
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate a 2D fast init path for include_self=False."""
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    code.writeline(
        "@triton.jit(do_not_specialize=["
        "'N','K','index_stride_0','out_stride_0'])"
    )

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("index,")
        code.writeline("out,")
        code.writeline("N,")
        code.writeline("K,")
        code.writeline("index_stride_0: int,")
        code.writeline("out_stride_0: int,")
        code.writeline("IS_PROD: tl.constexpr,")
        code.writeline("IS_AMAX: tl.constexpr,")
        code.writeline("IS_AMIN: tl.constexpr,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("LOOP: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("row = offsets // K")
            code.writeline("col = offsets - row * K")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("row = row.to(tl.int32)")
                code.writeline("col = col.to(tl.int32)")
                code.writeline("index_stride_0 = index_stride_0.to(tl.int32)")
                code.writeline("out_stride_0 = out_stride_0.to(tl.int32)")
            code.writeline("index_offsets = row * index_stride_0 + col")
            code.writeline("cur_index = tl.load(index + index_offsets, mask=mask, other=0)")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
            code.writeline("out_offsets = row * out_stride_0 + cur_index")
            code.writeline("identity = tl.zeros((BLOCK,), dtype=tl.float32)")
            code.writeline("if IS_PROD:")
            with code.indent():
                code.writeline("identity = identity + 1.0")
            code.writeline("elif IS_AMAX:")
            with code.indent():
                code.writeline("identity = identity - float('inf')")
            code.writeline("elif IS_AMIN:")
            with code.indent():
                code.writeline("identity = identity + float('inf')")
            code.writeline("tl.store(out + out_offsets, identity, mask=mask)")
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_mean_init_finalize_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate kernel to initialize mean counts or finalize mean output."""
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    out_stride_vars = ",".join(f"'out_stride_{i}'" for i in range(rank))
    count_stride_vars = ",".join(f"'count_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N_OUT',"
        f"{out_stride_vars},{count_stride_vars},{shape_vars}])"
    )

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("out,")
            code.writeline("count,")

            stride_args = ", ".join(f"out_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for out")

            stride_args = ", ".join(f"count_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for count")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("N_OUT,")
            code.writeline("INCLUDE_SELF: tl.constexpr,")
            code.writeline("IS_INIT: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N_OUT")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("out_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("count_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("out_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("count_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"out_stride_{i} = out_stride_{i}.to(tl.int32)")
                    code.writeline(f"count_stride_{i} = count_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"out_offsets += mod * out_stride_{i}")
                code.writeline(f"count_offsets += mod * count_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            code.writeline("if IS_INIT:")
            with code.indent():
                code.writeline("base = tl.full((BLOCK,), 0, dtype=tl.int32)")
                code.writeline("if INCLUDE_SELF:")
                with code.indent():
                    code.writeline("base += 1")
                code.writeline("tl.store(count + count_offsets, base, mask=mask)")
            code.writeline("else:")
            with code.indent():
                code.writeline(
                    "cur_count = tl.load(count + count_offsets, mask=mask, other=1)"
                )
                code.writeline("denom = tl.where(cur_count > 0, cur_count, 1)")
                code.writeline(
                    "cur_out = tl.load(out + out_offsets, mask=mask, other=0)"
                )
                code.writeline(
                    "tl.store(out + out_offsets, cur_out / denom, mask=mask)"
                )
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_mean_init_finalize_2d_lastdim_kernel(
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate a contiguous 2D mean count init/finalize fast path."""
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    code.writeline("@triton.jit(do_not_specialize=['N_OUT'])")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("out,")
        code.writeline("count,")
        code.writeline("N_OUT,")
        code.writeline("INCLUDE_SELF: tl.constexpr,")
        code.writeline("IS_INIT: tl.constexpr,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("LOOP: tl.constexpr,")
        code.writeline("INT32_OFFSET: tl.constexpr")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N_OUT")
            code.writeline("if IS_INIT:")
            with code.indent():
                code.writeline("base = tl.full((BLOCK,), 0, dtype=tl.int32)")
                code.writeline("if INCLUDE_SELF:")
                with code.indent():
                    code.writeline("base += 1")
                code.writeline("tl.store(count + offsets, base, mask=mask)")
            code.writeline("else:")
            with code.indent():
                code.writeline("cur_count = tl.load(count + offsets, mask=mask, other=1)")
                code.writeline("denom = tl.where(cur_count > 0, cur_count, 1)")
                code.writeline("cur_out = tl.load(out + offsets, mask=mask, other=0)")
                code.writeline("tl.store(out + offsets, cur_out / denom, mask=mask)")
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    parameters.append("reduce: tl.constexpr=None")
    parameters.append("include_self: tl.constexpr=True")
    parameters.append("int32_offset: tl.constexpr=None")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    count_kernel_name: str,
    init_kernel_name: str,
    mean_init_finalize_kernel_name: str,
    fast_kernel_name: str,
    fast_init_kernel_name: str,
    fast_mean_init_finalize_kernel_name: str,
    fast_key_init_kernel_name: str,
    fast_key_scatter_kernel_name: str,
    fast_key_finalize_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(inp.stride())")
        code.writeline("out_strides = out.stride()")
        code.writeline("index_strides = index.stride()")
        code.writeline("src_strides = src_strided.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("out_shapes = list(out.shape)")
        code.writeline("inp_size_dim = dim_size")
        code.writeline("stride_dim = dim_stride")

        code.writeline('IS_SUM = reduce == "sum"')
        code.writeline('IS_PROD = reduce == "prod"')
        code.writeline('IS_AMAX = reduce == "amax"')
        code.writeline('IS_AMIN = reduce == "amin"')
        code.writeline('IS_MEAN = reduce == "mean"')
        code.writeline("IS_FLOAT32 = out.dtype == torch.float32")
        code.writeline("int32_offset = True if int32_offset is None else int32_offset")

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]), ')
        code.writeline(")")

        if rank == 2:
            code.writeline("if (")
            with code.indent():
                code.writeline("N > 0")
                code.writeline("and len(index_shapes) == 2")
                code.writeline("and len(out_shapes) == 2")
                code.writeline("and index_shapes[1] > 0")
                code.writeline("and stride_dim == 1")
                code.writeline("and inp_strides[1] == 0")
                code.writeline("and inp_strides[0] == out_strides[0]")
                code.writeline("and out_strides[1] == 1")
                code.writeline("and out_strides[0] == out_shapes[1]")
                code.writeline("and index_strides[1] == 1")
                code.writeline("and src_strides[1] == 1")
            code.writeline("):")
            with code.indent():
                code.writeline("K = index_shapes[1]")
                code.writeline("index_stride_0 = index_strides[0]")
                code.writeline("src_stride_0 = src_strides[0]")
                code.writeline("out_stride_0 = out_strides[0]")
                code.writeline("if not include_self:")
                with code.indent():
                    init_launch: str = f"{fast_init_kernel_name}[grid]("
                    code.writeline(init_launch)
                    with code.indent():
                        code.writeline("index, out,")
                        code.writeline("N,")
                        code.writeline("K,")
                        code.writeline("index_stride_0,")
                        code.writeline("out_stride_0,")
                        code.writeline("IS_PROD,")
                        code.writeline("IS_AMAX,")
                        code.writeline("IS_AMIN,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                code.writeline("if (")
                with code.indent():
                    code.writeline("(IS_AMAX or IS_AMIN)")
                    code.writeline("and out.dtype in (torch.float16, torch.float32)")
                    code.writeline("and N >= 262144")
                    code.writeline("and out.numel() >= 4096")
                code.writeline("):")
                with code.indent():
                    code.writeline("keys = torch.empty_strided(")
                    with code.indent():
                        code.writeline("out.shape,")
                        code.writeline("out.stride(),")
                        code.writeline("device=out.device,")
                        code.writeline("dtype=torch.int32,")
                    code.writeline(")")
                    code.writeline("nan_flags = torch.empty_strided(")
                    with code.indent():
                        code.writeline("out.shape,")
                        code.writeline("out.stride(),")
                        code.writeline("device=out.device,")
                        code.writeline("dtype=torch.int32,")
                    code.writeline(")")
                    code.writeline("N_OUT = out.numel()")
                    code.writeline("out_grid = lambda meta: (")
                    with code.indent():
                        code.writeline(
                            'triton.cdiv(N_OUT, meta["BLOCK"] * meta["LOOP"]), '
                        )
                    code.writeline(")")
                    key_init_launch: str = f"{fast_key_init_kernel_name}[out_grid]("
                    code.writeline(key_init_launch)
                    with code.indent():
                        code.writeline("out, keys, nan_flags,")
                        code.writeline("N_OUT,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                    key_scatter_launch: str = (
                        f"{fast_key_scatter_kernel_name}[grid]("
                    )
                    code.writeline(key_scatter_launch)
                    with code.indent():
                        code.writeline("src_strided, index, keys, nan_flags,")
                        code.writeline("N,")
                        code.writeline("K,")
                        code.writeline("index_stride_0,")
                        code.writeline("src_stride_0,")
                        code.writeline("out_stride_0,")
                        code.writeline("IS_AMAX,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                    key_finalize_launch: str = (
                        f"{fast_key_finalize_kernel_name}[out_grid]("
                    )
                    code.writeline(key_finalize_launch)
                    with code.indent():
                        code.writeline("keys, nan_flags, out,")
                        code.writeline("N_OUT,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                    code.writeline("return out")
                code.writeline("if IS_MEAN:")
                with code.indent():
                    code.writeline("count = torch.empty_strided(")
                    with code.indent():
                        code.writeline("out.shape,")
                        code.writeline("out.stride(),")
                        code.writeline("device=out.device,")
                        code.writeline("dtype=torch.int32,")
                    code.writeline(")")
                    code.writeline("N_OUT = out.numel()")
                    code.writeline("out_grid = lambda meta: (")
                    with code.indent():
                        code.writeline(
                            'triton.cdiv(N_OUT, meta["BLOCK"] * meta["LOOP"]), '
                        )
                    code.writeline(")")
                    mean_init_launch: str = (
                        f"{fast_mean_init_finalize_kernel_name}[out_grid]("
                    )
                    code.writeline(mean_init_launch)
                    with code.indent():
                        code.writeline("out, count,")
                        code.writeline("N_OUT,")
                        code.writeline("include_self,")
                        code.writeline("True,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                    kernel_launch: str = f"{fast_kernel_name}[grid]("
                    code.writeline(kernel_launch)
                    with code.indent():
                        code.writeline("src_strided, index, out, count,")
                        code.writeline("N,")
                        code.writeline("K,")
                        code.writeline("index_stride_0,")
                        code.writeline("src_stride_0,")
                        code.writeline("out_stride_0,")
                        code.writeline("IS_SUM,")
                        code.writeline("IS_PROD,")
                        code.writeline("IS_AMAX,")
                        code.writeline("IS_AMIN,")
                        code.writeline("IS_MEAN,")
                        code.writeline("IS_FLOAT32,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                    mean_finalize_launch: str = (
                        f"{fast_mean_init_finalize_kernel_name}[out_grid]("
                    )
                    code.writeline(mean_finalize_launch)
                    with code.indent():
                        code.writeline("out, count,")
                        code.writeline("N_OUT,")
                        code.writeline("include_self,")
                        code.writeline("False,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                code.writeline("else:")
                with code.indent():
                    kernel_launch: str = f"{fast_kernel_name}[grid]("
                    code.writeline(kernel_launch)
                    with code.indent():
                        code.writeline("src_strided, index, out, out,")
                        code.writeline("N,")
                        code.writeline("K,")
                        code.writeline("index_stride_0,")
                        code.writeline("src_stride_0,")
                        code.writeline("out_stride_0,")
                        code.writeline("IS_SUM,")
                        code.writeline("IS_PROD,")
                        code.writeline("IS_AMAX,")
                        code.writeline("IS_AMIN,")
                        code.writeline("IS_MEAN,")
                        code.writeline("IS_FLOAT32,")
                        code.writeline("INT32_OFFSET=int32_offset,")
                    code.writeline(")")
                code.writeline("return out")

        code.writeline("if not include_self:")
        with code.indent():
            init_launch: str = f"{init_kernel_name}[grid]("
            code.writeline(init_launch)
            with code.indent():
                code.writeline("index, out, ")
                if rank > 0:
                    s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    code.writeline("inp_size_dim,")
                    code.writeline("stride_dim,")
                    code.writeline("N,")
                    code.writeline("IS_PROD,")
                    code.writeline("IS_AMAX,")
                    code.writeline("IS_AMIN,")
                    code.writeline("INT32_OFFSET=int32_offset,")
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
                code.writeline("IS_SUM,")
                code.writeline("IS_PROD,")
                code.writeline("IS_AMAX,")
                code.writeline("IS_AMIN,")
                code.writeline("IS_MEAN,")
                code.writeline("IS_FLOAT32,")
                code.writeline("INT32_OFFSET=int32_offset,")
        code.writeline(")")

        # Handle mean finalization through generated kernels.
        code.writeline("if IS_MEAN:")
        with code.indent():
            code.writeline("count = torch.empty_strided(")
            with code.indent():
                code.writeline("out.shape,")
                code.writeline("out.stride(),")
                code.writeline("device=out.device,")
                code.writeline("dtype=torch.int32,")
            code.writeline(")")
            code.writeline("count_strides = count.stride()")
            code.writeline("N_OUT = out.numel()")
            code.writeline("out_grid = lambda meta: (")
            with code.indent():
                code.writeline('triton.cdiv(N_OUT, meta["BLOCK"] * meta["LOOP"]), ')
            code.writeline(")")
            count_init_launch: str = f"{mean_init_finalize_kernel_name}[out_grid]("
            code.writeline(count_init_launch)
            with code.indent():
                code.writeline("out, count, ")
                if rank > 0:
                    s = ", ".join(f"out_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"count_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"out_shapes[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    code.writeline("N_OUT,")
                    code.writeline("include_self,")
                    code.writeline("True,")
                    code.writeline("INT32_OFFSET=int32_offset,")
            code.writeline(")")
            count_launch: str = f"{count_kernel_name}[grid]("
            code.writeline(count_launch)
            with code.indent():
                code.writeline("index, count, ")
                if rank > 0:
                    s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    code.writeline("inp_size_dim,")
                    code.writeline("stride_dim,")
                    code.writeline("N,")
                    code.writeline("INT32_OFFSET=int32_offset,")
            code.writeline(")")
            count_finalize_launch: str = f"{mean_init_finalize_kernel_name}[out_grid]("
            code.writeline(count_finalize_launch)
            with code.indent():
                code.writeline("out, count, ")
                if rank > 0:
                    s = ", ".join(f"out_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"count_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"out_shapes[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    code.writeline("N_OUT,")
                    code.writeline("include_self,")
                    code.writeline("False,")
                    code.writeline("INT32_OFFSET=int32_offset,")
            code.writeline(")")

        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    count_kernel_name: str,
    init_kernel_name: str,
    mean_init_finalize_kernel_name: str,
    fast_kernel_name: str,
    fast_init_kernel_name: str,
    fast_mean_init_finalize_kernel_name: str,
    fast_key_init_kernel_name: str,
    fast_key_scatter_kernel_name: str,
    fast_key_finalize_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_reduce_kernel(rank, kernel_name, code)
    if rank == 2:
        code = generate_scatter_reduce_2d_lastdim_kernel(fast_kernel_name, code)
        code = generate_scatter_reduce_2d_lastdim_key_kernels(
            fast_key_init_kernel_name,
            fast_key_scatter_kernel_name,
            fast_key_finalize_kernel_name,
            code,
        )
    code = generate_count_kernel(rank, count_kernel_name, code)
    code = generate_init_kernel(rank, init_kernel_name, code)
    if rank == 2:
        code = generate_init_2d_lastdim_kernel(fast_init_kernel_name, code)
    code = generate_mean_init_finalize_kernel(
        rank, mean_init_finalize_kernel_name, code
    )
    if rank == 2:
        code = generate_mean_init_finalize_2d_lastdim_kernel(
            fast_mean_init_finalize_kernel_name, code
        )
    code = generate_destination_passing_wrapper(
        rank,
        wrapper_name,
        kernel_name,
        count_kernel_name,
        init_kernel_name,
        mean_init_finalize_kernel_name,
        fast_kernel_name,
        fast_init_kernel_name,
        fast_mean_init_finalize_kernel_name,
        fast_key_init_kernel_name,
        fast_key_scatter_kernel_name,
        fast_key_finalize_kernel_name,
        code,
    )
    return code


class ScatterReduceFunction:
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
                "_scatter_reduce_wrapper",
                "_scatter_reduce_jit_function",
                "_scatter_reduce_count_jit_function",
                "_scatter_reduce_init_jit_function",
                "_scatter_reduce_mean_init_finalize_jit_function",
                "_scatter_reduce_2d_lastdim_jit_function",
                "_scatter_reduce_init_2d_lastdim_jit_function",
                "_scatter_reduce_mean_init_finalize_2d_lastdim_jit_function",
                "_scatter_reduce_key_init_2d_lastdim_jit_function",
                "_scatter_reduce_key_scatter_2d_lastdim_jit_function",
                "_scatter_reduce_key_finalize_2d_lastdim_jit_function",
                code,
            )

            file_name = f"scatter_reduce_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_scatter_reduce_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_reduce_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_reduce_func = ScatterReduceFunction()


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE_")
    out = inp

    # Validate reduce operation
    assert reduce in (
        "sum",
        "prod",
        "mean",
        "amax",
        "amin",
    ), f"Unsupported reduce operation: {reduce}"

    # Check for internal overlapping
    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    def int32_size_dim(x):
        return x.stride(dim) * x.size(dim) < 2**32

    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))

    _scatter_reduce_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        include_self,
        int32_offset=use_int32_offset,
    )

    return inp
