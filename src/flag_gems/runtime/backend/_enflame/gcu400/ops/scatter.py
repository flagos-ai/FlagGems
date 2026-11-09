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

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    # code.writeline("from flag_gems.utils import triton_lang_extension as tle")
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
        code.writeline("return 32")
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
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("num_ctas = tl.num_programs(0)")
        code.writeline("num_jobs = tl.cdiv(N, BLOCK * LOOP)")
        code.writeline("for job_id in range(pid, num_jobs, num_ctas):")
        with code.indent():
            code.writeline("if not INT32_OFFSET:")
            with code.indent():
                code.writeline("job_id = job_id.to(tl.int64)")
            code.writeline("offsets = job_id * LOOP * BLOCK + tl.arange(0, BLOCK)")

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
                        code.writeline(
                            f"index_stride_{i} = index_stride_{i}.to(tl.int32)"
                        )
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
                        code.writeline(
                            "res = tl.where(stop, cur_inp, cur_inp * cur_src)"
                        )
                        code.writeline(
                            "cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res, sem='relaxed')"
                        )
                        code.writeline("stop |= cur_inp == cas_res")
                        code.writeline(
                            "block_stop = tl.sum(stop.to(tl.int32)) == BLOCK"
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

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline(
                f'min(triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]), {MAX_GRID_DIM}), '
            )
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
    logger.debug("GEMS_ENFLAME SCATTER")

    orig_dtype = inp.dtype
    needs_upcast = reduce == "multiply" and orig_dtype == torch.float16
    if needs_upcast:
        inp = inp.to(torch.float32)
        src = src.to(torch.float32)

    out = inp.clone()

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    src_strided = src.as_strided(index.shape, src.stride())
    # src 必须用 as_strided 之后的视图判断：原始 src 的前导维可能更大
    if reduce in (None, "add") and is_last_dim_flat(out, dim, index, src_strided):
        scatter_last_dim(out, dim, index, src_strided, reduce == "add")
        if needs_upcast:
            out = out.to(orig_dtype)
        return out
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


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_ENFLAME SCATTER_")

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

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    src_restrided = src.as_strided(index.shape, src.stride())
    if reduce in (None, "add") and is_last_dim_flat(out, dim, index, src_restrided):
        scatter_last_dim(out, dim, index, src_restrided, reduce == "add")
        if needs_upcast:
            inp.copy_(out.to(orig_dtype))
        return inp
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


# ---------------------------------------------------------------------------
# 尾维 scatter 快速路径
#
# benchmark 与绝大多数真实调用都是沿最后一维 scatter：index/src 除 scatter
# 维外的位置与输出对应，写地址 = 行基址 + index 值。此时不再需要逐元素做
# rank 次 `/` `%` 的多维坐标拆解——把前导维压成「行」、尾维当「列」，用
# BLOCK_M × BLOCK_N 的 2D tile 覆盖，tile 内部只有一次乘加。
# ---------------------------------------------------------------------------

LAST_DIM_TILE_BYTES = 4 * 1024
LAST_DIM_MAX_BLOCK_N = 2048
LAST_DIM_ATOMIC_BLOCK_N = 4096


@triton.jit(
    do_not_specialize=[
        "rows",
        "width",
        "idx_row_stride",
        "src_row_stride",
        "out_row_stride",
        "num_col_tile",
    ]
)
def scatter_last_dim_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    rows,
    width,
    idx_row_stride,
    src_row_stride,
    out_row_stride,
    num_col_tile,
    IS_ADD: tl.constexpr,
    INT32: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """沿尾维 scatter：tile = BLOCK_M 行 × BLOCK_N 列。

    rows/width 为 index 展平后的行数与行宽；三个 row_stride 分别是
    index/src/out 展平后的行步长（要求前导维链式齐次，见 is_last_dim_flat）。
    """
    pid = tl.program_id(0)
    num_cta = tl.num_programs(0)
    num_tile = tl.cdiv(rows, BLOCK_M) * num_col_tile
    for tile in range(pid, num_tile, num_cta):
        # 每个 tile 只在这里做一次除法，摊到 BLOCK_M*BLOCK_N 个元素上
        row_tile = tile // num_col_tile
        col_tile = tile - row_tile * num_col_tile
        row_idx = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        col_idx = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_off = row_idx[:, None] * idx_row_stride + col_idx[None, :]
        src_off = row_idx[:, None] * src_row_stride + col_idx[None, :]
        if EVEN:
            # 整 tile 落在计算域内：省掉逐元素谓词，原子路径尤其划算
            index_val = tl.load(index_ptr + idx_off)
            src_val = tl.load(src_ptr + src_off)
        else:
            mask = (row_idx[:, None] < rows) & (col_idx[None, :] < width)
            index_val = tl.load(index_ptr + idx_off, mask=mask, other=0)
            src_val = tl.load(src_ptr + src_off, mask=mask, other=0)
        if INT32:
            out_off = row_idx[:, None].to(tl.int32) * out_row_stride
            out_off += index_val.to(tl.int32)
        else:
            out_off = row_idx[:, None].to(tl.int64) * out_row_stride
            out_off += index_val
        if IS_ADD:
            if EVEN:
                tl.atomic_add(out_ptr + out_off, src_val, sem="relaxed")
            else:
                tl.atomic_add(out_ptr + out_off, src_val, mask=mask, sem="relaxed")
        else:
            if EVEN:
                tl.store(out_ptr + out_off, src_val)
            else:
                tl.store(out_ptr + out_off, src_val, mask=mask)


def _max_offset(tensor):
    """张量可能访问到的最大线性偏移，用于判断能否用 int32 计算地址。"""
    return sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    )


def _flat_leading(tensor, rank):
    """前导维是否链式齐次：可把 rank-1 个前导维压成一个「行」维。"""
    if tensor.stride(rank - 1) != 1:
        return False
    for i in range(rank - 2):
        if tensor.stride(i) != tensor.shape(i + 1) * tensor.stride(i + 1):
            return False
    return True


def is_last_dim_flat(out, dim, index, src):
    """判断能否走尾维 2D tile 快速路径。

    行的展平以 index 的前导维为基准，因此三个张量的前导维形状必须与
    index 完全一致（维度更大时行号无法用一个步长映射过去）。
    """
    rank = index.dim()
    if rank < 2:
        return False
    # dim 可能为负值（-1 最常见），先归一化再判断是否为尾维
    if dim % rank != rank - 1:
        return False
    if out.dim() != rank or src.dim() != rank:
        return False
    if index.shape[-1] == 0 or index.numel() == 0:
        return False
    leading = tuple(index.shape[:-1])
    for tensor in (out, index, src):
        if tuple(tensor.shape[:-1]) != leading:
            return False
        if not _flat_leading(tensor, rank):
            return False
    return True


def _last_dim_tile(rows, width, element_size, add):
    """选 (BLOCK_M, BLOCK_N)。

    事务性写（普通 scatter）按 16 KiB 字节预算铺 tile；原子写（scatter_add）
    的耗时完全由原子次数决定，实测单行覆盖 4096 列比 2048 列更快，因此走
    单行 + 大列块的形状。并行度不足时先收窄列块、再收窄行块。
    """
    budget = max(1, LAST_DIM_TILE_BYTES // element_size)

    def _rows_per_tile(col_block):
        """非原子路径按字节预算铺行；行数足够多时单行 tile 更快。"""
        if rows >= 8 * MAX_GRID_DIM:
            return 1
        return max(1, min(triton.next_power_of_2(rows), budget // col_block))

    if add:
        block_n = min(triton.next_power_of_2(width), LAST_DIM_ATOMIC_BLOCK_N)
        block_m = 1
    else:
        block_n = min(triton.next_power_of_2(width), LAST_DIM_MAX_BLOCK_N)
        block_m = _rows_per_tile(block_n)
    num_col_tile = triton.cdiv(width, block_n)
    # 并行度不足时只收窄列块：小 case 把行块也切碎会让 tile 循环开销翻上来
    while triton.cdiv(rows, block_m) * num_col_tile < MAX_GRID_DIM and block_n > 64:
        block_n //= 2
        num_col_tile = triton.cdiv(width, block_n)
        if not add:
            block_m = _rows_per_tile(block_n)
    return block_m, block_n, num_col_tile


def scatter_last_dim(out, dim, index, src, add):
    """尾维 scatter 的 host 侧派发，就地写 out 并返回它。"""
    rank = index.dim()
    width = index.shape[-1]
    rows = index.numel() // width
    element_size = max(out.element_size(), index.element_size())
    block_m, block_n, num_col_tile = _last_dim_tile(rows, width, element_size, add)
    grid = min(MAX_GRID_DIM, triton.cdiv(rows, block_m) * num_col_tile)
    even = rows % block_m == 0 and width % block_n == 0
    int32 = all(_max_offset(t) < 2**31 for t in (out, index, src))
    scatter_last_dim_kernel[(grid,)](
        src,
        index,
        out,
        rows,
        width,
        index.stride(rank - 2),
        src.stride(rank - 2),
        out.stride(rank - 2),
        num_col_tile,
        IS_ADD=add,
        INT32=int32,
        EVEN=even,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        # 原子写吞吐随 warp 数线性增长并在 4 处饱和，普通写 2 warp 已够
        num_warps=4 if add else 2,
    )
    return out
