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

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    num_prog_x = tl.num_programs(axis=0)

    for k in range(0, tl.cdiv(M, num_prog_x * BLOCK_M)):
        row_block_id = k * num_prog_x + pid_x

        rows_offsets = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        rows_mask = rows_offsets < M
        cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

        out_mask = rows_mask & (cols_offsets < index_len)

        indices = tl.load(
            index + cols_offsets, mask=(cols_offsets < index_len), other=0
        )
        inp_off = rows_offsets * N + indices[None, :]
        out_off = rows_offsets * index_len + cols_offsets[None, :]

        selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
        tl.store(out + out_off, selected, mask=out_mask)


@libentry()
@triton.jit
def index_select_contiguous_kernel(
    inp,  # contiguous，逻辑 shape (outer, N, inner)
    out,  # contiguous，逻辑 shape (outer, index_len, inner)
    index,
    N,
    inner,
    index_len,
    total_p,
    BLOCK_I: tl.constexpr,
    GRID_X: tl.constexpr,
    OUTER_ONE: tl.constexpr,
):
    # 直接在输入原始布局上做连续访存，替代 dim_compress 的整张量搬运：
    # 固定 (o, j) 时，inner 维是连续的一段，可用一次连续 load/store 完成。
    # grid.x 固定为 GRID_X，用跨步循环覆盖全部 (o, j) 对；全展开
    # （grid.x=total_p）实测劣化 0.66~0.92x（CTA 过多、调度开销），
    # 固定 grid 更贴合 GCU 计算单元数。
    pid_x = tl.program_id(0)
    pid_i = tl.program_id(1)
    i_offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offs < inner
    for p in range(pid_x, total_p, GRID_X):
        if OUTER_ONE:
            # dim == 0 时 outer 恒为 1，p 就是 j，省掉整数除法/取模
            o = 0
            j = p
        else:
            o = p // index_len
            j = p % index_len
        idx = tl.load(index + j).to(tl.int32)
        in_off = (o * N + idx) * inner + i_offs
        out_off = p * inner + i_offs
        val = tl.load(inp + in_off, mask=i_mask, other=0)
        tl.store(out + out_off, val, mask=i_mask)


def make_contiguous_with_correct_stride(tensor):
    """确保 contiguous 后的 tensor 具有正确的 stride"""
    if tensor.numel() == 0:
        # 对于空 tensor，手动创建具有正确 stride 的版本
        new_tensor = tensor.flatten().view(tensor.shape)
        return new_tensor.contiguous()
    return tensor.contiguous()


def index_select(inp, dim, index):
    logger.debug("GEMS_ENFLAME INDEX_SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    # 不要写成 `assert (... for i in index)`：generator 创建时就会 iter(index)
    # 触发 Tensor.unbind，实测单次 0.31ms（占端到端 90%+），且 generator 恒为
    # truthy 本身就是无效检查。通用实现 src/flag_gems/ops/index_select.py 也
    # 无此校验；index 范围由调用方保证。

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()
    N = inp_shape[dim]

    inner = 1
    for s in inp_shape[dim + 1 :]:
        inner *= s
    outer = inp.numel() // (N * inner)
    total_p = outer * index_len

    # dim 不是最后一维且 dim 之后的连续块足够大时，直接在原布局上做连续访存，
    # 省掉 dim_compress 的 permute().contiguous() 整张量拷贝以及输出侧的
    # permute().contiguous() 拷贝（两次 O(numel) 搬运是主要瓶颈）。
    if dim != inp.ndim - 1 and inner >= 16 and index_len > 0:
        out_shape = list(inp_shape)
        out_shape[dim] = index_len
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        # grid.x 固定贴合 GCU 计算单元数：全展开（grid.x=total_p）实测劣化
        # 0.66x（CTA 过多、调度开销），固定 48 配跨步循环最优。
        grid_x = min(total_p, MAX_GRID_DIM)
        block_i = min(triton.next_power_of_2(inner), 8192)
        index_select_contiguous_kernel[(grid_x, triton.cdiv(inner, block_i))](
            inp,
            out,
            index,
            N,
            inner,
            index_len,
            total_p,
            block_i,
            grid_x,
            outer == 1,
            # 大 inner（单次连续传输 >= 8192 元素）用 4 warps 吞吐更高，
            # 中小 inner 用 2 warps 启动/占用更省（实测分路径优于单一值）。
            num_warps=4 if inner >= 8192 else 2,
        )
        return out

    # with dim_compress（dim == last 时为恒等 permute，无拷贝）
    inp = dim_compress(inp, dim)
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        min(triton.cdiv(M, meta["BLOCK_M"]), MAX_GRID_DIM),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        res = make_contiguous_with_correct_stride(out.permute(order))
        return res.reshape(res.shape)
    else:
        return out
