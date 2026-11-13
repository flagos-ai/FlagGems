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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# --- TUNING PARAMS START (CONJ_TUNING) ---
# 共轭 kernel 的启动期配置：block/grid/warps 由实测扫描确定，均为编译期常量
# 或 launch 参数，不参与数值语义。
CONJ_TUNING = {
    "block_size": 4096,
    "num_warps": 8,
    "fixed_grid": 96,
    "num_stages": 1,
    "negate_mode": "where",
    "grid_mode": "fixed",
}
# --- TUNING PARAMS END (CONJ_TUNING) ---


@libentry()
@triton.jit
def _conj_flat_kernel(
    in_ptr,
    out_ptr,
    n_floats,
    BLOCK_SIZE: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    NEGATE_MODE: tl.constexpr,
):
    # 固定 grid + 核内跨步：grid 与问题规模解耦，大 shape 不产生海量 CTA。
    tile_count = tl.cdiv(n_floats, BLOCK_SIZE)
    for tile_id in tl.range(tl.program_id(0), tile_count, GRID_STRIDE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_floats
        values = tl.load(in_ptr + offsets, mask=mask, other=0)
        # 交错布局 (real, imag, ...)：奇数下标为虚部，位与代替取模。
        is_imaginary = (offsets & 1) == 1
        if NEGATE_MODE == "multiply":
            sign = tl.where(is_imaginary, -1.0, 1.0).to(values.dtype)
            result = values * sign
        else:
            result = tl.where(is_imaginary, -values, values)
        tl.store(out_ptr + offsets, result, mask=mask)


def _launch_conj_flat(src: torch.Tensor, dst: torch.Tensor, tuning: dict) -> None:
    """把 contiguous complex 张量按平坦实数流共轭写入 dst。"""
    n_floats = src.numel() * 2
    if n_floats == 0:
        return
    float_dtype = _float_dtype(src)
    in_flat = src.reshape(-1).view(float_dtype)
    out_flat = dst.reshape(-1).view(float_dtype)
    block_size = tuning["block_size"]
    tile_count = triton.cdiv(n_floats, block_size)
    if tuning["grid_mode"] == "full":
        grid_stride = tile_count
    else:
        grid_stride = min(tuning["fixed_grid"], tile_count)
    _conj_flat_kernel[(grid_stride,)](
        in_flat,
        out_flat,
        n_floats,
        BLOCK_SIZE=block_size,
        GRID_STRIDE=grid_stride,
        NEGATE_MODE=tuning["negate_mode"],
        num_warps=tuning["num_warps"],
        num_stages=tuning["num_stages"],
    )


def _float_dtype(x: torch.Tensor) -> torch.dtype:
    # complex32 / complex64 / complex128 各自对应的实数成分 dtype。
    if x.dtype == torch.complex32:
        return torch.float16
    if x.dtype == torch.complex64:
        return torch.float32
    if x.dtype == torch.complex128:
        return torch.float64
    raise RuntimeError("conj only supports complex tensors")


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    # conj bit 置位时逻辑值已是共轭：先物化逻辑值，再取共轭还原存储数据。
    if input.is_conj():
        resolved = input.resolve_conj()
        return torch.conj_physical(resolved)

    src = input if input.is_contiguous() else input.contiguous()
    output = torch.empty_like(src)
    _launch_conj_flat(src, output, CONJ_TUNING)
    return output
