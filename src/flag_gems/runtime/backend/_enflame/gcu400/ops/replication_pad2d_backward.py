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

"""replication_pad2d 反向（Enflame GCU400）：单 launch、无原子、行网格跨步实现。

2026-11-15 修复 + 优化
=====================
1) **grid.x 溢出崩溃（功能缺陷）**
   旧实现有两条路径：interior_elements ≤ 16384 时走「一行一 program 的融合 kernel」，
   否则回退到「interior copy + 边缘 atomic add」双 kernel。回退路径的 grid 是
   `nc_total * H * cdiv(W, INTERIOR_BLOCK)` / `cdiv(interior_elements, BLOCK)`，
   (4,64,256,512) 需要 131072、(8,128,1024,1024) 更多，超过硬件 grid.x ≤ 65535，
   直接 `OutOfResources: out of resource: grid.x`（tests 里 4x64x256x512 的 18 个
   用例全挂）。新实现把「输入行」循环放进 kernel 做 grid-stride
   （`for row in range(pid, ROWS, G)`），grid 只由行数与调参上限决定，任何 shape
   都不会溢出；同时删掉了双 kernel 回退路径（其精度缺陷见第 2 点）。
2) **fp16/bf16 边缘精度**
   旧边缘 kernel 用 `tl.atomic_add` 直接在低精度输出上逐次累加，每个 pad 元素都会
   舍入一次（用随机 grad_output 与 CPU float64 参考比对，边界 0.1% 元素差到
   0.047）。新实现每个输入元素在寄存器里用 fp32 累加所有映射到它的输出元素，
   最后一次 cast，且完全不需要 atomic。
3) **ph 维度改成顺序累加（主要性能来源）**
   `[BLOCK_PH, BLOCK_W]` 2D tile 再 `tl.sum(axis=0)` 在 GCU 上会变成跨 lane 归约，
   是这条 kernel 的最大开销；改成 `tl.static_range(BLOCK_PH)` 顺序累加到 1D 寄存器
   向量后，同一组用例 geomean 17.19× → 37.59×（+2.2×，见报告）。列方向按
   `ELEM_CAP` 分块，超宽行（W > BLOCK_W）不会寄存器爆炸。

映射语义（replication 反向）：输入行 (nc, ih) 承接的输出行闭区间为
[oh_lo, oh_hi] = [ih == 0 ? 0 : PT + ih, ih == H-1 ? OH-1 : PT+ih]；列方向上左 pad
列折叠到 iw = 0、右 pad 列折叠到 iw = W-1。每个 grad_output 元素只被读一次，
读写流量即 OH*OW 读 + H*W 写（最优）。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ===== 调参常量（2026-11-15 交错 A/B 实测标定，min-of-3 轮）=====
# grid = min(行数, SPC * GRID_MULT, 65535)：GRID_MULT=1（48 个 program）最优，
# 2/4/8 倍分别 -2.7%/-5.8%/-11.9%。
GRID_MULT = 1
SPC = 48
MAX_GRID = 65535
# 单次 load 的元素预算：8192 与 16384/32768 等价（受 next_pow2(W) 约束），
# 1024/2048 分别 -31%/-11%。
ELEM_CAP = 8192
# num_warps：1 最优，2 时 -26%。
WARPS = 1
# ==============================================================


@triton.jit
def replication_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    ROWS,
    H,
    W,
    OH,
    OW,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    GO_ROW: tl.constexpr,
    GO_NC: tl.constexpr,
    GI_ROW: tl.constexpr,
    GI_NC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    NCHUNK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    iw0 = tl.arange(0, BLOCK_W)
    pw = tl.arange(0, BLOCK_PAD)

    for row in range(pid, ROWS, G):
        nc = row // H
        ih = row - nc * H
        # 该输入行承接的输出行闭区间（H == 1 时两个 where 合并后覆盖整个 OH）
        oh_lo = tl.where(ih == 0, 0, PT + ih)
        oh_hi = tl.where(ih == H - 1, OH - 1, PT + ih)
        go_nc = grad_output_ptr + nc * GO_NC
        gi_row = grad_input_ptr + row * GI_ROW

        for chunk in tl.static_range(NCHUNK):
            iw = chunk * BLOCK_W + iw0
            iwv = iw < W
            total = tl.zeros([BLOCK_W], dtype=tl.float32)
            # ph 维度顺序累加（避免 2D tile 的跨 lane 归约）
            for p in tl.static_range(BLOCK_PH):
                oh = oh_lo + p
                if oh <= oh_hi:
                    total += tl.load(
                        go_nc + oh * GO_ROW + (PL + iw), mask=iwv, other=0.0
                    ).to(tl.float32)
            # 左右 pad 列：整行只有 iw=0 / iw=W-1 两个位置承接
            if PL > 0:
                if chunk == 0:
                    left_sum = 0.0
                    for p in tl.static_range(BLOCK_PH):
                        oh = oh_lo + p
                        if oh <= oh_hi:
                            left_sum += tl.sum(
                                tl.load(
                                    go_nc + oh * GO_ROW + pw,
                                    mask=pw < PL,
                                    other=0.0,
                                ).to(tl.float32)
                            )
                    total += tl.where(iw0 == 0, left_sum, 0.0)
            if PR > 0:
                if chunk == NCHUNK - 1:
                    right_sum = 0.0
                    for p in tl.static_range(BLOCK_PH):
                        oh = oh_lo + p
                        if oh <= oh_hi:
                            right_sum += tl.sum(
                                tl.load(
                                    go_nc + oh * GO_ROW + (PL + W + pw),
                                    mask=pw < PR,
                                    other=0.0,
                                ).to(tl.float32)
                            )
                    total += tl.where(iw == W - 1, right_sum, 0.0)
            tl.store(gi_row + iw, total.to(OUT_DTYPE), mask=iwv)


def _dtype_constexpr(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64
    return tl.float32


def _next_pow2(n):
    value = 1
    while value < n:
        value *= 2
    return value


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        N = 1
        grad_output_4d = grad_output.view(
            1, C, grad_output.shape[-2], grad_output.shape[-1]
        )
    elif self.ndim == 4:
        N, C, H, W = self.shape
        grad_output_4d = grad_output.contiguous()
    else:
        raise ValueError(
            "replication_pad2d_backward expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    expected_grad_shape = (N, C, OH, OW)
    if tuple(grad_output_4d.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output_4d.shape)} does not match "
            f"expected shape {expected_grad_shape}"
        )

    # 无 padding 时反向就是恒等拷贝
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        result = grad_output_4d.contiguous().to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    rows = N * C * H
    if rows == 0:
        if out is not None:
            return out
        return torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)

    # kernel 用常量 stride 直接写连续缓冲；out 非连续时先算到连续缓冲再拷回
    write_back = None
    if out is None:
        target = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
    elif out.is_contiguous():
        target = out
    else:
        target = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
        write_back = out

    block_ph = _next_pow2(1 + pad_top + pad_bottom)
    block_w = min(_next_pow2(W), max(16, ELEM_CAP // block_ph))
    nchunk = triton.cdiv(W, block_w)
    block_pad = _next_pow2(max(pad_left, pad_right, 2))

    grid_rows = min(rows, SPC * GRID_MULT, MAX_GRID)
    with torch_device_fn.device(self.device):
        replication_pad2d_backward_kernel[(grid_rows,)](
            grad_output_4d,
            target,
            rows,
            H,
            W,
            OH,
            OW,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            OW,
            OH * OW,
            W,
            H * W,
            BLOCK_W=block_w,
            BLOCK_PH=block_ph,
            BLOCK_PAD=block_pad,
            NCHUNK=nchunk,
            OUT_DTYPE=_dtype_constexpr(self.dtype),
            num_warps=WARPS,
        )

    result = target.view(C, H, W) if is_3d else target
    if write_back is not None:
        write_back.copy_(result)
        return write_back
    return result


def replication_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding)


def replication_pad2d_backward_grad_input(grad_output, self, padding, *, grad_input):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
