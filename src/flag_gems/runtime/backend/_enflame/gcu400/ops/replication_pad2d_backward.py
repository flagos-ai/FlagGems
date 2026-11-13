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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ===== Step 4 轮次脚本改写的调参常量 =====
BASELINE_MODE = 0
INTERIOR_BLOCK = 256
EDGE_BLOCK = 128
NUM_WARPS = 4
EDGE_NUM_WARPS = 4
FUSED_MODE = 1
FUSED_BLOCK = 128
FUSED_WARPS = 1
FUSED_LIMIT = 16384
# ========================================


@triton.jit
def _interior_copy_kernel(
    grad_output_ptr,
    grad_input_ptr,
    OW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    INTERIOR_ELEMS: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat interior kernel for W < BLOCK_SIZE. Each element decodes its
    (nc, ih, iw) coordinate with constexpr stride arithmetic."""
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < INTERIOR_ELEMS

    iw = idx % W
    rest = idx // W
    ih = rest % H
    nc = rest // H

    oh = PAD_TOP + ih
    ow = PAD_LEFT + iw

    out_offset = nc * OH_OW_STRIDE + oh * OW + ow
    in_offset = nc * H_W_STRIDE + ih * W + iw

    grad = tl.load(grad_output_ptr + out_offset, mask=mask, other=0.0)
    tl.store(grad_input_ptr + in_offset, grad, mask=mask)


@triton.jit
def _interior_copy_kernel_row(
    grad_output_ptr,
    grad_input_ptr,
    OW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS_PER_ROW: tl.constexpr,
):
    """Row-based interior kernel for W >= BLOCK_SIZE. Each block covers
    BLOCK_SIZE consecutive elements within a single row — no per-element division."""
    pid = tl.program_id(0)
    row_id = pid // CHUNKS_PER_ROW
    chunk = pid % CHUNKS_PER_ROW

    nc = row_id // H
    ih = row_id % H

    base_iw = chunk * BLOCK_SIZE
    iw = base_iw + tl.arange(0, BLOCK_SIZE)
    mask = iw < W

    oh = PAD_TOP + ih
    ow = PAD_LEFT + iw

    out_base = nc * OH_OW_STRIDE + oh * OW
    in_base = nc * H_W_STRIDE + ih * W

    grad = tl.load(grad_output_ptr + out_base + ow, mask=mask, other=0.0)
    tl.store(grad_input_ptr + in_base + iw, grad, mask=mask)


@triton.jit
def _edges_atomic_kernel(
    grad_output_ptr,
    grad_input_ptr,
    OH,
    OW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    PAD_RIGHT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_BOTTOM: tl.constexpr,
    EDGE_ELEMS: tl.constexpr,
    NC_TOTAL: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Process edge regions (left/right/top/bottom padding) where multiple output
    positions may map to the same input position, requiring atomic_add."""
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < EDGE_ELEMS

    pad_rows = PAD_TOP + PAD_BOTTOM
    left_size = NC_TOTAL * OH * PAD_LEFT
    right_size = NC_TOTAL * OH * PAD_RIGHT

    in_left = idx < left_size
    in_right = (idx >= left_size) & (idx < left_size + right_size)

    safe_w = tl.maximum(1, W)
    safe_oh = tl.maximum(1, OH)
    safe_pad_l = tl.maximum(1, PAD_LEFT)
    safe_pad_r = tl.maximum(1, PAD_RIGHT)
    safe_pad_rows = tl.maximum(1, pad_rows)

    left_idx = idx
    left_ow = left_idx % safe_pad_l
    left_oh = (left_idx // safe_pad_l) % safe_oh
    left_nc = left_idx // (safe_pad_l * safe_oh)

    right_idx = idx - left_size
    right_ow = right_idx % safe_pad_r
    right_oh = (right_idx // safe_pad_r) % safe_oh
    right_nc = right_idx // (safe_pad_r * safe_oh)

    edge_col_idx = idx - left_size - right_size
    ecol = edge_col_idx % safe_w
    erow = (edge_col_idx // safe_w) % safe_pad_rows
    enc = edge_col_idx // (safe_w * safe_pad_rows)

    ow = tl.where(
        in_left, left_ow, tl.where(in_right, OW - PAD_RIGHT + right_ow, PAD_LEFT + ecol)
    )

    oh = tl.where(
        in_left,
        left_oh,
        tl.where(
            in_right,
            right_oh,
            tl.where(erow < PAD_TOP, erow, OH - PAD_BOTTOM + (erow - PAD_TOP)),
        ),
    )

    nc = tl.where(in_left, left_nc, tl.where(in_right, right_nc, enc))

    ih = oh - PAD_TOP
    ih = tl.maximum(0, tl.minimum(H - 1, ih))
    iw = ow - PAD_LEFT
    iw = tl.maximum(0, tl.minimum(W - 1, iw))

    out_offset = nc * OH_OW_STRIDE + oh * OW + ow
    in_offset = nc * H_W_STRIDE + ih * W + iw

    grad = tl.load(grad_output_ptr + out_offset, mask=mask, other=0.0)
    tl.atomic_add(grad_input_ptr + in_offset, grad, mask=mask)


@triton.jit
def _pad2d_bwd_row_fused_kernel(
    grad_output_ptr,
    grad_input_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    GO_NC: tl.constexpr,
    GO_H: tl.constexpr,
    GO_W: tl.constexpr,
    GI_NC: tl.constexpr,
    GI_H: tl.constexpr,
    GI_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """单 launch 无原子的 replication_pad2d 反向。

    一个 program 负责一条输入行 (nc, ih)，直接累加所有 clamp 映射到该行的
    输出元素：输出行 oh 的集合是 [oh_lo, oh_hi]（顶/底 padding 行会折叠到
    ih=0 / ih=H-1），每个 oh 行的中间列直接取值，左/右 padding 列分别折叠到
    iw=0 / iw=W-1。角落区域因此自动被包含，不再需要 atomic_add。
    """
    pid = tl.program_id(0)
    nc = pid // H
    ih = pid % H

    # clamp(oh - PT, 0, H-1) == ih 的输出行闭区间。
    # H == 1 时 ih 同时为 0 与 H-1，两个 where 合并后恰好覆盖整个 OH。
    oh_lo = tl.where(ih == 0, 0, PT + ih)
    oh_hi = tl.where(ih == H - 1, OH - 1, PT + ih)

    ph = tl.arange(0, BLOCK_PH)
    oh = oh_lo + ph
    oh_valid = oh <= oh_hi
    iw = tl.arange(0, BLOCK_W)
    iw_valid = iw < W

    go_base = grad_output_ptr + nc * GO_NC
    interior_ptrs = go_base + oh[:, None] * GO_H + (PL + iw[None, :]) * GO_W
    interior_vals = tl.load(
        interior_ptrs,
        mask=oh_valid[:, None] & iw_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(interior_vals, axis=0)

    if PL > 0:
        pw = tl.arange(0, BLOCK_PAD)
        left_ptrs = go_base + oh[:, None] * GO_H + pw[None, :] * GO_W
        left_vals = tl.load(
            left_ptrs,
            mask=oh_valid[:, None] & (pw[None, :] < PL),
            other=0.0,
        ).to(tl.float32)
        total += tl.where(iw == 0, tl.sum(left_vals), 0.0)

    if PR > 0:
        pw = tl.arange(0, BLOCK_PAD)
        right_ptrs = go_base + oh[:, None] * GO_H + (PL + W + pw[None, :]) * GO_W
        right_vals = tl.load(
            right_ptrs,
            mask=oh_valid[:, None] & (pw[None, :] < PR),
            other=0.0,
        ).to(tl.float32)
        total += tl.where(iw == W - 1, tl.sum(right_vals), 0.0)

    tl.store(
        grad_input_ptr + nc * GI_NC + ih * GI_H + iw * GI_W,
        total.to(OUT_DTYPE),
        mask=iw_valid,
    )


def _dtype_constexpr(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64
    return tl.float32


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

    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        result = grad_output_4d.contiguous().to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    output_dtype = self.dtype
    nc_total = N * C
    interior_elements = nc_total * H * W
    edge_elements = nc_total * OH * OW - interior_elements

    oh_ow_stride = OH * OW
    h_w_stride = H * W

    BLOCK_SIZE = INTERIOR_BLOCK
    BLOCK_SIZE_EDGES = EDGE_BLOCK

    # BASELINE_MODE 下完全复现公共实现的 launch 参数（不显式传 num_warps），
    # 用于校准「显式 num_warps」本身带来的收益。
    if BASELINE_MODE:
        interior_launch_opts = {}
        edge_launch_opts = {}
    else:
        interior_launch_opts = {"num_warps": NUM_WARPS}
        edge_launch_opts = {"num_warps": EDGE_NUM_WARPS}

    if out is not None:
        accum = out
    else:
        accum = torch.empty((N, C, H, W), device=self.device, dtype=output_dtype)

    # 无原子融合路径：一次 launch 完成 interior copy 与 padding 行/列归约。
    # 只有输入/输出张量连续时才能用编译期常量 stride，否则回退公共双 kernel 路径。
    if (
        FUSED_MODE == 1
        and accum.is_contiguous()
        and (FUSED_LIMIT == 0 or interior_elements <= FUSED_LIMIT)
    ):
        block_w = triton.next_power_of_2(max(W, 1))
        block_ph = triton.next_power_of_2(max(1 + pad_top + pad_bottom, 1))
        block_pad = triton.next_power_of_2(max(pad_left, pad_right, 1))
        with torch_device_fn.device(self.device):
            _pad2d_bwd_row_fused_kernel[(nc_total * H,)](
                grad_output_4d,
                accum,
                H,
                W,
                OH,
                OW,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                oh_ow_stride,
                OW,
                1,
                h_w_stride,
                W,
                1,
                block_w,
                block_ph,
                block_pad,
                _dtype_constexpr(output_dtype),
                num_warps=FUSED_WARPS,
            )
        if is_3d:
            result = accum.view(C, H, W)
        else:
            result = accum
        if out is not None:
            return out
        return result

    with torch_device_fn.device(self.device):
        if interior_elements > 0:
            if W >= BLOCK_SIZE:
                interior_chunks_per_row = triton.cdiv(W, BLOCK_SIZE)
                interior_grid = (nc_total * H * interior_chunks_per_row,)
                _interior_copy_kernel_row[interior_grid](
                    grad_output_4d,
                    accum,
                    OW,
                    H,
                    W,
                    pad_left,
                    pad_top,
                    oh_ow_stride,
                    h_w_stride,
                    BLOCK_SIZE,
                    interior_chunks_per_row,
                    **interior_launch_opts,
                )
            else:
                interior_grid = (triton.cdiv(interior_elements, BLOCK_SIZE),)
                _interior_copy_kernel[interior_grid](
                    grad_output_4d,
                    accum,
                    OW,
                    H,
                    W,
                    pad_left,
                    pad_top,
                    interior_elements,
                    oh_ow_stride,
                    h_w_stride,
                    BLOCK_SIZE,
                    **interior_launch_opts,
                )

        if edge_elements > 0:
            edge_grid = (triton.cdiv(edge_elements, BLOCK_SIZE_EDGES),)
            _edges_atomic_kernel[edge_grid](
                grad_output_4d,
                accum,
                OH,
                OW,
                H,
                W,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                edge_elements,
                nc_total,
                oh_ow_stride,
                h_w_stride,
                BLOCK_SIZE_EDGES,
                **edge_launch_opts,
            )

    if is_3d:
        accum = accum.view(C, H, W)
    return accum


def replication_pad2d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, grad_input: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
