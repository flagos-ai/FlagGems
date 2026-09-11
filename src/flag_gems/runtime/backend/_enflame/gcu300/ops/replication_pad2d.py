# Copyright 2026, The FlagOS Contributors.
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
#
# GCU300-specific replication_pad2d.
#
# Differences vs the generic implementation (flag_gems/ops/replication_pad2d.py):
#   * int32 indexing: GCU300 cannot legalize i64 index arithmetic in triton
#     kernels unless i64 support is explicitly enabled.
#   * non-contiguous inputs are supported (materialized via .contiguous()),
#     matching torch.nn.functional.pad behavior.
#   * BLOCK_SIZE grows when the grid would exceed the GCU grid.x limit (65535).
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_GCU_GRID_X_LIMIT = 65535


@triton.jit
def replication_pad2d_kernel(
    in_ptr,  # *Pointer* to input tensor
    out_ptr,  # *Pointer* to output tensor
    N,
    C,
    H,
    W,  # input dimensions
    OH,
    OW,  # output H and W
    PAD_LEFT,
    PAD_TOP,  # padding sizes
    TOTAL_ELEMS,  # total number of output elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMS

    # int32 indexing (i64 index arithmetic is not legalizable on GCU300
    # unless i64 support is explicitly enabled).
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    ih = oh - PAD_TOP
    iw = ow - PAD_LEFT

    ih = tl.maximum(0, tl.minimum(H - 1, ih))
    iw = tl.maximum(0, tl.minimum(W - 1, iw))

    in_index = ((n * C + c) * H + ih) * W + iw

    x = tl.load(in_ptr + in_index, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _prepare_dims_and_out(input: torch.Tensor, padding, out):
    if not isinstance(padding, (tuple, list)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of 4 integers: "
            "(pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("replication_pad2d does not support negative padding")

    if input.dim() == 4:
        N, C, H, W = input.shape
        out_shape = (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        kernel_N, kernel_C = N, C
    elif input.dim() == 3:
        C, H, W = input.shape
        out_shape = (C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        kernel_N, kernel_C = 1, C
    else:
        raise ValueError(
            "replication_pad2d expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    if H <= 0 or W <= 0:
        raise ValueError(
            "Input height and width must be greater than 0 for replication padding"
        )

    if out is None:
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    else:
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(
                f"Provided out tensor has shape {tuple(out.shape)}, "
                f"expected {out_shape}"
            )
        if out.device != input.device:
            raise ValueError("Input and out must be on the same device")
        if out.dtype != input.dtype:
            raise ValueError("Input and out must have the same dtype")

    return (
        kernel_N,
        kernel_C,
        H,
        W,
        out.shape[-2],
        out.shape[-1],
        pad_left,
        pad_top,
    ), out


def _launch_replication_pad2d_kernel(
    input: torch.Tensor, out: torch.Tensor, kernel_params
):
    if input.device != out.device:
        raise ValueError("Input and out must be on the same device")
    # The kernel indexes the input as a flat contiguous buffer; materialize
    # non-contiguous inputs (torch.nn.functional.pad accepts them too).
    if not input.is_contiguous():
        input = input.contiguous()
    if not out.is_contiguous():
        raise ValueError("Only contiguous out tensors are supported")

    N, C, H, W, OH, OW, pad_left, pad_top = kernel_params
    total_elems = out.numel()
    if total_elems == 0:
        return out

    BLOCK_SIZE = 1024
    # GCU grid.x hardware limit: grow BLOCK_SIZE so the grid fits.
    if triton.cdiv(total_elems, BLOCK_SIZE) > _GCU_GRID_X_LIMIT:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(total_elems, _GCU_GRID_X_LIMIT))
    grid = (triton.cdiv(total_elems, BLOCK_SIZE),)

    replication_pad2d_kernel[grid](
        input,
        out,
        N,
        C,
        H,
        W,
        OH,
        OW,
        pad_left,
        pad_top,
        total_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replication_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D")
    kernel_params, out = _prepare_dims_and_out(input, padding, out=None)
    return _launch_replication_pad2d_kernel(input, out, kernel_params)


def replication_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_OUT")
    kernel_params, out = _prepare_dims_and_out(input, padding, out=out)
    return _launch_replication_pad2d_kernel(input, out, kernel_params)
