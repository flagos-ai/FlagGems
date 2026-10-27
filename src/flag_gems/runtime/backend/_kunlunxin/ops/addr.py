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

# addr = beta*input + alpha*(vec1 (x) vec2), i.e.
#   out[m, n] = beta*input[m, n] + alpha*vec1[m]*vec2[n]
#
# History / why this file no longer uses `pointwise_dynamic`:
#
# The previous revision reformulated addr as a broadcasted `pointwise_dynamic`
# op. That fixed the discrete-access problem of the original hand-written
# 32x32 2D-tile kernel (XPU OffsetAnalysis marked every 2D offset
# `offsetState=-1`), but it carried two structural defects:
#
#   1. `prefer_1d_tile` computes
#          tile_size = next_power_of_2(cdiv(num_tasks, 12))
#      with NO upper bound (see `gen_task_partition_1d` in
#      `_kunlunxin/utils/pointwise_dynamic.py`). A (5333, 497) output holds
#      2_650_501 elements, so it asked for a 262_144-element tile and the ELF
#      stack check rejected the kernel with "Failed to tune buffer size." --
#      i.e. the previous implementation did not even compile for that shape.
#   2. Its Python wrapper costs ~98us of HOST time per call on this stack
#      (`StridedBuffer` construction, `broadcasted_stride`, `type_promotion`,
#      plus a ~20-argument launch). These shapes are launch-bound (the device
#      kernel is only ~7-11us), so the wrapper -- not the kernel -- set the
#      measured latency.
#
# This revision keeps the access pattern that XPU lowers well (flat 1D tile;
# input/output contiguous -> block DMA; `vec1` is a stride-0 broadcast and
# `vec2` a contiguous run, both of which OffsetAnalysis folds), but bounds the
# tile and launches the kernel directly from a thin wrapper.
#
# Tile policy: aim for a handful of programs (`_GRID_TARGET`) while never
# exceeding `_MAX_TILE`, which is what keeps the ELF stack inside its budget.
# Large outputs therefore fall back to a bounded tile with a larger grid
# instead of a huge tile with 12 programs.

_MAX_TILE = 2048
_MIN_TILE = 2048
_GRID_TARGET = 8

# The XPU backend rejects a kernel whose ELF stack exceeds its budget with
# "Failed to tune buffer size." / "TLE kernel stack is over the local-memory
# budget". The stack scales with the tile, but the budget that is actually
# available is not purely a function of the tile: the same (tile, num_warps)
# pair compiles in a fresh process and is rejected in a process that has already
# compiled other kernels. Rather than encode that, halve the tile and retry.
_TILE_RETRY_FLOOR = 128
_STACK_ERRORS = (
    "Failed to tune buffer size",
    "stack is over the local-memory budget",
    "out of resource: uni_sram",
)


@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_kernel(
    inp_ptr,
    v1_ptr,
    v2_ptr,
    out_ptr,
    beta,
    alpha,
    in0_s0: tl.constexpr,
    in0_s1: tl.constexpr,
    in1_s0: tl.constexpr,
    in1_s1: tl.constexpr,
    in2_s0: tl.constexpr,
    in2_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    s1: tl.constexpr,
    num_tasks: tl.constexpr,
    TILE: tl.constexpr,
):
    tid = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = tid < num_tasks
    i1 = tid % s1
    i0 = tid // s1

    x = tl.load(inp_ptr + i0 * in0_s0 + i1 * in0_s1, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.load(v1_ptr + i0 * in1_s0 + i1 * in1_s1, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(v2_ptr + i0 * in2_s0 + i1 * in2_s1, mask=mask, other=0.0).to(tl.float32)

    # Compute in fp32 (matches the accuracy the earlier explicit `.to(tl.float32)`
    # kernels achieved) and store back in the output dtype.
    y = beta * x + alpha * (s * w)
    tl.store(
        out_ptr + i0 * out_s0 + i1 * out_s1, y.to(out_ptr.dtype.element_ty), mask=mask
    )


def _broadcast_strides(tensor, task_shape):
    """Strides of `tensor` after broadcasting it to `task_shape`.

    A size-1 dimension that is expanded contributes stride 0, so the kernel
    reads the same element for every position along that axis.
    """
    strides = tensor.stride()
    return tuple(
        0 if (tensor.shape[d] == 1 and task_shape[d] != 1) else strides[d]
        for d in range(len(task_shape))
    )


def addr(input, vec1, vec2, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDR")
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("addr: expected 1-D vectors")

    # `beta`/`alpha` may arrive as 0-dim tensors (aten::addr accepts them).
    # `do_not_specialize` keeps them runtime scalars; a 0-dim tensor cannot be
    # handed to the kernel as a scalar, so materialise the value once here.
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    M = vec1.shape[0]
    N = vec2.shape[0]
    output_shape = (M, N)

    try:
        input_broadcasted = torch.broadcast_to(input, output_shape)
    except RuntimeError:
        raise ValueError(
            f"addr: input tensor of shape {input.shape} cannot be broadcast "
            f"to output shape {output_shape}"
        )

    out = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    num_tasks = M * N
    if num_tasks == 0:
        return out

    v1 = vec1.reshape(M, 1)
    v2 = vec2.reshape(1, N)

    tile = triton.next_power_of_2(triton.cdiv(num_tasks, _GRID_TARGET))
    tile = max(_MIN_TILE, min(tile, _MAX_TILE))

    s_i = _broadcast_strides(input_broadcasted, output_shape)
    s_1 = _broadcast_strides(v1, output_shape)
    s_2 = _broadcast_strides(v2, output_shape)
    s_o = _broadcast_strides(out, output_shape)

    while True:
        grid = (triton.cdiv(num_tasks, tile),)
        num_warps = 4 if tile <= 2048 else 8
        try:
            with torch_device_fn.device(input.device):
                addr_kernel[grid](
                    input_broadcasted,
                    v1,
                    v2,
                    out,
                    beta,
                    alpha,
                    s_i[0],
                    s_i[1],
                    s_1[0],
                    s_1[1],
                    s_2[0],
                    s_2[1],
                    s_o[0],
                    s_o[1],
                    N,
                    num_tasks,
                    TILE=tile,
                    num_warps=num_warps,
                )
            return out
        except RuntimeError as exc:
            if tile <= _TILE_RETRY_FLOOR or not any(
                msg in str(exc) for msg in _STACK_ERRORS
            ):
                raise
            logger.debug(
                "GEMS_KUNLUNXIN ADDR: tile %d rejected (%s); retrying smaller",
                tile,
                exc,
            )
            tile = max(_TILE_RETRY_FLOOR, tile // 2)
