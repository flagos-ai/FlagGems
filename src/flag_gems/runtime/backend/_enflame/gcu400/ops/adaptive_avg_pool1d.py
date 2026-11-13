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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_avg_pool1d_kernel(
    X,
    Y,
    L_IN,
    L_OUT,
    BLOCKS_PER_PLANE,
    MAX_WINDOW: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # One program handles BLOCK_OUT output positions of a single (N, C) plane.
    # aten's bin edges are floor(i*L_IN/L_OUT) and ceil((i+1)*L_IN/L_OUT);
    # loading a MAX_WINDOW-wide 2D tile lets all bins reduce in one tl.sum.
    # GCU400 caps grid.y at 255, while N*C can be far larger. Encode both the
    # output-block index and the plane index into a single grid.x dimension.
    pid = tl.program_id(0)
    out_block = pid % BLOCKS_PER_PLANE
    plane = pid // BLOCKS_PER_PLANE
    out_pos = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

    bin_start = (out_pos * L_IN) // L_OUT
    bin_end = ((out_pos + 1) * L_IN + L_OUT - 1) // L_OUT
    window = tl.arange(0, MAX_WINDOW)
    in_pos = bin_start[:, None] + window[None, :]
    valid = (out_pos[:, None] < L_OUT) & (in_pos < bin_end[:, None]) & (in_pos < L_IN)
    values = tl.load(X + plane * L_IN + in_pos, mask=valid, other=0.0).to(tl.float32)
    count = tl.maximum(bin_end - bin_start, 1)
    mean = tl.sum(values, 1) / count.to(tl.float32)
    tl.store(
        Y + plane * L_OUT + out_pos, mean.to(Y.dtype.element_ty), mask=out_pos < L_OUT
    )


def adaptive_avg_pool1d(input: torch.Tensor, output_size):
    """Adaptive average pooling over 1D input.

    The shared ``adaptive_avg_pool2d`` path lowers through
    ``_adaptive_pool_windows``, whose pointer-array tile makes the GCU400
    PassManager fail for every dtype. Calling ``aten::adaptive_avg_pool1d``
    directly is not an option either: FlagGems registers that very overload, so
    it recurses back here. This backend-specific kernel computes the bins
    directly from the 1D layout instead.
    """
    logger.debug("GEMS_ENFLAME ADAPTIVE_AVG_POOL1D")

    assert input.ndim == 3, f"adaptive_avg_pool1d expects 3D input, got {input.ndim}D"
    if isinstance(output_size, (list, tuple)):
        output_size = output_size[0]

    num_planes, _, length_in = input.shape
    length_out = output_size
    out = torch.empty(
        (input.shape[0], input.shape[1], length_out),
        dtype=input.dtype,
        device=input.device,
    )
    if length_out == 0:
        return out

    max_window = triton.next_power_of_2(triton.cdiv(length_in, length_out) + 1)
    block_out = min(64, triton.next_power_of_2(length_out))
    blocks_per_plane = triton.cdiv(length_out, block_out)
    grid = (blocks_per_plane * num_planes * input.shape[1],)
    _adaptive_avg_pool1d_kernel[grid](
        input,
        out,
        length_in,
        length_out,
        blocks_per_plane,
        max_window,
        block_out,
        num_warps=8,
    )
    return out
