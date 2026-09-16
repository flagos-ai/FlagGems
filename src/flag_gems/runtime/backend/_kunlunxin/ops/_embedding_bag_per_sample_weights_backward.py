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


@libentry()
@triton.jit
def _ebpswb_kernel(
    grad_ptr,
    weight_ptr,
    indices_ptr,
    offset2bag_ptr,
    output_ptr,
    embedding_dim,
    stride_grad,
    stride_weight,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program owns one sample: the whole dot product is accumulated in
    # registers and written with a single scalar store after the load-only
    # block loop (a store inside a ``range(NUM_BLOCKS)`` loop fails
    # TritonXPUUnrollControl on this backend).
    pid = tl.program_id(0)
    idx = tl.load(indices_ptr + pid).to(tl.int32)
    bag = tl.load(offset2bag_ptr + pid).to(tl.int32)
    result = 0.0
    for b in range(NUM_BLOCKS):
        offs = b * BLOCK_D + tl.arange(0, BLOCK_D)
        m = offs < embedding_dim
        # Clamp to a legal address and gate afterwards: no ``other=`` on loads.
        safe = tl.where(m, offs, 0)
        w = tl.load(weight_ptr + idx * stride_weight + safe).to(tl.float32)
        g = tl.load(grad_ptr + bag * stride_grad + safe).to(tl.float32)
        w = tl.where(m, w, 0.0)
        g = tl.where(m, g, 0.0)
        result += tl.sum(g * w)
    tl.store(output_ptr + pid, result)


def _embedding_bag_per_sample_weights_backward(
    grad: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    mode: int,
    padding_idx: int = -1,
) -> torch.Tensor:
    """Gradient of the embedding bag w.r.t. per_sample_weights.

    Same contract as ``torch.ops.aten._embedding_bag_per_sample_weights_backward``:
    ``output[i] = <grad[offset2bag[i]], weight[indices[i]]>`` for each sample,
    with ``padding_idx`` zeroing applied afterwards.
    """
    logger.debug("GEMS_KUNLUNXIN _EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD")

    if int(mode) != 0:
        raise RuntimeError(
            "embedding_bag_backward: per_sample_weights only supported for mode='sum'"
        )
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    assert (
        grad.device == weight.device == indices.device == offset2bag.device
    ), "All inputs must be on the same device."

    num_samples = indices.numel()
    embedding_dim = weight.shape[1]

    assert num_samples > 0, "num_samples must be positive"
    assert embedding_dim > 0, "embedding_dim must be positive"

    output = torch.empty(num_samples, device=grad.device, dtype=torch.float32)

    grad = grad.contiguous()
    weight = weight.contiguous()
    indices = indices.contiguous()
    offset2bag = offset2bag.contiguous()

    BLOCK_D = triton.next_power_of_2(min(embedding_dim, 1024))
    NUM_BLOCKS = triton.cdiv(embedding_dim, BLOCK_D)
    grid = (num_samples,)

    _ebpswb_kernel[grid](
        grad,
        weight,
        indices,
        offset2bag,
        output,
        embedding_dim,
        grad.stride(0),
        weight.stride(0),
        NUM_BLOCKS=NUM_BLOCKS,
        BLOCK_D=BLOCK_D,
    )

    if int(padding_idx) >= 0:
        output[indices == int(padding_idx)] = 0.0

    return output.to(grad.dtype)
