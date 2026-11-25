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

from flag_gems.ops.embedding_bag_backward import (
    _eb_backward_finish,
    _eb_backward_validate,
    _embedding_bag_backward_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _masked_launch(kernel, packed_kernel, grid, pointers, metadata, **options):
    if kernel is _eb_backward_validate:
        # Keep mapping, frequency counts and accumulator initialization; disable
        # only the device assertion, leaving all indirect access masks intact.
        metadata = (*metadata[:18], False, *metadata[19:])
        options["debug"] = False
    return kernel[grid](*pointers, *metadata, **options)


@libentry()
@triton.jit
def _max_zero(acc, count: tl.constexpr, block: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    tl.store(acc + x, 0, x < count)


@libentry()
@triton.jit
def _max_scatter(
    grad,
    sizes,
    maximum,
    acc,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    padding: tl.constexpr,
    sg0: tl.constexpr,  # gradient bag stride
    sg1: tl.constexpr,  # gradient feature stride
    sb: tl.constexpr,  # bag-size stride
    sx0: tl.constexpr,  # argmax bag stride
    sx1: tl.constexpr,  # argmax feature stride
    block: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    bag = x // embedding_dim
    col = x % embedding_dim
    in_bounds = x < num_bags * embedding_dim
    row = tl.load(maximum + bag * sx0 + col * sx1, in_bounds, other=-1).to(tl.int64)
    size = tl.load(sizes + bag * sb, in_bounds, other=0)
    active = (
        in_bounds & (row >= 0) & (row < num_weights) & (row != padding) & (size > 0)
    )
    value = tl.load(grad + bag * sg0 + col * sg1, active, other=0).to(tl.float32)
    tl.atomic_add(acc + row * embedding_dim + col, value, active, sem="relaxed")


def _compute_max(grad, indices, offsets, mapping, sizes, maximum, num_weights, padding):
    # MAX uses saved argmax rows, not a per-occurrence mapping or frequency.
    # Host metadata and deterministic-mode checks remain in the generic entry.
    if grad.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    bags, dim = grad.shape
    acc = torch.empty((num_weights, dim), dtype=torch.float32, device=grad.device)
    result = (
        acc if grad.dtype == torch.float32 else torch.empty_like(acc, dtype=grad.dtype)
    )
    count = num_weights * dim
    with torch_device_fn.device(grad.device):
        if count:
            _max_zero[(triton.cdiv(count, 4096),)](acc, count, 4096, debug=False)
        if indices.numel() and bags and dim and num_weights:
            _max_scatter[(triton.cdiv(bags * dim, 256),)](
                grad,
                sizes,
                maximum,
                acc,
                bags,
                dim,
                num_weights,
                padding,
                *grad.stride(),
                sizes.stride(0),
                *maximum.stride(),
                256,
                debug=False,
            )
        if count and result is not acc:
            _eb_backward_finish[(triton.cdiv(count, 4096),)](
                acc,
                indices,
                result,
                count,
                dim,
                False,
                4096,
                debug=False,
            )
    return result


def _embedding_bag_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights=None,
    padding_idx=-1,
):
    logger.debug("GEMS_ILUVATAR _EMBEDDING_BAG_BACKWARD")
    result = _embedding_bag_backward_impl(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        padding_idx,
        device_assert_enabled=True,
        fused_init_enabled=True,
        launch_fn=_masked_launch,
        ignore_device_errors=True,
        max_fn=_compute_max,
    )
    return result
