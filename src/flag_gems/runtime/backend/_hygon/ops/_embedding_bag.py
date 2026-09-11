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

from flag_gems.ops._embedding_bag import _embedding_bag_forward_kernel as _generic_entry
from flag_gems.ops._embedding_bag import _embedding_bag_impl as _generic_impl
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_shared_forward_jit = _generic_entry.jit_function


@libentry()
@triton.jit(debug=True)
def _hygon_embedding_bag_forward_kernel(
    weight,
    indices,
    offsets,
    per_sample_weights,
    output,
    offset_to_bag,
    bag_size,
    max_indices,  # maximum-value embedding indices
    error,
    num_indices: tl.constexpr,
    num_offsets: tl.constexpr,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    stride_w0: tl.constexpr,  # weight row stride
    stride_w1: tl.constexpr,  # weight feature stride
    stride_i: tl.constexpr,  # indices stride
    stride_o: tl.constexpr,  # offsets stride
    stride_p: tl.constexpr,  # per-sample-weight stride
    padding: tl.constexpr,
    mode: tl.constexpr,
    has_per_sample: tl.constexpr,
    block_l: tl.constexpr,  # bag-length block size
    block_d: tl.constexpr,  # embedding-dimension block size
    ascend_abi: tl.constexpr = False,  # Ascend auxiliary-output application binary interface
    bag_size_b: tl.constexpr = False,  # bag-size output uses num_bags entries
    search_steps: tl.constexpr = 0,
    index_block: tl.constexpr = 128,
    async_assert: tl.constexpr = False,  # asynchronous device assertion
    serial_max: tl.constexpr = False,  # serial maximum reduction
    key_max: tl.constexpr = False,  # maximum reduction using ordered keys
    reciprocal_mean: tl.constexpr = False,
    skip_pad_scale: tl.constexpr = False,  # skip padding-scale adjustment
):
    _shared_forward_jit(
        weight,
        indices,
        offsets,
        per_sample_weights,
        output,
        offset_to_bag,
        bag_size,
        max_indices,
        error,
        num_indices,
        num_offsets,
        num_bags,
        embedding_dim,
        num_weights,
        stride_w0,
        stride_w1,
        stride_i,
        stride_o,
        stride_p,
        padding,
        mode,
        has_per_sample,
        block_l,
        block_d,
        ascend_abi=ascend_abi,
        bag_size_b=bag_size_b,
        search_steps=search_steps,
        index_block=index_block,
        async_assert=async_assert,
        serial_max=False,
        key_max=mode == 2,
        reciprocal_mean=mode == 1,
        skip_pad_scale=skip_pad_scale,
    )


def _embedding_bag_forward_only(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS_HYGON _EMBEDDING_BAG_FORWARD_ONLY")
    kernel = block_config = None
    if (
        weight.dtype == torch.float64
        and weight.ndim == 2
        and weight.shape[1] >= 512
        and indices.ndim == 1
        and offsets.ndim == 1
    ):
        bags = offsets.numel() - int(include_last_offset)
        if bags >= 128 and indices.numel() <= bags * 16:
            # Short wide bags benefit from fewer feature tiles. Integer argmax
            # keys avoid double-precision reductions; a scalar reciprocal keeps
            # MEAN division out of the feature vector.
            kernel = _hygon_embedding_bag_forward_kernel
            if mode == 2:
                block_config = (8, 256, 128, 4)
            else:
                block_config = (8, 1024, 128, 4 if mode == 1 else 8)
    return _generic_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        forward_only=True,
        _kernel=kernel,
        _block_config=block_config,
    )
