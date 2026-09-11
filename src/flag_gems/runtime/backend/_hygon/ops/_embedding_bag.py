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
    Weight,
    Indices,
    Offsets,
    PerSample,
    Output,
    Offset2Bag,
    BagSize,
    MaxIndices,
    Error,
    N: tl.constexpr,
    O: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    STRIDE_W0: tl.constexpr,
    STRIDE_W1: tl.constexpr,
    STRIDE_I: tl.constexpr,
    STRIDE_O: tl.constexpr,
    STRIDE_P: tl.constexpr,
    PADDING: tl.constexpr,
    MODE: tl.constexpr,
    HAS_PER_SAMPLE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ASCEND_ABI: tl.constexpr = False,
    BAG_SIZE_B: tl.constexpr = False,
    SEARCH_STEPS: tl.constexpr = 0,
    INDEX_BLOCK: tl.constexpr = 128,
    ASYNC_ASSERT: tl.constexpr = False,
    SERIAL_MAX: tl.constexpr = False,
    KEY_MAX: tl.constexpr = False,
    RECIPROCAL_MEAN: tl.constexpr = False,
    SKIP_PAD_SCALE: tl.constexpr = False,
):
    _shared_forward_jit(
        Weight,
        Indices,
        Offsets,
        PerSample,
        Output,
        Offset2Bag,
        BagSize,
        MaxIndices,
        Error,
        N,
        O,
        B,
        D,
        V,
        STRIDE_W0,
        STRIDE_W1,
        STRIDE_I,
        STRIDE_O,
        STRIDE_P,
        PADDING,
        MODE,
        HAS_PER_SAMPLE,
        BLOCK_L,
        BLOCK_D,
        ASCEND_ABI=ASCEND_ABI,
        BAG_SIZE_B=BAG_SIZE_B,
        SEARCH_STEPS=SEARCH_STEPS,
        INDEX_BLOCK=INDEX_BLOCK,
        ASYNC_ASSERT=ASYNC_ASSERT,
        SERIAL_MAX=False,
        KEY_MAX=MODE == 2,
        RECIPROCAL_MEAN=MODE == 1,
        SKIP_PAD_SCALE=SKIP_PAD_SCALE,
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
    logger.debug("GEMS _EMBEDDING_BAG_FORWARD_ONLY")
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
