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

import copy
import logging

import torch
import triton
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

# f32 represents every integer with |n| <= 2**24 exactly (in [2**23, 2**24)
# the spacing is 1.0, so each integer is still exact; inexactness only starts
# above 2**24). Under precision-mode 0/1 i32 strides are demoted to f32
# arithmetic, so address intermediates >= 2**24 lose precision and elements
# land at wrong offsets. Every intermediate in the flip address chain is
# bounded by max_abs_offset, so gating on 2**24 keeps the whole chain exact —
# tightening to 2**23 would only cost performance, not gain correctness.
# Force pm=2 to keep exact integer arithmetic beyond that bound.
_F32_PRECISION_THRESHOLD = 2**24


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_TSINGMICRO FLIP")
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        assert not flip_dims_b[
            dim
        ], "dim {} appears multiple times in the list of dims".format(dim)
        flip_dims_b[dim] = True
    n = 0
    per_dim_offsets = [0] * len(flip_dims_b)
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            per_dim_offsets[i] = A.shape[i] - 1
            strides[i] = -strides[i]
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()
    # flip's linear offset can far exceed numel (dilated inputs double the
    # strides), so gate integer-precision protection on the max absolute offset.
    # The tile-decode tid itself reaches numel, so broadcast views with a small
    # max_abs_offset (e.g. A.expand(2**26, 10)) still need gating on numel.
    max_abs_offset = sum((A.shape[i] - 1) * abs(A.stride(i)) for i in range(A.dim()))
    needs_i64 = (
        A.numel() > _F32_PRECISION_THRESHOLD
        or max_abs_offset > _F32_PRECISION_THRESHOLD
        or any(abs(A.stride(i)) > _F32_PRECISION_THRESHOLD for i in range(A.dim()))
    )
    precision_token = set_precision_mode("2") if needs_i64 else None
    try:
        out = torch.empty_like(A)
        # a flipped view of A: keep the base pointer unshifted and hand the per-dim
        # offsets to the kernel. Folding them into the data pointer moves the DMA
        # base out of the allocation for views whose declared shape exceeds the
        # underlying storage, and the driver's OOB check then reads a bogus header
        # through address aliasing (or faults on unmapped memory).
        flipped_A = StridedBuffer(A, strides=strides, offsets=per_dim_offsets)

        # Negative strides + offsets cannot be expressed with block pointers
        # (boundary_check clips tiles whose offsets exceed the logical shape), so
        # flip always uses the masked-load kernel variant.
        config = copy.copy(copy_func.config)
        config.prefer_block_pointer = False
        copy_func.config = config

        # TODO: flip op can have a custom task simplification method, but we skip it now and just use A's rank.
        overload = copy_func.instantiate(A.ndim)
        overload(flipped_A, out0=out)
        return out
    finally:
        if precision_token is not None:
            restore_precision_mode(precision_token)
