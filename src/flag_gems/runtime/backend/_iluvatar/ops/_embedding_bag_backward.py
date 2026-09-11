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

from flag_gems.ops._embedding_bag_backward import (
    _eb_backward_finish,
    _eb_backward_validate,
    _eb_backward_validate_body,
    _embedding_bag_backward_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._embedding_bag import _check_corex_error

logger = logging.getLogger(__name__)


class _BackwardLaunch:
    def __init__(self):
        self.error = None
        self.finish = None

    def __call__(self, kernel, packed_kernel, grid, pointers, metadata, **options):
        if kernel is _eb_backward_finish:
            self.finish = (pointers, metadata)
            return None
        if kernel is _eb_backward_validate:
            self.error = pointers[9]
            # Later CoreX launches can consume the SDK's error status. Report
            # the validated flags after all gradient kernels have been queued.
            metadata = (*metadata[:18], False, *metadata[19:])
            options["debug"] = False
        return kernel[grid](*pointers, *metadata, **options)

    def check(self):
        if self.error is not None:
            _check_corex_error(self.error, self.error.numel(), self.finish)


@libentry()
@triton.jit
def _max_initialize(
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    error,
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_validate_body(
        indices,
        offsets,
        offset_to_bag,
        bag_size,
        maximum_indices,
        indices,
        indices,
        indices,
        indices,
        error,
        out,
        meta.value[2] * meta.value[3],
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
        meta.value[4],
        meta.value[5],
        meta.value[8],
        meta.value[9],
        meta.value[10],
        meta.value[11],
        meta.value[12],
        meta.value[13],
        meta.value[14],
        2,
        False,
        False,
        False,
        False,
        True,
        0,
        256,
        True,
    )


@libentry()
@triton.jit
def max_scatter_direct(
    grad,  # output gradient
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    num_bags: tl.constexpr = meta.value[1]
    embedding_dim: tl.constexpr = meta.value[2]
    num_weights: tl.constexpr = meta.value[3]
    pad: tl.constexpr = meta.value[5]
    sg0: tl.constexpr = meta.value[6]
    sg1: tl.constexpr = meta.value[7]
    sb: tl.constexpr = meta.value[11]
    sx0: tl.constexpr = meta.value[12]
    sx1: tl.constexpr = meta.value[13]
    block: tl.constexpr = meta.value[19]
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    if embedding_dim > 0:
        bags = x // embedding_dim
        cols = x % embedding_dim
        maximum = tl.load(
            maximum_indices + bags * sx0 + cols * sx1,
            x < num_bags * embedding_dim,
            other=-1,
        ).to(tl.int64)
        sizes = tl.load(bag_size + bags * sb, x < num_bags * embedding_dim, other=0)
        active = (
            (x < num_bags * embedding_dim)
            & (maximum >= 0)
            & (maximum < num_weights)
            & (maximum != pad)
            & (sizes > 0)
        )
        values = tl.load(grad + bags * sg0 + cols * sg1, active, other=0).to(
            out.dtype.element_ty
        )
        tl.atomic_add(
            out + maximum * embedding_dim + cols, values, active, sem="relaxed"
        )


def _compute_max(grad, indices, offsets, mapping, sizes, maximum, num_weights, padding):
    b, d = grad.shape
    n = indices.numel()
    if (
        not n
        or not d
        or grad.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    block, warps = 128, 4
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    cast = grad.dtype in (torch.float16, torch.bfloat16)
    accumulator = (
        torch.empty((num_weights, d), dtype=torch.float32, device=grad.device)
        if cast
        else output
    )
    blocks = max(
        triton.cdiv(num_weights * d, 1024),
        triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256),
    )
    error = torch.empty((blocks,), device=grad.device, dtype=torch.int32)
    meta = (
        n,
        b,
        d,
        num_weights,
        offsets.numel(),
        padding,
        *grad.stride(),
        indices.stride(0),
        offsets.stride(0),
        mapping.stride(0),
        sizes.stride(0),
        *maximum.stride(),
        mapping.numel() != 0 and d != 0,
        grad.dtype == torch.float64,
        False,
        0,
        0,
        block,
    )
    with torch_device_fn.device(grad.device):
        _max_initialize[(blocks,)](
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            accumulator,
            error,
            meta,
            debug=False,
        )
        if b and d and num_weights:
            max_scatter_direct[(triton.cdiv(b * d, block),)](
                grad, sizes, maximum, accumulator, meta, num_warps=warps
            )
        finish = (
            ((accumulator, error, output), (num_weights * d, max(d, 1), False, 4096))
            if cast
            else None
        )
        _check_corex_error(error, blocks, finish)
    return output


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
    logger.debug("GEMS _EMBEDDING_BAG_BACKWARD")
    launcher = _BackwardLaunch()
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
        launch_fn=launcher,
        max_fn=_compute_max,
    )
    if not sparse:
        launcher.check()
    return result
