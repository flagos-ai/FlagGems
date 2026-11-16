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

from flag_gems.ops.segment_reduce import _prepare_common, _prod
from flag_gems.ops.segment_reduce import (
    _segment_reduce_backward as _generic_segment_reduce_backward,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _segment_reduce_max_min_backward_kernel(
    grad,
    output,
    data,
    offsets,
    grad_input,
    segment_count,
    inner_size,
    data_size_axis,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    inner_idx = pid % inner_size
    row_idx = pid // inner_size
    dim_idx = row_idx % segment_count
    outer_idx = row_idx // segment_count

    offsets_base = outer_idx * (segment_count + 1) + dim_idx
    segment_start = tl.load(offsets + offsets_base)
    segment_end = tl.load(offsets + offsets_base + 1)

    if segment_start < segment_end:
        grad_value = tl.load(grad + pid).to(compute_dtype)
        output_value = tl.load(output + pid).to(compute_dtype)
        match_count = tl.full((), 0, dtype=tl.int32)

        pos = segment_start
        while pos < segment_end:
            data_offset = (
                outer_idx * data_size_axis * inner_size + pos * inner_size + inner_idx
            )
            value = tl.load(data + data_offset).to(compute_dtype)
            is_match = (value != value) | (value == output_value)
            match_count += is_match.to(tl.int32)
            pos += 1

        store_value = tl.where(
            (match_count >= 2) & (grad_value > 0),
            grad_value / match_count,
            grad_value,
        )
        pos = segment_start
        while pos < segment_end:
            data_offset = (
                outer_idx * data_size_axis * inner_size + pos * inner_size + inner_idx
            )
            value = tl.load(data + data_offset).to(compute_dtype)
            is_match = (value != value) | (value == output_value)
            tl.store(grad_input + data_offset, store_value, mask=is_match)
            pos += 1


def _segment_reduce_backward(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
):
    logger.debug("GEMS_ASCEND _SEGMENT_REDUCE_BACKWARD")
    if reduce not in ("max", "min"):
        return _generic_segment_reduce_backward(
            grad,
            output,
            data,
            reduce,
            lengths=lengths,
            offsets=offsets,
            axis=axis,
            initial=initial,
        )

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, None, axis, True
    )
    data_contig = data.contiguous()
    grad_contig = grad.contiguous()
    output_contig = output.contiguous()
    grad_input = torch.zeros(data_contig.shape, dtype=grad.dtype, device=grad.device)
    if output_contig.numel() == 0:
        return grad_input

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    grid = (output_contig.numel(),)
    with torch_device_fn.device(data.device):
        _segment_reduce_max_min_backward_kernel[grid](
            grad_contig,
            output_contig,
            data_contig,
            offsets_contig,
            grad_input,
            segment_count,
            inner_size,
            data_contig.shape[axis],
        )
    return grad_input


def _segment_reduce_backward_out(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
    out,
):
    logger.debug("GEMS_ASCEND _SEGMENT_REDUCE_BACKWARD_OUT")
    result = _segment_reduce_backward(
        grad,
        output,
        data,
        reduce,
        lengths=lengths,
        offsets=offsets,
        axis=axis,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out
