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

from flag_gems.ops.segment_reduce import (
    _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH,
    _check_index_tensor,
    _check_reduce_and_dtype,
    _get_uniform_segment_length,
)
from flag_gems.ops.segment_reduce import (
    _segment_reduce_backward as _generic_segment_reduce_backward,
)
from flag_gems.ops.segment_reduce import _wrap_axis
from flag_gems.ops.segment_reduce import segment_reduce as _generic_segment_reduce

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _native_prod(inp, dim):
    return torch.ops.aten.prod.dim_int.redispatch(
        _FALLBACK_KEYSET, inp, dim, False, dtype=None
    )


def _long_uniform_prod(data, reduce, lengths, indices, offsets, axis, initial):
    if (
        reduce != "prod"
        or initial is not None
        or lengths is None
        or offsets is not None
        or indices is not None
    ):
        return None

    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    _check_index_tensor(data, lengths, "lengths", axis)
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None or segment_length <= _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH:
        return None

    segment_count = lengths.shape[-1]
    view_shape = (
        data.shape[:axis] + (segment_count, segment_length) + data.shape[axis + 1 :]
    )
    reshaped = data.contiguous().reshape(view_shape)
    return _native_prod(reshaped, axis + 1)


def segment_reduce(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
):
    logger.debug("GEMS_METAX SEGMENT_REDUCE")
    result = _long_uniform_prod(data, reduce, lengths, indices, offsets, axis, initial)
    if result is not None:
        return result
    return _generic_segment_reduce(
        data,
        reduce,
        lengths=lengths,
        indices=indices,
        offsets=offsets,
        axis=axis,
        unsafe=unsafe,
        initial=initial,
    )


def segment_reduce_out(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
    out,
):
    logger.debug("GEMS_METAX SEGMENT_REDUCE_OUT")
    result = segment_reduce(
        data,
        reduce,
        lengths=lengths,
        indices=indices,
        offsets=offsets,
        axis=axis,
        unsafe=unsafe,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out


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
    logger.debug("GEMS_METAX _SEGMENT_REDUCE_BACKWARD")
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
    logger.debug("GEMS_METAX _SEGMENT_REDUCE_BACKWARD_OUT")
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
