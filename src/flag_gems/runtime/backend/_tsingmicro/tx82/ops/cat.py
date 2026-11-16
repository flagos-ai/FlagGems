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
import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from flag_gems.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tensor_wrapper import StridedBuffer
from .copy import _int64_large_strided_unsafe
from .copy import copy_ as copy_tensor

logger = logging.getLogger(__name__)

# f32 can only represent integers up to 2**24 exactly; under precision-mode 0/1
# i32 strides are demoted to f32 arithmetic, so larger address intermediates
# lose precision and elements land at wrong offsets. Force pm=2 for exact
# integer arithmetic (same rationale as flip).
_F32_PRECISION_THRESHOLD = 2**24


def _needs_i64_precision(inputs, out0):
    in_max = 0
    for a in inputs:
        if any(abs(a.stride(i)) > _F32_PRECISION_THRESHOLD for i in range(a.ndim)):
            return True
        in_max = max(
            in_max,
            sum((a.shape[i] - 1) * abs(a.stride(i)) for i in range(a.ndim)),
        )
    # output views are freshly allocated, so the store's max physical offset
    # (logical - offset) * stride is bounded by out0.numel()
    return max(in_max, out0.numel()) > _F32_PRECISION_THRESHOLD


config_ = CodeGenConfig(
    1024,
    (16, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_TSINGMICRO CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # Find max ndim among all tensors
    max_ndim = max(_.ndim for _ in A)

    assert dim >= -max_ndim and dim < max_ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive (relative to max_ndim)
    dim = dim % max_ndim

    # Handle mixed-rank tensors: PyTorch allows cat when a lower-rank
    # tensor is empty (has 0 in some dimension). It implicitly unsqueezes
    # the empty tensor to match max_ndim, placing its 0-size at the dim position.
    if any(_.ndim != max_ndim for _ in A):
        # Find a reference tensor with max_ndim for non-dim sizes
        ref = next(_ for _ in A if _.ndim == max_ndim)
        ref_shape = ref.shape

        for i, a in enumerate(A):
            if a.ndim < max_ndim:
                new_shape = list(ref_shape)
                new_shape[dim] = a.shape[dim % a.ndim]
                A[i] = a.reshape(new_shape)

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    out0_dim_offsets = list(
        itertools.accumulate([s[dim] for s in inp_shapes[:-1]], initial=0)
    )

    # Per-dim offsets cannot be expressed with block pointers (boundary_check
    # clips tiles whose offsets exceed the logical shape), so always use the
    # masked-load kernel variant.
    config = copy.copy(copy_func.config)
    config.prefer_block_pointer = False
    copy_func.config = config

    needs_i64 = _needs_i64_precision(A, out0)
    precision_token = set_precision_mode("2") if needs_i64 else None
    try:
        for a, out0_dim_offset in zip(A, out0_dim_offsets):
            out_tensor_view = out0.narrow(dim, out0_dim_offset, a.shape[dim])
            if _int64_large_strided_unsafe(out_tensor_view, a):
                # TX Triton cannot lower some int64 pointwise copies with a large
                # non-contiguous stride. Reuse copy_'s ATen redispatch fallback.
                copy_tensor(out_tensor_view, a)
                continue
            per_dim_offsets = [0] * len(a.shape)
            # negative: store computes addr = base + (logical - offset) * stride,
            # so a negative offset shifts the write region forward along `dim`
            per_dim_offsets[dim] = -out0_dim_offset
            in_view = StridedBuffer(a, a.shape, a.stride())
            out_view = StridedBuffer(
                out0, a.shape, out0.stride(), offsets=per_dim_offsets
            )
            copy_func.instantiate(a.ndim)(in_view, out0=out_view)
    finally:
        if precision_token is not None:
            restore_precision_mode(precision_token)
    return out0
