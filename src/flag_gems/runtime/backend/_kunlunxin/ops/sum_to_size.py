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
"""Kunlunxin sum_to_size override.

The generic implementation delegates the reduction to ``flag_gems.ops.sum``:
the flat two-stage kernel fails to compile on XPU3 for every benchmark shape
(``tl.sum`` over a masked 1-D tile of ``next_pow2(M)`` lanes), and the generic
``sum_dim`` produces wrong values for a middle-dim reduction of a 3-D tensor.
The vendor ``sum``/``sum_dim`` kernels are already tuned (TLE row reduce) and
exact on this backend, so this override only reimplements the shape bookkeeping
and routes the reduction through them.
"""

import logging

import torch

from flag_gems.runtime.backend._kunlunxin.ops.sum import sum_dim as _vendor_sum_dim

logger = logging.getLogger(__name__)


def sum_to_size(inp, *size):
    logger.debug("GEMS_KUNLUNXIN SUM_TO_SIZE")
    # ``size`` may arrive as a single sequence (the aten schema passes a
    # SymInt[] list) or as a variadic list of ints (Tensor.sum_to_size(*size)).
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    else:
        size = tuple(size)

    target_shape = [int(s) for s in size]
    inp_shape = list(inp.shape)

    # Reverse the broadcast: sum over the leading dims ``size`` does not cover
    # and over the trailing dims where the target is 1 but the input is not,
    # keeping those dims so the result reshapes back to ``size``.
    leading_dims = inp.ndim - len(target_shape)
    reduce_dims = list(range(leading_dims))
    for i in range(leading_dims, inp.ndim):
        if target_shape[i - leading_dims] == 1 and inp_shape[i] != 1:
            reduce_dims.append(i)

    if len(reduce_dims) == 0:
        return inp.reshape(target_shape)

    if inp.dtype == torch.bool:
        inp = inp.to(torch.int64)

    out = _vendor_sum_dim(inp, reduce_dims, keepdim=True)
    return out.reshape(target_shape)
