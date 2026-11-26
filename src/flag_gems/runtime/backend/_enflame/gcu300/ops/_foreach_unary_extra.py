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

"""GCU300 fallbacks for ``_foreach_asin_`` and ``_foreach_erfc_``.

The single-tensor ``asin`` / ``erfc`` kernels abort on GCU300 (rc=134), so the
generic foreach path cannot use them.  These two entry points evaluate the same
functions from kernels that are healthy on this backend:

* ``asin(x) = atan2(x, sqrt(1 - x*x))``;
* ``erfc(x) = 1 - erf(x)``.

Inputs are staged to dense form and computed in float32 for the 16-bit dtypes;
the result is copied back into the original storage, so gappy views write
through.  The dtype/error contract (in-place promotion refusal, overlap
refusal, empty-list error) is taken from the generic unary table.
"""

import logging

import torch

import flag_gems
from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
)

logger = logging.getLogger(__name__)


def _check(name, tensors):
    op = UNARY_OPS[name]
    for t in tensors:
        if t.dtype.is_complex:
            raise NotImplementedError(
                f"foreach executor has no complex kernel for {t.dtype}"
            )
        if op.allowed is not None and t.dtype not in op.allowed:
            check_dtype_supported(t.dtype, op.allowed)
        out_dtype = op.out_dtype_fn(t.dtype)
        if out_dtype != t.dtype:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(t.dtype).replace('torch.', '').capitalize()}"
            )
        if has_internal_overlap(t):
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
    return op


def _work(t):
    return t if t.is_contiguous() else t.contiguous()


def _compute_dtype(t):
    # ATen evaluates these transcendental kernels in float32 for the 16-bit
    # float types; doing the same avoids the extra rounding of a fp16
    # subtraction/square in the asin and erfc identities.
    if t.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return t.dtype


def _foreach_asin_(self):
    logger.debug("GEMS _FOREACH_ASIN_")
    tensors = check_tensor_list(self)
    _check("asin", tensors)
    for t in tensors:
        work = _work(t)
        dt = _compute_dtype(t)
        x = work.to(dt)
        res = flag_gems.atan2(x, flag_gems.sqrt(1.0 - x * x))
        t.copy_(res.to(t.dtype))
    return None


def _foreach_erfc_(self):
    logger.debug("GEMS _FOREACH_ERFC_")
    tensors = check_tensor_list(self)
    _check("erfc", tensors)
    for t in tensors:
        work = _work(t)
        dt = _compute_dtype(t)
        x = work.to(dt)
        res = 1.0 - flag_gems.erf(x)
        t.copy_(res.to(t.dtype))
    return None


__all__ = ["_foreach_asin_", "_foreach_erfc_"]
