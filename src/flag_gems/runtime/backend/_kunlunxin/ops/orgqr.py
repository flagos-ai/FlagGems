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

"""Kunlunxin/XPU implementation of ``orgqr`` / ``orgqr.out``.

``torch.orgqr(A, tau)`` is exactly ``torch.linalg.householder_product(A, tau)``:
given the Householder reflectors (``A``, ``tau``) produced by ``torch.geqrf``,
build ``Q = H(0) H(1) ... H(k-1)``.

The generic ``flag_gems.ops.orgqr`` kernel keeps Q in a 2-D register tile and
reduces it along axis 0 (``tl.sum(v[:, None] * q, axis=0)``) inside a dynamic
loop that also stores and re-loads the tile.  TritonXPU rejects that shape
(``'arith.muli' op requires the same encoding`` during ``TritonXPULegalize``)
and cannot lower ``tt.trans`` as a workaround (see
``harness/solution/inverse/README.md``).

The backend already ships a validated pure-Triton Householder product
(``_kunlunxin/ops/linalg_householder_product.py``: one column per program,
1-D scalar reductions, clamped addresses, no store/reload loop).  This module
reuses it for the numerical work and only adds the ``orgqr``/``orgqr.out``
contract handling (shape/dtype/device validation, out resize, final cast).
"""

import importlib
import logging

import torch

from flag_gems.ops.copy import copy_

logger = logging.getLogger(__name__)


def _backend_hp_module():
    # Submodule import, not the same-named function (see slogdet.py).
    return importlib.import_module(".linalg_householder_product", __package__)


def _compute(input, tau):
    return _backend_hp_module().linalg_householder_product(input, tau)


def _validate(input, tau, out=None):
    if input.ndim < 2:
        raise RuntimeError("orgqr: input must have at least 2 dimensions")
    if tau.ndim != input.ndim - 1 or tau.shape[:-1] != input.shape[:-2]:
        raise RuntimeError("orgqr: tau batch dimensions must match input")
    if input.shape[-2] < input.shape[-1]:
        raise RuntimeError(
            "orgqr: input.shape[-2] must be greater than or equal to input.shape[-1]"
        )
    if tau.shape[-1] > input.shape[-1]:
        raise RuntimeError(
            "orgqr: input.shape[-1] must be greater than or equal to tau.shape[-1]"
        )
    if input.dtype != tau.dtype or input.device != tau.device:
        raise RuntimeError("orgqr: input and tau must have the same dtype and device")
    if input.is_complex():
        raise RuntimeError(
            "orgqr: complex dtypes are not supported by this Triton kernel"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("orgqr: only float32 and float64 inputs are supported")
    if out is not None:
        if out.device != input.device:
            raise RuntimeError("orgqr: input and out must be on the same device")
        if not torch.can_cast(input.dtype, out.dtype):
            raise RuntimeError(
                f"orgqr: result dtype {input.dtype} cannot be safely cast to {out.dtype}"
            )


def _orgqr_impl(input, tau, out=None):
    _validate(input, tau, out)
    result = _compute(input, tau)

    if out is None:
        return result

    # torch.orgqr(out=...) resizes an empty/short out and then writes the
    # result, preserving the storage of a correctly shaped out (including
    # non-contiguous batch layouts and an out aliasing the input).
    if out.shape != input.shape:
        out.resize_(input.shape)
    copy_(out, result)
    return out


def orgqr(input, tau):
    """Compute real float32/float64 Q from Householder reflectors."""
    logger.debug("GEMS_KUNLUNXIN ORGQR")
    return _orgqr_impl(input, tau)


def orgqr_out(input, tau, *, out):
    """Write Q into ``out``, preserving its storage and batch strides."""
    logger.debug("GEMS_KUNLUNXIN ORGQR_OUT")
    return _orgqr_impl(input, tau, out)
