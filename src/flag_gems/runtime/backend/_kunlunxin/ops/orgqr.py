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

"""Kunlunxin XPU3 override for ``orgqr`` / ``orgqr_out``.

The generic kernel keeps the whole ``[M, N]`` block in one tile and walks the
reflectors with a runtime loop.  This backend rejects that tile in
``TritonXPUCoreTiling`` (``'arith.muli' op requires the same encoding for all
operands and results``), so the generic file does not compile here at all.

The override keeps one output column per program and reduces over a 1-D vector of
rows, which the backend accepts.  Two properties of this backend shape the rest:

* the row extent must cover the whole matrix in a single program -- splitting it
  into tiles makes the backend store to the wrong rows -- so the row tile is
  always ``next_power_of_2(M)`` and the grid only spans batch and column;
* a masked ``tl.load`` whose mask combines two conditions returns real memory for
  the masked-off lanes, so the reflector vector is loaded with a single
  ``rows < M`` mask and the ``rows > k`` / ``rows == k`` parts are selected with
  ``tl.where``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def _orgqr_kernel(inp, tau, out, M, N, K, BLOCK_M: tl.constexpr):
    """Build one column of Q from the Householder reflectors of one batch item."""
    batch = tl.program_id(0)
    col = tl.program_id(1)

    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < M
    base = batch * M * N

    q = tl.where(rows == col, 1.0, 0.0).to(out.dtype.element_ty)
    for k in range(K - 1, -1, -1):
        tau_k = tl.load(tau + batch * K + k)
        # A single-mask load: the backend returns real memory for lanes excluded
        # by a compound mask, so the tail is selected arithmetically instead.
        raw = tl.load(inp + base + rows * N + k, mask=row_mask, other=0.0)
        v = tl.where(rows == k, 1.0, tl.where(rows > k, raw, 0.0))
        q = q - tau_k * v * tl.sum(v * q, axis=0)

    tl.store(out + base + rows * N + col, q, mask=row_mask)


def _validate_orgqr(input, tau, out=None):
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


def _orgqr_impl(input, tau, out):
    _validate_orgqr(input, tau, out)
    if out.shape != input.shape:
        out.resize_(input.shape)

    m, n = input.shape[-2:]
    k = tau.shape[-1]
    batch_size = input.numel() // max(1, m * n)
    if out.numel() == 0 or n == 0:
        return out

    input_work = input.contiguous().view(batch_size, m, n)
    tau_work = tau.contiguous().view(batch_size, k)

    if input_work.data_ptr() == out.data_ptr():
        # The reflectors are read while Q is written, so an aliased out needs the
        # input preserved first.
        input_work = input_work.clone()

    # The kernel walks a row-major [batch, M, N] tile, so Q is always built in a
    # fresh contiguous workspace and copied into out afterwards.  Writing through
    # a view of out would silently assume out's strides are row-major, which is
    # not true for the transposed outputs callers pass in.
    out_work = torch.empty((batch_size, m, n), dtype=input.dtype, device=out.device)

    block_m = triton.next_power_of_2(m)
    with torch_device_fn.device(input.device):
        _orgqr_kernel[(batch_size, n)](
            input_work,
            tau_work,
            out_work,
            m,
            n,
            k,
            BLOCK_M=block_m,
        )
        if out.dtype != input.dtype or out_work.data_ptr() != out.data_ptr():
            # A dtype cast and a strided out both need an explicit write-back.
            copy_(out, out_work.view(out.shape))
    return out


def orgqr(input, tau):
    """Compute real float32/float64 Q from Householder reflectors."""
    logger.debug("GEMS_KUNLUNXIN ORGQR")
    _validate_orgqr(input, tau)
    out = torch.empty_like(input)
    return _orgqr_impl(input, tau, out)


def orgqr_out(input, tau, *, out):
    """Write Q into ``out``, preserving its storage and arbitrary batch strides."""
    logger.debug("GEMS_KUNLUNXIN ORGQR_OUT")
    return _orgqr_impl(input, tau, out)


__all__ = ["orgqr", "orgqr_out"]
