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

"""Kunlunxin backend override for ``logdet``.

The generic register-resident LU kernel does not lower on TritonXPU at any
matrix order -- n = 2 already fails with the same frontend error as the larger
sizes:

    PassManager::run failed ... ConvertTritonToTritonXPU
    triton.runtime.errors.OutOfResources: out of resource: uni_sram
      Required: 0, Hardware limit: 0

so this is not a tile-size problem and shrinking cannot help. The backend
already ships a working ``linalg_slogdet`` Triton implementation, so this
override reuses it and only adds the ``slogdet -> logdet`` post-processing in
its own kernel.

The post-processing is deliberately *not* expressed with ``torch.where``: the
backend's ``where`` lowers the select as an arithmetic blend
``self * c + other * (1 - c)``, and ``nan * 0`` is ``nan``, so a ``nan`` in the
branch the condition does not select still poisons the result. That is exactly
the shape of ``logdet`` (``nan`` for a negative determinant), so the selection
is done inside the kernel with ``tl.where`` instead.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.logdet import _REGISTER_TILE_LIMIT
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_BLOCK = 256


@libentry()
@triton.jit
def _logdet_post_kernel(
    sign_ptr,
    logabsdet_ptr,
    out_ptr,
    M,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    sign = tl.load(sign_ptr + offs, mask=mask, other=1.0)
    logabsdet = tl.load(logabsdet_ptr + offs, mask=mask, other=0.0)
    # torch.logdet: nan for a negative determinant, -inf for a singular matrix.
    result = tl.where(sign > 0.0, logabsdet, float("nan"))
    result = tl.where(sign == 0.0, -float("inf"), result)
    tl.store(out_ptr + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


def logdet(inp):
    logger.debug("GEMS_KUNLUNXIN LOGDET")

    if inp.dim() < 2 or inp.shape[-1] != inp.shape[-2]:
        raise RuntimeError("logdet: input must be batches of square matrices")
    if inp.is_complex():
        raise RuntimeError("logdet: complex inputs are not supported by this kernel")
    if inp.requires_grad:
        raise RuntimeError("logdet: autograd is not supported by this kernel")
    if inp.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(f"logdet: unsupported dtype {inp.dtype}")

    n = inp.shape[-1]
    batch_shape = tuple(inp.shape[:-2])
    batch_count = math.prod(batch_shape)

    fp64_unsupported = inp.dtype == torch.float64 and n > 16
    if n > _REGISTER_TILE_LIMIT or fp64_unsupported:
        raise RuntimeError(
            f"logdet: {inp.dtype} matrices of size {n} are not supported by "
            "the Triton kernel"
        )

    if n == 0:
        return torch.zeros(batch_shape, dtype=inp.dtype, device=inp.device)
    if batch_count == 0:
        return torch.empty(batch_shape, dtype=inp.dtype, device=inp.device)

    sign, logabsdet = torch.linalg.slogdet(inp)
    # ``logabsdet`` is freshly allocated by ``slogdet`` and not aliased by the
    # caller, so it doubles as the output buffer: the post-processing rewrites it
    # in place and saves one allocation on the host-bound small-shape path.
    flat = logabsdet.reshape(-1)
    _logdet_post_kernel[(triton.cdiv(batch_count, _BLOCK),)](
        sign.reshape(-1),
        flat,
        flat,
        batch_count,
        BLOCK=_BLOCK,
        num_warps=1,
    )
    return logabsdet


__all__ = ["logdet"]
