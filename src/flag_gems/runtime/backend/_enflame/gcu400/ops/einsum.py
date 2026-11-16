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

"""GCU400 einsum specialization: dispatch common equations to Triton paths inside the use_gems() context.

aten's generic einsum is split into a chain of pointwise + generic reductions on the enflame backend;
`i,i->` on 65536 elements compiles a single-block reduction kernel needing 2.6MB DSM (hardware limit
917504B), which directly raises OutOfResources; `i,j->ij` lands on the broadcast-mul gather path.
This module takes over only within the FlagGems registration context; the native reference still uses the original aten
implementation.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_ORIGINAL_EINSUM = torch.einsum
_DOT_BLOCK = 1024


@triton.jit
def _einsum_dot_partial(x_ptr, y_ptr, partial_ptr, N, BLOCK: tl.constexpr):
    # Each program reduces one BLOCK segment; the in-block state vector has fixed length, so DSM is decoupled from N
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    xv = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    yv = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid, tl.sum(xv * yv))


@triton.jit
def _einsum_dot_final(partial_ptr, out_ptr, M, BLOCK_M: tl.constexpr):
    offsets = tl.arange(0, BLOCK_M)
    vals = tl.load(partial_ptr + offsets, mask=offsets < M, other=0.0)
    tl.store(out_ptr, tl.sum(vals))


def _einsum_dot(x, y):
    # Deferred import avoids circular dependencies during flag_gems initialization; two-stage tiled reduction decouples
    # DSM from N
    from flag_gems.runtime import torch_device_fn

    num_elements = x.numel()
    block = min(_DOT_BLOCK, triton.next_power_of_2(num_elements))
    num_partials = triton.cdiv(num_elements, block)
    partial = torch.empty((num_partials,), dtype=torch.float32, device=x.device)
    out = torch.empty((), dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _einsum_dot_partial[(num_partials,)](
            x, y, partial, num_elements, BLOCK=block, num_warps=8
        )
        _einsum_dot_final[(1,)](
            partial,
            out,
            num_partials,
            BLOCK_M=triton.next_power_of_2(num_partials),
            num_warps=4,
        )
    return out


def _gems_is_active():
    # use_gems().__enter__ sets this global via enable() and deletes it on exit
    try:
        import flag_gems

        return getattr(flag_gems, "current_work_registrar", None) is not None
    except Exception:
        return False


def _dispatch(equation, operands):
    eq = equation.replace(" ", "")
    if eq == "i,i->":
        return _einsum_dot(operands[0], operands[1])
    if eq == "i,j->ij":
        # Call enflame's row-streaming Triton kernel directly, avoiding aten's broadcast-mul path
        import flag_gems

        return flag_gems.outer(operands[0], operands[1])
    if eq in ("ij,jk->ik", "bij,bjk->bik", "...ij,...jk->...ik"):
        return torch.matmul(operands[0], operands[1])
    if eq == "ii->":
        return torch.diagonal(operands[0]).sum()
    if eq == "ii->i":
        return torch.diagonal(operands[0])
    if eq == "ij->ji":
        return operands[0].transpose(0, 1)
    if eq == "ijk->":
        return operands[0].sum()
    if eq == "ijk->j":
        return operands[0].sum(dim=(0, 2))
    # Uncovered equations keep native aten behavior to avoid affecting other operators
    return _ORIGINAL_EINSUM(equation, *operands)


def einsum(equation, *operands):
    if _gems_is_active():
        try:
            return _dispatch(equation, operands)
        except Exception:
            logger.debug("GCU400 einsum fell back for equation %s", equation)
            return _ORIGINAL_EINSUM(equation, *operands)
    return _ORIGINAL_EINSUM(equation, *operands)


torch.einsum = einsum
