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

"""Kunlunxin XPU3 override for ``aten::ravel``.

The VIEW path (``inp.is_contiguous()`` -> ``inp.view(-1)``) is kept unchanged:
it is zero-copy and needs no kernel.

Only the COPY path changes.  Generic ``flag_gems/ops/ravel.py`` dispatches the
copy on the canonical layout shape and, for layouts whose outermost canonical
dim has stride 1 (e.g. a transposed contiguous tensor), uses
``_ravel_tile_kernel``: a ``(BR, BI0)`` 2-D transposing tile built from
``tl.arange`` 2-D tensors.  On XPU3 that tile fails to compile
(``arith.addi`` type mismatch / ``uni_sram`` OutOfResources), and the rank-1
``_ravel_strided1d_kernel`` specialisation returns wrong values for bf16
strided inputs.  This is the ``pattern-xpu3-2d-tile-silent-miscompile`` family.

The existing rank-agnostic 1-D flat gather (``_ravel_copy_kernel``) is correct
on XPU3 for every tested layout (it is the route generic already takes for the
expanded / rank>=2 non-inner-stride layouts), so the override routes *all*
non-contiguous, non-complex inputs through it.  Complex inputs keep the generic
1-D complex kernel.

The benchmark only feeds contiguous tensors, so the VIEW path (0 kernels) is
what the performance number measures; the copy path is a correctness fix.
"""

import logging

import torch
import triton

from flag_gems.ops.ravel import (
    _canonical_dims,
    _MAX_DIMS,
    _pack,
    _ravel_copy_cplx_kernel,
    _ravel_copy_kernel,
)

logger = logging.getLogger(__name__)

__all__ = ["ravel"]

_BLOCK = 1024
_NUM_WARPS = 4


def _copy_impl(inp):
    numel = inp.numel()
    out = torch.empty(numel, dtype=inp.dtype, device=inp.device)
    if numel == 0:
        return out

    dims = _canonical_dims(inp)
    rank = len(dims)
    if rank == 0:  # a single element is always contiguous: unreachable
        return out
    if rank > _MAX_DIMS:
        raise RuntimeError(
            "ravel: unsupported strided layout rank %d (max %d)" % (rank, _MAX_DIMS)
        )

    neg = bool(inp.is_neg())
    packed = _pack(dims)

    if inp.dtype.is_complex:
        _ravel_copy_cplx_kernel[(triton.cdiv(numel, _BLOCK),)](
            inp.data_ptr(),
            out.data_ptr(),
            numel,
            *packed,
            R=rank,
            WIDE=(inp.dtype == torch.complex128),
            NEG=neg,
            CONJ=bool(inp.is_conj()),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        return out

    _ravel_copy_kernel[(triton.cdiv(numel, _BLOCK),)](
        inp,
        out,
        numel,
        *packed,
        R=rank,
        NEG=neg,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


class _RavelCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.input_shape = inp.shape
        return _copy_impl(inp)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.reshape(ctx.input_shape)


def ravel(self):
    """``aten::ravel(Tensor(a) self) -> Tensor(a)``."""
    logger.debug("GEMS_KUNLUNXIN RAVEL")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "ravel(): argument 'self' must be a Tensor, not %s" % type(self).__name__
        )
    if self.layout != torch.strided:
        raise RuntimeError("unsupported memory format option Contiguous")

    if self.is_contiguous():
        return self.view(-1)

    if self.requires_grad and torch.is_grad_enabled():
        return _RavelCopy.apply(self)
    return _copy_impl(self)
