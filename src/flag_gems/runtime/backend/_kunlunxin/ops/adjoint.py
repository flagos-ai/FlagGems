# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kunlunxin ``adjoint``: a true zero-copy conjugate-transpose view.

``aten::adjoint`` is a pure metadata operator (``Tensor(a) -> Tensor(a)``):
the result must alias the input's storage, swap the last two strides and, for
complex inputs, carry the *lazy* conjugate bit.  No element is read, moved or
written.

Why a backend override is required
----------------------------------
The generic FlagGems implementation composes ``self.mT`` / ``self.mH``.  Both
``aten::mT`` and ``aten::mH`` are registered by ``FlagGems`` and the complex
path of ``mH`` ultimately calls ``Tensor.conj`` -> ``aten::conj``, which this
backend overrides with a *materialising* ``_conj`` kernel.  In production
dispatch (``flag_gems.use_gems()``) that turns ``adjoint`` on complex tensors
into a **copy**: the result loses the storage alias (``data_ptr`` changes),
loses the lazy conjugate bit, and launches a device kernel for an op that
should touch no data.  The same applies to the real path, where ``self.mT``
issues a per-rank "axis order" kernel.

This override builds the view directly from metadata, so it is correct under
both plain and ``use_gems`` dispatch:

* last-two axes swapped via ``torch.as_strided`` (a metadata view over the
  input's own storage, preserving ``storage_offset`` and alias);
* complex dtypes receive the lazy conjugate bit through
  ``torch._C._set_conj`` — the same per-tensor flag native ATen sets — toggled
  relative to the input so ``adjoint(conj(x))`` cancels exactly like ATen;
* real dtypes need nothing else (conjugation is the identity).

Rank rules mirror ATen: rank >= 2 -> the view; rank == 1 -> the same
"matrices or batches of matrices" ``RuntimeError``; rank == 0 -> the deprecated
notice plus a conj view (identity for real, lazy-conj for complex).
"""

import logging
import warnings

import torch

logger = logging.getLogger(__name__)

__all__ = ["adjoint"]

_1D_MESSAGE = (
    "tensor.adjoint() is only supported on matrices or batches of matrices. "
    "Got {}-D tensor."
)
_0D_MESSAGE = "adjoint() is deprecated on 0-D tensors. Consider using x.conj()."

_AS_STRIDED = torch.as_strided
_SET_CONJ = torch._C._set_conj


def _conj_bit_view(tensor: torch.Tensor, src_is_conj: bool) -> torch.Tensor:
    """Toggle the lazy conjugate bit of the fresh view ``tensor``.

    The returned view inherits (or not) whatever flag ``as_strided`` carried;
    we set it explicitly to ``not src_is_conj`` so that ``adjoint`` toggles the
    bit exactly like native ATen (``adjoint(conj(x))`` clears it).
    """
    _SET_CONJ(tensor, not src_is_conj)
    return tensor


def adjoint(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ADJOINT")

    ndim = self.dim()
    if ndim == 0:
        # Deprecated aten path: transpose is undefined, degrade to a conj view.
        warnings.warn(_0D_MESSAGE, UserWarning, stacklevel=2)
        out = _AS_STRIDED(self, (), (), self.storage_offset())
        if self.dtype.is_complex:
            out = _conj_bit_view(out, self.is_conj())
        return out

    if ndim < 2:
        raise RuntimeError(_1D_MESSAGE.format(ndim))

    shape = list(self.shape)
    strides = list(self.stride())
    shape[-2], shape[-1] = shape[-1], shape[-2]
    strides[-2], strides[-1] = strides[-1], strides[-2]
    out = _AS_STRIDED(self, shape, strides, self.storage_offset())

    if self.dtype.is_complex:
        out = _conj_bit_view(out, self.is_conj())
    return out
