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

"""GCU400 ``matrix_H`` specialization.

``matrix_H`` is metadata-only: it returns a transposed (and lazily conjugated)
view and never launches a kernel, so the whole measured cost is the Python host
path. The generic implementation routes every call through a shared helper that
re-reads the rank/messages from module state and emits a log record; this
version binds the C-level Tensor descriptors once at import, inlines the rank
contract of ``aten::matrix_H`` and skips the per-call log record.

Contract (identical to the generic implementation):
  * rank 1 -> RuntimeError, rank 0 -> deprecated warn-once path degrading to a
    lazy ``conj()``;
  * rank > 2 -> RuntimeError (batches belong to ``aten::mH``);
  * complex input -> lazy conjugate bit toggled with the backend dispatch key
    excluded, so the view is never materialized by the FlagGems ``_conj``
    kernel;
  * real input -> plain stride exchange.
"""

import warnings

import torch

from flag_gems import runtime

# Bind the C-level descriptors once: a per-call ``t.conj()`` / ``t.transpose()``
# would pay a type-dictionary lookup plus a bound-method creation.
_CONJ = torch.Tensor.conj
_TRANSPOSE = torch.Tensor.transpose
_IS_COMPLEX = torch.Tensor.is_complex

# ``aten::matrix_H`` rejects rank-1 and batched inputs; the messages mirror aten
# so that callers matching on the text keep working.
_MATRIX_H_1D_MESSAGE = (
    "tensor.H is only supported on matrices (2-D tensors). Got 1-D tensor."
)
_MATRIX_H_0D_MESSAGE = "Tensor.H is deprecated on 0-D tensors. Consider using x.conj()."

# aten warns once per process on the 0-D path; mirror that so the hot path does
# not pay the warnings machinery on every call.
_WARN_STATE = {"warned": False}

# The lazy conjugate bit must be toggled without entering the FlagGems
# ``_conj`` kernel, which would materialize a physical copy.
_BACKEND_KEYSET = torch._C.DispatchKeySet(
    torch._C._parse_dispatch_key(runtime.device.dispatch_key)
)
_EXCLUDE_BACKEND = torch._C._ExcludeDispatchKeyGuard


def _lazy_conj(self):
    """Toggle the lazy conjugate bit with the backend key excluded.

    The guard must cover only ``conj``: the following ``transpose`` (an
    ``as_strided`` relayout) needs the PrivateUse1 key, so it stays outside.
    """
    with _EXCLUDE_BACKEND(_BACKEND_KEYSET):
        return _CONJ(self)


def matrix_H(self):
    ndim = self.ndim
    if ndim == 2:
        # Hot path first: 2-D real input is a single stride exchange.
        if _IS_COMPLEX(self):
            return _TRANSPOSE(_lazy_conj(self), -2, -1)
        return _TRANSPOSE(self, -2, -1)
    if ndim == 1:
        raise RuntimeError(_MATRIX_H_1D_MESSAGE)
    if ndim == 0:
        if not _WARN_STATE["warned"]:
            _WARN_STATE["warned"] = True
            warnings.warn(_MATRIX_H_0D_MESSAGE, UserWarning, stacklevel=2)
        return _lazy_conj(self)
    raise RuntimeError(
        f"tensor.H is only supported on matrices (2-D tensors). Got {ndim}-D "
        "tensor. For batches of matrices, consider using tensor.mH"
    )
