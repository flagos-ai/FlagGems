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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kunlunxin backend override for ``linalg_inv``.

``aten::linalg_inv`` is the same mathematical operation as ``aten::inverse``
(batched ``A^{-1}``) reached through a different ATen entry point.  The generic
``flag_gems/ops/linalg_inv.py`` kernel lowers the whole elimination into
register tiles and reduces over arithmetic 2-D tiles (``tl.sum(..., axis=1)``)
plus a tuple ``tl.reduce``; TritonXPU cannot legalize that body at all:

    TritonXPULegalize ... Failures have been detected while processing an MLIR
    pass pipeline

The backend already ships a gather-free, pure-Triton Gauss-Jordan
(``_kunlunxin/ops/inverse.py``, see ``harness/solution/inverse/README.md``).
Rather than duplicate the kernel, this module keeps only the ``linalg_inv``
specific argument contract (single ``A`` argument, ATen's exact error messages
for non-square / <2-D / half precision) and delegates the numerics to that
implementation.  No CPU/ATen/native fallback is involved: ``inverse`` is the
backend-local Triton path.
"""

import torch

from .inverse import inverse as _inverse


def linalg_inv(A):
    """Invert batches of square matrices, matching ``aten::linalg_inv``."""
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.inv: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "linalg.inv: A must be batches of square matrices, but they are "
            "%d by %d matrices" % (A.shape[-2], A.shape[-1])
        )
    if A.dtype in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "linalg.inv: Low precision dtypes not supported. Got %s"
            % ("Half" if A.dtype == torch.float16 else "BFloat16")
        )
    return _inverse(A)


__all__ = ["linalg_inv"]
