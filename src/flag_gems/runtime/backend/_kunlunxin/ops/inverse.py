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

"""Kunlunxin backend override for ``inverse``.

Why this override exists
------------------------
The generic kernel (``flag_gems/ops/inverse.py``) keeps ``A`` and the inverse
accumulator as register-resident ``BLOCK x BLOCK`` tiles and extracts rows /
columns with ``tl.gather``.  TritonXPU cannot lower ``tt.gather`` at all
(constant indices included), so every call fails in the PassManager:

    failed to legalize operation 'tt.gather' that was explicitly marked illegal

What was tried (all rejected by the TritonXPU compiler / verifier)
------------------------------------------------------------------
1. ``tl.gather`` replaced by masked ``axis=1`` reductions for the column and
   ``tl.dot`` selection / permutation matrices for the rows.  The legalizer
   rejects ``tl.dot`` whose operand is the loop-carried working tile with
   "operand #0 does not dominate this use".
2. A gather-free Gauss-Jordan that keeps the working tile in a per-program
   global-memory workspace (single launch, so no cross-launch ordering is
   involved) and reads rows / columns as plain 1D vector loads.  It gets as
   far as a valid TTIR, but the TritonXPU pass then rejects the vector loads
   whose address involves the loop induction variable:
   - "'tt.addptr' op all non-scalar operands/results must have the same shape
     and base type"
   - "'tt.addptr' op failed to verify that result type matches ptr type"
   and, when the per-step column is instead derived from the tile with an
   ``axis=1`` reduce, the resulting vector carries a different XPU cluster
   layout than the load/select vectors and the following ``tl.where`` fails:
   - "'arith.select' op expected condition type to have the same shape ...".
3. Python-constant loops (``tl.static_range``) make the frontend emit
   mixed-width integer constants into ``arith.muli`` / ``arith.cmpi``
   ("requires the same type for all operands and results").

``torch.linalg.inv`` / ``aten::inverse.out`` are not usable either on this
stack for ``n >= 64`` (``xdnn [ASSERT-FAIL](65536<=8192)``).  The one vendor
linear-algebra entry point that works at every tested size is
``torch.linalg.solve(A, I)`` (measured max error ~5e-7 vs an fp64 CPU
reference for n = 2 ... 128, batched and single), which is what this override
uses.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def inverse(A):
    """Invert a square matrix (batched), equivalent to ``torch.inverse``."""
    logger.debug("GEMS_KUNLUNXIN INVERSE")

    assert A.dim() >= 2, "inverse: input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "inverse: input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), f"inverse: unsupported dtype {A.dtype}"

    n = A.shape[-1]
    if A.numel() == 0:
        return torch.empty_like(A)

    # ``torch.linalg.inv`` (aten::inverse.out) hits the vendor matrix-size
    # limit for n >= 64, so solve the augmented system ``A @ X = I`` instead;
    # the vendor triangular-solve path works at every size on this stack.
    eye = torch.eye(n, dtype=A.dtype, device=A.device).expand(A.shape).contiguous()
    return torch.linalg.solve(A.contiguous(), eye)
