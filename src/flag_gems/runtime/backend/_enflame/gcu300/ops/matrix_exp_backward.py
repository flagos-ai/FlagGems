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

import logging
import math

import torch

from flag_gems.ops.matrix_exp_backward import _launch_axpy, _launch_scale

logger = logging.getLogger(__name__)


def _matrix_exp(X, num_terms=None):
    """Scaling-and-squaring Taylor series, as in the generic module.

    The only difference is that the matrix products go through the registered
    `bmm` instead of importing the generic kernel: this backend's tuned config
    is written for its own `bmm` signature, so the generic one cannot consume
    it and fails in the jit binder.
    """
    from flag_gems import bmm

    batch, m, _ = X.shape
    if num_terms is None:
        num_terms = 13

    norm = X.abs().sum(dim=-1).amax(dim=-1).amax()
    norm_val = float(norm.item())
    if norm_val > 0.0 and math.isfinite(norm_val):
        s = max(0, int(math.ceil(math.log2(norm_val))))
    else:
        s = 0
    scale = float(2.0**s)

    Xs = _launch_scale(X, 1.0 / scale) if s > 0 else X.clone()

    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    result = eye.unsqueeze(0).expand(batch, m, m).contiguous()
    term = result.clone()

    for k in range(1, num_terms + 1):
        term = bmm(term, Xs)
        if k > 1:
            term = _launch_scale(term, 1.0 / k)
        _launch_axpy(result, term, 1.0)

    for _ in range(s):
        result = bmm(result, result)

    return result


def matrix_exp_backward(self, grad):
    logger.debug("GEMS_ENFLAME MATRIX_EXP_BACKWARD")
    assert self.shape == grad.shape, "self and grad must have the same shape"
    assert (
        self.ndim >= 2 and self.shape[-1] == self.shape[-2]
    ), "matrix_exp_backward expects square matrices"

    orig_shape = self.shape
    orig_dtype = self.dtype
    n = orig_shape[-1]
    n2 = 2 * n

    # Compute in float32 for half-precision inputs (float32 kept as-is).
    compute_dtype = (
        torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    )

    self_c = self.reshape(-1, n, n).to(compute_dtype)
    grad_c = grad.reshape(-1, n, n).to(compute_dtype).contiguous()
    batch = self_c.shape[0]

    at = self_c.transpose(-1, -2).contiguous()

    # Assemble the block matrix M = [[A^T, G], [0, A^T]] with slice copies.
    # The generic version does this in one kernel that derives the row/column
    # of every lane from `offs % n2` and `offs // n2`; on this backend that
    # index reconstruction returns the wrong row for the lower half of the
    # matrix, so M came out with A^T only in the top-left block and the result
    # was wrong for every n > 2. Slice assignment goes through the aten copy
    # kernel instead, which is exact.
    block = torch.zeros((batch, n2, n2), dtype=compute_dtype, device=self.device)
    block[:, :n, :n] = at
    block[:, :n, n:] = grad_c
    block[:, n:, n:] = at

    expm = _matrix_exp(block)

    # The gradient is the top-right n x n block of exp(M).
    out = expm[:, :n, n:].contiguous()

    return out.reshape(orig_shape).to(orig_dtype)
