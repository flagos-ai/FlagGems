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

"""Kunlunxin override for ``aten::native_norm``.

The generic module binds ``flag_gems.ops.vector_norm.vector_norm`` at import
time, so its fallback always runs the generic two-stage reduction and never
reaches the Kunlunxin ``vector_norm``.  That generic path costs two launches
plus a workspace allocation for every input above 16384 elements, which is 4 of
the 5 benchmark shapes, leaving large shapes at 0.04-0.14x.

This override keeps the same semantics but resolves the reduction through the
backend's own ``vector_norm``, which has a single-launch flat L2 fast path and
a clustered stage-1 for large inputs.
"""

import logging

import torch

from flag_gems.ops.native_norm import native_norm as _generic_native_norm

from .vector_norm import _l2_flat, vector_norm

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# The backend's single-program TLE flat path builds a [1, YBLOCK] descriptor
# with YBLOCK up to 32768 lanes.  At 32768 lanes the descriptor buffer reaches
# 128 KB for float32 (64 KB for the 16-bit dtypes) and the launch faults with an
# illegal memory access, so the affected window stays on the tuned non-TLE
# ``_l2_flat`` path.  Above the single-program limit (131072 elements)
# ``vector_norm`` switches to a row-reduce plan whose tiles are always small.
_TLE_SINGLE_SAFE_MAX = {
    torch.float32: 1 << 14,   # YBLOCK <= 16384 (64 KB)
    torch.float16: 1 << 16,   # YBLOCK = 32768 (64 KB)
    torch.bfloat16: 1 << 16,
}
_TLE_SINGLE_LIMIT = 1 << 17


def native_norm(self, p=2):
    """p-norm over all elements (full reduction to a scalar)."""
    logger.debug("GEMS_KUNLUNXIN NATIVE_NORM")
    if float(p) == 2.0 and self.dtype in _SUPPORTED_DTYPES:
        n = self.numel()
        safe_max = _TLE_SINGLE_SAFE_MAX.get(self.dtype, 0)
        if safe_max < n <= _TLE_SINGLE_LIMIT:
            x = self.contiguous()
            out = torch.empty_strided((), (), dtype=self.dtype, device=x.device)
            _l2_flat(x, out)
            return out
        # The backend vector_norm owns the tuned flat L2 fast paths (single
        # TLE program for small inputs, row-reduce + scalarise above the TLE
        # single-program limit).
        return vector_norm(self, ord=p, dim=None, keepdim=False, dtype=self.dtype)
    # Other orders (and float64) keep the generic behaviour, including its
    # NotImplementedError for unsupported dtypes.
    return _generic_native_norm(self, p)


__all__ = ["native_norm"]
