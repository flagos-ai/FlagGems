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

"""In-place T-Head PPU batched GEMM with a fused bias epilogue."""

import logging

from flag_gems.ops.baddbmm_ import baddbmm_ as _generic_baddbmm_

from .baddbmm import _can_use_ppu_baddbmm, baddbmm
from .bmm import _dispatch_ppu_bmm
from .gemm_utils import _output_overlaps_inputs

logger = logging.getLogger(__name__)


def baddbmm_(self, A, B, *, beta=1.0, alpha=1.0):
    """Fuse the in-place bias read and output write when shapes match."""
    logger.debug("GEMS_THEAD BADDBMM_")
    if (
        self.ndim == 3
        and A.ndim == B.ndim == 3
        and self.shape == (A.shape[0], A.shape[1], B.shape[2])
        and self.is_contiguous()
        and not self.requires_grad
        and _can_use_ppu_baddbmm(self, A, B, self)
    ):
        if _output_overlaps_inputs(self, A, B):
            return self.copy_(baddbmm(self, A, B, beta=beta, alpha=alpha))
        # Candidate measurements may write self more than once before launch.
        bias = self if beta == 0 else self.clone()
        return _dispatch_ppu_bmm(A, B, self, bias=bias, alpha=alpha, beta=beta)
    return _generic_baddbmm_(self, A, B, beta=beta, alpha=alpha)


__all__ = ["baddbmm_"]
