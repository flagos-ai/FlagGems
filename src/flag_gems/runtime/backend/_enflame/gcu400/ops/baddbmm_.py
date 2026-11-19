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

import logging

from flag_gems.ops.baddbmm import baddbmm  # noqa: E402  (generic; fallback path only)

from .baddbmm import baddbmm_out  # noqa: E402

logger = logging.getLogger(__name__)


def baddbmm_(self, batch1, batch2, *, beta=1, alpha=1):
    logger.debug("GEMS_ENFLAME BADDBMM_")
    # In-place fast path: reuse the gcu400 baddbmm kernel with C == bias ==
    # self. Every program reads only its own bias tile before storing the
    # same tile, so the aliasing is safe; this removes the temporary
    # (B, M, N) allocation and the whole copy-back pass of the generic
    # out-of-place route, and inherits the native-kernel delegation line
    # plus the stride-based broadcast bias epilogue of the local baddbmm.
    # Guard notes:
    #   * aten::baddbmm_ meta already enforces self.shape == (batch, M, N)
    #     and same dtypes; broadcast/other shapes are rejected upstream.
    #   * self must be row-major contiguous: the local kernel's masked C
    #     store on a column-major out (e.g. a transposed self view) is a
    #     documented silent-wrong-value trap (dot loop + stride_cm == 1 +
    #     masked store, see knowledge compiler_traps addmm_ entry), so
    #     non-contiguous selves keep the out-of-place + copy_ route.
    batch, M, K = batch1.shape
    N = batch2.shape[2]
    if (
        self.shape == (batch, M, N)
        and self.dtype == batch1.dtype == batch2.dtype
        and self.is_contiguous()
    ):
        return baddbmm_out(self, batch1, batch2, beta=beta, alpha=alpha, out=self)
    # Defensive fallback (aten meta already rejects these cases): keep the
    # out-of-place compute + copy_ semantics of the generic implementation.
    result = baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)
    return self.copy_(result)
