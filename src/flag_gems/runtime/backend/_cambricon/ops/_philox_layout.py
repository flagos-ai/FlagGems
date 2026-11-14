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

import triton
import triton.language as tl


@triton.jit
def philox_lane(r, i: tl.constexpr):
    """Return the i-th of the 4 philox outputs as a 1-D [BLOCK] tensor.

    ``libdevice.philox(BLOCK, ...)`` has been observed with two layouts:

    * ``[BLOCK, 4]`` (element-major), e.g. triton-mlu 3.0 / 3.4 (see #6955);
    * ``[4, BLOCK]`` (lane-major), the layout assumed by #6259.

    The layout is chosen at compile time from ``r.shape``: if dim 0 is 4 and
    dim 1 is not, the result is lane-major and we take ``r[i, :]``; otherwise
    we take ``r[:, i]``.  When BLOCK == 4 the shape is ``[4, 4]`` and the two
    layouts cannot be told apart; we fall back to element-major, the layout
    produced by the triton-mlu releases we have verified.  Both choices yield
    4 uniformly distributed, independent streams, so only the exact sample
    order (not correctness of the distribution) depends on this choice.
    """
    if r.shape[0] == 4:
        if r.shape[1] != 4:
            return r[i, :]
        else:
            return r[:, i]
    else:
        return r[:, i]
