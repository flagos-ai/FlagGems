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

"""Kunlunxin backend override for ``aten::argwhere``.

``torch.argwhere(x)`` is, by definition, ``torch.nonzero(x, as_tuple=False)``
(verified bit-for-bit against the CPU ATen reference for every dtype / shape /
0-dim / empty / non-contiguous / NaN / -0.0 case in ``tests/test_argwhere.py``).
The generic ``flag_gems.ops.argwhere`` re-implements the whole compaction with a
generic three-kernel design whose scatter step lowers to a *data-dependent*
per-lane global store, which on XPU3 is both very slow (~3.1 ms/call for
1024x1024, 96.6% of the op) and, in this environment, numerically wrong
(over-counts the tail block and writes a shifted / out-of-bounds result).

The Kunlunxin ``nonzero`` backend already ships a correct, tuned compaction
(``tle.raw`` LM-compaction payload with a generic Triton fallback; see
``_kunlunxin/ops/nonzero.py``). Reusing it keeps the whole computation in a real
XPU kernel -- no ATen / CPU / composite fallback, no redispatcher -- and makes
``argwhere`` share ``nonzero``'s correctness and performance.
"""

import logging

from .nonzero import nonzero

logger = logging.getLogger(__name__)


def argwhere(input):
    """Return the ``(num_nonzero, ndim)`` int64 indices of non-zero elements.

    ``argwhere`` is exactly ``nonzero(as_tuple=False)``; delegating to the
    vendored XPU ``nonzero`` kernel gives row-major ordering, int64 coordinates
    and the exact output shape (including the ``(0, ndim)`` empty case).
    """
    logger.debug("GEMS_KUNLUNXIN ARGWHERE")
    return nonzero(input, as_tuple=False)
