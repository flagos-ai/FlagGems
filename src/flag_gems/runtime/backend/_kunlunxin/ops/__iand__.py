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

from .bitwise_and import bitwise_and_scalar_, bitwise_and_tensor_

# Use the generic op's logger name for consistency with the other dunder
# overrides (the functional test has no caplog assertion for __iand__, but the
# record still carries the vendor prefix for the dispatch liveness probe).
logger = logging.getLogger("flag_gems.ops.__iand__")


def __iand___tensor(self, other):
    """``__iand__.Tensor`` for Kunlunxin.

    ``__iand__.Tensor`` and ``bitwise_and_.Tensor`` are the same ATen operation's
    two entry points.  The generic ``__iand___tensor`` calls
    ``flag_gems.ops.bitwise_and.bitwise_and_tensor_``, which is bound at import
    time *before* SpecOpRegistrar swaps the namespace, so it always runs the slow
    generic kernel.  Route straight to the XPU-tuned in-place ``bitwise_and_``
    kernel instead.
    """
    logger.debug("GEMS_KUNLUNXIN __IAND___TENSOR")
    return bitwise_and_tensor_(self, other)


def __iand___scalar(self, other):
    """``__iand__.Scalar`` for Kunlunxin (see ``__iand___tensor``)."""
    logger.debug("GEMS_KUNLUNXIN __IAND___SCALAR")
    return bitwise_and_scalar_(self, other)
