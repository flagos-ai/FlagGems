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

from .bitwise_or import bitwise_or_tensor

# Use the generic op's logger name: the functional test asserts the record under
# ``gems_log_logger(flag_gems.or_tensor)`` == "flag_gems.ops.or_tensor" and the
# vendor prefix "GEMS_KUNLUNXIN OR_TENSOR".
logger = logging.getLogger("flag_gems.ops.or_tensor")


def or_tensor(self, other):
    """``__or__.Tensor`` for Kunlunxin.

    ``__or__.Tensor`` and ``bitwise_or.Tensor`` are the same ATen operation's
    two entry points.  The generic ``or_tensor`` calls
    ``flag_gems.ops.bitwise_or.bitwise_or_tensor``, bound at import time before
    SpecOpRegistrar swaps the namespace, so it always runs the slow generic
    kernel.  Route straight to the XPU-tuned ``bitwise_or_tensor`` (which packs
    bool/int16 into int32 on large tensors).
    """
    logger.debug("GEMS_KUNLUNXIN OR_TENSOR")
    return bitwise_or_tensor(self, other)
