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

from .bitwise_or import bitwise_or_scalar

logger = logging.getLogger(__name__)


def or_scalar(self, other):
    """``__or__.Scalar`` for Kunlunxin.

    ``__or__.Scalar`` and ``bitwise_or.Scalar`` are the same ATen operation's
    two entry points; the generic ``or_scalar`` is bound to the slow generic
    bitwise_or kernel at import time.  Route straight to the XPU-tuned
    ``bitwise_or_scalar`` (which packs bool/int16 into int32 on large tensors),
    mirroring ``or_tensor``.
    """
    logger.debug("GEMS_KUNLUNXIN OR_SCALAR")
    return bitwise_or_scalar(self, other)
