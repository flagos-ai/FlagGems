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

from flag_gems.ops.ne import ne_func, ne_func_scalar

logger = logging.getLogger(__name__)


def ne_(A, B):
    logger.debug("GEMS NE_")
    ne_func(A, B, out0=A)
    return A


def ne_scalar_(A, B):
    logger.debug("GEMS NE_ SCALAR")
    ne_func_scalar(A, B, out0=A)
    return A
