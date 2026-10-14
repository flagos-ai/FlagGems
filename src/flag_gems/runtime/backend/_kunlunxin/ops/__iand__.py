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

logger = logging.getLogger(__name__)


def __iand___tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN __IAND___TENSOR")
    return bitwise_and_tensor_(self, other)


def __iand___scalar(self, other):
    logger.debug("GEMS_KUNLUNXIN __IAND___SCALAR")
    return bitwise_and_scalar_(self, other)
