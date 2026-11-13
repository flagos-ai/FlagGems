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

import torch

logger = logging.getLogger(__name__)


def _reshape_alias(self: torch.Tensor, size, stride):
    """Create a view of ``self`` with the given ``size`` and ``stride``.

    This is an internal reshape helper that skips the validity checks that a
    normal reshape performs. It always returns a view that shares storage with
    ``self`` (reusing ``self``'s storage offset), mirroring
    ``aten::_reshape_alias``.
    """
    logger.debug("GEMS _RESHAPE_ALIAS")
    return self.as_strided(size, stride, self.storage_offset())
