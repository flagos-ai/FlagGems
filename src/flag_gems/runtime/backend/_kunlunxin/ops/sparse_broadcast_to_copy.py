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

"""Kunlunxin override of ``aten::_sparse_broadcast_to_copy``.

The copy variant shares the broadcasting semantics of
``aten::_sparse_broadcast_to`` (materialised by the backend override) and adds
the ``out=`` form.  The generic module imports the generic
``sparse_broadcast_to`` implementation at import time, so it would keep using
the kernel that fails MLIR legalization on this backend; this override routes
through the backend implementation instead.
"""

import logging

from flag_gems.runtime.backend._kunlunxin.ops.sparse_broadcast_to import (
    sparse_broadcast_to,
)

logger = logging.getLogger(__name__)


def sparse_broadcast_to_copy(self, size, out=None):
    logger.debug("GEMS SPARSE BROADCAST TO COPY (kunlunxin)")
    result = sparse_broadcast_to(self, size)
    if out is not None:
        out.copy_(result)
        return out
    return result
