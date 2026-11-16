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

"""Triton implementation of ``aten::_sparse_broadcast_to_copy`` for sparse COO.

The copy variant shares the broadcasting semantics of
``aten::_sparse_broadcast_to`` — replicating stored entries over the newly
prepended and expanded sparse dimensions while copying payload values bitwise
through the target dense strides — but always materialises a fresh tensor and
additionally provides the ``out=`` form.

The KernelGen-qualified ``sparse_broadcast_to`` kernel already builds fresh
indices and values (it never aliases the input storage), so this module
reuses it directly and only adds the copy-specific ``out=`` handling.
"""

import logging

from flag_gems.ops.sparse_broadcast_to import sparse_broadcast_to

logger = logging.getLogger(__name__)


def sparse_broadcast_to_copy(self, size, out=None):
    logger.debug("GEMS SPARSE BROADCAST TO COPY")
    result = sparse_broadcast_to(self, size)
    if out is not None:
        out.copy_(result)
        return out
    return result
