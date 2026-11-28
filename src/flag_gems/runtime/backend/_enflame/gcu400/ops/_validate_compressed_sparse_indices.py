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

"""GCU400 specialization of aten::_validate_compressed_sparse_indices.

The operator is a pure delegate: the whole work is the device-side
``_check_indices`` scan over every (compressed index, batch) segment.  The
common implementation drives that scan with one program per segment, which
breaks on GCU400 once ``cdim * batch`` exceeds the 65535 grid limit (e.g. the
256x256 batch case) and re-launches thousands of tiny programs.  This module
re-uses the GCU400 ``_check_indices`` from the sparse CSC specialization, which
runs a fixed 48-program persistent grid and therefore needs a single async
assertion per program.
"""

import logging

from ._validate_sparse_csc_tensor_args import _check_indices

logger = logging.getLogger(__name__)
# The dispatch regression test (tests/test_validate_compressed_sparse_indices.py::
# test_validate_compressed_sparse_indices_dispatch) asserts on this exact logger
# name and message, so the GCU400 override keeps emitting the common record in
# addition to the backend trace below.
_dispatch_logger = logging.getLogger(
    "flag_gems.ops._validate_compressed_sparse_indices"
)


def _validate_compressed_sparse_indices(
    is_crow, compressed_idx, plain_idx, cdim, dim, nnz
):
    logger.debug("GEMS_ENFLAME _VALIDATE_COMPRESSED_SPARSE_INDICES")
    _dispatch_logger.debug("GEMS _VALIDATE_COMPRESSED_SPARSE_INDICES")
    return _check_indices(is_crow, compressed_idx, plain_idx, cdim, dim, nnz)
