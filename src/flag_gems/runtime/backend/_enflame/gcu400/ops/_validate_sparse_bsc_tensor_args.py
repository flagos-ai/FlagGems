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

"""GCU400 specialization of aten::_validate_sparse_bsc_tensor_args.

Block-column-major variant of the sparse compressed validation.  Every metadata
check and the device-side index scan live in ``_validate_metadata``; the GCU400
copy of that helper (shared with the CSR/CSC validators) drives the scan with a
fixed persistent grid, so

* the ``max(cdim, 1) * batch`` grid of the portable module no longer overflows
  the GCU400 ``grid.x`` limit (65535) for large batched block tensors, and
* an invalid index is reported through a per-program verdict slot read back by
  the host instead of a device assertion, because a GCU400 SIP assertion aborts
  the whole process and hangs when ``RLIMIT_CORE`` is zero.

The logger name is pinned by
``tests/test_validate_sparse_bsc_tensor_args.py::test_..._dispatch``, which also
requires a record whose message is exactly the portable wording. The backend
convention additionally requires the ``GEMS_ENFLAME`` token on every message, so
both records are emitted (see the wrapper below).
"""

import logging

import torch

from ._validate_sparse_csc_tensor_args import _validate_metadata

logger = logging.getLogger("flag_gems.ops._validate_sparse_bsc_tensor_args")


def _validate_sparse_bsc_tensor_args(
    ccol_indices, row_indices, values, size, check_pinning=None
):
    # The Enflame backend convention requires the "GEMS_ENFLAME" token, while
    # tests/test_validate_sparse_bsc_tensor_args.py::test_..._dispatch verifies
    # the ATen dispatch by matching the portable wording with an equality check.
    # A single message cannot satisfy both, and the shared test must not be
    # edited, so emit the backend-tagged record plus the portable token.
    logger.debug("GEMS_ENFLAME_VALIDATE_SPARSE_BSC_TENSOR_ARGS")
    logger.debug("GEMS _VALIDATE_SPARSE_BSC_TENSOR_ARGS")
    return _validate_metadata(
        ccol_indices, row_indices, values, size, torch.sparse_bsc, check_pinning
    )
