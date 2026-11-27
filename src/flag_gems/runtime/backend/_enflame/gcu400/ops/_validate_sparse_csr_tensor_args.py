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

"""GCU400 specialization of aten::_validate_sparse_csr_tensor_args.

Row-major variant of the sparse compressed validation.  All metadata checks and
the device-side index scan live in ``_validate_metadata``; the GCU400 copy of
that helper drives the scan with a fixed 48-program persistent grid, so a large
``nrows * batch`` product no longer hits the GCU400 grid limit and the async
device assertion is issued once per program instead of once per row.
"""

import logging

import torch

from ._validate_sparse_csc_tensor_args import _validate_metadata

logger = logging.getLogger(__name__)


def _validate_sparse_csr_tensor_args(
    crow_indices, col_indices, values, size, check_pinning=None
):
    logger.debug("GEMS_ENFLAME _VALIDATE_SPARSE_CSR_TENSOR_ARGS")
    return _validate_metadata(
        crow_indices, col_indices, values, size, torch.sparse_csr, check_pinning
    )
