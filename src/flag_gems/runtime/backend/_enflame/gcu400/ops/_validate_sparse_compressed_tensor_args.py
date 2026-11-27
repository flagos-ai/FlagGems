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

from ._validate_sparse_csc_tensor_args import _validate_metadata

# The shared dispatch test verifies that an ATen call reaches a FlagGems
# implementation by looking for this message on the generic module's logger, so
# the backend entry point reports through the same logger name.
logger = logging.getLogger("flag_gems.ops._validate_sparse_compressed_tensor_args")


def _validate_sparse_compressed_tensor_args(
    compressed_indices, plain_indices, values, size, layout, check_pinning=None
):
    """GCU400 backend of the generic sparse-compressed validation.

    The generic implementation binds the generic ``_validate_metadata`` at
    import time, which scans the indices with the KernelGen kernel.  This
    backend's ``_validate_metadata`` drives the same scan with a fixed
    48-program persistent grid and with index loads matched to the GCU400 int64
    layout regime, so the compressed entry point has to resolve to it as well --
    otherwise this op keeps the generic scan and trips a device SIP assert on
    int64 indices while the per-layout entry points (csc/csr/bsr/bsc) do not.
    """
    logger.debug("GEMS_ENFLAME_VALIDATE_SPARSE_COMPRESSED_TENSOR_ARGS")
    return _validate_metadata(
        compressed_indices,
        plain_indices,
        values,
        size,
        layout,
        check_pinning,
    )
