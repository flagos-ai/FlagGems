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


def sym_numel(A: torch.Tensor) -> int:
    """Implements ``aten::sym_numel(Tensor self) -> SymInt``.

    The element count is a pure shape-metadata query: the product of the
    dimensions is available host-side and needs no device kernel (the generic
    implementation reduces the shape with a Triton kernel and returns the same
    value as a plain int).
    """
    logger.debug("GEMS_KUNLUNXIN SYM_NUMEL")
    return A.numel()
