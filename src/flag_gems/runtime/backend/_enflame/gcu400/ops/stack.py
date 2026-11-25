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

"""GCU400-specific ``aten::stack`` implementation.

Native ATen's ``stack`` is a thin wrapper over ``_stack`` (composite decomposition); this module keeps
the same decomposition: directly compose and reuse GCU400's optimized ``_stack`` (multiplicative-addressing three-path
copy,
legacy int64 division interleaving kernel only as a restricted-dtype fallback). Previously ``stack`` carried
a legacy kernel byte-for-byte identical to old ``_stack``, which degraded to ~4ms at 16.7M elements
on the new toolchain (vs aten ~0.07ms); reuse gives the same performance as ``_stack``; it also inherits
``empty_strided`` allocation (saving the extra zero-init launch of GEMS's ``empty`` override),
complex handling, zero-volume early return and unconditional dim range validation.
"""

import logging
from typing import List, Tuple, Union

import torch

from ._stack import _stack

logger = logging.getLogger(__name__)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME STACK")
    return _stack(tensors, dim)
