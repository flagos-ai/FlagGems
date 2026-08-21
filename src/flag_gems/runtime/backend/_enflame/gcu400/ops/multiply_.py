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

from .mul import mul_

logger = logging.getLogger(__name__)


def multiply_(A, B):
    """In-place multiply (multiply_), an alias for the enflame gcu400 mul_.

    The generic ``flag_gems.ops.multiply_`` binds the *generic* ``mul_`` at
    import time, so it never reaches this backend's optimized ``mul_``. On GCU
    the generic path then tries ``aten::mul.out.redispatch`` with a
    CompositeExplicitAutograd keyset, which TopsRider torch cannot resolve
    (NotImplementedError: no fallback registered for aten::mul.out).
    Overriding ``multiply_`` here routes it to the gcu400 ``mul_`` instead.
    """
    logger.debug("GEMS_ENFLAME MULTIPLY_")
    if not isinstance(A, torch.Tensor):
        raise ValueError("Unreachable.")
    return mul_(A, B)
