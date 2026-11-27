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

"""GCU400-specific _foreach_div_.ScalarList (in-place).

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.
"""

import logging

from ._foreach_binary import _apply_list, _apply_scalar, _apply_tensor

logger = logging.getLogger(__name__)


def _foreach_div_List(self, other):
    return _apply_list("div", self, other, False)


def _foreach_div_Scalar(self, scalar):
    return _apply_scalar("div", self, scalar, False)


def _foreach_div_ScalarList(self, scalars):
    return _apply_scalar("div", self, scalars, False)


def _foreach_div_Tensor(self, other):
    return _apply_tensor("div", self, other, False)


def _foreach_div__List(self, other):
    return _apply_list("div", self, other, True)


def _foreach_div__Scalar(self, scalar):
    return _apply_scalar("div", self, scalar, True)


def _foreach_div__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_DIV__SCALARLIST")
    return _apply_scalar("div", self, scalars, True)


def _foreach_div__Tensor(self, other):
    return _apply_tensor("div", self, other, True)
