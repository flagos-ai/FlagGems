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
#
# Kunlunxin XPU specialization of ``special_expm1``.
#
# ``torch.special.expm1`` / ``aten::special_expm1`` is the same computation as
# ``aten::expm1``.  This forwards to the backend's existing tuned ``expm1``
# implementation so both entry points share the fast kernel.
import logging

from .expm1 import expm1 as _kunlunxin_expm1

logger = logging.getLogger(__name__)


def special_expm1(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_EXPM1")
    return _kunlunxin_expm1(A)
