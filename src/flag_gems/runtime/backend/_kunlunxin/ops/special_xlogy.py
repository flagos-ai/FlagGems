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

"""Kunlunxin override for ``aten::special_xlogy``.

``special_xlogy`` and ``xlogy`` share one definition (NaN when ``y`` is NaN,
0 when ``x == 0``, otherwise ``x * log(y)``), and this backend already carries a
tuned ``xlogy`` implementation: a ``pointwise_dynamic`` fast path with a fixed
codegen config for large inputs plus wide unmasked raw kernels for the rest.

The generic ``special_xlogy`` instead goes through the stock
``pointwise_dynamic`` wrapper, whose tile is derived from ``numel`` and whose
body uses a nested ``tl.where`` with a ``float("nan")`` literal -- both of which
collapse the vectorised memory path on XPU and leave large shapes at ~0.004x.

Reusing the backend ``xlogy`` keeps the semantics (its predicate order matches)
and inherits all of its tuning.
"""

import logging

from .xlogy import xlogy as _xlogy

logger = logging.getLogger(__name__)


def special_xlogy(a, b):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_XLOGY")
    return _xlogy(a, b)


__all__ = ["special_xlogy"]
