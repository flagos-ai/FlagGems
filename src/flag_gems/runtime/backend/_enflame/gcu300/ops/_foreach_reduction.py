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

"""GCU300 fallback for ``_foreach_norm``.

The shared reduction kernel in :mod:`flag_gems.ops._foreach_reduction` packs
int64 device pointers into a metadata buffer and aborts on GCU300.  This
fallback applies the existing single-tensor ``norm`` to one tensor at a time.

``dtype`` is applied to the *input* before the norm, matching the generic
``_pow_sum_list`` (``work = t if dtype is None else t.to(dtype)``); casting the
result instead loses the extra precision the caller asked for.
"""

import logging

import flag_gems
from flag_gems.utils.foreach import check_tensor_list

logger = logging.getLogger(__name__)


def _foreach_norm(self, ord=2, dtype=None):
    logger.debug("GEMS _FOREACH_NORM")
    tensors = check_tensor_list(self)
    src = tensors if dtype is None else [t.to(dtype) for t in tensors]
    return [flag_gems.norm(t, p=ord) for t in src]


# ``tests/test_foreach_ops.py`` maps the ATen key ``_foreach_norm.Scalar`` to
# the attribute ``_foreach_norm_Scalar`` when it overlays vendor wrappers.  The
# registered function must keep the ATen function name ``_foreach_norm``; the
# alias only makes the test exercise this implementation instead of the
# generic executor.
_foreach_norm_Scalar = _foreach_norm


__all__ = ["_foreach_norm", "_foreach_norm_Scalar"]
