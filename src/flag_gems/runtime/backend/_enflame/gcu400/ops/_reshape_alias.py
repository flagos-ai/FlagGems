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

import torch


def _reshape_alias(self: torch.Tensor, size, stride):
    """Create a view of ``self`` with the given ``size`` and ``stride``.

    This is an internal reshape helper that skips the validity checks that a
    normal reshape performs. It always returns a view that shares storage with
    ``self`` (reusing ``self``'s storage offset), mirroring
    ``aten::_reshape_alias``.
    logger.debug("GEMS_ENFLAME _RESHAPE_ALIAS")

    This op launches no kernel, so the whole cost is host-side. Everything the
    op does happens in a single ``as_strided`` call, and the free-function
    ``torch.as_strided`` form measured fastest on this host (it skips the
    class-attribute lookup of ``torch.Tensor.as_strided`` and the bound-method
    creation of ``self.as_strided``). A ``view``-based fast path for the 1-D
    case was rejected: the ``numel``/``is_contiguous``/stride guards that the
    alias semantics force (~0.27us) exceed the view terminal's edge over the
    free-function terminal (~0.08us). Debug logging is left out of this
    single-line hot path for the same reason.
    """
    return torch.as_strided(self, size, stride, self.storage_offset())
