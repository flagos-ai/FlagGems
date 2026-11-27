# Copyright 2026, The FlagOS Contributors.
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
"""Triton implementation of ``aten::__irshift__`` (in-place right shift).

``__irshift__.Tensor`` / ``__irshift__.Scalar`` and ``bitwise_right_shift_`` are two ATen entries
with the same in-place right-shift semantics (confirmed by schema and numerical A/B). This
reuses the layered enflame ``bitwise_right_shift_`` implementation: contiguous storage takes a
single 1-D flat launch (streaming tile for large work sets, parallel ladder for small ones),
while non-contiguous / broadcast / mixed-width falls back to the stride-aware pointwise path.

Previously this file carried its own default-config ``pointwise_dynamic`` kernel, paying the
rank-specialized block-pointer launch plus full promotion/stride cost on every call, and even
large tensors (>=16 MB, bandwidth bound) only reached pointwise tile efficiency; the flat path
now covers both tensor and scalar right shifts with one launch, nothing left to duplicate here.
"""

import logging

import torch

from .bitwise_right_shift import bitwise_right_shift_

logger = logging.getLogger(__name__)


def __irshift__(self: torch.Tensor, other) -> torch.Tensor:
    """In-place right shift ``self >>= other``, returning the modified ``self``."""
    logger.debug("GEMS_ENFLAME IRSHIFT")

    # In-place semantics: out is fixed to self, and ATen's __irshift__ accepts only this form.
    bitwise_right_shift_(self, other)
    return self
