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

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


def bitwise_not(A):
    logger.debug("GEMS_KUNLUNXIN BITWISE_NOT")
    # The Triton integer paths are not reliable in the current XPU toolchain:
    # ``~x`` emits an instruction rejected by elfconv for large tiles, while the
    # arithmetic identity ``-x - 1`` can hang even on a 266-element int32 tile.
    # The vendor kernel covers integer and bool dtypes and is also faster.
    return torch.ops.aten.bitwise_not.default.redispatch(_FALLBACK_KEYSET, A)


def bitwise_not_(A):
    logger.debug("GEMS_KUNLUNXIN BITWISE_NOT_")
    return torch.ops.aten.bitwise_not_.default.redispatch(_FALLBACK_KEYSET, A)
