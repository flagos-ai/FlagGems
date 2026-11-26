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

from flag_gems.ops.copy import copy_ as _generic_copy_

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    # copy_ is a fallthrough for PyTorch's Conjugate/Negative dispatch keys, so
    # the backend implementation owns resolving lazy conjugate/negative views.
    # PyTorch's MathOpFallback materialises every math-bit input with
    # at::clone, which routes back through copy_. A pointer-based kernel that
    # reads raw storage therefore leaks un-resolved physical values into every
    # operator that takes the fallback. Defer those cases to PyTorch, which
    # resolves the bit correctly.
    if isinstance(src, torch.Tensor) and (
        dst.is_neg() or dst.is_conj() or src.is_neg() or src.is_conj()
    ):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )
    return _generic_copy_(dst, src, non_blocking)
