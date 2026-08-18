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
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def copy_kernel(src_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    src = tl.load(src_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, src, mask=mask)


def _list_to_tensor(self):
    """Triton implementation of aten::_list_to_tensor.

    Note: aten::_list_to_tensor is a JIT-only prim operator with no c10
    dispatcher kernel. It is listed in _FULL_CONFIG for consistency with the
    standard FlagGems operator registration flow, but the dispatcher cannot
    route calls to that entry. The implementation is also exposed directly as
    ``flag_gems._list_to_tensor`` so it can be exercised explicitly.
    """
    logger.debug("GEMS _LIST_TO_TENSOR")
    n = len(self)
    device = runtime.device.name
    out = torch.empty((n,), dtype=torch.int32, device=device)
    if n == 0:
        return out

    src = torch.tensor(self, dtype=torch.int32, device=device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    with torch_device_fn.device(out.device):
        copy_kernel[grid](src, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
