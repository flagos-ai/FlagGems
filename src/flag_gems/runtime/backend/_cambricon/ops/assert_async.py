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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _assert_async_kernel(x_ptr, scratch_ptr, MSG: tl.constexpr):
    val = tl.load(x_ptr)
    cond = val != 0
    tl.device_assert(cond, MSG)
    tl.store(scratch_ptr, cond)


def _assert_async(tensor: torch.Tensor, msg: str = "Assertion failed"):
    logger.debug("GEMS_CAMBRICON ASSERT_ASYNC")
    if tensor.numel() != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with shape {list(tensor.shape)} is ambiguous"
        )
    scratch = torch.empty((), dtype=torch.bool, device=tensor.device)
    with torch_device_fn.device(tensor.device):
        _assert_async_kernel[(1,)](tensor, scratch, MSG=msg)
