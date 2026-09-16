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

# Kunlunxin(XPU) override of flag_gems.ops.functional_assert_async.
#
# Same root cause and fix as _kunlunxin/ops/assert_async.py: the Triton
# XPU backend lowers `tl.device_assert` to a no-op, so a false assertion
# never raises. The condition is written to a device scratch buffer and
# checked on the host after an explicit sync so that
# `_functional_assert_async.msg` keeps "raise when the value is falsy".

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _functional_assert_async_kernel(x_ptr, scratch_ptr, MSG: tl.constexpr):
    val = tl.load(x_ptr)
    cond = val != 0
    # Works on backends with a real device assert; no-op on XPU.
    tl.device_assert(cond, MSG)
    tl.store(scratch_ptr, cond)


def _functional_assert_async(
    tensor: torch.Tensor, assert_msg: str, dep_token: torch.Tensor
) -> torch.Tensor:
    """
    Functional version of _assert_async that takes a dependency token and returns a new token.

    This is part of PyTorch's functionalization framework for tracking operation dependencies.

    Args:
        tensor: Single-element tensor to check (non-zero = pass, zero = fail)
        assert_msg: Error message if assertion fails
        dep_token: Input dependency token (typically an empty tensor)

    Returns:
        A new empty tensor serving as the output dependency token
    """
    logger.debug("GEMS_KUNLUNXIN FUNCTIONAL_ASSERT_ASYNC")

    if tensor.numel() != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with shape {list(tensor.shape)} is ambiguous"
        )

    scratch = torch.empty((), dtype=torch.bool, device=tensor.device)
    with torch_device_fn.device(tensor.device):
        _functional_assert_async_kernel[(1,)](tensor, scratch, MSG=assert_msg)
    torch_device_fn.synchronize()
    if not scratch.item():
        raise RuntimeError(assert_msg)

    # Return a new dependency token (empty tensor with same dtype/device as input token)
    return torch.empty(0, dtype=dep_token.dtype, device=dep_token.device)
