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

from flag_gems.ops.copy import copy_ as _triton_copy_
from flag_gems.ops.logit_ import logit_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.logit_")


def logit_(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN LOGIT_")
    if len(args) == 0:
        raise TypeError("logit_ expected at least 1 argument (got 0)")
    x = args[0]
    eps = None
    if len(args) > 1:
        eps = args[1]
    if "eps" in kwargs:
        eps = kwargs["eps"]

    if not isinstance(x, torch.Tensor):
        raise TypeError("logit_ expects a torch.Tensor as the first argument")
    if not x.is_floating_point():
        raise TypeError("logit_ expects a floating point tensor")

    has_eps = eps is not None
    eps_value = float(eps) if has_eps else 0.0

    needs_copy_back = not x.is_contiguous()
    buf = x if not needs_copy_back else x.contiguous()

    n_elements = buf.numel()
    if n_elements == 0:
        return x

    dtype = buf.dtype
    compute_in_fp32 = dtype in (torch.float16, torch.bfloat16)
    compute_in_fp64 = dtype == torch.float64

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        logit_kernel[grid](
            buf,
            n_elements,
            eps_value,
            has_eps=has_eps,
            COMPUTE_FP32=compute_in_fp32,
            COMPUTE_FP64=compute_in_fp64,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    if needs_copy_back:
        _triton_copy_(x, buf)

    return x
