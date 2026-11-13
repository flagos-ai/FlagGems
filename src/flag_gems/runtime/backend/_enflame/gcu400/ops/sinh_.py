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

from .sinh import _launch_sinh_flat, sinh, sinh_kernel

logger = logging.getLogger(__name__)

# In-place sinh on GCU400: the flat grid-stride kernels (dtype-split math
# bodies) live in sinh.py; this module keeps only the aten-facing argument
# handling.  Non-float16/bfloat16/float32 dtypes fall back to the codegen
# pointwise kernel, non-contiguous tensors to a copy_ round-trip, instead of
# raising like the previous fixed-config (BLOCK=8192, nw4, 2-exp body)
# implementation, which also left the large 512M bucket 2.7x above the
# tuned flat config.


def sinh_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME SINH_")
    x = args[0] if args else kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise TypeError("sinh_ expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("sinh_ expected a torch.Tensor")

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        sinh_kernel(x, out0=x)
        return x
    if not x.is_contiguous():
        x.copy_(sinh(x))
        return x
    N_total = x.numel()
    if N_total == 0:
        return x

    _launch_sinh_flat(x, x, N_total, x.element_size(), x.dtype)
    return x
