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

from .full import check_dtype, full_func, full_func_scalar

logger = logging.getLogger(__name__)


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logger.debug("GEMS_ENFLAME FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    size = x.size()
    out = torch.empty(size, device=device, dtype=dtype)
    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value)
    else:
        return full_func_scalar(out, fill_value)
