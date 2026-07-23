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

from flag_gems.runtime import device

from .eye_m import _fill_diagonal

logger = logging.getLogger(__name__)
device_ = device


def eye(size, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye(n): bulk zero-fill + diagonal write.
    """
    logger.debug("GEMS_KUNLUNXIN EYE")

    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye.")

    out = torch.zeros(
        (size, size), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    return _fill_diagonal(out, size, size)
