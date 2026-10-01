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

from flag_gems.ops.cauchy import UNROLL, cauchy_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger("flag_gems.ops.cauchy")


def _cauchy_grid(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"] * UNROLL),)


def cauchy_(self, median=0, sigma=1, *, generator=None):
    """
    In-place Cauchy distribution sampler.
    """
    logger.debug("GEMS_KUNLUNXIN CAUCHY_")
    shape = self.shape
    device = self.device
    N = volume(shape)
    if N == 0:
        return self
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        cauchy_kernel[_cauchy_grid](self, N, median, sigma, philox_seed, philox_offset)
    return self


def cauchy(self, median=0, sigma=1, *, generator=None):
    """
    Out-of-place Cauchy distribution sampler.
    """
    logger.debug("GEMS_KUNLUNXIN CAUCHY")
    out = torch.empty_like(self)
    cauchy_(out, median, sigma, generator=generator)
    return out
