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
import os

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import broadcast_shapes, volume

from ..utils.pointwise_dynamic import pointwise_dynamic
from .randn import (
    _native_cpu_randn,
    _philox_to_dsa_seed_scalars,
    _should_cpu_fallback,
    randn_kernel,
)

logger = logging.getLogger(__name__)

_CPU_FALLBACK_MAX = int(os.environ.get("FLAGGEMS_NORMAL_CPU_FALLBACK_MAX", "32768"))


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_float(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_float(val, std, mean):
    return val * std + mean


def normal_distribution(shape, device, *, generator=None, out=None):
    logger.debug("GEMS_TSINGMICRO NORMAL_DISTRIBUTION")
    N = volume(shape)
    if N == 0:
        if out is None:
            return torch.empty(shape, device=device, dtype=torch.float32)
        return out

    increment = max(N, 1)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    # float64 output: DSA randn is float32-only; keep CPU for correctness.
    if out is not None and out.dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO NORMAL f64 out → CPU")
        cpu = _native_cpu_randn(shape, torch.float64, device)
        out.copy_(cpu)
        return out

    if _should_cpu_fallback(N, device, _CPU_FALLBACK_MAX):
        logger.debug(
            "GEMS_TSINGMICRO NORMAL cpu fallback N=%s (max=%s)", N, _CPU_FALLBACK_MAX
        )
        cpu = _native_cpu_randn(shape, torch.float32, device)
        if out is None:
            return cpu
        out.copy_(cpu)
        return out

    if out is None:
        out = torch.empty(shape, device=device, dtype=torch.float32)
    seed0, seed1 = _philox_to_dsa_seed_scalars(philox_seed, philox_offset)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out.view(-1), seed0, seed1, N)
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_TSINGMICRO NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    # float64: DSA base is float32-only; keep CPU for correctness.
    if mean.dtype is torch.float64 or std.dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO NORMAL_TENSOR_TENSOR f64 → CPU")
        return torch.ops.aten.normal.Tensor_Tensor(
            mean.cpu(), std.cpu(), generator=generator
        ).to(device=device)
    out = normal_distribution(shape, device)
    return transform_func_tensor_tensor(out, std, mean)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS_TSINGMICRO NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    # float64: DSA base is float32-only; keep CPU for correctness.
    if mean.dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO NORMAL_TENSOR_FLOAT f64 → CPU")
        return torch.ops.aten.normal.Tensor_float(
            mean.cpu(), std, generator=generator
        ).to(device=device)
    out = normal_distribution(shape, device)
    return transform_func_tensor_float(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_TSINGMICRO NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    # float64: DSA base is float32-only; keep CPU for correctness.
    if std.dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO NORMAL_FLOAT_TENSOR f64 → CPU")
        return torch.ops.aten.normal.float_Tensor(
            mean, std.cpu(), generator=generator
        ).to(device=device)
    out = normal_distribution(shape, device)
    return transform_func_float_tensor(out, std, mean)


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS_TSINGMICRO NORMAL_")
    shape = self.shape
    device = self.device
    self = normal_distribution(shape, device, generator=None, out=self)
    transform_func_float_float(self, std, mean, out0=self)
    return self
