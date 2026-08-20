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

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import CodeGenConfig

logger = logging.getLogger(__name__)

MAX_GRID_SIZES = (65535, 65535, 65535)
config_f16 = CodeGenConfig(
    max_tile_size=1024,
    max_grid_size=MAX_GRID_SIZES,
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
)


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def clamp_func_tensor(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")], config=config_f16)
@triton.jit
def clamp_func_tensor_f16(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min_tensor(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_f16)
@triton.jit
def clamp_func_min_tensor_f16(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max_tensor(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_f16)
@triton.jit
def clamp_func_max_tensor_f16(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_SUNRISE CLAMP_TENSOR")
    if A.dtype == torch.half:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max_tensor_f16(A, maxi)
        elif maxi is None:
            return clamp_func_min_tensor_f16(A, mini)
        else:
            return clamp_func_tensor_f16(A, mini, maxi)
    else:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max_tensor(A, maxi)
        elif maxi is None:
            return clamp_func_min_tensor(A, mini)
        else:
            return clamp_func_tensor(A, mini, maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_SUNRISE CLAMP_TENSOR_")
    if A.dtype == torch.half:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max_tensor_f16(A, maxi, out0=A)
        elif maxi is None:
            return clamp_func_min_tensor_f16(A, mini, out0=A)
        else:
            return clamp_func_tensor_f16(A, mini, maxi, out0=A)
    else:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max_tensor(A, maxi, out0=A)
        elif maxi is None:
            return clamp_func_min_tensor(A, mini, out0=A)
        else:
            return clamp_func_tensor(A, mini, maxi, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_f16,
)
@triton.jit
def clamp_func_f16(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_f16
)
@triton.jit
def clamp_func_min_f16(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_f16
)
@triton.jit
def clamp_func_max_f16(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clamp_min(A, mini):
    logger.debug("GEMS_SUNRISE CLAMP_MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    if isinstance(mini, torch.Tensor):
        if A.dtype == torch.half:
            return clamp_func_min_tensor_f16(A, mini)
        return clamp_func_min_tensor(A, mini)
    return clamp_func_min(A, mini)


def clamp_min_(A, mini):
    logger.debug("GEMS_SUNRISE CLAMP_MIN_")
    if mini is None:
        raise ValueError("Mini must not be None")
    if isinstance(mini, torch.Tensor):
        if A.dtype == torch.half:
            return clamp_func_min_tensor_f16(A, mini, out0=A)
        return clamp_func_min_tensor(A, mini, out0=A)
    return clamp_func_min(A, mini, out0=A)


def clamp_min_out(A, mini, *, out=None):
    logger.debug("GEMS_SUNRISE CLAMP_MIN_OUT")
    if mini is None:
        raise ValueError("Mini must not be None")
    if isinstance(mini, torch.Tensor):
        if A.dtype == torch.half:
            return clamp_func_min_tensor_f16(A, mini, out0=out)
        return clamp_func_min_tensor(A, mini, out0=out)
    return clamp_func_min(A, mini, out0=out)


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_SUNRISE CLAMP")
    if A.dtype == torch.half:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max_f16(A, maxi)
        elif maxi is None:
            return clamp_func_min_f16(A, mini)
        else:
            return clamp_func_f16(A, mini, maxi)
    else:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max(A, maxi)
        elif maxi is None:
            return clamp_func_min(A, mini)
        else:
            return clamp_func(A, mini, maxi)


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_SUNRISE CLAMP_")
    if A.dtype == torch.half:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max_f16(A, maxi, out0=A)
        elif maxi is None:
            return clamp_func_min_f16(A, mini, out0=A)
        else:
            return clamp_func_f16(A, mini, maxi, out0=A)
    else:
        if mini is None and maxi is None:
            raise ValueError("At least one of mini or maxi must not be None")
        elif mini is None:
            return clamp_func_max(A, maxi, out0=A)
        elif maxi is None:
            return clamp_func_min(A, mini, out0=A)
        else:
            return clamp_func(A, mini, maxi, out0=A)
