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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.nan_to_num import nan_to_num as _generic_nan_to_num
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
_isnan = tl_extra_shim.isnan
_FLOAT32_MAX = torch.finfo(torch.float32).max
_MAX_ELEMENTS = (1 << 31) - 1


@libentry()
@triton.jit
def _nan_to_num_contiguous_kernel(
    X, Y, N, nan, posinf, neginf, BLOCK: tl.constexpr, FULL: tl.constexpr
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if FULL:
        x = tl.load(X + offsets)
    else:
        active = offsets < N
        x = tl.load(X + offsets, mask=active, other=0.0)
    # Classify the original input: replacements may themselves be NaN or Inf.
    x_nan = _isnan(x)
    x_posinf = x == float("inf")
    x_neginf = x == -float("inf")
    y = tl.where(x_nan, nan, x)
    y = tl.where(x_posinf, posinf, y)
    y = tl.where(x_neginf, neginf, y)
    if FULL:
        tl.store(Y + offsets, y)
    else:
        tl.store(Y + offsets, y, mask=active)


def _can_use_contiguous_path(A):
    if (
        type(A) is not torch.Tensor
        or A.layout != torch.strided
        or A.dtype != torch.float32
        or A.requires_grad
        or A.is_conj()
        or A.is_neg()
        or not A.is_contiguous()
    ):
        return False
    n = A.numel()
    if type(n) is not int or not 0 < n <= _MAX_ELEMENTS:
        return False
    return torch.autograd.forward_ad.unpack_dual(A).tangent is None


def _normalize_replacements(nan, posinf, neginf):
    replacements = []
    for value, default in zip(
        (nan, posinf, neginf), (0.0, _FLOAT32_MAX, -_FLOAT32_MAX)
    ):
        value = default if value is None else value
        # Keep unknown scalar classes and overflow/error behavior on the
        # generic path. Signed zero and nonfinite replacements are supported.
        if type(value) not in (int, float):
            return None
        try:
            value = float(value)
        except (OverflowError, ValueError):
            return None
        if math.isfinite(value) and abs(value) > _FLOAT32_MAX:
            return None
        replacements.append(value)
    return tuple(replacements)


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_HYGON NAN_TO_NUM")
    if not _can_use_contiguous_path(A):
        return _generic_nan_to_num(A, nan=nan, posinf=posinf, neginf=neginf)
    replacements = _normalize_replacements(nan, posinf, neginf)
    if replacements is None:
        return _generic_nan_to_num(A, nan=nan, posinf=posinf, neginf=neginf)

    n = A.numel()
    block, warps = (512, 2) if n <= 65536 else (1024, 4)
    with torch_device_fn.device(A.device):
        out = torch.empty_like(A)
        _nan_to_num_contiguous_kernel[(triton.cdiv(n, block),)](
            A,
            out,
            n,
            *replacements,
            BLOCK=block,
            FULL=(n % block == 0),
            num_warps=warps,
            num_stages=2,
            waves_per_eu=1,
            allow_flush_denorm=False,
        )
    return out
