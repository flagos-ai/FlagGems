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

from flag_gems.ops.floor import floor as _generic_floor
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_MAX_ELEMENTS = (1 << 31) - 1


@libentry()
@triton.jit
def _floor_contiguous_kernel(X, Y, N, BLOCK: tl.constexpr, FULL: tl.constexpr):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if FULL:
        x = tl.load(X + offsets)
    else:
        active = offsets < N
        x = tl.load(X + offsets, mask=active, other=0.0)
    y = tl.floor(x)
    if FULL:
        tl.store(Y + offsets, y)
    else:
        tl.store(Y + offsets, y, mask=active)


def _can_use_contiguous_path(A):
    # Metadata only: keep general strides, dtypes and tensor subclasses on the
    # existing pointwise implementation. No assumption about input values.
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


def floor(A):
    logger.debug("GEMS_HYGON FLOOR")
    if not _can_use_contiguous_path(A):
        return _generic_floor(A)

    n = A.numel()
    with torch_device_fn.device(A.device):
        out = torch.empty_like(A)
        _floor_contiguous_kernel[(triton.cdiv(n, _BLOCK_SIZE),)](
            A,
            out,
            n,
            BLOCK=_BLOCK_SIZE,
            FULL=(n % _BLOCK_SIZE == 0),
            num_warps=4,
            num_stages=2,
            waves_per_eu=1,
            allow_flush_denorm=False,
        )
    return out
