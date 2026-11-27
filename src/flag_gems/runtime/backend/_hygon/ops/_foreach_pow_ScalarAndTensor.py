# Copyright (c) 2024 FlagGems. All rights reserved.
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

"""Scalar-base foreach power with one packed output and one fused launch."""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import registered_wrappers
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.foreach import _device_meta, check_tensor_list, tl_dtype

"Native zero-copy result views; numerical work remains in Triton."
_hygon_foreach_views__NATIVE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_hygon_foreach_views__UNBIND = torch.ops.aten.unbind.int


def _hygon_foreach_views__unbind_views(tensor):
    return _hygon_foreach_views__UNBIND.redispatch(
        _hygon_foreach_views__NATIVE_KEYSET, tensor, 0
    )


unbind_views = _hygon_foreach_views__unbind_views
logger = logging.getLogger(__name__)
_generic_pow = registered_wrappers()["_foreach_pow.ScalarAndTensor"]


@libentry()
@triton.jit
def _pow_packed(
    Meta,
    Out,
    N: tl.constexpr,
    Base,
    LogBase,
    DT: tl.constexpr,
    FAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tensor = tl.program_id(1)
    ptr = tl.load(Meta + tensor).to(tl.pointer_type(DT))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + offsets, offsets < N, 0).to(tl.float32)
    if FAST:
        value = tl.exp2(x * LogBase)
    else:
        value = tl_extra_shim.pow(Base.to(tl.float32), x)
    tl.store(Out + tensor.to(tl.int64) * N + offsets, value, offsets < N)


def _foreach_pow_ScalarAndTensor(self, exponent):
    logger.debug("GEMS_HYGON _FOREACH_POW_SCALARANDTENSOR")
    tensors = check_tensor_list(exponent)
    first = tensors[0]
    dtype = first.dtype
    if (
        isinstance(self, torch.Tensor)
        or dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(
            (
                t.dtype != dtype
                or t.device != first.device
                or t.shape != first.shape
                or (not t.is_contiguous())
                for t in tensors
            )
        )
    ):
        return _generic_pow(self, exponent)
    base = float(self)
    fast = base > 0.0 and base != 1.0 and math.isfinite(base)
    log_base = math.log2(base) if fast else 0.0
    n = first.numel()
    out = torch.empty((len(tensors),) + first.shape, dtype=dtype, device=first.device)
    if n:
        meta = _device_meta([t.data_ptr() for t in tensors], first.device)
        _pow_packed[triton.cdiv(n, 1024), len(tensors)](
            meta, out, n, base, log_base, tl_dtype(dtype), fast, 1024, num_warps=4
        )
    return list(unbind_views(out))
