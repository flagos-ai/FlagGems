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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    BINARY_OPS as _hygon_foreach_binary_fast__BINARY_OPS,
)
from flag_gems.ops._foreach_binary import (
    registered_wrappers as _hygon_foreach_binary_fast__registered_wrappers,
)
from flag_gems.utils import libentry as _hygon_foreach_binary_fast__libentry
from flag_gems.utils.foreach import (
    check_tensor_list as _hygon_foreach_binary_fast__check_tensor_list,
)

"Native zero-copy result views; numerical work remains in Triton."
_hygon_foreach_views__NATIVE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_hygon_foreach_views__UNBIND = torch.ops.aten.unbind.int


def _hygon_foreach_views__unbind_views(tensor):
    return _hygon_foreach_views__UNBIND.redispatch(
        _hygon_foreach_views__NATIVE_KEYSET, tensor, 0
    )


_hygon_foreach_binary_fast__unbind_views = _hygon_foreach_views__unbind_views
_hygon_foreach_binary_fast__GENERIC = _hygon_foreach_binary_fast__registered_wrappers()
_hygon_foreach_binary_fast__FLOATS = (torch.float16, torch.bfloat16, torch.float32)
_hygon_foreach_binary_fast__KERNELS = {}


@triton.jit
def _hygon_foreach_binary_fast__select_ptr(pointers, slot):
    ptr = pointers[0]
    for i in tl.static_range(1, len(pointers)):
        ptr = tl.where(slot == i, pointers[i], ptr)
    return ptr


@_hygon_foreach_binary_fast__libentry()
@triton.jit
def _hygon_foreach_binary_fast__binary_uniform(
    A,
    B,
    C,
    Out,
    scalar,
    alpha,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    FN: tl.constexpr,
    MODE: tl.constexpr,
    INPLACE: tl.constexpr,
):
    slot = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ap = _hygon_foreach_binary_fast__select_ptr(A, slot)
    mask = offsets < N
    if N % BLOCK == 0:
        a = tl.load(ap + offsets).to(tl.float32)
    else:
        a = tl.load(ap + offsets, mask, 0).to(tl.float32)
    if MODE == 1:
        b = scalar
    elif MODE == 2:
        b = tl.load(B[0]).to(tl.float32)
    else:
        bp = _hygon_foreach_binary_fast__select_ptr(B, slot)
        if N % BLOCK == 0:
            b = tl.load(bp + offsets).to(tl.float32)
        else:
            b = tl.load(bp + offsets, mask, 0).to(tl.float32)
    if MODE == 3:
        cp = _hygon_foreach_binary_fast__select_ptr(C, slot)
        if N % BLOCK == 0:
            c = tl.load(cp + offsets).to(tl.float32)
        else:
            c = tl.load(cp + offsets, mask, 0).to(tl.float32)
        result = FN(a, b, c, scalar)
    else:
        result = FN(a, b * alpha)
    if INPLACE:
        dst = ap + offsets
    else:
        dst = Out + slot.to(tl.int64) * N + offsets
    if N % BLOCK == 0:
        tl.store(dst, result)
    else:
        tl.store(dst, result, mask)


def _hygon_foreach_binary_fast__uniform(seq, first, shape):
    dtype = first.dtype
    device = first.device
    return all(
        (
            t.dtype == dtype
            and t.device == device
            and (t.shape == shape)
            and t.is_contiguous()
            for t in seq
        )
    )


def _hygon_foreach_binary_fast__number(value):
    return type(value) in (int, float)


def _hygon_foreach_binary_fast__scalar_value(value, count):
    if isinstance(value, (list, tuple)):
        if (
            len(value) != count
            or not value
            or (not _hygon_foreach_binary_fast__number(value[0]))
        ):
            return None
        if value.count(value[0]) != count or any((type(s) is bool for s in value)):
            return None
        return float(value[0])
    return float(value) if _hygon_foreach_binary_fast__number(value) else None


def _hygon_foreach_binary_fast__binary(
    attr, name, overload, tensors, other, inplace=False, alpha=1, third=None, value=None
):
    fallback = _hygon_foreach_binary_fast__GENERIC[attr]
    kwargs = {"alpha": alpha} if name in ("add", "sub") and overload == "List" else {}

    def slow():
        if third is not None:
            return fallback(tensors, other, third, value)
        return fallback(tensors, other, **kwargs)

    tensors = _hygon_foreach_binary_fast__check_tensor_list(tensors)
    first = tensors[0]
    count = len(tensors)
    shape = first.shape
    if (
        count > 16
        or first.dtype not in _hygon_foreach_binary_fast__FLOATS
        or first.device.type == "cpu"
        or (not _hygon_foreach_binary_fast__uniform(tensors, first, shape))
    ):
        return slow()
    if inplace and len({t.data_ptr() for t in tensors}) != count:
        return slow()
    mode = 0
    bp = ()
    cp = ()
    scalar = 0.0
    if third is not None:
        if (
            not isinstance(other, (list, tuple))
            or not isinstance(third, (list, tuple))
            or len(other) != count
            or (len(third) != count)
        ):
            return slow()
        if (
            not all((isinstance(t, torch.Tensor) for t in (*other, *third)))
            or not _hygon_foreach_binary_fast__uniform(other, first, shape)
            or (not _hygon_foreach_binary_fast__uniform(third, first, shape))
        ):
            return slow()
        vals = value
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.ndim != 1 or value.numel() != count:
                return slow()
            vals = value.tolist()
        scalar = _hygon_foreach_binary_fast__scalar_value(vals, count)
        if scalar is None:
            return slow()
        bp, cp, mode = (tuple(other), tuple(third), 3)
    elif overload in ("Scalar", "ScalarList"):
        scalar = _hygon_foreach_binary_fast__scalar_value(other, count)
        if scalar is None:
            return slow()
        mode = 1
    elif overload == "Tensor":
        if not isinstance(other, torch.Tensor) or other.device != first.device:
            return slow()
        if other.numel() == 1 and (
            other.dtype == first.dtype
            or (other.ndim == 0 and other.dtype in _hygon_foreach_binary_fast__FLOATS)
        ):
            bp, mode = ((other,), 2)
        elif (
            other.dtype == first.dtype
            and other.shape == shape
            and other.is_contiguous()
        ):
            bp = (other,) * count
        else:
            return slow()
    else:
        if (
            not isinstance(other, (list, tuple))
            or len(other) != count
            or (not all((isinstance(t, torch.Tensor) for t in other)))
            or (not _hygon_foreach_binary_fast__uniform(other, first, shape))
        ):
            return slow()
        bp = tuple(other)
    n = first.numel()
    output = (
        first
        if inplace
        else torch.empty((count,) + shape, dtype=first.dtype, device=first.device)
    )
    if n:
        fn = _hygon_foreach_binary_fast__BINARY_OPS[name].fn
        block = max(256, 4096 // first.element_size())
        args = (
            tuple(tensors),
            bp,
            cp,
            output,
            scalar,
            float(alpha),
            n,
            block,
            fn,
            mode,
            inplace,
        )
        alignment = tuple((t.data_ptr() % 16 == 0 for t in (*tensors, *bp, *cp)))
        key = (
            first.get_device(),
            first.dtype,
            n,
            count,
            block,
            fn,
            mode,
            inplace,
            bp[0].dtype if bp else None,
            alignment,
        )
        grid = (triton.cdiv(n, block), count, 1)
        kernel = _hygon_foreach_binary_fast__KERNELS.get(key)
        if kernel is None:
            compiled = _hygon_foreach_binary_fast__binary_uniform[grid](
                *args, num_warps=4, enable_fp_fusion=False
            )
            kernel = compiled[0] if isinstance(compiled, tuple) else compiled
            if kernel is not None:
                if len(_hygon_foreach_binary_fast__KERNELS) >= 256:
                    _hygon_foreach_binary_fast__KERNELS.pop(
                        next(iter(_hygon_foreach_binary_fast__KERNELS))
                    )
                _hygon_foreach_binary_fast__KERNELS[key] = kernel
        else:
            kernel[grid](*args, stream=torch.cuda.current_stream().cuda_stream)
    return None if inplace else list(_hygon_foreach_binary_fast__unbind_views(output))


binary = _hygon_foreach_binary_fast__binary
logger = logging.getLogger(__name__)


def _foreach_div__Tensor(self, other):
    logger.debug("GEMS_HYGON _FOREACH_DIV__TENSOR")
    return binary("_foreach_div_.Tensor", "div", "Tensor", self, other, inplace=True)
