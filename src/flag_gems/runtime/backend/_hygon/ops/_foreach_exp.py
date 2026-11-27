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

from flag_gems.ops._foreach_unary import (
    UNARY_OPS as _hygon_foreach_unary_fast__UNARY_OPS,
)
from flag_gems.ops._foreach_unary import (
    UNARY_OPS as _hygon_foreach_unary_fp64__UNARY_OPS,
)
from flag_gems.ops._foreach_unary import (
    apply_unary as _hygon_foreach_unary_fast__apply_unary,
)
from flag_gems.utils import libentry as _hygon_foreach_unary_fast__libentry
from flag_gems.utils import tl_extra_shim as _hygon_foreach_unary_fast__tl_extra_shim
from flag_gems.utils import tl_extra_shim as _hygon_foreach_unary_fp64__tl_extra_shim
from flag_gems.utils.foreach import _STATS as _hygon_foreach_unary_fast__STATS
from flag_gems.utils.foreach import (
    check_tensor_list as _hygon_foreach_unary_fast__check_tensor_list,
)
from flag_gems.utils.foreach import (
    foreach_unary as _hygon_foreach_unary_fast__foreach_unary,
)

"Keep double inputs in double opmath on semantic fallback paths."
_hygon_foreach_unary_fp64__NATIVE_ATAN = _hygon_foreach_unary_fp64__UNARY_OPS["atan"].fn


@triton.jit
def _hygon_foreach_unary_fp64__atan_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.atan(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_ATAN(x)


_hygon_foreach_unary_fp64__NATIVE_CEIL = _hygon_foreach_unary_fp64__UNARY_OPS["ceil"].fn


@triton.jit
def _hygon_foreach_unary_fp64__ceil_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.ceil(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_CEIL(x)


_hygon_foreach_unary_fp64__NATIVE_COS = _hygon_foreach_unary_fp64__UNARY_OPS["cos"].fn


@triton.jit
def _hygon_foreach_unary_fp64__cos_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.cos(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_COS(x)


_hygon_foreach_unary_fp64__NATIVE_ERF = _hygon_foreach_unary_fp64__UNARY_OPS["erf"].fn


@triton.jit
def _hygon_foreach_unary_fp64__erf_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.erf(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_ERF(x)


_hygon_foreach_unary_fp64__NATIVE_EXP = _hygon_foreach_unary_fp64__UNARY_OPS["exp"].fn


@triton.jit
def _hygon_foreach_unary_fp64__exp_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.exp(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_EXP(x)


_hygon_foreach_unary_fp64__NATIVE_EXPM1 = _hygon_foreach_unary_fp64__UNARY_OPS[
    "expm1"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__expm1_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.expm1(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_EXPM1(x)


_hygon_foreach_unary_fp64__NATIVE_FLOOR = _hygon_foreach_unary_fp64__UNARY_OPS[
    "floor"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__floor_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.floor(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_FLOOR(x)


_hygon_foreach_unary_fp64__NATIVE_FRAC = _hygon_foreach_unary_fp64__UNARY_OPS["frac"].fn


@triton.jit
def _hygon_foreach_unary_fp64__frac_wide(x):
    if x.dtype == tl.float64:
        return x - _hygon_foreach_unary_fp64__tl_extra_shim.trunc(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_FRAC(x)


_hygon_foreach_unary_fp64__NATIVE_LOG10 = _hygon_foreach_unary_fp64__UNARY_OPS[
    "log10"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__log10_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.log10(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_LOG10(x)


_hygon_foreach_unary_fp64__NATIVE_LOG2 = _hygon_foreach_unary_fp64__UNARY_OPS["log2"].fn


@triton.jit
def _hygon_foreach_unary_fp64__log2_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.log2(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_LOG2(x)


_hygon_foreach_unary_fp64__NATIVE_RECIPROCAL = _hygon_foreach_unary_fp64__UNARY_OPS[
    "reciprocal"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__reciprocal_wide(x):
    if x.dtype == tl.float64:
        return 1.0 / x
    else:
        return _hygon_foreach_unary_fp64__NATIVE_RECIPROCAL(x)


_hygon_foreach_unary_fp64__NATIVE_SIGMOID = _hygon_foreach_unary_fp64__UNARY_OPS[
    "sigmoid"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__sigmoid_wide(x):
    if x.dtype == tl.float64:
        return 1.0 / (1.0 + _hygon_foreach_unary_fp64__tl_extra_shim.exp(-x))
    else:
        return _hygon_foreach_unary_fp64__NATIVE_SIGMOID(x)


_hygon_foreach_unary_fp64__NATIVE_SIGN = _hygon_foreach_unary_fp64__UNARY_OPS["sign"].fn


@triton.jit
def _hygon_foreach_unary_fp64__sign_wide(x):
    if x.dtype == tl.float64:
        return tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0)).to(tl.float64)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_SIGN(x)


_hygon_foreach_unary_fp64__NATIVE_SIN = _hygon_foreach_unary_fp64__UNARY_OPS["sin"].fn


@triton.jit
def _hygon_foreach_unary_fp64__sin_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.sin(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_SIN(x)


_hygon_foreach_unary_fp64__NATIVE_TAN = _hygon_foreach_unary_fp64__UNARY_OPS["tan"].fn


@triton.jit
def _hygon_foreach_unary_fp64__tan_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.tan(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_TAN(x)


_hygon_foreach_unary_fp64__NATIVE_TRUNC = _hygon_foreach_unary_fp64__UNARY_OPS[
    "trunc"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__trunc_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.trunc(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_TRUNC(x)


_hygon_foreach_unary_fp64__NATIVE_LOG = _hygon_foreach_unary_fp64__UNARY_OPS["log"].fn


@triton.jit
def _hygon_foreach_unary_fp64__log_wide(x):
    if x.dtype == tl.float64:
        return _hygon_foreach_unary_fp64__tl_extra_shim.log(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_LOG(x)


_hygon_foreach_unary_fp64__NATIVE_SQRT = _hygon_foreach_unary_fp64__UNARY_OPS["sqrt"].fn


@triton.jit
def _hygon_foreach_unary_fp64__sqrt_wide(x):
    if x.dtype == tl.float64:
        return tl.sqrt(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_SQRT(x)


_hygon_foreach_unary_fp64__NATIVE_RSQRT = _hygon_foreach_unary_fp64__UNARY_OPS[
    "rsqrt"
].fn


@triton.jit
def _hygon_foreach_unary_fp64__rsqrt_wide(x):
    if x.dtype == tl.float64:
        return 1.0 / tl.sqrt(x)
    else:
        return _hygon_foreach_unary_fp64__NATIVE_RSQRT(x)


_hygon_foreach_unary_fp64__FUNCTIONS = {
    "log": _hygon_foreach_unary_fp64__log_wide,
    "sqrt": _hygon_foreach_unary_fp64__sqrt_wide,
    "rsqrt": _hygon_foreach_unary_fp64__rsqrt_wide,
    "atan": _hygon_foreach_unary_fp64__atan_wide,
    "ceil": _hygon_foreach_unary_fp64__ceil_wide,
    "cos": _hygon_foreach_unary_fp64__cos_wide,
    "erf": _hygon_foreach_unary_fp64__erf_wide,
    "exp": _hygon_foreach_unary_fp64__exp_wide,
    "expm1": _hygon_foreach_unary_fp64__expm1_wide,
    "floor": _hygon_foreach_unary_fp64__floor_wide,
    "frac": _hygon_foreach_unary_fp64__frac_wide,
    "log10": _hygon_foreach_unary_fp64__log10_wide,
    "log2": _hygon_foreach_unary_fp64__log2_wide,
    "reciprocal": _hygon_foreach_unary_fp64__reciprocal_wide,
    "sigmoid": _hygon_foreach_unary_fp64__sigmoid_wide,
    "sign": _hygon_foreach_unary_fp64__sign_wide,
    "sin": _hygon_foreach_unary_fp64__sin_wide,
    "tan": _hygon_foreach_unary_fp64__tan_wide,
    "trunc": _hygon_foreach_unary_fp64__trunc_wide,
}
"Native zero-copy result views; numerical work remains in Triton."
_hygon_foreach_views__NATIVE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_hygon_foreach_views__UNBIND = torch.ops.aten.unbind.int


def _hygon_foreach_views__unbind_views(tensor):
    return _hygon_foreach_views__UNBIND.redispatch(
        _hygon_foreach_views__NATIVE_KEYSET, tensor, 0
    )


"Fused dense foreach kernels with generic FlagGems semantic fallbacks."
_hygon_foreach_unary_fast__FP64_FUNCTIONS = _hygon_foreach_unary_fp64__FUNCTIONS
_hygon_foreach_unary_fast__unbind_views = _hygon_foreach_views__unbind_views
_hygon_foreach_unary_fast__FLOATS = (torch.float16, torch.bfloat16, torch.float32)
_hygon_foreach_unary_fast__INTS = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


@triton.jit
def _hygon_foreach_unary_fast__round_value(x):
    if x.dtype.is_floating():
        if x.dtype == tl.float64:
            return _hygon_foreach_unary_fast__tl_extra_shim.rint(x)
        else:
            return _hygon_foreach_unary_fast__tl_extra_shim.rint(x.to(tl.float32))
    else:
        return x


@_hygon_foreach_unary_fast__libentry()
@triton.jit
def _hygon_foreach_unary_fast__uniform_kernel(
    inputs,
    output,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    FN: tl.constexpr,
    INPLACE: tl.constexpr,
):
    slot = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ptr = inputs[0]
    for i in tl.static_range(1, len(inputs)):
        ptr = tl.where(slot == i, inputs[i], ptr)
    if N % BLOCK == 0:
        x = tl.load(ptr + offset)
    else:
        x = tl.load(ptr + offset, offset < N, 0)
    value = FN(x)
    if INPLACE:
        dst = ptr + offset
    else:
        dst = output + slot.to(tl.int64) * N + offset
    if N % BLOCK == 0:
        tl.store(dst, value)
    else:
        tl.store(dst, value, offset < N)


_hygon_foreach_unary_fast__KERNELS = {}


def _hygon_foreach_unary_fast__launch(tensors, output, n, block, fn, inplace):
    pointers = tuple((t.data_ptr() for t in tensors))
    alignment = tuple((pointer % 16 == 0 for pointer in pointers))
    key = (
        tensors[0].get_device(),
        tensors[0].dtype,
        n,
        len(tensors),
        block,
        fn,
        inplace,
        alignment,
    )
    args = (tuple(tensors), output, n, block, fn, inplace)
    grid = (triton.cdiv(n, block), len(tensors), 1)
    kernel = _hygon_foreach_unary_fast__KERNELS.get(key)
    if kernel is None:
        compiled = _hygon_foreach_unary_fast__uniform_kernel[grid](*args, num_warps=4)
        kernel = compiled[0] if isinstance(compiled, tuple) else compiled
        if kernel is not None:
            if len(_hygon_foreach_unary_fast__KERNELS) >= 128:
                _hygon_foreach_unary_fast__KERNELS.pop(
                    next(iter(_hygon_foreach_unary_fast__KERNELS))
                )
            _hygon_foreach_unary_fast__KERNELS[key] = kernel
    else:
        if tensors[0].get_device() == torch.cuda.current_device():
            args = (pointers, output.data_ptr(), n, block, fn, inplace)
        kernel[grid](*args, stream=torch.cuda.current_stream().cuda_stream)


def _hygon_foreach_unary_fast__unary(name, tensors, inplace=False):
    tensors = _hygon_foreach_unary_fast__check_tensor_list(tensors)
    first = tensors[0]
    dtype = first.dtype
    count = len(tensors)
    supported = dtype in _hygon_foreach_unary_fast__FLOATS
    if name in ("neg", "round"):
        supported = (
            supported
            or dtype == torch.float64
            or dtype in _hygon_foreach_unary_fast__INTS
        )
    shape = first.shape
    device = first.device
    if supported and device.type != "cpu" and (count <= 128):
        for t in tensors:
            if (
                t.dtype != dtype
                or t.device != device
                or t.shape != shape
                or (not t.is_contiguous())
            ):
                break
        else:
            if inplace and len({t.data_ptr() for t in tensors}) != count:
                for t in tensors:
                    _hygon_foreach_unary_fast__apply_unary(name, [t], inplace=True)
                return None
            n = first.numel()
            output = (
                first
                if inplace
                else (
                    torch.empty_like(first)
                    if count == 1
                    else torch.empty((count,) + shape, dtype=dtype, device=device)
                )
            )
            _hygon_foreach_unary_fast__STATS.reset()
            if n:
                fn = (
                    _hygon_foreach_unary_fast__round_value
                    if name == "round"
                    else _hygon_foreach_unary_fast__UNARY_OPS[name].fn
                )
                block = max(256, 4096 // dtype.itemsize)
                _hygon_foreach_unary_fast__launch(
                    tensors, output, n, block, fn, inplace
                )
                _hygon_foreach_unary_fast__STATS.executor_launches += 1
                _hygon_foreach_unary_fast__STATS.groups += 1
            return (
                None
                if inplace
                else (
                    [output]
                    if count == 1
                    else list(_hygon_foreach_unary_fast__unbind_views(output))
                )
            )
    if name in _hygon_foreach_unary_fast__FP64_FUNCTIONS and any(
        (t.dtype == torch.float64 for t in tensors)
    ):
        op = _hygon_foreach_unary_fast__UNARY_OPS[name]
        result = _hygon_foreach_unary_fast__foreach_unary(
            tensors,
            _hygon_foreach_unary_fast__FP64_FUNCTIONS[name],
            complex_fn=op.complex_fn,
            out_dtype_fn=op.out_dtype_fn,
            inplace=inplace,
            allowed_dtypes=op.allowed,
        )
        return None if inplace else result
    return _hygon_foreach_unary_fast__apply_unary(name, tensors, inplace=inplace)


unary = _hygon_foreach_unary_fast__unary
logger = logging.getLogger(__name__)


def _foreach_exp(self):
    logger.debug("GEMS_HYGON _FOREACH_EXP (GEMS _FOREACH_EXP)")
    return unary("exp", self, inplace=False)
