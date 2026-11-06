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
"""Native layer norm: strided-row Hygon launch path with generic fallback."""

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_HYGON = runtime.device.vendor_name == "hygon"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_LAUNCH_CACHE = {}
try:
    from c_operators import allocate_layer_norm_outputs_1d as ALLOCATE_1D
except ImportError:
    # Pure-Python installations remain usable without the optional C++ wheel.
    def ALLOCATE_1D(input):
        new_empty = input.new_empty
        shape = input.shape[:-1] + (1,)
        return (
            new_empty(input.shape),
            new_empty(shape, dtype=torch.float32),
            new_empty(shape, dtype=torch.float32),
        )


@libentry()
@triton.jit(do_not_specialize=["X", "Y", "W", "B", "Mean", "Rstd", "EPS"])
def _strided_row_layer_norm(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N: tl.constexpr,
    EPS,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    WS: tl.constexpr,
    BS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    remaining = row
    offset = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (remaining % SHAPE[axis]).to(tl.int64) * STRIDE[axis]
        remaining = remaining // SHAPE[axis]
    col = tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + offset + col, valid, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    delta = x - mean
    var = tl.sum(tl.where(valid, delta * delta, 0), 0) / N
    rstd = tl.rsqrt(var + EPS)
    out = delta * rstd
    if W is not None:
        out = out * tl.load(W + col * WS, valid, other=1).to(tl.float32)
    if B is not None:
        out = out + tl.load(B + col * BS, valid, other=0).to(tl.float32)
    tl.store(Y + row * N + col, out, valid)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


def _can_use_strided(input, normalized_shape, weight, bias):
    if not _HYGON or input.device.type not in ("cuda", "flagos", "privateuseone"):
        return False
    if input.dtype not in _DTYPES or len(normalized_shape) != 1 or input.ndim < 1:
        return False
    n = normalized_shape[0]
    if not (0 < n <= 4096) or input.shape[-1] != n or input.numel() == 0:
        return False
    if input.stride(-1) != 1 or input.ndim > 8:
        return False
    for affine in (weight, bias):
        if affine is not None and (
            affine.shape != (n,)
            or affine.dtype != input.dtype
            or affine.device != input.device
            or affine.stride(0) < 0
        ):
            return False
    return True


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # Leave compiler/subclass dispatch on the original implementation. The
    # prepared eager launch assumes concrete metadata and ordinary Tensors.
    if (
        type(input) is not torch.Tensor
        or torch.compiler.is_compiling()
        or (weight is not None and type(weight) is not torch.Tensor)
        or (bias is not None and type(bias) is not torch.Tensor)
    ):
        return _generic_native(input, normalized_shape, weight, bias, eps)
    # Materialize logical negative values before passing storage to Triton.
    if input.is_neg():
        input = input.resolve_neg()
    if weight is not None and weight.is_neg():
        weight = weight.resolve_neg()
    if bias is not None and bias.is_neg():
        bias = bias.resolve_neg()
    device, dtype = input.device, input.dtype
    shape, stride = input.shape, input.stride()
    wmeta = (
        None
        if weight is None
        else (weight.shape, weight.stride(), weight.dtype, weight.device)
    )
    bmeta = (
        None if bias is None else (bias.shape, bias.stride(), bias.dtype, bias.device)
    )
    # All metadata consulted by eligibility and compilation participates in the
    # key. A new Tensor with the same metadata is safe; a resized or re-strided
    # Tensor cannot silently reuse an incompatible validated plan.
    key = (
        device,
        dtype,
        shape,
        stride,
        tuple(normalized_shape),
        wmeta,
        bmeta,
        type(eps),
    )
    plan = _LAUNCH_CACHE.get(key)
    if plan is None:
        if not _can_use_strided(input, normalized_shape, weight, bias):
            return _generic_native(input, normalized_shape, weight, bias, eps)
        n = normalized_shape[0]
        stats_shape = shape[:-1] + (1,)
        block = 1 << (n - 1).bit_length()
        grid = (input.numel() // n, 1, 1)
        ws = 0 if weight is None else wmeta[1][0]
        bs = 0 if bias is None else bmeta[1][0]
        constants = (n, shape[:-1], stride[:-1], ws, bs, block)
        # Only immutable metadata and compiled objects are saved. Outputs,
        # current streams, Tensor objects and their addresses are never saved.
        plan = [stats_shape, grid, constants, None]
        if len(_LAUNCH_CACHE) >= 128:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = plan
    stats_shape, grid, constants, executable = plan
    out, mean, rstd = ALLOCATE_1D(input)
    n, leading_shape, leading_stride, ws, bs, block = constants
    args = (
        input,
        out,
        weight,
        bias,
        mean,
        rstd,
        n,
        eps,
        leading_shape,
        leading_stride,
        ws,
        bs,
        block,
    )
    # Only avoid an exchange when the current thread is already on the input
    # device. A real device change retains the restoring standard guard.
    if torch_device_fn.current_device() == device.index:
        _submit(plan, args, device.index)
    else:
        with torch_device_fn.device(device.index):
            _submit(plan, args, device.index)
    return out, mean, rstd


def _submit(plan, args, index):
    _, grid, _, executable = plan
    if executable is None:
        kernel, _ = _strided_row_layer_norm[grid](
            *args,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # Cache only the compiled kernel; construct its standard runner at
        # each warm invocation rather than retaining a separate callable.
        plan[3] = kernel
        return
    # Standard runner owns stream lookup, metadata, hooks and launch.
    kernel = executable
    runner = kernel[grid]
    runner(*args, stream=driver.active.get_current_stream(index))


def _generic_native(input, normalized_shape, weight, bias, eps):
    from flag_gems.ops.layernorm import _layer_norm_generic

    out, mean, rstd = _layer_norm_generic(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return out, mean.view(stats_shape), rstd.view(stats_shape)


layer_norm = native_layer_norm
