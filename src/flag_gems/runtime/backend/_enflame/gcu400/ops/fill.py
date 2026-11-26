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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_INT32_MAX = torch.iinfo(torch.int32).max
# Element-wise Triton stores of 64-bit dtypes are not reliable on this backend
# (the runtime replaces Long/Double with Int, corrupting every other element),
# so they stay on the generic broadcast kernel.  Complex lowering is just as
# fragile on this backend (see new_full).
_FLAT_UNSAFE_DTYPES = (
    torch.int64,
    torch.uint64,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
# GCU400 tuning for pure-write streams (new_full RSI, 2026-11-16): grids above
# the 24 schedulable SIPs never help pure writes (48 == 24) and 32 misfits the
# SIP count (~+20% on mid shapes); 32Ki-element tiles are the mid-band sweet
# spot and tie the ~3.25 TB/s DRAM write wall at stream sizes.
_FLAT_MAX_GRID = 24
_FLAT_MIN_BLOCK = 2048
_FLAT_SWEET_BLOCK = 32768
_LAUNCH_ATTEMPTS = 3


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


@libentry()
@triton.jit
def fill_scalar_flat_kernel(
    out_ptr,
    fill_value,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The fill value is a plain Python scalar: write the broadcast constant in
    # flat memory order with a grid-stride loop capped at the schedulable SIPs.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
        if EXACT:
            tl.store(out_ptr + offsets, val)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, val, mask=mask)


@libentry()
@triton.jit
def fill_tensor_flat_kernel(
    out_ptr,
    value_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The value is a 0-dim device tensor: load it once per CTA and broadcast
    # it on the store, instead of materializing a stride-0 block load per tile.
    fill_value = tl.load(value_ptr)
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
        if EXACT:
            tl.store(out_ptr + offsets, val)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, val, mask=mask)


def _flat_block(n_elements):
    if n_elements <= 8192:
        # Small sizes sit on the launch floor: one CTA with an exact tile.
        return max(2, triton.next_power_of_2(n_elements))
    # 32Ki-element tiles are the mid-band sweet spot (ramp-friendly: -1.4%~-4%
    # vs 64/128Ki caps at 4-64MB) and tie the DRAM write wall at stream sizes
    # (BLOCK is not the limiter once the 24 SIPs saturate writes).
    return min(
        _FLAT_SWEET_BLOCK,
        max(_FLAT_MIN_BLOCK, triton.next_power_of_2(-(-n_elements // 48))),
    )


_STRIDED_BLOCK = 1024
_STRIDED_MAX_GRID = 65535
_MAX_STRIDED_DIMS = 4


@libentry()
@triton.jit
def fill_strided_scalar_kernel(
    out_ptr,
    fill_value,
    start,
    n_elements,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Logical-index walk over a strided destination.  The generated pointwise
    # kernel drops the masked tail lanes of a non-dense 2-D store (filling
    # x[..., 358:] left the last 18 elements of the second row untouched), so
    # this path scatters a 1-D lane mask instead.  Dims/strides are constexpr
    # (the codegen is per-layout) and the leading dims are padded with size 1.
    idx = start + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    i3 = idx % D3
    r = idx // D3
    i2 = r % D2
    r = r // D2
    i1 = r % D1
    i0 = r // D1
    off = i0.to(tl.int64) * S0 + i1.to(tl.int64) * S1
    off += i2.to(tl.int64) * S2 + i3.to(tl.int64) * S3
    val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
    tl.store(out_ptr + off, val, mask=mask)


@libentry()
@triton.jit
def fill_strided_tensor_kernel(
    out_ptr,
    value_ptr,
    start,
    n_elements,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    fill_value = tl.load(value_ptr)
    idx = start + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    i3 = idx % D3
    r = idx // D3
    i2 = r % D2
    r = r // D2
    i1 = r % D1
    i0 = r // D1
    off = i0.to(tl.int64) * S0 + i1.to(tl.int64) * S1
    off += i2.to(tl.int64) * S2 + i3.to(tl.int64) * S3
    val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
    tl.store(out_ptr + off, val, mask=mask)


def _strided_layout(tensor):
    """Collapse the logical layout to <= 4 dims, or None when unsupported.

    Drops size-1 dims (they never move the address) and merges neighbouring
    dims that are contiguous relative to each other (``s_outer == s_inner *
    d_inner``), which turns a slice of a contiguous tensor into a handful of
    runs.  Returns the (dims, strides) pair padded to four dims, outermost
    first.
    """
    dims = []
    strides = []
    for d, s in zip(tensor.shape, tensor.stride()):
        if d != 1:
            dims.append(int(d))
            strides.append(int(s))
    if not dims:
        return None
    merged_dims = [dims[-1]]
    merged_strides = [strides[-1]]
    for d, s in zip(reversed(dims[:-1]), reversed(strides[:-1])):
        inner_d = merged_dims[-1]
        inner_s = merged_strides[-1]
        if s == inner_s * inner_d:
            merged_dims[-1] = d * inner_d
        else:
            merged_dims.append(d)
            merged_strides.append(s)
    if len(merged_dims) > _MAX_STRIDED_DIMS:
        return None
    while len(merged_dims) < _MAX_STRIDED_DIMS:
        merged_dims.append(1)
        merged_strides.append(0)
    merged_dims.reverse()
    merged_strides.reverse()
    return merged_dims, merged_strides


def _fill_strided(out, value, is_scalar):
    """Fill a non-dense destination in logical order; False when unsupported."""
    if out.layout != torch.strided:
        return False
    layout = _strided_layout(out)
    if layout is None:
        return False
    dims, strides = layout
    target = out
    if out.dtype == torch.bool:
        # Bool element stores lower to the narrow i1 lane on this backend;
        # writing through a same-width int8 view is exact.
        if out.stride(-1) != 1:
            return False
        target = out.view(torch.int8)
    n = target.numel()
    if n == 0:
        return True
    if is_scalar:
        if out.dtype == torch.bool:
            fill_value = 1 if value else 0
        else:
            fill_value = value
    else:
        fill_value = value.view(torch.int8) if out.dtype == torch.bool else value
    block = min(_STRIDED_BLOCK, triton.next_power_of_2(n))
    per_chunk = _STRIDED_MAX_GRID * block
    with torch_device_fn.device(out.device):
        start = 0
        while start < n:
            count = min(n - start, per_chunk)
            grid = (triton.cdiv(count, block),)
            if is_scalar:
                fill_strided_scalar_kernel[grid](
                    target,
                    fill_value,
                    start,
                    n,
                    D0=dims[0],
                    D1=dims[1],
                    D2=dims[2],
                    D3=dims[3],
                    S0=strides[0],
                    S1=strides[1],
                    S2=strides[2],
                    S3=strides[3],
                    BLOCK=block,
                    num_warps=4,
                )
            else:
                fill_strided_tensor_kernel[grid](
                    target,
                    fill_value,
                    start,
                    n,
                    D0=dims[0],
                    D1=dims[1],
                    D2=dims[2],
                    D3=dims[3],
                    S0=strides[0],
                    S1=strides[1],
                    S2=strides[2],
                    S3=strides[3],
                    BLOCK=block,
                    num_warps=4,
                )
            start += count
    return True


def _launch_flat(kernel, store_target, args, n_elements, num_warps=2):
    block = _flat_block(n_elements)
    grid = (min(_FLAT_MAX_GRID, -(-n_elements // block)),)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level flake windows on shared nodes); retrying
    # the launch recompiles and succeeds (new_full / atan2_ / sinh precedents
    # for the same toolchain family).
    for attempt in range(_LAUNCH_ATTEMPTS):
        try:
            with torch_device_fn.device(store_target.device):
                kernel[grid](
                    store_target,
                    *args,
                    n_elements,
                    BLOCK=block,
                    EXACT=(n_elements % block == 0),
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == _LAUNCH_ATTEMPTS - 1:
                raise
            logger.warning(
                "fill flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_store_target(out):
    if out.dtype == torch.bool:
        # Store through a same-width int8 view: bool element stores lower to
        # the narrow i1 lane on this backend.
        return out.view(torch.int8)
    return out


def _fill_scalar_flat_eligible_fresh(input, out):
    # ``out`` is freshly allocated by ``torch.empty_like`` so it always owns a
    # dense, non-overlapping storage; a cheap layout check replaces the aten
    # dense-query (which costs a dispatcher hop on every call).
    if input.layout != torch.strided:
        return False
    if out.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    return out.numel() <= _INT32_MAX


def _fill_scalar_flat_eligible_into(input, out):
    if out.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    if out.numel() > _INT32_MAX:
        return False
    if out.shape != input.shape:
        return False
    # A uniform fill writes the same value to every element, so any dense
    # non-overlapping layout can be filled in flat memory order.
    return torch.ops.aten.is_non_overlapping_and_dense(out)


def _fill_scalar_flat(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return
    store_target = _flat_store_target(out)
    if out.dtype == torch.bool:
        value = 1 if value else 0
    _launch_flat(fill_scalar_flat_kernel, store_target, (value,), n_elements)


def _fill_tensor_flat_eligible_fresh(input, out, value):
    if not _fill_scalar_flat_eligible_fresh(input, out):
        return False
    if value.dtype in _FLAT_UNSAFE_DTYPES or value.dtype != out.dtype:
        return False
    return True


def _fill_tensor_flat_eligible_into(input, out, value):
    if out.dtype in _FLAT_UNSAFE_DTYPES or value.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    if value.dtype != out.dtype:
        return False
    if out.numel() > _INT32_MAX:
        return False
    if out.shape != input.shape:
        return False
    return torch.ops.aten.is_non_overlapping_and_dense(out)


def _fill_tensor_flat(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return
    store_target = _flat_store_target(out)
    if out.dtype == torch.bool:
        value = value.view(torch.int8)
    _launch_flat(fill_tensor_flat_kernel, store_target, (value,), n_elements)


def _fill_scalar_impl(input, value):
    logger.debug("GEMS_ENFLAME FILL")
    out = torch.empty_like(input)
    if _fill_scalar_flat_eligible_fresh(input, out):
        _fill_scalar_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def _fill_tensor_impl(input, value):
    if not value.is_cuda:
        return _fill_scalar_impl(input, value.item())
    logger.debug("GEMS_ENFLAME FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    if _fill_tensor_flat_eligible_fresh(input, out, value):
        _fill_tensor_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    if _fill_scalar_flat_eligible_into(input, out):
        _fill_scalar_flat(out, value)
        return out
    if _fill_strided(out, value, True):
        return out
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _fill_tensor_flat_eligible_into(input, out, value):
        _fill_tensor_flat(out, value)
        return out
    if _fill_strided(out, value, False):
        return out
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS_ENFLAME FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _fill_tensor_flat_eligible_into(self, self, value):
        _fill_tensor_flat(self, value)
        return self
    if _fill_strided(self, value, False):
        return self
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


def fill_scalar_(self, value=0):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_")
    if _fill_scalar_flat_eligible_into(self, self):
        _fill_scalar_flat(self, value)
        return self
    if _fill_strided(self, value, True):
        return self
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self


# ---------------------------------------------------------------------------
# Autograd bridge
#
# ``fill`` allocates its output, so a requiring-grad input would silently lose
# its graph and the backward tests (plain, mixed-dtype value gradient and the
# double backward) would fail.  The gradient w.r.t. the input is exactly zero
# and the gradient w.r.t. the value is the sum of the upstream gradient, which
# is what the native formula produces.  Both halves are produced by this
# backend: the zero gradient by the tuned flat write kernel below, and the
# value gradient by the backend ``sum`` kernel instead of the shared one.
from flag_gems.runtime.backend._enflame.gcu400.ops.sum import (  # noqa: E402
    sum as _backend_sum,
)


def _zero_grad(input):
    """Constant (zero) gradient, written by the tuned flat write kernel.

    This is the zero element of the gradient, not the operator's math, but
    ``torch.zeros_like`` under gems dispatch runs the shared ``zeros_like``
    kernel, which caps the grid at 24 programs and launches a single warp
    (~1733 GB/s measured on a 16 Mi-element fp16 gradient, 19.4us against
    15.1us for the flat write).  The flat write kernel in this file is already
    tuned against the DRAM write wall (``_flat_block`` / ``_FLAT_MAX_GRID``), so
    reusing it keeps this half of the backward a pure bandwidth stream.  64-bit
    / complex element stores are unreliable on this backend, so those dtypes
    keep the native allocation.
    """
    out = torch.empty_like(input)
    if _fill_scalar_flat_eligible_fresh(input, out):
        _fill_scalar_flat(out, 0.0)
        return out
    return torch.zeros_like(input)


class _ValueGradSum(torch.autograd.Function):
    """Full reduction of the upstream gradient, kept differentiable.

    The shared bridge reduces with ``flag_gems.ops.sum``: 275.8us / 122 GB/s on
    a 16 Mi-element fp16 gradient, i.e. the whole fill backward, against 25.0us
    / 1340 GB/s for the backend ``sum`` kernel that also serves ``aten::sum``.
    The autograd wrapper is preserved so the double-backward path still goes
    through ``_FillTensor``.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.shape = input.shape
        ctx.dtype = input.dtype
        ctx.device = input.device
        return _backend_sum(input)

    @staticmethod
    def backward(ctx, grad_output):
        empty = torch.empty(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        return _FillTensor.apply(empty, grad_output)


def _value_gradient(grad_output, value_device):
    grad_value = (
        _ValueGradSum.apply(grad_output)
        if torch.is_grad_enabled()
        else _backend_sum(grad_output)
    )
    if grad_value.device != value_device:
        grad_value = grad_value.to(value_device)
    return grad_value


class _FillScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value):
        return _fill_scalar_impl(input, value)

    @staticmethod
    def backward(ctx, grad_output):
        return _zero_grad(grad_output), None


class _FillTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value):
        ctx.value_device = value.device
        return _fill_tensor_impl(input, value)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[1]:
            # Nothing differentiates w.r.t. the value tensor, so the reduction
            # that builds its gradient would only cost a launch (~100us from
            # the engine thread on gcu400).
            return _zero_grad(grad_output), None
        return _zero_grad(grad_output), _value_gradient(
            grad_output, ctx.value_device
        )


def fill_scalar(input, value):
    logger.debug("GEMS_ENFLAME FILL")
    if torch.is_grad_enabled() and input.requires_grad:
        return _FillScalar.apply(input, value)
    return _fill_scalar_impl(input, value)


def fill_tensor(input, value):
    logger.debug("GEMS_ENFLAME FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    # The grad check has to come first: the host-value shortcut below reads the
    # value through ``.item()``, which would drop its graph edge (the mixed
    # dtype and double-backward tests differentiate w.r.t. that tensor).
    if torch.is_grad_enabled() and (input.requires_grad or value.requires_grad):
        return _FillTensor.apply(input, value)
    if not value.is_cuda:
        return _fill_scalar_impl(input, value.item())
    return _fill_tensor_impl(input, value)
