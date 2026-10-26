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
from triton.runtime import driver

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _fmin_ignore_nan(x, y):
    # torch.fmin semantics (ignore NaN) via a single branch-free select:
    #     fmin(x, y) = (x <= y) | isnan(y) ? x : y
    # Only the y operand needs a NaN test: an ordered x<=y already yields y when
    # x is NaN, and isnan(y) covers the cases where y is NaN. This avoids a
    # second select and (for bf16) an extra widening.
    #
    # NaN is detected via integer bit patterns (exponent all-ones, non-zero
    # mantissa) because the XPU backend cannot do unordered fcmp (isnan /
    # y != y) on wide vectors.
    if x.dtype == tl.float32:
        yi = y.to(tl.int32, bitcast=True)
        ynan = (yi & 0x7FFFFFFF) > 0x7F800000
    elif x.dtype == tl.float16:
        yi = y.to(tl.int16, bitcast=True)
        ynan = (yi & 0x7FFF) > 0x7C00
    elif x.dtype == tl.bfloat16:
        # bf16 -> int16 bitcast is unsupported; detect NaN via exact fp32 widen.
        yi = y.to(tl.float32).to(tl.int32, bitcast=True)
        ynan = (yi & 0x7FFFFFFF) > 0x7F800000
    else:
        return tl.minimum(x, y)
    return tl.where((x <= y) | ynan, x, y)


@triton.jit
def fmin_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        # Masked path is costlier; only used for tail / small shapes where
        # n_elements is not a multiple of BLOCK_SIZE.
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = _fmin_ignore_nan(x, y)
        tl.store(out_ptr + offsets, out, mask=mask)
    else:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        out = _fmin_ignore_nan(x, y)
        tl.store(out_ptr + offsets, out)


# =============================================================================
# Small-shape fast path (flat_launcher).
#
# For tiny shapes the generic `fn[grid]` launch re-specializes / reloads the
# compiled kernel per call on this backend, dominating runtime. Binding the
# compiled kernel once via `driver.active.flat_launchers` and replaying it by
# ABI skips that per-call reload.
#
# Only small, contiguous, same-shape/same-float-dtype calls take this path;
# broadcast / dtype-promotion / large shapes fall through to the already-tuned
# generic path.
# =============================================================================
_FLAT_MISS = object()
_FLAT_LAUNCHERS = _FLAT_MISS
_FLAT_MAX_NUMEL = 65536  # above this the generic path's tile/cta heuristic wins
_FLAT_BLOCK = 4096  # one tile per program; [64,64] is a single program


def _flat_launchers():
    """Resolve `driver.active.flat_launchers` once (`driver.active` is a lazy
    proxy, so repeated attribute walks are not free)."""
    global _FLAT_LAUNCHERS
    if _FLAT_LAUNCHERS is _FLAT_MISS:
        _FLAT_LAUNCHERS = getattr(driver.active, "flat_launchers", None)
    return _FLAT_LAUNCHERS


@triton.jit(
    # Keep the compiled kernel independent of the keys below; otherwise a
    # pointer's alignment class or n_elements' divisibility would send the first
    # call through a different specialization than the bound launcher.
    do_not_specialize=["n_elements"],
    do_not_specialize_on_alignment=["x_ptr", "y_ptr", "out_ptr"],
)
def _fmin_flat_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Masked store always on, so an n_elements that BLOCK does not divide is
    # served by the same binary (small shape -> mask cost negligible).
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, _fmin_ignore_nan(x, y), mask=mask)


def _flat_fmin(a, b, out, out_provided):
    """Launch `_fmin_flat_kernel` for the cases it covers; None otherwise.

    Key = (numel, dtype, BLOCK): numel+BLOCK fix the grid (part of compilation
    on XPU) and dtype fixes the signature; complete because this is the whole
    kernel (one flat extent, no stride/broadcast) and pointers are excluded from
    specialization (`do_not_specialize*`).
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return None  # broadcast / dtype promotion: handled by generic path
    if a.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return None
    if not (a.is_contiguous() and b.is_contiguous()):
        return None
    numel = a.numel()
    if numel == 0 or numel > _FLAT_MAX_NUMEL:
        return None
    if a.device != b.device or a.device.index != torch_device_fn.current_device():
        return None
    launchers = _flat_launchers()
    if launchers is None:  # triton without the launcher cache: correct, just slower
        return None
    if out_provided:
        if (
            out.shape != a.shape
            or out.dtype != a.dtype
            or not out.is_contiguous()
            or out.device != a.device
        ):
            return None
    else:
        out = torch.empty(a.shape, dtype=a.dtype, device=a.device)

    key = (numel, a.dtype, _FLAT_BLOCK)
    grid = (-(-numel // _FLAT_BLOCK),)
    launch, stream = launchers.acquire(_fmin_flat_kernel, key)
    if launch is None:  # first call for this key: compile, bind, replay next time
        kernel = _fmin_flat_kernel[grid](a, b, out, numel, BLOCK=_FLAT_BLOCK)
        launchers.bind(_fmin_flat_kernel, key, kernel, grid)
        return out
    launch(stream, a, b, out, numel)
    return out


def _to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
        if device is not None and t.device != device:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t
    return torch.tensor(x, device=device, dtype=dtype)


def _prepare_inputs(a, b, out=None):
    dev = None
    if isinstance(out, torch.Tensor):
        dev = out.device
    else:
        if isinstance(a, torch.Tensor):
            dev = a.device
        if isinstance(b, torch.Tensor):
            dev = b.device if dev is None else dev
    if dev is None:
        dev = torch.device("cuda")
    a = _to_tensor(a, device=dev)
    b = _to_tensor(b, device=dev)
    a_b, b_b = torch.broadcast_tensors(a, b)
    out_dtype = torch.result_type(a_b, b_b)
    if out_dtype.is_complex:
        raise TypeError("fmin does not support complex dtypes.")
    compute_dtype = torch.int8 if out_dtype == torch.bool else out_dtype
    a_c = a_b.to(compute_dtype).contiguous()
    b_c = b_b.to(compute_dtype).contiguous()
    return a_c, b_c, out_dtype, compute_dtype


def _pick_block_size(n_elements):
    # Memory-bound pointwise (2 loads + 1 store). BLOCK=16384 is the sweet spot
    # for large shapes; a much wider tile regressed fp32/bf16. Small shapes use
    # more, smaller programs since they are host-bound. The launch picks the
    # no-mask kernel whenever n is a multiple of BLOCK (masked path is costlier).
    if n_elements >= 32768:
        return 16384, 8
    return 2048, 4


def fmin(a, b):
    logger.debug("GEMS_KUNLUNXIN FMIN")
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        flat = _flat_fmin(a, b, None, False)
        if flat is not None:
            return flat
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=None)
    out_shape = a_c.shape
    if compute_dtype == out_dtype:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = out
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = torch.empty(out_shape, dtype=compute_dtype, device=a_c.device)
    n_elements = out_c.numel()
    block_size, num_warps = _pick_block_size(n_elements)
    need_mask = 1 if (n_elements % block_size) else 0
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(a_c.device):
        fmin_kernel[grid](
            a_c,
            b_c,
            out_c,
            n_elements,
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
            num_warps=num_warps,
        )
    if out_c.dtype != out.dtype:
        out.copy_(out_c.to(out_dtype))
    return out


def fmin_out(a, b, out):
    logger.debug("GEMS_KUNLUNXIN FMIN_OUT")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        flat = _flat_fmin(a, b, out, True)
        if flat is not None:
            return flat
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=out)
    expected_shape = a_c.shape
    if out.device != a_c.device:
        raise ValueError("out tensor must be on the same device as inputs.")
    if out.dtype != out_dtype:
        raise TypeError(f"out tensor has dtype {out.dtype}, expected {out_dtype}.")
    if tuple(out.shape) != tuple(expected_shape):
        raise ValueError(
            f"out tensor has shape {tuple(out.shape)}, expected {tuple(expected_shape)} after broadcasting."
        )
    if compute_dtype == out_dtype and out.is_contiguous():
        out_c = out
    else:
        out_c = torch.empty(expected_shape, dtype=compute_dtype, device=out.device)
    n_elements = out_c.numel()
    block_size, num_warps = _pick_block_size(n_elements)
    need_mask = 1 if (n_elements % block_size) else 0
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        fmin_kernel[grid](
            a_c,
            b_c,
            out_c,
            n_elements,
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
            num_warps=num_warps,
        )
    if out_c is not out:
        if out_c.dtype != out.dtype:
            out.copy_(out_c.to(out.dtype))
        else:
            if out.is_contiguous():
                out.copy_(out_c)
            else:
                out.view_as(out.contiguous()).copy_(out_c)
    return out
