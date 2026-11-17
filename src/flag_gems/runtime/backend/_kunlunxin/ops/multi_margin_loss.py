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
"""Kunlunxin XPU3 override for the ``multi_margin_loss`` out-variants.

The generic kernels reduce inside a *runtime* class loop
(``for class_start in range(0, C, BLOCK_C): ... tl.sum(...)``), which
TritonXPU refuses to legalise:

    failed to legalize operation 'tt.reduce' that was explicitly marked illegal
    Pipeline failed while executing [`ConvertTritonXPUToLLVM`]

so ``multi_margin_loss.out`` and ``multi_margin_loss_backward.grad_input`` are
unusable on this backend.

This override keeps the exact generic mathematics (per-row weight ``w[target]``,
``z = margin - x[target] + x[c]``, ``p`` in {1, 2}, none/mean/sum reductions) but
reshapes every kernel to the forms already proven on XPU3:

* **one program per row, pure 1-D tile** -- no 2-D row blocks (they silently
  mis-store rows on this backend);
* **at most one ``tl.sum`` per kernel, outside any loop** -- when a problem is
  wider than one tile the blocks are folded with an *element-wise vector
  accumulator* and reduced once at the end;
* the out-of-range target check is done by a small synchronous kernel (cached
  per target tensor) instead of ``tl.device_assert``, which does not combine
  with the loop-carried accumulation here.

Every masked load/store clamps its addresses into the allocation, and the
``_out`` contract (``result is out``, resize of an empty ``out``, non-contiguous
``out``) is preserved.
"""

import logging
import sys
import weakref

import torch
import triton
import triton.language as tl

from flag_gems.ops.multi_margin_loss import (
    _check_grad_output,
    _check_inputs,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# One tile must cover a whole row (or the whole row-loss vector) so the only
# reduction sits outside any loop.  Wider problems use a 128-lane loop, the
# width the toolchain's unroll control allows inside a dynamic loop.
_MAX_SINGLE_TILE = 8192
_LOOP_BLOCK = 128
_TARGET_VALIDATION_CACHE_LIMIT = 128
_TARGET_VALIDATION_CACHE = {}


@triton.jit(do_not_specialize=["N", "C"])
def _validate_target_kernel(Target, Invalid, N, C, BLOCK: tl.constexpr):
    """Count out-of-range targets; vector accumulate + one reduce."""
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        target = tl.load(Target + tl.minimum(offs, N - 1), mask=mask, other=0)
        acc += (mask & ((target < 0) | (target >= C))).to(tl.int32)
    tl.store(Invalid, tl.sum(acc, axis=0))


@triton.jit(do_not_specialize=["C", "margin"])
def _row_loss_kernel(
    Input,
    Target,
    Weight,
    Out,
    C,
    margin,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: ``out[row] = w[t] * sum_c z^p / C`` (z>0, c != t)."""
    acc_dtype = tl.float64 if Input.dtype.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    row = tl.program_id(0)
    target = tl.load(Target + row)
    valid = (target >= 0) & (target < C)
    safe = tl.where(valid, target, 0)
    row_off = row * C
    target_value = tl.load(Input + row_off + safe, mask=valid, other=0.0).to(acc_dtype)

    offs = tl.arange(0, BLOCK)
    cmask = offs < C
    values = tl.load(
        Input + row_off + tl.minimum(offs, C - 1), mask=cmask, other=0.0
    ).to(acc_dtype)
    z = margin - target_value + values
    active = cmask & valid & (offs != safe) & (z > 0)
    term = tl.where(active, z, 0.0)
    if P == 2:
        term = term * term
    loss_sum = tl.sum(term, axis=0)

    if HAS_WEIGHT:
        target_weight = tl.load(Weight + safe, mask=valid, other=0.0).to(acc_dtype)
    else:
        target_weight = tl.full((), 1.0, acc_dtype)
    tl.store(Out + row, (target_weight * loss_sum / C.to(acc_dtype)).to(Out.dtype.element_ty))


@triton.jit(do_not_specialize=["C", "margin"])
def _row_loss_loop_kernel(
    Input,
    Target,
    Weight,
    Out,
    C,
    margin,
    P: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``C`` wider than one tile: vector-accumulate class blocks, one reduce."""
    acc_dtype = tl.float64 if Input.dtype.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    row = tl.program_id(0)
    target = tl.load(Target + row)
    valid = (target >= 0) & (target < C)
    safe = tl.where(valid, target, 0)
    row_off = row * C
    target_value = tl.load(Input + row_off + safe, mask=valid, other=0.0).to(acc_dtype)

    acc = tl.zeros((BLOCK,), dtype=acc_dtype)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        cmask = offs < C
        values = tl.load(
            Input + row_off + tl.minimum(offs, C - 1), mask=cmask, other=0.0
        ).to(acc_dtype)
        z = margin - target_value + values
        active = cmask & valid & (offs != safe) & (z > 0)
        term = tl.where(active, z, 0.0)
        if P == 2:
            term = term * term
        acc += term
    loss_sum = tl.sum(acc, axis=0)

    if HAS_WEIGHT:
        target_weight = tl.load(Weight + safe, mask=valid, other=0.0).to(acc_dtype)
    else:
        target_weight = tl.full((), 1.0, acc_dtype)
    tl.store(
        Out + row,
        (target_weight * loss_sum / C.to(acc_dtype)).to(Out.dtype.element_ty),
    )


@triton.jit(do_not_specialize=["N", "C", "margin"])
def _row_grad_kernel(
    GradOut,
    Input,
    Target,
    Weight,
    GradIn,
    N,
    C,
    margin,
    P: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: per-class grad, ``grad[target] = -sum``."""
    acc_dtype = tl.float64 if Input.dtype.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    row = tl.program_id(0)
    target = tl.load(Target + row)
    valid = (target >= 0) & (target < C)
    safe = tl.where(valid, target, 0)
    row_off = row * C
    target_value = tl.load(Input + row_off + safe, mask=valid, other=0.0).to(acc_dtype)

    if HAS_WEIGHT:
        target_weight = tl.load(Weight + safe, mask=valid, other=0.0).to(acc_dtype)
    else:
        target_weight = tl.full((), 1.0, acc_dtype)

    if REDUCTION == 0:
        grad_output = tl.load(GradOut + row).to(acc_dtype)
    else:
        grad_output = tl.load(GradOut).to(acc_dtype)

    scale = grad_output * target_weight / C.to(acc_dtype)
    if REDUCTION == 1:
        scale /= N.to(acc_dtype)

    offs = tl.arange(0, BLOCK)
    cmask = offs < C
    values = tl.load(
        Input + row_off + tl.minimum(offs, C - 1), mask=cmask, other=0.0
    ).to(acc_dtype)
    z = margin - target_value + values
    active = cmask & valid & (offs != safe) & (z > 0)
    if P == 1:
        grad = tl.where(active, scale, 0.0)
    else:
        grad = tl.where(active, 2.0 * z * scale, 0.0)
    target_grad = tl.zeros((), dtype=acc_dtype) - tl.sum(grad, axis=0)
    grad = tl.where(cmask & (offs == safe), target_grad, grad)
    tl.store(
        GradIn + row_off + tl.minimum(offs, C - 1),
        grad.to(GradIn.dtype.element_ty),
        mask=cmask,
    )


@triton.jit(do_not_specialize=["N", "C", "margin"])
def _row_grad_loop_kernel(
    GradOut,
    Input,
    Target,
    Weight,
    GradIn,
    N,
    C,
    margin,
    P: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``C`` wider than one tile: vector-accumulate, one reduce, one store each.

    The per-class store happens per block (disjoint addresses); the target entry
    is written after the loop.
    """
    acc_dtype = tl.float64 if Input.dtype.element_ty == tl.float64 else tl.float32
    margin = margin.to(acc_dtype)
    row = tl.program_id(0)
    target = tl.load(Target + row)
    valid = (target >= 0) & (target < C)
    safe = tl.where(valid, target, 0)
    row_off = row * C
    target_value = tl.load(Input + row_off + safe, mask=valid, other=0.0).to(acc_dtype)

    if HAS_WEIGHT:
        target_weight = tl.load(Weight + safe, mask=valid, other=0.0).to(acc_dtype)
    else:
        target_weight = tl.full((), 1.0, acc_dtype)

    if REDUCTION == 0:
        grad_output = tl.load(GradOut + row).to(acc_dtype)
    else:
        grad_output = tl.load(GradOut).to(acc_dtype)

    scale = grad_output * target_weight / C.to(acc_dtype)
    if REDUCTION == 1:
        scale /= N.to(acc_dtype)

    acc = tl.zeros((BLOCK,), dtype=acc_dtype)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        cmask = offs < C
        values = tl.load(
            Input + row_off + tl.minimum(offs, C - 1), mask=cmask, other=0.0
        ).to(acc_dtype)
        z = margin - target_value + values
        active = cmask & valid & (offs != safe) & (z > 0)
        if P == 1:
            grad = tl.where(active, scale, 0.0)
        else:
            grad = tl.where(active, 2.0 * z * scale, 0.0)
        acc += grad
        tl.store(
            GradIn + row_off + tl.minimum(offs, C - 1),
            grad.to(GradIn.dtype.element_ty),
            mask=cmask,
        )
    target_grad = tl.zeros((), dtype=acc_dtype) - tl.sum(acc, axis=0)
    tl.store(
        GradIn + row_off + safe,
        target_grad.to(GradIn.dtype.element_ty),
        mask=valid,
    )


@triton.jit(do_not_specialize=["N"])
def _reduce_rows_kernel(RowLoss, Out, N, REDUCTION: tl.constexpr, BLOCK: tl.constexpr):
    """Sum (or mean) row losses: one load, one reduce, one store."""
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    values = tl.load(RowLoss + tl.minimum(offs, N - 1), mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    if REDUCTION == 1:
        total = total / N.to(tl.float32)
    tl.store(Out, total.to(Out.dtype.element_ty))


@triton.jit(do_not_specialize=["N"])
def _reduce_rows_loop_kernel(
    RowLoss, Out, N, REDUCTION: tl.constexpr, BLOCK: tl.constexpr
):
    """``N`` wider than one tile: vector-accumulate rows, one reduce."""
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        acc += tl.load(RowLoss + tl.minimum(offs, N - 1), mask=mask, other=0.0)
    total = tl.sum(acc, axis=0)
    if REDUCTION == 1:
        total = total / N.to(tl.float32)
    tl.store(Out, total.to(Out.dtype.element_ty))


@triton.jit
def _fill_scalar_kernel(Out, NAN: tl.constexpr):
    value = 0.0
    if NAN:
        value = float("nan")
    tl.store(Out, tl.full((), value, Out.dtype.element_ty))


def _target_validation_key(target, N, C):
    return (
        target.device.type,
        target.device.index,
        target.data_ptr(),
        target.numel(),
        target._version,
        N,
        C,
    )


def _validate_target_range(target, N, C):
    """Synchronous out-of-range target check, cached per (pointer, version)."""
    if N == 0:
        return
    key = _target_validation_key(target, N, C)
    cached = _TARGET_VALIDATION_CACHE.get(key)
    if cached is not None and cached() is target:
        return
    invalid = torch.empty((), dtype=torch.int32, device=target.device)
    block = min(_LOOP_BLOCK, triton.next_power_of_2(N))
    with torch_device_fn.device(target.device):
        _validate_target_kernel[(1,)](target, invalid, N, C, BLOCK=block)
    if invalid.item() != 0:
        raise RuntimeError("multi_margin_loss: target index out of bounds")
    if len(_TARGET_VALIDATION_CACHE) >= _TARGET_VALIDATION_CACHE_LIMIT:
        _TARGET_VALIDATION_CACHE.clear()
    _TARGET_VALIDATION_CACHE[key] = weakref.ref(target)


def _launch_row_loss(input, target, weight, N, C, p, margin, out):
    """Per-row losses -> contiguous ``out`` of shape (N,)."""
    weight_ptr = input if weight is None else weight
    with torch_device_fn.device(input.device):
        if C <= _MAX_SINGLE_TILE:
            _row_loss_kernel[(N,)](
                input,
                target,
                weight_ptr,
                out,
                C,
                margin,
                P=p,
                HAS_WEIGHT=weight is not None,
                BLOCK=triton.next_power_of_2(C),
            )
        else:
            _row_loss_loop_kernel[(N,)](
                input,
                target,
                weight_ptr,
                out,
                C,
                margin,
                P=p,
                HAS_WEIGHT=weight is not None,
                BLOCK=_LOOP_BLOCK,
            )


def _forward(input, target, weight, N, C, is_batched, p, margin, reduction, out):
    """Fill ``out`` (shape/contiguity contract already established)."""
    if N == 0:
        if reduction == 0:
            return out
        with torch_device_fn.device(input.device):
            _fill_scalar_kernel[(1,)](out, reduction == 1)
        return out

    if reduction == 0:
        if out.is_contiguous():
            _launch_row_loss(input, target, weight, N, C, p, margin, out)
            return out
        # Non-contiguous 1-D ``out``: compute contiguous, then scatter rows.
        row_losses = torch.empty((N,), dtype=out.dtype, device=input.device)
        _launch_row_loss(input, target, weight, N, C, p, margin, row_losses)
        _copy_strided(row_losses, out)
        return out

    row_losses = torch.empty((N,), dtype=torch.float32, device=input.device)
    _launch_row_loss(input, target, weight, N, C, p, margin, row_losses)
    red = 1 if reduction == 1 else 2
    with torch_device_fn.device(input.device):
        if N <= _MAX_SINGLE_TILE:
            _reduce_rows_kernel[(1,)](
                row_losses,
                out,
                N,
                REDUCTION=red,
                BLOCK=triton.next_power_of_2(N),
            )
        else:
            _reduce_rows_loop_kernel[(1,)](
                row_losses, out, N, REDUCTION=red, BLOCK=_LOOP_BLOCK
            )
    return out


@triton.jit(do_not_specialize=["N"])
def _copy_row_kernel(Src, Dst, N, DST_STRIDE, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    value = tl.load(Src + tl.minimum(offs, N - 1), mask=mask, other=0.0)
    tl.store(Dst + tl.minimum(offs, N - 1) * DST_STRIDE, value, mask=mask)


def _copy_strided(src, dst):
    """Row-wise copy into a non-contiguous 1-D destination."""
    n = src.numel()
    if n == 0:
        return
    block = 256
    with torch_device_fn.device(src.device):
        _copy_row_kernel[(triton.cdiv(n, block),)](
            src, dst, n, dst.stride(0), BLOCK=block
        )


def multi_margin_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    """Functional forward (no ``out``); reuses the same XPU-safe kernels.

    The generic forward is not covered by the ``*_out`` override, and its
    kernel contains ``tl.device_assert`` which lowers to ``__assert_fail`` and
    fails at link time on XPU3 (wiki
    ``pattern-xpu3-device-assert-unlinkable``).  The marker's
    ``forward_backward_consistency`` case calls this entry directly, so it
    needs the same implementation.
    """
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    _validate_target_range(target, N, C)
    shape = _output_shape(N, is_batched, reduction)
    out = torch.empty(shape, dtype=input.dtype, device=input.device)
    return _forward(input, target, weight, N, C, is_batched, p, margin, reduction, out)


def multi_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS_OUT")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    if out.device != input.device:
        raise RuntimeError("multi_margin_loss.out: out must be on the input device")
    if out.dtype != input.dtype:
        raise RuntimeError("multi_margin_loss.out: out must have the input dtype")
    _validate_target_range(target, N, C)

    shape = _output_shape(N, is_batched, reduction)
    if tuple(out.shape) != shape:
        out.resize_(shape)
    return _forward(input, target, weight, N, C, is_batched, p, margin, reduction, out)


def _launch_backward(grad_output, input, target, weight, N, C, p, margin, reduction, out):
    if N == 0:
        return out
    weight_ptr = input if weight is None else weight
    with torch_device_fn.device(input.device):
        if C <= _MAX_SINGLE_TILE:
            _row_grad_kernel[(N,)](
                grad_output,
                input,
                target,
                weight_ptr,
                out,
                N,
                C,
                margin,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=weight is not None,
                BLOCK=triton.next_power_of_2(C),
            )
        else:
            _row_grad_loop_kernel[(N,)](
                grad_output,
                input,
                target,
                weight_ptr,
                out,
                N,
                C,
                margin,
                P=p,
                REDUCTION=reduction,
                HAS_WEIGHT=weight is not None,
                BLOCK=_LOOP_BLOCK,
            )
    return out


def multi_margin_loss_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS_BACKWARD_OUT")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    if grad_input.device != input.device:
        raise RuntimeError(
            "multi_margin_loss_backward.grad_input: output must be on the input device"
        )
    if grad_input.dtype != input.dtype:
        raise RuntimeError(
            "multi_margin_loss_backward.grad_input: output must have the input dtype"
        )
    if tuple(grad_input.shape) != tuple(input.shape):
        grad_input.resize_(input.shape)
    _validate_target_range(target, N, C)

    if not grad_input.is_contiguous():
        temp = torch.empty_like(input)
        _launch_backward(
            grad_output, input, target, weight, N, C, p, margin, reduction, temp
        )
        _copy_back(grad_input, temp)
        return grad_input
    return _launch_backward(
        grad_output, input, target, weight, N, C, p, margin, reduction, grad_input
    )


def _copy_back(dst, src):
    """Element-wise copy into a non-contiguous 2-D destination."""
    if dst.numel() == 0:
        return
    rows = dst.shape[0] if dst.dim() > 1 else 1
    cols = dst.numel() // rows
    block = 256
    with torch_device_fn.device(dst.device):
        _copy_2d_kernel[(triton.cdiv(cols, block), rows)](
            src,
            dst,
            rows,
            cols,
            src.stride(0) if dst.dim() > 1 else cols,
            dst.stride(0),
            dst.stride(1) if dst.dim() > 1 else 1,
            BLOCK=block,
        )


@triton.jit(do_not_specialize=["ROWS", "COLS", "SRC_ROW", "DST_ROW", "DST_COL"])
def _copy_2d_kernel(
    Src,
    Dst,
    ROWS,
    COLS,
    SRC_ROW,
    DST_ROW,
    DST_COL,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < COLS
    safe = tl.minimum(offs, COLS - 1)
    value = tl.load(
        Src + row * SRC_ROW + safe, mask=mask, other=0.0
    )
    tl.store(Dst + row * DST_ROW + safe * DST_COL, value, mask=mask)


def multi_margin_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    p,
    margin,
    weight=None,
    reduction=1,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN MULTI_MARGIN_LOSS_BACKWARD")
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    grad_output = _check_grad_output(grad_output, input, N, is_batched, reduction)
    _validate_target_range(target, N, C)
    grad_input = torch.empty_like(input)
    _launch_backward(
        grad_output, input, target, weight, N, C, p, margin, reduction, grad_input
    )
    return grad_input


# The marker test (and the platform benchmark) bind their candidate from the
# *generic* module path (`flag_gems.ops.multi_margin_loss_out`) rather than
# `flag_gems.multi_margin_loss_out`.  The generic kernels cannot lower on XPU3
# (`tt.reduce`), so rebind the generic module attributes to this vendor
# implementation -- same pattern as `_kunlunxin/ops/_batch_norm_impl_index.py`
# and `_enflame/gcu300/ops/div.py`.
_VENDOR_OPS = {
    "multi_margin_loss": multi_margin_loss,
    "multi_margin_loss_out": multi_margin_loss_out,
    "multi_margin_loss_backward_out": multi_margin_loss_backward_out,
    "multi_margin_loss_backward": multi_margin_loss_backward,
}
# Both the package attribute (`flag_gems.ops.<name>`, used by the marker test)
# and the submodule attribute (`flag_gems.ops.multi_margin_loss.<name>`, used by
# `from flag_gems.ops.multi_margin_loss import ...` in the platform benchmark)
# have to resolve to this module.
for _mod_name in ("flag_gems.ops", "flag_gems.ops.multi_margin_loss"):
    _mod = sys.modules.get(_mod_name)
    if _mod is not None:
        for _name, _fn in _VENDOR_OPS.items():
            setattr(_mod, _name, _fn)
