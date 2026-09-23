import logging
import math

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.as_strided_copy import _launch_as_strided_copy
from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

from ..utils.pointwise_dynamic import pointwise_dynamic
from .expand_copy import _launch_bcast

logger = logging.getLogger(__name__)

_PROMOTION = ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT

_GATE = 1_048_576

# `pointwise_dynamic` derives its 1-D tile as ``next_power_of_2(numel // 12)`` and
# specialises the generated kernel on ``numel`` (``num_tasks`` is a
# ``tl.constexpr``), so its compile cost grows with the tensor size (one compile
# per distinct numel).  That is accepted for the main fast path -- measured 4.4x
# faster than the raw kernels at 16.7M elements -- but a *broadcast* call would
# additionally have to go through the general ndim codegen, so an oversized
# broadcast is materialised instead and then computed with the 1-D
# `pointwise_dynamic` kernel (measured 2.0 s for x(1,) vs y(2**28,)).
_BROADCAST_MATERIALIZE_MAX = 12_582_912  # -> pw 1-D tile <= 1 << 20

# The flat broadcast-gather kernel behind ``_launch_bcast`` addresses six
# dimensions; a deeper broadcast layout falls back to the rank-general gems
# copy kernel instead.
_MAX_FLAT_DIM = 6

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def _xlogy_fast(x, y):
    # Same precedence as the raw kernels / ATen:
    #   NaN if y is NaN; 0 if x == 0; otherwise x * log(y).
    # This path is also the broadcast / non-contiguous-out fallback, so it must
    # be bit-exact for every input, not only for the positive-y benchmark data
    # (x == 0 with y <= 0 / y == +-inf otherwise returns NaN instead of 0).
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    prod = x_f32 * tl.log(1.0000000000000000 * y_f32)
    res = tl.where(x_f32 == 0.0, 0.0, prod)
    y_bits = y_f32.to(tl.int32, bitcast=True)
    y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
    return tl.where(y_nan, float("nan"), res)


MIN_BLOCK = 2048
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False
_MASKED_FALLBACK_BLOCK = 8192


def _pick_block(n_elements):
    # NOTE on reachability: the tensor x tensor entry point (`_launch`) never
    # reaches the two >= 1M branches below, because it early-returns into
    # `pointwise_dynamic` at n >= _GATE.  The two *scalar* launchers do reach
    # them: `_launch_tensor_scalar` / `_launch_scalar_tensor` only short-circuit
    # for a 1-element Tensor operand and then hand any n to this function, so
    # `torch.xlogy(tensor, scalar)` with n >= 1M lowers to the plain
    # masked/unmasked scalar kernels here.  Dropping these branches therefore
    # silently regressed those two paths (measured on XPU: fp32 n=16.7M
    # 0.078 -> 0.131 ms, fp16 n=2**30 2.51 -> 7.54 ms), so they are kept.
    #
    # On the >= 65536-lane tiles: the "32768 is a known-bad tile width" note
    # that the `trunc` override carries does NOT reproduce for this operator --
    # it is a different codegen path (masked/unmasked scalar body, not trunc's).
    # Measured here, 65536/num_warps=16 is the fastest and 32768 is the
    # second-best tile (see solution/xlogy/README.md section 3/9), so those
    # widths are deliberately used for the large-n cells.
    if n_elements >= 16_777_216 and n_elements % 65536 == 0:
        return 65536, 16, False
    if n_elements >= 1_048_576:
        for tile in (32768, 16384, 8192, 4096, 2048):
            if n_elements % tile == 0:
                return tile, 4, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return MIN_BLOCK, 4, True
    return _MASKED_FALLBACK_BLOCK, 4, True


@triton.jit
def xlogy_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offset, mask=mask, other=1).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_kernel_unmasked(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr + offset).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


@triton.jit
def xlogy_tensor_scalar_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    y_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = y_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_tensor_scalar_kernel_unmasked(
    x_ptr,
    out_ptr,
    y_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = y_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


@triton.jit
def xlogy_tensor_scalar_ptr_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_tensor_scalar_ptr_kernel_unmasked(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


@triton.jit
def xlogy_scalar_tensor_kernel(
    y_ptr,
    out_ptr,
    n_elements,
    x_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    y = tl.load(y_ptr + offset, mask=mask, other=1).to(tl.float32)
    x = x_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_scalar_tensor_kernel_unmasked(
    y_ptr,
    out_ptr,
    x_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.load(y_ptr + offset).to(tl.float32)
    x = x_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


@triton.jit
def xlogy_scalar_tensor_ptr_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    y = tl.load(y_ptr + offset, mask=mask, other=1).to(tl.float32)
    x = tl.load(x_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_scalar_tensor_ptr_kernel_unmasked(
    y_ptr,
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.load(y_ptr + offset).to(tl.float32)
    x = tl.load(x_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


def _exact_tensor_scalar(n_elements, y_val):
    return n_elements < _GATE or not (0.0 < y_val and math.isfinite(y_val))


def _exact_scalar_tensor(n_elements, x_val):
    return n_elements < _GATE or x_val == 0.0


def _launch(x, y, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    if n_elements >= _GATE:
        _xlogy_fast(x, y, out0=out)
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    # The tensor-tensor body has to be exact for every size: the inexact
    # `x * log(y)` form returns NaN instead of 0 whenever x == 0 and y <= 0
    # (or y == +-inf), which ATen never does.
    exact = True
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        xlogy_kernel[grid](
            x,
            y,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        xlogy_kernel_unmasked[grid](
            x,
            y,
            out,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _launch_tensor_scalar(x, out, y_val):
    n_elements = x.numel()
    if n_elements == 0:
        return
    if isinstance(y_val, torch.Tensor) and y_val.numel() == 1 and n_elements < _GATE:
        block_size, num_warps, masked = _pick_block(n_elements)
        if masked:
            grid = (triton.cdiv(n_elements, block_size),)
            xlogy_tensor_scalar_ptr_kernel[grid](
                x,
                y_val,
                out,
                n_elements,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        else:
            grid = (n_elements // block_size,)
            xlogy_tensor_scalar_ptr_kernel_unmasked[grid](
                x,
                y_val,
                out,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        return
    y_val = float(y_val)
    block_size, num_warps, masked = _pick_block(n_elements)
    exact = _exact_tensor_scalar(n_elements, y_val)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        xlogy_tensor_scalar_kernel[grid](
            x,
            out,
            n_elements,
            y_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        xlogy_tensor_scalar_kernel_unmasked[grid](
            x,
            out,
            y_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _launch_scalar_tensor(y, out, x_val):
    n_elements = y.numel()
    if n_elements == 0:
        return
    if isinstance(x_val, torch.Tensor) and x_val.numel() == 1 and n_elements < _GATE:
        block_size, num_warps, masked = _pick_block(n_elements)
        if masked:
            grid = (triton.cdiv(n_elements, block_size),)
            xlogy_scalar_tensor_ptr_kernel[grid](
                y,
                x_val,
                out,
                n_elements,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        else:
            grid = (n_elements // block_size,)
            xlogy_scalar_tensor_ptr_kernel_unmasked[grid](
                y,
                x_val,
                out,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        return
    x_val = float(x_val)
    block_size, num_warps, masked = _pick_block(n_elements)
    exact = _exact_scalar_tensor(n_elements, x_val)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        xlogy_scalar_tensor_kernel[grid](
            y,
            out,
            n_elements,
            x_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        xlogy_scalar_tensor_kernel_unmasked[grid](
            y,
            out,
            x_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _flat_kernels_usable(x, y, out):
    """The raw kernels index ``x``/``y``/``out`` with one flat offset, i.e. they
    are only valid when all three tensors share the same shape and ``out`` is
    contiguous.  ATen instead broadcasts ``other`` and honours the strides of an
    ``out=`` tensor (and rejects an ``out`` of the wrong shape), so those cases
    have to go through the broadcast/stride aware pointwise kernel."""
    return x.shape == y.shape and out.shape == x.shape and out.is_contiguous()


def _out_shape_error(name, expected, out):
    return RuntimeError(
        f"xlogy out tensor shape is invalid, should be {tuple(expected)} "
        f"but is {tuple(out.shape)}!"
    )


def _materialize_broadcast(t, shape):
    """Contiguous copy of ``t`` broadcast to ``shape``, on gems Triton kernels.

    ``t.expand(shape).contiguous()`` must not be used here: inside ``use_gems``
    the materialising copy leaves the operator (and the vendor ``copy_`` is a
    plain aten path whose strided ``copy_slice`` branch faults).  ``_launch_bcast``
    is the flat, block-tiled gather that backs ``expand_copy``: it decodes the
    flat destination index against the expanded view's *real* strides -- 0 on
    the broadcast dims -- so nothing has to be normalised first, and its store
    side stays contiguous (block DMA).  Only the rank-general gems copy kernel
    is used beyond the six dimensions that kernel addresses.
    """
    if tuple(t.shape) == tuple(shape) and t.is_contiguous():
        return t
    view = t.expand(tuple(shape))
    dst = torch.empty(tuple(shape), dtype=t.dtype, device=t.device)
    if dst.numel() == 0:
        return dst
    if view.dim() <= _MAX_FLAT_DIM:
        _launch_bcast(view.shape, view.stride(), view, dst, dst.numel())
    else:
        _gems_copy_(dst, view)
    return dst


def _write_back(dst, src):
    """``dst[...] = src`` through ``dst``'s own shape and strides, on gems
    Triton kernels.

    ``dst.copy_(src)`` may not be used here: inside ``use_gems`` it leaves the
    operator (and the vendor ``copy_`` is a plain aten path).  The scratch is
    always allocated in ``dst``'s dtype, because the compute kernels cast to the
    output element type on store -- so this reproduces ``dst.copy_(src)``'s cast
    without a second pass.  ``_launch_as_strided_copy`` is the generic gems
    Triton strided-copy launcher: it reads ``src`` flat and stores through
    ``dst``'s real strides, so a transposed / strided destination is written
    element by element instead of as a single flat run.
    """
    if src.dtype != dst.dtype:
        raise NotImplementedError(
            "xlogy: a strided out tensor must have the result dtype "
            f"({src.dtype}), but got {dst.dtype}"
        )
    if src.layout != torch.strided or dst.layout != torch.strided:
        raise NotImplementedError("xlogy: only strided tensors are supported as out")
    if dst.numel() == 0:
        return dst
    _launch_as_strided_copy(src, dst)
    return dst


def _launch_broadcast(x, y, out=None):
    """Exact result for broadcasting operands and/or a strided ``out``.

    ``pointwise_dynamic`` is the only kernel here that understands broadcasting
    and arbitrary ``out`` strides (the raw kernels index every tensor with one
    flat offset); it also validates the ``out`` shape and rejects internally
    overlapping operands, like ATen.  Above ``_BROADCAST_MATERIALIZE_MAX`` the
    broadcast is materialised first and the result is then produced by the 1-D
    ``pointwise_dynamic`` kernel, to avoid paying the general ndim codegen for
    an oversized broadcast.  (Materialised sizes always exceed ``_GATE``, so
    ``_launch`` routes them to the 1-D pw kernel, not to the raw kernels.)
    """
    shape = torch.broadcast_shapes(x.shape, y.shape)
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    if n_elements <= _BROADCAST_MATERIALIZE_MAX:
        if out is None:
            return _xlogy_fast(x, y)
        _xlogy_fast(x, y, out0=out)
        return out
    if out is not None and tuple(out.shape) != tuple(shape):
        raise _out_shape_error("xlogy", shape, out)
    xb = _materialize_broadcast(x, shape)
    yb = _materialize_broadcast(y, shape)
    _, result_dtype = type_promotion(xb, yb, type_promotion=_PROMOTION)
    if out is not None:
        # the compute kernels cast on store, so computing straight into the
        # destination's dtype is what ``out.copy_(res)`` used to do
        result_dtype = out.dtype
    res = torch.empty(shape, dtype=result_dtype, device=xb.device)
    _launch(xb, yb, res)
    if out is None:
        return res
    _write_back(out, res)
    return out


def xlogy(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY")
    x = self.contiguous()
    y = other.contiguous()
    if x.shape != y.shape:
        # ATen broadcasts ``other``; the raw kernels would index it flatly.
        return _launch_broadcast(x, y)
    _, result_dtype = type_promotion(x, y, type_promotion=_PROMOTION)
    out = torch.empty_like(x, dtype=result_dtype)
    _launch(x, y, out)
    return out


def xlogy_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_OUT")
    x = self.contiguous()
    y = other.contiguous()
    if not _flat_kernels_usable(x, y, out):
        # handles broadcasting, strided ``out`` and raises on a bad out shape
        return _launch_broadcast(x, y, out)
    _launch(x, y, out)
    return out


def xlogy_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_")
    x = self.contiguous()
    y = other.contiguous()
    if x.shape != y.shape:
        # ATen requires ``other`` to broadcast onto ``self``; compute the
        # broadcast result straight through ``self``'s layout (``_launch_broadcast``
        # validates that the broadcast fits ``self`` and writes with its strides).
        return _launch_broadcast(x, y, out=self)
    _launch(x, y, x)
    if x.data_ptr() != self.data_ptr():
        self.copy_(x.view(self.shape))
    return self


def xlogy_tensor_scalar(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR")
    x = self.contiguous()
    _, result_dtype = type_promotion(x, other, type_promotion=_PROMOTION)
    out = torch.empty_like(x, dtype=result_dtype)
    _launch_tensor_scalar(x, out, other)
    return out


def xlogy_tensor_scalar_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR_OUT")
    x = self.contiguous()
    if out.shape != x.shape:
        raise _out_shape_error("xlogy", x.shape, out)
    if not out.is_contiguous():
        # compute on a contiguous buffer, then scatter with ``out``'s strides
        res = torch.empty_like(x, dtype=out.dtype)
        _launch_tensor_scalar(x, res, other)
        _write_back(out, res)
        return out
    _launch_tensor_scalar(x, out, other)
    return out


def xlogy_tensor_scalar_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR_")
    x = self.contiguous()
    _launch_tensor_scalar(x, x, float(other))
    if x.data_ptr() != self.data_ptr():
        self.copy_(x.view(self.shape))
    return self


def xlogy_scalar_tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_SCALAR_TENSOR")
    y = other.contiguous()
    _, result_dtype = type_promotion(self, other, type_promotion=_PROMOTION)
    out = torch.empty_like(y, dtype=result_dtype)
    _launch_scalar_tensor(y, out, self)
    return out


def xlogy_scalar_tensor_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_SCALAR_TENSOR_OUT")
    y = other.contiguous()
    if out.shape != y.shape:
        raise _out_shape_error("xlogy", y.shape, out)
    if not out.is_contiguous():
        # compute on a contiguous buffer, then scatter with ``out``'s strides
        res = torch.empty_like(y, dtype=out.dtype)
        _launch_scalar_tensor(y, res, self)
        _write_back(out, res)
        return out
    _launch_scalar_tensor(y, out, self)
    return out
