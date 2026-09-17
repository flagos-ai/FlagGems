import logging
import math

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_PROMOTION = ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT

_GATE = 1_048_576

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
    return x.to(tl.float32) * tl.log(1.0000000000000000 * y.to(tl.float32))


MIN_BLOCK = 2048
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False
_MASKED_FALLBACK_BLOCK = 8192


def _pick_block(n_elements):
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


def _exact_tensor_tensor(n_elements):
    return n_elements < _GATE


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
    exact = _exact_tensor_tensor(n_elements)
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


def xlogy(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY")
    x = self.contiguous()
    y = other.contiguous()
    _, result_dtype = type_promotion(x, y, type_promotion=_PROMOTION)
    out = torch.empty_like(x, dtype=result_dtype)
    _launch(x, y, out)
    return out


def xlogy_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_OUT")
    x = self.contiguous()
    y = other.contiguous()
    _launch(x, y, out)
    return out


def xlogy_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_")
    x = self.contiguous()
    y = other.contiguous()
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
    _launch_scalar_tensor(y, out, self)
    return out
