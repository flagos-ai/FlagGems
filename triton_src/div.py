import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def true_div_kernel(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a / b
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def trunc_divide_kernel(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    div = a / b
    c = tl.where(div >= 0, tl.math.floor(div), tl.math.ceil(div))
    # div = tl_extra_shim.div_rz(a, b)
    # c = tl_extra_shim.trunc(div)
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def float_floordiv_kernel(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    div = a / b
    result = tl.math.floor(div)
    c = tl.where(b == 0.0, div, result)

    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def int_floordiv_kernel(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.int32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.int32)
    r = a % b
    c1 = r != 0
    c2 = (a < 0) ^ (b < 0)
    q = a // b
    result = tl.where(c1 & c2, q - 1, q)
    result = tl.where(b == 0, -1, result)
    tl.store(c_ptr + offsets, result, mask=mask)


@triton.jit
def remainder_kernel(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    r = a % b
    c = (r != 0.0) & ((r < 0.0) != (b < 0.0))
    result = tl.where(c, r + b, r)
    tl.store(c_ptr + offsets, result, mask=mask)


@triton.jit
def true_div_kernel_(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a / b
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def trunc_divide_kernel_(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    div = a / b
    c = tl.where(div >= 0, tl.math.floor(div), tl.math.ceil(div))
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def float_floordiv_kernel_(a_ptr, b_ptr, c_ptr, size, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    col = offsets % N

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + col, mask=mask)

    div = a / b
    q = tl.math.floor(div)

    diff = div - q
    threshold = 1e-6
    fix = (diff < -threshold) & ((a < 0) != (b < 0))
    q = tl.where(fix, q - 1.0, q)

    q = tl.where(b == 0.0, div, q)
    tl.store(c_ptr + offsets, q, mask=mask)


@triton.jit
def int_floordiv_kernel_(a_ptr, b_ptr, c_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask).to(tl.int32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.int32)

    q = a // b
    r = a % b
    need_fix = (r != 0) & ((a < 0) != (b < 0))
    q = tl.where(need_fix, q - 1, q)

    q = tl.where(b == 0, 0, q)
    tl.store(c_ptr + offsets, q, mask=mask)
