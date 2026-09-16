import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# "1.0" as int32-lane bit patterns (little-endian):
#   fp16 1.0 = 0x3C00  -> pair 0x3C003C00
#   bf16 1.0 = 0x3F80  -> pair 0x3F803F80
#   fp32 1.0 = 0x3F800000
_FP16_ONE_PAIR = 0x3C003C00
_BF16_ONE_PAIR = 0x3F803F80
_FP32_ONE = 0x3F800000
# Single 16-bit lane for the odd-tail path (16-bit dtypes).
_BF16_ONE_BITS = 0x3F80
_FP16_ONE_BITS = 0x3C00


@triton.jit
def new_ones_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Generic fallback (non-bitwise dtypes): masked store of the typed 1.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    one = tl.full((BLOCK_SIZE,), 1, dtype=output_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, one, mask=mask)


@triton.jit
def new_ones_bits_kernel(output_ptr, n_elements, bits, BLOCK_SIZE: tl.constexpr):
    # Writes a raw 16-bit pattern through an int16 view (odd-N 16-bit dtypes).
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.full((BLOCK_SIZE,), bits, dtype=tl.int16)
    tl.store(output_ptr + offsets, v, mask=mask)


@triton.jit
def new_ones_bits32_exact_kernel(output_ptr, bits, BLOCK_SIZE: tl.constexpr):
    # Unmasked 32-bit pattern store (n must be a multiple of BLOCK_SIZE).
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.full((BLOCK_SIZE,), bits, dtype=tl.int32)
    tl.store(output_ptr + offsets, v)


@triton.jit
def new_ones_bits32_kernel(output_ptr, n_elements, bits, BLOCK_SIZE: tl.constexpr):
    # Masked 32-bit pattern store (tail block).
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.full((BLOCK_SIZE,), bits, dtype=tl.int32)
    tl.store(output_ptr + offsets, v, mask=mask)


def _launch_config(n32):
    if n32 <= 1024:
        return 256, 2
    if n32 <= 8192:
        return 1024, 4
    if n32 <= 65536:
        return 4096, 8
    return 16384, 8


def _fill_exact_or_masked(view32, n32, bits, device):
    block_size, num_warps = _launch_config(n32)
    if n32 % block_size == 0:
        grid = (n32 // block_size,)
        with torch_device_fn.device(device):
            new_ones_bits32_exact_kernel[grid](
                view32, bits, BLOCK_SIZE=block_size, num_warps=num_warps
            )
    else:
        grid = (triton.cdiv(n32, block_size),)
        with torch_device_fn.device(device):
            new_ones_bits32_kernel[grid](
                view32, n32, bits, BLOCK_SIZE=block_size, num_warps=num_warps
            )


def new_ones(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_KUNLUNXIN NEW_ONES")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype

    out = torch.empty(size, device=device, dtype=dtype)
    N = out.numel()
    if N == 0:
        return out

    if dtype in (torch.float16, torch.bfloat16):
        if N & 1:
            # Odd N: no int32 view possible; fill through an int16 view with
            # the single-lane bits (still a fast block DMA, ~1.16 TB/s).
            block_size, num_warps = _launch_config(N)
            grid = (triton.cdiv(N, block_size),)
            bits16 = _FP16_ONE_BITS if dtype == torch.float16 else _BF16_ONE_BITS
            with torch_device_fn.device(device):
                new_ones_bits_kernel[grid](
                    out.reshape(-1).view(torch.int16),
                    N,
                    bits16,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
        else:
            # out is contiguous -> reshape(-1) is a zero-copy view; the int32
            # view stores two 16-bit lanes per word (fast block DMA path).
            bits = _FP16_ONE_PAIR if dtype == torch.float16 else _BF16_ONE_PAIR
            _fill_exact_or_masked(
                out.reshape(-1).view(torch.int32), N // 2, bits, device
            )
    elif dtype == torch.float32:
        _fill_exact_or_masked(out.reshape(-1).view(torch.int32), N, _FP32_ONE, device)
    else:
        block_size, num_warps = _launch_config(N)
        grid = (triton.cdiv(N, block_size),)
        with torch_device_fn.device(device):
            new_ones_kernel[grid](out, N, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out
