# Kunlunxin (XPU) override of new_ones.
#
# The generic new_ones launches the pure-write `ones_kernel` with a hard-coded
# BLOCK_SIZE=1024 -> grid = cdiv(N, 1024). For large N this is millions of tiny
# programs, so the fill is LAUNCH-BOUND (~10-20 GB/s) regardless of size: the
# IR (ir-new_ones-dev6.log) is dominated by `tensor<1024x...>` tiles. Large
# shapes ([4096,4096], [268435456], [10000,65536]) run at gems speedup
# ~0.04-0.11.
#
# Fix: size-banded BLOCK_SIZE / num_warps (the proven `arange` generation-kernel
# heuristic on this backend) so each program writes a wide contiguous block DMA
# and the grid stays small.
#
# bf16 store is a native-dtype trap: XPU triton materialises a bf16 `tl.store`
# ~6x slower than fp16/fp32 for the SAME byte count ([4096,4096] 0.178ms vs
# 0.032ms), so bf16 stayed at ~0.11 while fp16/fp32 reached ~0.72. Fix: for bf16
# reinterpret the buffer as int16 and store the bit pattern of 1.0 (0x3F80 =
# 16256) -- an int16 store IS a fast block DMA (measured 0.030ms, == fp16). Pure
# write -> zero correctness risk.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# bf16 1.0 = 0 01111111 0000000 = 0x3F80 -> 16256 as a signed int16.
_BF16_ONE_BITS = 0x3F80


@triton.jit
def new_ones_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    one = tl.full((BLOCK_SIZE,), 1, dtype=output_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, one, mask=mask)


@triton.jit
def new_ones_bits_kernel(output_ptr, n_elements, bits, BLOCK_SIZE: tl.constexpr):
    # Writes a raw 16-bit pattern through an int16 view (bf16 fast path).
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.full((BLOCK_SIZE,), bits, dtype=tl.int16)
    tl.store(output_ptr + offsets, v, mask=mask)


def _launch_config(n):
    if n <= 1024:
        return 256, 2
    if n <= 8192:
        return 1024, 4
    if n <= 65536:
        return 4096, 8
    return 16384, 8


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

    block_size, num_warps = _launch_config(N)
    grid = (triton.cdiv(N, block_size),)
    with torch_device_fn.device(device):
        if dtype == torch.bfloat16:
            new_ones_bits_kernel[grid](
                out.view(torch.int16),
                N,
                _BF16_ONE_BITS,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        else:
            new_ones_kernel[grid](out, N, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out
