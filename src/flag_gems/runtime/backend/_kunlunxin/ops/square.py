import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# NOTE: every tile must be >= 2048 lanes for the bf16 store path. On XPU the
# fp32->bf16 down-cast uses round-to-nearest only when the compiled tile is
# >= 2048 lanes; with a 512/1024 tile the compiler emits a truncating cast
# that disagrees with torch.square on ~43% of bf16 elements (tests use exact
# bit-match). The smallest bucket is 2048, so this always holds.
MIN_BLOCK = 2048
MAX_BLOCK = 131072
# Compile-time knobs that matter on XPU: bigger DMA buffer + unrolled vector
# loads keep the memory pipeline saturated; async memory is disabled because
# the launch/completion of async copies dominates for memory-bound kernels.
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into one of 3 unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~4 times total (no per-shape recompilation / IR
    # explosion). Two rules:
    #  1. When n_elements divides the tile exactly the kernel runs WITHOUT a
    #     mask. A (runtime) always-true mask forces the slow masked memory
    #     path on XPU (~1.8-2.4x penalty measured for fp16/bf16, ~2x for
    #     fp32 on 16M elements) even though the condition is trivially true.
    #  2. Big tiles (up to 131072) are better than small ones for
    #     bandwidth-bound flat copies: grid = n/tile stays well above the 12
    #     XPU clusters while each program streams a large contiguous chunk.
    #     The multi-program launch floor (~0.006ms) still bounds small tensors.
    if n_elements >= 1_048_576 and n_elements % MAX_BLOCK == 0:
        return MAX_BLOCK, 32, False
    if n_elements >= 262_144 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        # Small tensors: a light single-block tile keeps launch cheap (~6us
        # floor) instead of spawning one heavyweight 16384-lane program.
        return 2048, 4, True
    return 16384, 8, True


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    # torch.square computes in fp32 and rounds once to the output dtype (true
    # for both scalar and vectorized paths). Match it exactly (tests use
    # gems_assert_equal) by upcasting to fp32 before the multiply.
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    out = x * x
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def square_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    out = x * x
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    with torch_device_fn.device(x.device):
        if masked:
            grid = (triton.cdiv(n_elements, block_size),)
            square_kernel[grid](
                x, out, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps,
                unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        else:
            grid = (n_elements // block_size,)
            square_kernel_unmasked[grid](
                x, out, BLOCK_SIZE=block_size, num_warps=num_warps,
                unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )


def square(A):
    logger.debug("GEMS_KUNLUNXIN SQUARE")
    x = A.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out


def square_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN SQUARE_OUT")
    if out is None:
        return square(A)
    x = A.contiguous()
    if out.is_contiguous():
        _launch(x, out)
    else:
        tmp = torch.empty_like(x)
        _launch(x, tmp)
        out.copy_(tmp.view(out.shape))
    return out


def square_(A):
    logger.debug("GEMS_KUNLUNXIN SQUARE_")
    x = A.contiguous()
    _launch(x, x)
    if x.data_ptr() != A.data_ptr():
        A.copy_(x.view(A.shape))
    return A
