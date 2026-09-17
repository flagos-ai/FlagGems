import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK = 8192
# NOTE: MIN_BLOCK must stay >= 2048. On XPU the fp32->bf16 down-cast in the
# store uses round-to-nearest only when the compiled tile is >= 2048 lanes;
# with a 512/1024 tile the compiler emits a truncating cast that disagrees
# with torch.square on ~43% of bf16 elements (tests use exact bit-match).
MIN_BLOCK = 2048
NUM_CLUSTERS = 12


def _pick_block(n_elements):
    # Bucket the tile to a power of two in [MIN_BLOCK, MAX_BLOCK] so the kernel
    # compiles at most ~3 times total (no per-shape recompilation / IR
    # explosion) while still splitting small/medium tensors across the 12 XPU
    # clusters instead of running on 1-2 programs.
    target = triton.next_power_of_2(max(1, triton.cdiv(n_elements, NUM_CLUSTERS)))
    return max(MIN_BLOCK, min(MAX_BLOCK, target))


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


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size = _pick_block(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(x.device):
        square_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)


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
