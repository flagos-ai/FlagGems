import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow
_tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_kernel(x):
    return _tanh(x.to(tl.float32))


_TANH_BW_MAX_BLOCK = 65536
_TANH_BW_BF16_FLAT_MAX_NUMEL = 1 << 16


@triton.jit
def tanh_backward_flat_kernel(y_ptr, dy_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, (dy * (1.0 - y * y)).to(out_ptr.dtype.element_ty))


@triton.jit
def tanh_backward_flat_masked_kernel(
    y_ptr,
    dy_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    tl.store(
        out_ptr + offs,
        (dy * (1.0 - y * y)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _tanh_backward_pick_block(n_elements):
    if n_elements <= 4096:
        return min(triton.next_power_of_2(n_elements), _TANH_BW_MAX_BLOCK)
    if n_elements <= 131072:
        ctas = 8
    elif n_elements <= 2097152:
        ctas = 32
    else:
        ctas = 128
    block = (n_elements + ctas - 1) // ctas
    return min(triton.next_power_of_2(block), _TANH_BW_MAX_BLOCK)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y, dy):
    y = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y * y)


# --------------------------------------------------------------------------
# tanh_backward flat stride-1 fast path (2026-09-09, harness/solution/tanh_backward)
#
# Why: `tanh_backward_kernel` above is a bare `pointwise_dynamic` with **no
# `config=`**, so it inherits the kunlunxin default `CodeGenConfig(512, ...)`
# from `_kunlunxin/utils/codegen_config_utils.py` (max_tile_size=512,
# `unroll_num=0`, `buffer_size_limit=0`). Measured cost of that path (shared
# card, us per call, `harness/solution/tanh_backward/micro_ab.py`):
#
#   n            vendor kernel   generic pointwise_dynamic
#   1K .. 1M       19 .. 22 us         107 .. 110 us   <- fixed host-side cost
#   16.7M (fp16)        75 us               162 us     (~610 GB/s)
#   67.1M (fp16)       242 us               334 us     (~1190 GB/s)
#
# So the generic path pays a ~107 us floor **per call** that is independent of
# size: eight of the twelve benchmark shapes are <=4M elements and are therefore
# 5x slower than the vendor kernel purely from dispatch overhead, before any
# memory traffic. A flat stride-1 kernel launched directly cuts that floor to
# ~37-50 us.
#
# in_grad = dy * (1 - y*y) is pure pointwise, two inputs / one output, no
# reduction and no `tl.where` (HARNESS_SUMMARY section 4.0: `tl.where` is the
# most expensive primitive on this backend, so none is introduced). The fp32
# expression below is written in exactly the same order as
# `tanh_backward_kernel`, so the fast path is bit-identical to the generic path
# for every dtype.
#
# Access shape is identical to `atan2` after its select-free rewrite (2 loads +
# 1 store, stride-1, memory bound), so the launch options are taken from
# `atan2.py`. The **tile policy is not** -- `atan2`'s 2048/8192 buckets were
# re-measured here and are far off optimum for this shape ladder; see the sweep
# below.
#
# Tile sweep (2026-09-09, shared card, us/call, 2 loads + 1 store, all three
# float dtypes; `harness/solution/tanh_backward/sweep_ladder.py`). BLOCK is the
# only variable that matters: `num_warps` 8/16/32, `unroll_num` 8/16/32 and
# `buffer_size_limit` 8192/16384 all land inside +-1% of each other.
#
#   fp16 16.7M : 2048 514  4096 307  8192 190  16384 137  32768 117  65536 112
#   bf16 16.7M : 2048 573  4096 328  8192 215  16384 173  32768 145  65536 134
#   fp32 16.7M : 2048 540  4096 292  8192 195  16384 181  32768 167  65536 158
#   fp16 67.1M : 2048 1939 4096 1041 8192 582  16384 390  32768 312  65536 284
#   fp16 262K  :                     8192  55  16384  42  32768  48  65536  50
#   fp16 4096  : 2048  58  4096  61  8192  59  16384  54  32768  53  65536 193
#
# => three buckets: <=16K -> 2048 (launch-floor bound, ~50 us for every tile
# width, so take the narrowest and waste no lanes), <=1M -> 16384, above -> 65536.
# The 8192-lane tile `atan2` uses would have **regressed** the four >=16.7M cases
# of the benchmark matrix against the generic path (fp16 67.1M: 582 us vs the
# generic path's 421 us); that regression is exactly what the sweep found and
# removed.
#
# Tile width stops at 65536 on purpose. HARNESS_SUMMARY section 3 records
# `tl.store(bf16_ptr, fp32_val.to(tl.bfloat16))` hanging a whole card at
# TILE=131072, and section 3.1 records a 32768-lane unmasked tile doing the same
# in `special_ndtr`. 65536 was therefore stepped up one width at a time under
# `timeout`, tiny grid first, and has since run the full 12-shape x 3-dtype
# ladder plus the equivalence and test suites clean. Note also that dropping
# `buffer_size_limit` (i.e. the backend default `buffer_size_limit=0,
# unroll_num=0`) makes even BLOCK=8192 fail to compile with
# `triton.runtime.errors.OutOfResources: out of resource: uni_sram` -- the
# bounded SRAM buffer is what makes the wide tiles legal at all.
#
# GEMS_KUNLUNXIN_TANH_BACKWARD_LEGACY=1 forces the original
# `pointwise_dynamic` path. This is the one-constant arm switch of
# HARNESS_SUMMARY section 6.7: it lets the BASE and POST benchmark arms run
# back-to-back inside a single exclusive card claim (exclusive windows on this
# machine are minutes long at best) instead of relying on a source checkout
# between arms. Default (unset) is the fast path.
# --------------------------------------------------------------------------
_LEGACY_TANH_BACKWARD = os.environ.get("GEMS_KUNLUNXIN_TANH_BACKWARD_LEGACY") == "1"

_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False

# bf16 upper bound for the fast path (2026-09-09,
# `harness/solution/tanh_backward/bf16_crossover.py`, us/call, spread over three
# repeats < 1%):
#
#   n        bf16 fast  bf16 generic | fp16 fast  fp16 generic
#    8 Mi        91         133      |     65          128
#   16 Mi       124         163      |     94          157
#   32 Mi       202         221      |    149          213
#   64 Mi       363         339      |    262          328
#
# Same byte traffic as fp16, but bf16 costs ~35 % more in the flat kernel: the
# `fp32 -> bf16` store conversion, not memory, is the limit there, and past
# 32 Mi elements the generic `pointwise_dynamic` path wins. So bf16 above
# 32 Mi keeps the generic path -- this is a bucketing decision between two XPU
# Triton kernels, no host/ATen fallback. The obvious "just widen the tile"
# answer is not available: HARNESS_SUMMARY section 3 records exactly
# `tl.store(bf16_ptr, fp32_val.to(tl.bfloat16))` at TILE=131072 hanging a card.
_BF16_FAST_MAX_NUMEL = 32 << 20


def _pick_block(n_elements):
    # Three buckets only -> at most 6 compiled variants (masked/unmasked x 3
    # tiles) per dtype, so the compile cost stays bounded. Boundaries and widths
    # come from the sweep table in the header comment.
    if n_elements <= 16384:
        return 2048, 4, n_elements % 2048 != 0
    if n_elements <= 1048576:
        return 16384, 8, n_elements % 16384 != 0
    return 65536, 8, n_elements % 65536 != 0


@triton.jit
def tanh_backward_flat_kernel(
    y_ptr, dy_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0).to(tl.float32)
    res = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def tanh_backward_flat_kernel_unmasked(y_ptr, dy_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    res = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty))


def _tanh_backward_fast_eligible(grad_output, output):
    if _LEGACY_TANH_BACKWARD:
        return False
    if not (torch.is_tensor(grad_output) and torch.is_tensor(output)):
        return False
    if output.dtype is not grad_output.dtype or output.dtype not in _FAST_DTYPES:
        return False
    if output.shape != grad_output.shape:
        return False
    # 0-dim tensors go to the generic path (one element, launch bound anyway).
    if output.dim() == 0 or output.numel() == 0:
        return False
    if output.dtype is torch.bfloat16 and output.numel() > _BF16_FAST_MAX_NUMEL:
        return False
    return output.is_contiguous() and grad_output.is_contiguous()


def _tanh_backward_fast(grad_output, output):
    n_elements = output.numel()
    out = torch.empty_like(output)
    block, warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block),)
        tanh_backward_flat_kernel[grid](
            output,
            grad_output,
            out,
            n_elements,
            BLOCK=block,
            num_warps=warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block,)
        tanh_backward_flat_kernel_unmasked[grid](
            output,
            grad_output,
            out,
            BLOCK=block,
            num_warps=warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    return out


def tanh(self):
    logger.debug("GEMS_KUNLUNXIN TANH")
    out = tanh_kernel(self)
    return out


def _tanh_backward_flat(grad_output, output):
    numel = output.numel()
    if numel == 0:
        return torch.empty_like(output)
    grad_input = torch.empty_strided(
        output.shape, output.stride(), dtype=output.dtype, device=output.device
    )
    block = _tanh_backward_pick_block(numel)
    if numel % block == 0:
        tanh_backward_flat_kernel[(numel // block,)](
            output,
            grad_output,
            grad_input,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    else:
        tanh_backward_flat_masked_kernel[(triton.cdiv(numel, block),)](
            output,
            grad_output,
            grad_input,
            numel,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    return grad_input


def tanh_backward(grad_output, output):
    logger.debug("GEMS_KUNLUNXIN TANH_BACKWARD")
    if _tanh_backward_fast_eligible(grad_output, output):
        return _tanh_backward_fast(grad_output, output)
    in_grad = tanh_backward_kernel(output, grad_output)
    return in_grad


def tanh_(A):
    logger.debug("GEMS_KUNLUNXIN TANH_")
    out = tanh_kernel(A, out0=A)
    return out
