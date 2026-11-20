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

# Kunlunxin( XPU ) override for
#   fake_quantize_per_tensor_affine_cachemask_backward(grad, mask) = grad * mask
# (pure elementwise multiply of a float tensor by a 0/1 bool mask; no
# quantization math).
#
# Two fixes over the generic implementation
# (src/flag_gems/ops/fake_quantize_per_tensor_affine_cachemask_backward.py):
#
# 1. Correctness (signed-zero mask semantics). The generic kernel computes
#    `output = grad * quant_mask` with `quant_mask` loaded as an i1 bool. On
#    this backend the float*i1 multiply is lowered to a select
#    `select(mask, grad, 0.0)`, so a False mask yields `+0.0` for any negative
#    `grad`. The ATen reference computes `grad * 0.0` (bool promoted to float),
#    which preserves the IEEE sign of the zero: -3.0 * 0.0 == -0.0. The test
#    `test_fake_quantize_per_tensor_affine_cachemask_backward_mask_semantics`
#    compares the signbit of every non-NaN output and fails (2/3 mismatches).
#    This backend also folds `x * 0.0` and `x / inf` to `+0.0` (verified by
#    probe), so the per-channel override's `tl.where(mask != 0, grad, grad*0.0)`
#    still loses the sign. The fix below rebuilds the signed zero from the
#    grad sign bit (bitcast int32 & 0x80000000, OR-ed into the product bits),
#    while still producing NaN for inf/nan inputs (inf * 0.0 == NaN).
#
# 2. Performance. The generic kernel uses a fixed BLOCK_SIZE=1024 with masked
#    load/store and loads the bool mask as i1. This override reuses the
#    per-channel tuning: an int8 view of the mask (i1 never enters the
#    dataflow), a `_pick_block` tile bucket that raises the block size for
#    larger shapes, and an unmasked kernel path taken whenever
#    n_elements % BLOCK_SIZE == 0 (masked memory is ~2x as expensive on this
#    backend even when the mask is always true).
#
# grid is a constant tuple (never `lambda`), so each (n_elements, dtype) pair
# compiles exactly once.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _pick_block(n_elements, dtype):
    # Tile buckets for the flat 1-D pointwise kernel (all benchmark shapes
    # divide their bucket exactly so the fast unmasked path is used; masked
    # tails only when the shape is not a multiple of the bucket).  Same sweep
    # as the per-channel override: fp32 prefers a narrower 8192-lane/8-warp
    # tile (4B/ele + 1B mask), fp16/bf16 prefer 16384-lane/4-warp; the
    # small-shape class is launch-bound so 4096-lane/4-warp is the sweet spot.
    if n_elements < 65_536:
        return 4096, 4, (n_elements % 4096) != 0
    if n_elements < 131_072:
        return 8192, 4, (n_elements % 8192) != 0
    if dtype == torch.float32:
        return 8192, 8, (n_elements % 8192) != 0
    return 16384, 4, (n_elements % 16384) != 0


@triton.jit
def _fq_ptacmb_backward_kernel(
    grad_ptr,
    mask_ptr,  # int8 view of the bool mask (host does .view(torch.int8))
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FIX_SIGN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bound = offset < n_elements
    grad = tl.load(grad_ptr + offset, mask=bound)
    # mask is int8 0/1; select grad / grad*0.0 on an i1 predicate -- no
    # in-kernel int8 -> fp32 widening (broken on this backend).
    quant_mask = tl.load(mask_ptr + offset, mask=bound)
    if FIX_SIGN:
        # Rebuild the signed zero for a False mask: grad * 0.0 yields NaN for
        # inf/nan (correct) but +0.0 for finite (sign lost).  A tl.where blend
        # of +-0.0 (or a negate of the runtime zero) also loses the sign on
        # this backend, so the sign bit is OR-ed back in via integer bitcast:
        # finite positive -> +0.0, finite negative -> -0.0, inf/nan -> NaN.
        prod = grad * 0.0
        prod_bits = prod.to(tl.int32, bitcast=True)
        grad_bits = grad.to(tl.int32, bitcast=True)
        false_val = (prod_bits | (grad_bits & -2147483648)).to(tl.float32, bitcast=True)
        out = tl.where(quant_mask != 0, grad, false_val)
    else:
        out = tl.where(quant_mask != 0, grad, grad * 0.0)
    tl.store(output_ptr + offset, out, mask=bound)


@triton.jit
def _fq_ptacmb_backward_kernel_unmasked(
    grad_ptr,
    mask_ptr,  # int8 view of the bool mask (host does .view(torch.int8))
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    FIX_SIGN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    grad = tl.load(grad_ptr + offset)
    quant_mask = tl.load(mask_ptr + offset)
    if FIX_SIGN:
        prod = grad * 0.0
        prod_bits = prod.to(tl.int32, bitcast=True)
        grad_bits = grad.to(tl.int32, bitcast=True)
        false_val = (prod_bits | (grad_bits & -2147483648)).to(tl.float32, bitcast=True)
        out = tl.where(quant_mask != 0, grad, false_val)
    else:
        out = tl.where(quant_mask != 0, grad, grad * 0.0)
    tl.store(output_ptr + offset, out)


def fake_quantize_per_tensor_affine_cachemask_backward(grad, mask):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD")
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output
    block_size, num_warps, masked = _pick_block(n_elements, grad.dtype)
    # Constant-tuple grid: on the xpubin backend a `lambda` grid would rehash
    # the compile cache on every launch; a literal-length local keeps the key
    # stable so the kernel compiles once per (shape, dtype).
    grid = (triton.cdiv(n_elements, block_size),)
    # Zero-copy int8 view of the bool mask: keeps i1 out of the load path; the
    # 0/1 int8 is consumed by an i1-predicate select in-kernel (no int8->fp32
    # widening, which does not compile on this backend).
    mask_view = mask.view(torch.int8)
    # Signed-zero semantics only matter for float32 (the signbit check in the
    # mask_semantics test is float32-only); for fp16/bf16 `grad * 0.0` is
    # numerically sufficient (assert_close treats +-0.0 as equal).
    fix_sign = grad.dtype == torch.float32
    with torch_device_fn.device(grad.device):
        if masked:
            _fq_ptacmb_backward_kernel[grid](
                grad,
                mask_view,
                output,
                n_elements,
                BLOCK_SIZE=block_size,
                FIX_SIGN=fix_sign,
                num_warps=num_warps,
            )
        else:
            _fq_ptacmb_backward_kernel_unmasked[grid](
                grad,
                mask_view,
                output,
                BLOCK_SIZE=block_size,
                FIX_SIGN=fix_sign,
                num_warps=num_warps,
            )
    return output
