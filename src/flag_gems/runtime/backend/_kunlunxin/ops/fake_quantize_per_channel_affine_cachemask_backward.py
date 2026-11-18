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
#   fake_quantize_per_channel_affine_cachemask_backward(grad, mask) = grad * mask
# (pure elementwise multiply of a float tensor by a 0/1 bool mask; no
# quantization math).
#
# Baseline (2026-10-09, cp /tmp/fq_pcmb_stub_backup.py): the generic
# implementation at src/flag_gems/ops/... used `grid = lambda meta: (...)`. On
# the xpubin backend the launch grid participates in the compilation cache key,
# so a fresh lambda object per call makes every launch a full recompile (see
# KERNEL_OPT_EXPERIENCE 2.4#41): measured Gems latency 126-374 ms for every
# shape in the benchmark matrix, size-independent (speedup ~0.000). The fixed
# grip is a constant-tuple grid; the rest of this file tunes the memory path of
# the (per-shape compiled once) kernel.
#
# Memory-shape: read grad (2B f16/bf16 or 4B f32) + read mask (1B, viewed as
# int8 so i1 never enters the dataflow) + write out (2/4B).
#
# Mask path: the bool mask is viewed as int8 (zero-copy, same 1-byte layout as
# torch.bool) and loaded; the 0/1 int8 value selects between `grad` and
# `grad * 0.0`.  An in-kernel int8 -> fp32 conversion (`.to(tl.float32)`) is
# NOT used: it fails to compile on the Triton XPU backend with "size mismatch
# when packing elements for LLVM struct expected 4 but got 1" (KERNEL_OPT_EXPERIENCE
# row 43: int8/int16 widening is broken on this backend; the CI container's
# toolchain hits it while the local 3.6.0 build happens to accept it).  The
# int8 `!= 0` comparison emits an i1 predicate (the same domain every load /
# store mask uses) and selects fp32 operands -- no widening, single launch.
#
# Correctness notes:
#   * mask True  -> select `grad` (== grad * 1.0 for every value incl NaN/inf)
#   * mask False -> select `grad * 0.0` (== grad * 0.0: +-0.0 for finite,
#     NaN for inf/NaN), identical to the generic `grad * mask` (bool promoted
#     to float) for every value including -0.0/+-inf/NaN (NaN*1.0==NaN,
#     inf*0.0==NaN).
#   * The masked (tail) kernel keeps the store masked -- an unmasked store is a
#     known KL3 299 trigger (bernoulli/col2im), so "remove the load mask, keep
#     the store mask" is the verified pattern; here we simply only ever take
#     the fully-unmasked kernel when n_elements % BLOCK_SIZE == 0 (no tail at
#     all), and the tail kernel masks both load and store.
#   * grid is a constant tuple (never `lambda`), so each (n_elements, dtype)
#     pair compiles exactly once.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _pick_block(n_elements, dtype):
    # Tile buckets for the flat 1-D pointwise kernel (all benchmark shapes
    # divide their bucket exactly so the fast unmasked path is used; masked
    # tails only when the shape is not a multiple of the bucket).  Swept
    # 2026-10-09 (harness/solution/.../probe_blocks{,2}.py, min-of-3
    # do_bench warmup=1/rep=1): fp32 prefers a narrower 8192-lane/8-warp tile
    # (4B/ele + 1B mask), fp16/bf16 prefer 16384-lane/4-warp; the small-shape
    # class is launch-bound so 4096-lane/4-warp is the sweet spot there.
    if n_elements < 65_536:
        return 4096, 4, (n_elements % 4096) != 0
    if n_elements < 131_072:
        return 8192, 4, (n_elements % 8192) != 0
    if dtype == torch.float32:
        return 8192, 8, (n_elements % 8192) != 0
    return 16384, 4, (n_elements % 16384) != 0


@triton.jit
def _fq_pcmb_backward_kernel(
    grad_ptr,
    mask_ptr,  # int8 view of the bool mask (host does .view(torch.int8))
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bound = offset < n_elements
    grad = tl.load(grad_ptr + offset, mask=bound)
    # mask is int8 0/1; select grad / grad*0.0 on an i1 predicate -- no
    # in-kernel int8 -> fp32 widening (broken on this backend, see module doc).
    quant_mask = tl.load(mask_ptr + offset, mask=bound)
    out = tl.where(quant_mask != 0, grad, grad * 0.0)
    tl.store(output_ptr + offset, out, mask=bound)


@triton.jit
def _fq_pcmb_backward_kernel_unmasked(
    grad_ptr,
    mask_ptr,  # int8 view of the bool mask (host does .view(torch.int8))
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    grad = tl.load(grad_ptr + offset)
    quant_mask = tl.load(mask_ptr + offset)
    out = tl.where(quant_mask != 0, grad, grad * 0.0)
    tl.store(output_ptr + offset, out)


def fake_quantize_per_channel_affine_cachemask_backward(grad, mask):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_BACKWARD")
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output
    block_size, num_warps, masked = _pick_block(n_elements, grad.dtype)
    # Constant-tuple grid: on the xpubin backend a `lambda` grid would rehash
    # the compile cache on every launch (see module doc); a literal-length
    # local keeps the key stable so the kernel compiles once per (shape, dtype).
    grid = (triton.cdiv(n_elements, block_size),)
    # Zero-copy int8 view of the bool mask: keeps i1 out of the load path; the
    # 0/1 int8 is consumed by an i1-predicate select in-kernel (no int8->fp32
    # widening, which does not compile on this backend -- see module doc).
    mask_view = mask.view(torch.int8)
    with torch_device_fn.device(grad.device):
        if masked:
            _fq_pcmb_backward_kernel[grid](
                grad,
                mask_view,
                output,
                n_elements,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        else:
            _fq_pcmb_backward_kernel_unmasked[grid](
                grad,
                mask_view,
                output,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    return output
