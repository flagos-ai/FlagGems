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

"""Kunlunxin (XPU) override for ``aten::_foreach_erf``.

The generic path is both compute- and memory-bound on this backend: it uses
``tl.math.erf`` (a slow libdevice extern) and the shared executor's narrow,
always-masked tiles. This override reuses the executor's metadata/grouping
(an int64 table of input ptrs | output ptrs | numels over a 2D grid) but owns
the launch to apply the single-tensor ``erf`` recipe:
  * deg-12 odd polynomial with a branch-free NaN-safe clamp to [-3, 3]; no
    transcendental and no select. Max abs err 3.38e-5, within atol 1e-4;
    NaN propagates, +/-inf saturates to +/-0.99997.
  * wide unmasked tiles + ``buffer_size_limit``, with a ``NEED_MASK`` constexpr
    so homogeneous/divisible lists skip the ~2x-costlier masked memory path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import NO_COMPLEX
from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
    int_to_float,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

UNROLL_NUM = 16
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(max_numel):
    # Wide unmasked tiles to saturate block-DMA; 2D grid keeps program count sane.
    if max_numel >= 1_048_576:
        return 131072, 32, 8192
    if max_numel >= 262_144:
        return 32768, 8, 8192
    if max_numel >= 16384:
        return 16384, 8, 4096
    return 2048, 4, 0


@triton.jit
def _foreach_erf_kernel(
    meta_ptr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # meta layout (int64): [0,NT) in ptrs | [NT,2NT) out ptrs | [2NT,3NT) numels
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(in_ptr + idx).to(tl.float32)

    # Branch-free clamp to [-3, 3] that also propagates NaN (tl.maximum does not
    # propagate NaN here): clamp(x) == min(-min(-x, 3), 3).
    xc = tl.minimum(-tl.minimum(-x, 3.0), 3.0)
    t2 = xc * xc
    p = 2.0958534e-12
    p = p * t2 + -1.4200718e-10
    p = p * t2 + 4.475963e-09
    p = p * t2 + -8.813736e-08
    p = p * t2 + 1.2336866e-06
    p = p * t2 + -1.3277817e-05
    p = p * t2 + 0.00011584865
    p = p * t2 + -0.00084552215
    p = p * t2 + 0.005211773
    p = p * t2 + -0.0268563
    p = p * t2 + 0.112833545
    p = p * t2 + -0.37612554
    p = p * t2 + 1.1283791
    r = xc * p

    if NEED_MASK:
        tl.store(out_ptr + idx, r.to(OUT_DT), mask=mask)
    else:
        tl.store(out_ptr + idx, r.to(OUT_DT))


def _launch_group(srcs, dsts, in_dtype, out_dtype, device):
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    nt = len(srcs)
    block, num_warps, bsl = _pick_block(max_numel)
    # Unmasked only when all tensors share a length that divides the tile;
    # otherwise an unmasked program would overrun a shorter tensor.
    homogeneous = all(n == max_numel for n in numels)
    need_mask = (not homogeneous) or (max_numel % block != 0)

    meta_cpu = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    )
    meta = meta_cpu.to(device, non_blocking=True)

    grid = (nt, triton.cdiv(max_numel, block))
    kwargs = dict(
        num_warps=num_warps,
        unroll_num=UNROLL_NUM,
        isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
    )
    if bsl:
        kwargs["buffer_size_limit"] = bsl
    _foreach_erf_kernel[grid](
        meta,
        nt,
        tl_dtype(in_dtype),
        tl_dtype(out_dtype),
        block,
        need_mask,
        **kwargs,
    )


def _run(tensors, inplace):
    tensors = check_tensor_list(tensors)
    results = [None] * len(tensors)
    buckets = {}  # (device, in_dtype) -> (srcs, dsts)
    writebacks = []

    for i, t in enumerate(tensors):
        dtype = t.dtype
        check_dtype_supported(dtype, NO_COMPLEX)
        out_dtype = int_to_float(dtype)
        key = (t.device, dtype)

        if inplace:
            if out_dtype != dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if is_flat_addressable(t):
                src = dst = t
            else:
                src = dst = t.contiguous()
                writebacks.append((t, src))
            results[i] = t
        else:
            src = t if is_flat_addressable(t) else t.contiguous()
            dst = torch.empty_like(src, dtype=out_dtype)
            results[i] = dst

        bucket = buckets.setdefault(key, ([], []))
        bucket[0].append(src)
        bucket[1].append(dst)

    for (device, in_dtype), (srcs, dsts) in buckets.items():
        _launch_group(srcs, dsts, in_dtype, dsts[0].dtype, device)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)

    return results


def _foreach_erf(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ERF")
    return _run(self, inplace=False)


def _foreach_erf_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ERF_")
    _run(self, inplace=True)
    return None
