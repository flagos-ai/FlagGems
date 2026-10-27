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

"""Kunlunxin (XPU) override for the unary ``_foreach_rsqrt`` family.

The generic executor uses ``1.0 / tl.sqrt(x)`` (divide is a separate slow op
here) and caps the block at 1024, producing many tiny masked programs. This
override keeps the generic host orchestration but swaps in a wider,
mask-free-when-uniform kernel using the hardware SFU ``xpu.rsqrt``. Complex
inputs defer to the reference executor.
"""

import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.ops._foreach_unary import NO_COMPLEX32, c_rsqrt
from flag_gems.utils.foreach import (
    _STATS,
    check_dtype_supported,
    check_tensor_list,
    foreach_unary,
    has_internal_overlap,
    int_to_float,
    is_flat_addressable,
    tl_dtype,
)

# Use the generic foreach logger name and record format so the repository's
# registration-liveness test (watching ``GEMS _FOREACH_RSQRT``) still passes
# when this override is the live implementation.
logger = logging.getLogger("flag_gems.ops._foreach_unary")


@triton.jit
def _rsqrt_sfu(x):
    # Compute in fp32 through the SFU; the store casts back. ``x`` is already
    # promoted to the output dtype by the executor.
    return xpu.rsqrt(x.to(tl.float32))


@triton.jit
def _rsqrt_flat_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One tensor, pointers passed as direct kernel args.

    Direct base pointers (vs loading from a metadata buffer at runtime) let the
    compiler prove alignment and emit a wide coalesced block-DMA; the indirect
    form caps bandwidth on a pure-bandwidth op like rsqrt.
    """
    offset = tl.program_id(0) * BLOCK
    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, xpu.rsqrt(x.to(tl.float32)).to(OUT_DT), mask=mask)
    else:
        x = tl.load(in_ptr + idx)
        tl.store(out_ptr + idx, xpu.rsqrt(x.to(tl.float32)).to(OUT_DT))


class _bf16_fast:
    """Enable the faster bf16 store path.

    The env is read at triton compile time, so wrapping the launch that first
    compiles a bf16 kernel is enough; it is restored afterward.
    """

    def __init__(self, on):
        self.on = on
        self.prev = None

    def __enter__(self):
        if self.on:
            self.prev = os.environ.get("TRITONXPU_BF16_FAST")
            os.environ["TRITONXPU_BF16_FAST"] = "1"

    def __exit__(self, *exc):
        if self.on:
            if self.prev is None:
                os.environ.pop("TRITONXPU_BF16_FAST", None)
            else:
                os.environ["TRITONXPU_BF16_FAST"] = self.prev
        return False


@triton.jit
def _foreach_unary_kernel_xpu(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One launch, ``NT`` tensors; 2D grid ``(NT, chunks)``.

    ``meta_ptr`` (int64): ``[0,NT)`` input ptrs, ``[NT,2NT)`` output ptrs,
    ``[2NT,3NT)`` element counts. When every tensor shares one numel that
    divides ``BLOCK``, the grid covers each exactly, so ``NEED_MASK`` is False
    and load/store run mask-free (the masked path is slower even when the mask
    is trivially true).
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        n_elements = tl.load(meta_ptr + 2 * NT + t)
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT), mask=mask)
    else:
        x = tl.load(in_ptr + idx)
        tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT))


_BIG = 1 << 20  # per-tensor direct-arg launch at/above this numel


def _pick_block_small(max_numel):
    """(block, num_warps, buffer_size_limit) for the fused metadata path.

    Only mid/small groups reach here (big ones use the direct-arg path). Small
    tensors get one chunk each -- they are launch/host bound, so a wider tile
    buys nothing.
    """
    if max_numel >= (1 << 14):
        return 8192, 8, 4096
    block = min(8192, triton.next_power_of_2(max_numel))
    return block, 4, 0


def _launch_direct(srcs, dsts, out_dtype):
    """Big shapes: one direct-arg launch per tensor.

    rsqrt is pure-bandwidth and needs direct pointers to hit full block-DMA. The
    per-tensor launch host cost is negligible against a 16M-element transfer.
    """
    BLOCK, num_warps, bsl = 131072, 32, 8192
    out_tl = tl_dtype(out_dtype)
    with _bf16_fast(out_dtype == torch.bfloat16):
        for s, d in zip(srcs, dsts):
            n = s.numel()
            need_mask = (n % BLOCK) != 0
            grid = (triton.cdiv(n, BLOCK),)
            _rsqrt_flat_kernel[grid](
                s,
                d,
                n,
                out_tl,
                BLOCK,
                need_mask,
                num_warps=num_warps,
                buffer_size_limit=bsl,
            )


def _launch_fused(srcs, dsts, in_dtype, out_dtype, device):
    """Mid/small shapes: one fused launch over the whole group (NT tensors).

    Pointers come through a metadata buffer. The per-launch host cost of a
    per-tensor split dominates at these sizes, so fusing wins even though the
    indirect pointer caps bandwidth.
    """
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    block, num_warps, bsl = _pick_block_small(max_numel)
    nt = len(srcs)
    uniform = min(numels) == max_numel
    need_mask = not (uniform and (max_numel % block == 0))
    meta = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, triton.cdiv(max_numel, block))
    kwargs = {"num_warps": num_warps}
    if bsl:
        kwargs["buffer_size_limit"] = bsl
    with _bf16_fast(out_dtype == torch.bfloat16):
        _foreach_unary_kernel_xpu[grid](
            meta,
            _rsqrt_sfu,
            nt,
            tl_dtype(in_dtype),
            tl_dtype(out_dtype),
            block,
            need_mask,
            **kwargs,
        )


def _launch_group_xpu(srcs, dsts, in_dtype, device):
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    out_dtype = dsts[0].dtype
    if max_numel >= _BIG:
        _launch_direct(srcs, dsts, out_dtype)
    else:
        _launch_fused(srcs, dsts, in_dtype, out_dtype, device)
    _STATS.executor_launches += 1
    _STATS.groups += 1


def _foreach_rsqrt_impl(tensors, inplace):
    tensors = check_tensor_list(tensors)
    # Complex is rare; defer to the reference executor so the complex kernel and
    # dtype policy stay bit-identical.
    if any(t.dtype.is_complex for t in tensors):
        return foreach_unary(
            tensors,
            _rsqrt_sfu,
            complex_fn=c_rsqrt,
            out_dtype_fn=int_to_float,
            inplace=inplace,
            allowed_dtypes=NO_COMPLEX32,
        )

    _STATS.reset()
    results = [None] * len(tensors)
    buckets = {}
    writebacks = []
    for i, t in enumerate(tensors):
        dtype = t.dtype
        check_dtype_supported(dtype, NO_COMPLEX32)
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
                _STATS.staged_inputs += 1
            results[i] = t
        else:
            src = t if is_flat_addressable(t) else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            dst = torch.empty_like(src, dtype=out_dtype)
            results[i] = dst
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [])
        bucket[0].append(src)
        bucket[1].append(dst)

    for (device, dtype), (srcs, dsts) in buckets.items():
        _launch_group_xpu(srcs, dsts, dtype, device)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1

    return results


def _foreach_rsqrt(self):
    logger.debug("GEMS _FOREACH_RSQRT")
    return _foreach_rsqrt_impl(self, inplace=False)


def _foreach_rsqrt_(self):
    logger.debug("GEMS _FOREACH_RSQRT_")
    _foreach_rsqrt_impl(self, inplace=True)
    # In-place schema returns ``()``; returning the list makes the dispatcher
    # reject the kernel.
    return None
