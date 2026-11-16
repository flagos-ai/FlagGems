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

"""Kunlunxin (XPU) implementation of ``max_pool1d`` (values only, no indices).

Two problems with the generic kernel on this backend:

1. Correctness.  The generic kernel implements the -inf padding with
   ``tl.load(..., mask=..., other=min_val)``, and this backend silently ignores
   the ``other`` clause of masked loads (KERNEL_OPT_EXPERIENCE #4.3-6).  Every
   output window that touches the left padding therefore reads neighbouring
   memory and can return a value outside the window (the same defect that the
   existing ``max_pool1d_with_indices`` override fixes).  Here the address is
   clamped in-bounds first and the window membership is applied explicitly.
2. Speed.  ``@triton.autotune`` re-benchmarks 5 configs for every new
   (out_l, total, k, s) key (do_bench also zeroes a 256 MB cache-flush buffer,
   which dominated the pre-profile), and the per-lane masked loads pay the
   documented masked-memory tax.

Fast path: for the dominant geometry ``k=3, s=2, p=1, dilation=1,
ceil_mode=False, L % 2 == 0`` (out_len == L // 2) the window of output ``o`` is
``[2o-1, 2o, 2o+1]``.  With the input viewed as pairs (fp32 -> int64, fp16/bf16
-> int32) the flat output index equals the flat pair index, so the three taps
are recovered from two unit-stride streams:

    self = pair[q]    (lo=x[2o], hi=x[2o+1])   -> xp + q
    left = pair[q-1]  (hi=x[2o-1])             -> xp + q - 1

``pair[-1]`` of the very first output is out of the allocation, so the head of
the output runs on a clamped (de-vectorised) launch while the rest stays purely
affine -- the same split the proven ``avg_pool1d`` pair kernel uses.

Every other geometry runs the flat one-lane-per-output kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .max_pool1d_with_indices import _parse_pool_params, max_pool1d_output_size

logger = logging.getLogger(__name__)

# outputs covered by the clamped head launch (same measured sweet spot as the
# avg_pool1d pair kernel: <=64 lanes is latency bound, >=512 lanes re-introduces
# enough clamped-index work to cost more than the extra launch)
HEAD_OUT = 256
# below this output count the whole operator is launch bound, so a single
# clamped launch is faster than head + affine body
SINGLE_LAUNCH_MAX = 4096


def _pair_block(total: int) -> int:
    # Measured on P800/XPU3 (SAFE-launch sweep, all three dtypes): 2048 wins up
    # to ~256K outputs, 4096 above that; 8192 is 10-15% worse everywhere.
    if total <= 4096:
        return 1024
    if total <= 262144:
        return 2048
    return 4096


def _flat_block(total: int, elem_size: int) -> int:
    """Measured medians on P800/XPU3 (5 rounds, dil2 + ceil x 3 dtypes).

    512 for launch-bound totals, 1024 for the mid range, 2048 as the general
    sweet spot; 16-bit dtypes past ~256K outputs prefer 4096 (their loads pack
    two elements per word) while fp32 stays at 2048 there.
    """
    if total <= 2048:
        return 512
    if total <= 16384:
        return 1024
    if elem_size >= 4:
        return 1024 if total <= 163840 else 2048
    if total > 262144:
        return 4096
    return 2048


@libentry()
@triton.jit
def _max_pool1d_pair_kernel(
    xp,
    yp,
    total,
    off,
    OW: tl.constexpr,
    SAFE: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``max_pool1d(k=3, s=2, p=1, dilation=1, ceil_mode=False, L % 2 == 0)``.

    MODE 0: fp32 input viewed as int64 pairs.
    MODE 1: fp16 input viewed as int32 pairs.
    MODE 2: bf16 input viewed as int32 pairs.
    """
    lan = tl.arange(0, BLOCK)
    q = (tl.program_id(0) * BLOCK + lan + off).to(tl.int32)
    m = q < total
    if SAFE:
        # clamped index (de-vectorises): only the head launch / small outputs,
        # where lane 0 would otherwise read pair[-1] out of the allocation.
        left = tl.maximum(q - 1, 0)
    else:
        left = q - 1
    v0 = tl.load(xp + q, mask=m, other=0)
    v1 = tl.load(xp + left, mask=m, other=0)
    if MODE == 0:
        v0 = v0.to(tl.uint64, bitcast=True)
        v1 = v1.to(tl.uint64, bitcast=True)
        lo0 = (v0 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        hi0 = (v0 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        hi1 = (v1 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    elif MODE == 1:
        v0 = v0.to(tl.uint32, bitcast=True)
        v1 = v1.to(tl.uint32, bitcast=True)
        lo0 = (v0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        hi0 = (v0 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        hi1 = (v1 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        v0 = v0.to(tl.uint32, bitcast=True)
        v1 = v1.to(tl.uint32, bitcast=True)
        lo0 = ((v0 & 0xFFFF) << 16).to(tl.float32, bitcast=True)
        hi0 = (v0 & 0xFFFF0000).to(tl.float32, bitcast=True)
        hi1 = (v1 & 0xFFFF0000).to(tl.float32, bitcast=True)

    # max of (lo0, hi0, hi1); the leading window of each row has no left tap.
    # NaN propagation matches ATen (a NaN tap always wins, first-max otherwise).
    mx = lo0
    mx = tl.where(hi0 != hi0, hi0, tl.where(hi0 > mx, hi0, mx))
    cand = tl.where(hi1 != hi1, hi1, tl.where(hi1 > mx, hi1, mx))
    mx = tl.where(q % OW > 0, cand, mx)

    if MODE == 0:
        tl.store(yp + q, mx, mask=m)
    elif MODE == 1:
        tl.store(yp + q, mx.to(tl.float16), mask=m)
    else:
        tl.store(yp + q, mx.to(tl.bfloat16), mask=m)


@libentry()
@triton.jit
def _max_pool1d_flat_kernel(
    input_ptr,
    output_ptr,
    total: tl.constexpr,
    planes: tl.constexpr,
    in_l: tl.constexpr,
    out_l: tl.constexpr,
    kernel_l: tl.constexpr,
    stride_l: tl.constexpr,
    padding_l: tl.constexpr,
    dilation_l: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One lane per output, ``kernel_l`` statically unrolled taps.

    Every load is unmasked (the masked-load ``other=`` fallback is ignored by
    this backend, which is the generic kernel's correctness defect).  The taps
    of a window live on the lattice ``start + m * dilation``; clamping a tap
    into ``[lo, hi]`` -- the first/last lattice points inside the input -- is
    numerically harmless because both ends are taps of the same window.  That
    removes the per-tap validity predicate entirely (2 min/max per tap instead
    of the compare/or/and/where chain).  Lanes past ``total`` are clamped to
    the last plane so their (discarded) loads stay in bounds.
    """
    ELEM64: tl.constexpr = input_ptr.dtype.element_ty == tl.float64
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if total > 0x7FFFFFFF or (total // out_l) * in_l > 0x7FFFFFFF:
        offsets = offsets.to(tl.int64)
    output_mask = offsets < total
    nc = tl.minimum(offsets // out_l, planes - 1)
    start = (offsets % out_l) * stride_l - padding_l

    if dilation_l == 1:
        lo = tl.maximum(start, 0)
    else:
        lo = tl.where(
            start >= 0,
            start,
            start + ((-start + dilation_l - 1) // dilation_l) * dilation_l,
        )
    # keeps `hi` in bounds even for ATen-illegal windows without an in-range
    # tap (padding > (kernel-1)*dilation/2); a no-op for legal geometries
    lo = tl.minimum(tl.maximum(lo, 0), in_l - 1)
    hi = lo + ((in_l - 1 - lo) // dilation_l) * dilation_l

    acc_dtype: tl.constexpr = tl.float64 if ELEM64 else tl.float32
    mx = tl.full((BLOCK,), float("-inf"), acc_dtype)
    base = nc * in_l
    for kl in tl.static_range(0, kernel_l):
        il = tl.minimum(tl.maximum(start + kl * dilation_l, lo), hi)
        # fp16/bf16 -> fp32 is exact: the result stays bit-identical.
        value = tl.load(input_ptr + base + il).to(acc_dtype)
        # maxnumf ignores NaNs, so restore ATen's NaN propagation explicitly.
        mx = tl.where(value != value, value, tl.maximum(mx, value))

    tl.store(
        output_ptr + offsets, mx.to(output_ptr.dtype.element_ty), mask=output_mask
    )


def _pair_fast_path(
    input: torch.Tensor, kernel_l, stride_l, padding_l, dilation_l, ceil_mode
) -> bool:
    if ceil_mode or kernel_l != 3 or stride_l != 2 or padding_l != 1:
        return False
    if dilation_l != 1:
        return False
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if input.shape[-1] % 2 != 0:
        return False
    if input.shape[-1] // 2 != max_pool1d_output_size(
        input.shape[-1], kernel_l, stride_l, padding_l, dilation_l, ceil_mode
    ):
        return False
    align = 8 if input.dtype == torch.float32 else 4
    if input.data_ptr() % align != 0:
        return False
    # the pair kernel indexes with int32
    if input.numel() // 2 > 0x7FFFFFFF:
        return False
    return True


def _max_pool1d_pair(input: torch.Tensor) -> torch.Tensor:
    inp = input.contiguous() if not input.is_contiguous() else input
    n, c, lin = inp.shape
    out_len = lin // 2
    out = torch.empty((n, c, out_len), dtype=inp.dtype, device=inp.device)
    total = n * c * out_len
    if total == 0:
        return out

    if inp.dtype == torch.float32:
        mode = 0
        xv = inp.view(torch.int64)
    elif inp.dtype == torch.float16:
        mode = 1
        xv = inp.view(torch.int32)
    else:
        mode = 2
        xv = inp.view(torch.int32)

    block = _pair_block(total)
    head = min(HEAD_OUT, total)
    with torch_device_fn.device(inp.device):
        if total <= SINGLE_LAUNCH_MAX:
            _max_pool1d_pair_kernel[(triton.cdiv(total, block),)](
                xv,
                out,
                total,
                0,
                OW=out_len,
                SAFE=True,
                MODE=mode,
                BLOCK=block,
                num_warps=4,
            )
        else:
            _max_pool1d_pair_kernel[(1,)](
                xv,
                out,
                total,
                0,
                OW=out_len,
                SAFE=True,
                MODE=mode,
                BLOCK=head,
                num_warps=4,
            )
            rest = total - head
            _max_pool1d_pair_kernel[(triton.cdiv(rest, block),)](
                xv,
                out,
                total,
                head,
                OW=out_len,
                SAFE=False,
                MODE=mode,
                BLOCK=block,
                num_warps=4,
            )
    return out


def _max_pool1d_flat(
    x: torch.Tensor, out: torch.Tensor, kernel_l, stride_l, padding_l, dilation_l
) -> None:
    in_l = x.shape[-1]
    out_l = out.shape[-1]
    total = out.numel()
    block = _flat_block(total, x.element_size())
    grid = (triton.cdiv(total, block),)
    with torch_device_fn.device(x.device):
        _max_pool1d_flat_kernel[grid](
            x,
            out,
            total,
            x.numel() // in_l,
            in_l,
            out_l,
            kernel_l,
            stride_l,
            padding_l,
            dilation_l,
            BLOCK=block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=False,
        )


def max_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL1D")

    kernel_l, stride_l, padding_l, dilation_l = _parse_pool_params(
        kernel_size, stride, padding, dilation
    )

    squeeze_batch = input.dim() == 2
    if squeeze_batch:
        input = input.unsqueeze(0)
    input = input.contiguous()
    in_n, in_c, in_l = input.shape
    out_l = max_pool1d_output_size(
        in_l, kernel_l, stride_l, padding_l, dilation_l, ceil_mode
    )

    if out_l <= 0:
        output = torch.empty((in_n, in_c, 0), device=input.device, dtype=input.dtype)
        return output.squeeze(0) if squeeze_batch else output

    if _pair_fast_path(input, kernel_l, stride_l, padding_l, dilation_l, ceil_mode):
        output = _max_pool1d_pair(input)
    else:
        output = torch.empty(
            (in_n, in_c, out_l), device=input.device, dtype=input.dtype
        )
        if output.numel() != 0:
            _max_pool1d_flat(
                input, output, kernel_l, stride_l, padding_l, dilation_l
            )

    return output.squeeze(0) if squeeze_batch else output
