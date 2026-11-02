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

import torch
import triton
import triton.language as tl


@triton.jit
def _special_round_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    MODE: tl.constexpr,  # 0: decimals == 0, 1: decimals > 0, 2: decimals < 0
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # .cg bypasses L1 for these streaming accesses; measured +1-1.5% bandwidth.
    x = tl.load(x_ptr + offs, mask=mask, cache_modifier=".cg")

    if IS_FP64:
        # Exact fp64 path (torch computes round with decimals in fp64 for fp64 inputs).
        if MODE == 0:
            xf = x
        elif MODE == 1:
            xf = x * scale
        else:
            xf = x / scale
        sign = tl.where(xf >= 0.0, 1.0, -1.0)
        ax = tl.math.abs(xf)
        fl = tl.math.floor(ax)
        frac = ax - fl
        even = (fl % 2.0) == 0.0
        ra = tl.where(
            frac > 0.5,
            fl + 1.0,
            tl.where(frac < 0.5, fl, tl.where(even, fl, fl + 1.0)),
        )
        res = sign * ra
        if MODE == 1:
            res = res / scale
        elif MODE == 2:
            res = res * scale
        res = tl.where(res == 0.0, 0.0, res)
        tl.store(out_ptr + offs, res, mask=mask, cache_modifier=".cg")
    else:
        # FP16/BF16/FP32: torch computes the intermediate product/quotient and the
        # final rescale in fp32.  rint is round-half-to-even, exactly torch's round.
        if MODE == 0:
            xf = x.to(tl.float32)
        elif MODE == 1:
            xf = x.to(tl.float32) * scale
        else:
            xf = x.to(tl.float32) / scale
        r = tl.extra.libdevice.rint(xf)
        if MODE == 1:
            r = r / scale
        elif MODE == 2:
            r = r * scale
        # The backend converts -0.0 to +inf when storing to fp16/bf16; clamp zeros.
        r = tl.where(r == 0.0, 0.0, r)
        tl.store(out_ptr + offs, r.to(x.dtype), mask=mask, cache_modifier=".cg")


def special_round_out(self, out, *, decimals=0):
    n = out.numel()
    if n == 0:
        return out
    if decimals == 0:
        mode, scale = 0, 1.0
    elif decimals > 0:
        mode, scale = 1, float(10**decimals)
    else:
        mode, scale = 2, float(10 ** (-decimals))
    # Launch geometry tuned on Iluvatar BI-V150 via target do_bench: fp32 large
    # tensors peak at 2 elems/thread (256/4 warps; 1024/16 warps ties at 1G and
    # wins ~0.4% at 16M); fp16/bf16 and small tensors do best with 1024/8 warps.
    if n <= 65536:
        block_size, num_warps = 1024, 8
    elif self.dtype == torch.float32:
        block_size, num_warps = 1024, 16
    else:
        block_size, num_warps = 1024, 8
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _special_round_kernel[grid](
        self,
        out,
        scale,
        n,
        MODE=mode,
        IS_FP64=(self.dtype == torch.float64),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
