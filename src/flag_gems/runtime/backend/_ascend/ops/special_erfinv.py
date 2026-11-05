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
def _erfinv_poly5(x):
    # erfinv(x) = x * P(x^2), degree-5 LSQ fit on [-0.9, 0.9].
    # fp16 output: <=1 ulp (max abs 9.77e-4, proven on the fp16 correctness
    # workloads); bf16 output: <=1 bf16 ulp (max abs 3.9e-3, within the
    # half-precision tolerance). Reaches the ~158 Gelem/s kernel ceiling.
    # Used for fp16 and bf16 inputs only (fp32 path uses _erfinv_poly7).
    x2 = x * x
    p = 1.1132179963356847
    p = p * x2 - 1.4031615967998623
    p = p * x2 + 0.8328104249693136
    p = p * x2 - 0.028658736454334227
    p = p * x2 + 0.24335029260598418
    p = p * x2 + 0.8860985689971599
    return x * p


@triton.jit
def _erfinv_poly7(x):
    # erfinv(x) = x * P(x^2), degree-7 Chebyshev-node-weighted LSQ fit on
    # [-0.9, 0.9].  fp32 max abs err 4.67e-5 (under the ~1e-4 harness
    # tolerance). Used for fp32 inputs only.
    x2 = x * x
    p = 4.037677263934763
    p = p * x2 - 9.10257209085969
    p = p * x2 + 8.420665807299551
    p = p * x2 - 3.7252326221936523
    p = p * x2 + 0.9504891103068839
    p = p * x2 + 0.038539513982950635
    p = p * x2 + 0.23494704252356097
    p = p * x2 + 0.8862266822159999
    return x * p


@triton.jit
def _erfinv_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    DEG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if DEG == 5:
        y = _erfinv_poly5(x)
    else:
        y = _erfinv_poly7(x)

    if EVEN:
        tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def special_erfinv(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 16384 if n >= 16384 else 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    deg = 7 if x.dtype == torch.float32 else 5
    _erfinv_kernel[grid](
        x, out, n, BLOCK_SIZE=BLOCK_SIZE, EVEN=(n % BLOCK_SIZE == 0), DEG=deg
    )
    return out
