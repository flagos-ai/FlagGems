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

from flag_gems.runtime import torch_device_fn


@triton.jit
def _clamp_rows(
    x_ptr,
    o_ptr,
    n_rows,
    n_cols,
    in_rs,
    min_v: tl.constexpr,
    max_v: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    MASK_R: tl.constexpr,
    MASK_C: tl.constexpr,
    HAS_MIN: tl.constexpr,
    HAS_MAX: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    r = pid_r * BR + tl.arange(0, BR)
    c = pid_c * BC + tl.arange(0, BC)
    in_off = r[:, None] * in_rs + c[None, :]
    out_off = r[:, None] * n_cols + c[None, :]
    if MASK_R or MASK_C:
        m = (r[:, None] < n_rows) & (c[None, :] < n_cols)
        x = tl.load(x_ptr + in_off, mask=m, other=0)
    else:
        x = tl.load(x_ptr + in_off)
    if HAS_MIN:
        x = tl.where(x < min_v, min_v, x)
    if HAS_MAX:
        x = tl.where(x > max_v, max_v, x)
    if MASK_R or MASK_C:
        tl.store(o_ptr + out_off, x, mask=m)
    else:
        tl.store(o_ptr + out_off, x)


def clamp(A, mini=None, maxi=None):
    # The fast path is for compact columns and scalar bounds exactly
    # representable in the operand dtype.
    def bound(v):
        return v is None or (
            type(v) in (int, float) and -128 <= v <= 128 and v == int(v)
        )

    if (
        A.device.type != "npu"
        or A.ndim != 2
        or A.layout != torch.strided
        or A.dtype not in (torch.bfloat16, torch.float32)
        or A.stride(1) != 1
        or A.numel() >= 2**31
        or A.shape[0] * A.stride(0) >= 2**31
        or A.is_neg()
        or not bound(mini)
        or not bound(maxi)
        or (mini is None and maxi is None)
    ):
        from flag_gems.ops.clamp import clamp as generic_clamp

        return generic_clamp(A, mini, maxi)
    rows, width = A.shape
    out = torch.empty((rows, width), dtype=A.dtype, device=A.device)
    if not A.numel():
        return out
    bc = min(1024, triton.next_power_of_2(width))
    br = min(4, triton.next_power_of_2(rows))
    with torch_device_fn.device(A.device):
        _clamp_rows[(triton.cdiv(width, bc), triton.cdiv(rows, br))](
            A,
            out,
            rows,
            width,
            A.stride(0),
            mini or 0,
            maxi or 0,
            br,
            bc,
            rows % br != 0,
            width % bc != 0,
            mini is not None,
            maxi is not None,
        )
    return out
