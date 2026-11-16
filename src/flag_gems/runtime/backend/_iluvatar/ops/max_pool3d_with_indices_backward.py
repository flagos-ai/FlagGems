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
def _scatter_f32_2d(
    grad_ptr,
    idx_ptr,
    acc_ptr,
    G_plane,
    DHW,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < G_plane
    base_g = nc.to(tl.int64) * G_plane
    g = tl.load(grad_ptr + base_g + offs, mask=mask, other=0.0).to(tl.float32)
    idx = tl.load(idx_ptr + base_g + offs, mask=mask, other=0)
    in_pos = nc.to(tl.int64) * DHW + idx
    tl.atomic_add(acc_ptr + in_pos, g, mask=mask, sem="relaxed")


@triton.jit
def _scatter_f32_1d(
    grad_ptr,
    idx_ptr,
    acc_ptr,
    G_plane,
    DHW,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    plane = offs // G_plane
    in_pos = plane.to(tl.int64) * DHW + idx
    tl.atomic_add(acc_ptr + in_pos, g, mask=mask, sem="relaxed")


@triton.jit
def _convert_kernel(
    acc_ptr,
    out_ptr,
    n_elements,
    DST_F16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    if DST_F16:
        v = v.to(tl.float16)
    else:
        v = v.to(tl.bfloat16)
    tl.store(out_ptr + offs, v, mask=mask, cache_modifier=".cs")


@triton.jit
def _convert_kernel_nomask(
    acc_ptr,
    out_ptr,
    DST_F16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(acc_ptr + offs)
    if DST_F16:
        v = v.to(tl.float16)
    else:
        v = v.to(tl.bfloat16)
    tl.store(out_ptr + offs, v, cache_modifier=".cs")


def _tri(v, dflt):
    if v is None:
        return dflt
    if isinstance(v, (tuple, list)):
        if len(v) == 0:
            return dflt
        return tuple(int(x) for x in v)
    return (int(v),) * 3


def max_pool3d_with_indices_backward(
    grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    n_planes = self.shape[0] * self.shape[1]
    G_plane = grad_output.shape[2] * grad_output.shape[3] * grad_output.shape[4]
    DHW = self.shape[2] * self.shape[3] * self.shape[4]
    n_g = grad_output.numel()
    n_i = self.numel()
    dt = self.dtype

    if dt == torch.float32:
        out = torch.zeros_like(self)
        acc = out
    else:
        out = torch.empty_like(self)
        acc = torch.zeros(self.shape, dtype=torch.float32, device=self.device)

    if n_i == 0 or n_g == 0:
        return out

    if G_plane >= 256:
        _scatter_f32_2d[(n_planes, triton.cdiv(G_plane, 128))](
            grad_output,
            indices,
            acc,
            G_plane,
            DHW,
            BLOCK=128,
            num_warps=4,
        )
    else:
        _scatter_f32_1d[(triton.cdiv(n_g, 1024),)](
            grad_output,
            indices,
            acc,
            G_plane,
            DHW,
            n_g,
            BLOCK=1024,
            num_warps=8,
        )
    if dt != torch.float32:
        if n_i % 1024 == 0:
            _convert_kernel_nomask[(n_i // 1024,)](
                acc,
                out,
                DST_F16=(dt == torch.float16),
                BLOCK=1024,
                num_warps=8,
            )
        else:
            _convert_kernel[(triton.cdiv(n_i, 1024),)](
                acc,
                out,
                n_i,
                DST_F16=(dt == torch.float16),
                BLOCK=1024,
                num_warps=8,
            )
    return out
