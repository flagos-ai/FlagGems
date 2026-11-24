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
def _fadd_flat(
    X,
    Y,
    OUT,
    n_elements,
    alpha,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    NATIVE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        if NATIVE:
            acc = x + y
        else:
            acc = x.to(tl.float32) + alpha * y.to(tl.float32)
        tl.store(OUT + offs, acc.to(OUT.dtype.element_ty))
    else:
        m = offs < n_elements
        x = tl.load(X + offs, mask=m)
        y = tl.load(Y + offs, mask=m)
        if NATIVE:
            acc = x + y
        else:
            acc = x.to(tl.float32) + alpha * y.to(tl.float32)
        tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _fadd_tiny3(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    """Single-program, three-argument form (n <= 8192, exact numeric path)."""
    offs = tl.arange(0, BLOCK)
    if BLOCK == N:
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        tl.store(OUT + offs, (x + y).to(OUT.dtype.element_ty))
    else:
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.load(Y + offs, mask=m)
        tl.store(OUT + offs, (x + y).to(OUT.dtype.element_ty), mask=m)


def foreach_add_list(self, other, *, alpha=1):
    if (
        alpha != 1
        or not isinstance(self, (list, tuple))
        or not isinstance(other, (list, tuple))
        or not self
        or len(self) != len(other)
        or any(
            x.device.type != "npu"
            or x.device != self[0].device
            or y.device != x.device
            or x.dtype != torch.bfloat16
            or y.dtype != x.dtype
            or x.shape != y.shape
            or not x.is_contiguous()
            or not y.is_contiguous()
            or x.numel() >= 2**31
            or x.is_neg()
            or y.is_neg()
            for x, y in zip(self, other)
        )
    ):
        from flag_gems.ops._foreach_binary import _foreach_add_List

        return _foreach_add_List(self, other, alpha=alpha)
    outputs = []
    with torch_device_fn.device(self[0].device):
        for x, y in zip(self, other):
            out = torch.empty_like(x)
            outputs.append(out)
            n = x.numel()
            if n and n <= 8192:
                _fadd_tiny3[(1,)](x, y, out, n, triton.next_power_of_2(n), num_warps=1)
            elif n:
                _fadd_flat[(triton.cdiv(n, 8192),)](
                    x, y, out, n, alpha, 8192, n % 8192 == 0, True, num_warps=2
                )
    return outputs
