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

BLOCK = 1024
SLOTS = 16
TBL_LEN = 64
SMALL_BYTES = 4 * 1024 * 1024
_DTYPE_CODES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_ELEM_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
_META = {}


def _meta(device):
    m = _META.get(device)
    if m is None:
        cpu = torch.empty(TBL_LEN, dtype=torch.int64, pin_memory=True)
        dev = torch.empty(TBL_LEN, dtype=torch.int64, device=device)
        m = (cpu, cpu.numpy(), dev)
        _META[device] = m
    return m


@triton.jit
def _single_entry(x_ptr, a_ptr, b_ptr, n, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = x.to(tl.float32) + value * a.to(tl.float32) * b.to(tl.float32)
    tl.store(x_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_table(tbl, value, DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    x_i = tl.load(tbl + 3 * slot)
    a_i = tl.load(tbl + 3 * slot + 1)
    b_i = tl.load(tbl + 3 * slot + 2)
    n = tl.load(tbl + 48 + slot).to(tl.int32)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if DTYPE == 0:
        xp = x_i.to(tl.pointer_type(tl.float16))
        ap = a_i.to(tl.pointer_type(tl.float16))
        bp = b_i.to(tl.pointer_type(tl.float16))
        x = tl.load(xp + offs, mask=mask)
        a = tl.load(ap + offs, mask=mask)
        b = tl.load(bp + offs, mask=mask)
        out = x.to(tl.float32) + value * a.to(tl.float32) * b.to(tl.float32)
        tl.store(xp + offs, out.to(tl.float16), mask=mask)
    elif DTYPE == 1:
        xp = x_i.to(tl.pointer_type(tl.bfloat16))
        ap = a_i.to(tl.pointer_type(tl.bfloat16))
        bp = b_i.to(tl.pointer_type(tl.bfloat16))
        x = tl.load(xp + offs, mask=mask)
        a = tl.load(ap + offs, mask=mask)
        b = tl.load(bp + offs, mask=mask)
        out = x.to(tl.float32) + value * a.to(tl.float32) * b.to(tl.float32)
        tl.store(xp + offs, out.to(tl.bfloat16), mask=mask)
    else:
        xp = x_i.to(tl.pointer_type(tl.float32))
        ap = a_i.to(tl.pointer_type(tl.float32))
        bp = b_i.to(tl.pointer_type(tl.float32))
        x = tl.load(xp + offs, mask=mask)
        a = tl.load(ap + offs, mask=mask)
        b = tl.load(bp + offs, mask=mask)
        out = x + value * a * b
        tl.store(xp + offs, out, mask=mask)


def foreach_addcmul_scalar_(self, tensor1, tensor2, value=1):
    n_slots = len(self)
    if n_slots == 0:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item()
    v = float(value)
    first = self[0]
    code = _DTYPE_CODES.get(first.dtype)
    base = 0
    while base < n_slots:
        end = base + SLOTS
        if end > n_slots:
            end = n_slots
        cnt = end - base
        max_n = 0
        total = 0
        same = code is not None
        for i in range(base, end):
            x = self[i]
            if x.dtype != first.dtype:
                same = False
            n = x.numel()
            if n > max_n:
                max_n = n
            total += n
        if max_n > 0:
            if same and total * _ELEM_BYTES[first.dtype] <= SMALL_BYTES:
                ptrs = []
                sizes = []
                for i in range(base, end):
                    ptrs.append(self[i].data_ptr())
                    ptrs.append(tensor1[i].data_ptr())
                    ptrs.append(tensor2[i].data_ptr())
                    sizes.append(self[i].numel())
                cpu, host_arr, dev = _meta(first.device)
                host_arr[:] = ptrs + [0] * (48 - 3 * cnt) + sizes + [0] * (SLOTS - cnt)
                dev.copy_(cpu, non_blocking=True)
                blocks = (max_n + BLOCK - 1) // BLOCK
                _fused_table[(blocks, cnt)](
                    dev, v, DTYPE=code, BLOCK=BLOCK, num_warps=4
                )
            else:
                for i in range(base, end):
                    x = self[i]
                    n = x.numel()
                    if n == 0:
                        continue
                    blocks = (n + BLOCK - 1) // BLOCK
                    _single_entry[(blocks,)](
                        x, tensor1[i], tensor2[i], n, v, BLOCK=BLOCK, num_warps=4
                    )
        base = end
    return None
