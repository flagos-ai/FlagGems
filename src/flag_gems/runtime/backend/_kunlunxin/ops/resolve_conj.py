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

# Kunlunxin (XPU) override of `resolve_conj`.
#
# `resolve_conj(A)` materializes a conjugated complex tensor: out = (re, -im).
# The generic path (`torch.complex(A.real, A.imag.neg())` / the generic triton
# kernels that take separate stride-2 x_real / x_imag pointers) does several
# strided (stride-2) float32 passes over the complex storage -> discrete access
# on XPU -> IR baseline `harness/perf_ir_3/ir-resolve_conj-dev5.log` gems speedup
# 0.007-0.040 ([4096,4096] 21.6ms, [10000,65536] 834ms).
#
# Fix: for a *contiguous* complex64 conj tensor the raw storage is one
# contiguous float32 stream [re0, im0, re1, im1, ...]. A single 1D contiguous
# kernel copies it while negating the odd (imaginary) lanes -> ONE stride-1
# block DMA in and out (no strided access). triton reads raw storage ignoring
# the lazy conj bit, so negating the imaginary lanes exactly materializes
# conj(A). Non-contiguous / non-complex64 inputs fall back to the correct
# torch.complex path.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _conj_flat_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    out = tl.where((i % 2) == 1, -x, x)
    tl.store(fout + i, out, mask=m)


def _f32_storage_view(A, nelems, off):
    st = A.untyped_storage()
    f = torch.empty(0, dtype=torch.float32, device=A.device)
    f.set_(st, off, (nelems,), (1,))
    return f


def resolve_conj(A: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN RESOLVE_CONJ")
    if not A.is_conj():
        return A
    if A.dtype == torch.complex64 and A.is_contiguous():
        n2 = A.numel() * 2
        fin = _f32_storage_view(A, n2, A.storage_offset() * 2)
        out = torch.empty_like(A)
        fout = out.view(torch.float32).reshape(-1)
        BLOCK = 8192
        grid = (triton.cdiv(n2, BLOCK),)
        with torch_device_fn.device(A.device):
            _conj_flat_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)
        return out
    return torch.complex(A.real, A.imag.neg())
