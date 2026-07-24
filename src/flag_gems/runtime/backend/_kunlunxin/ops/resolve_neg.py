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

# Kunlunxin (XPU) override of `resolve_neg`.
#
# `resolve_neg(A)` materializes a neg-bit tensor: out = -raw (matching CPU/CUDA
# PyTorch semantics; the generic path uses `neg_func(A)`). The benchmark input
# is `x.conj().imag` -> a stride-2 float32 view (into contiguous complex storage)
# with the neg bit set. The generic `neg_func` runs the elementwise neg on that
# strided view through the un-tuned generic kernel -> discrete (offsetState=-1)
# access -> IR baseline `harness/perf_ir_3/ir-resolve_neg-dev3.log` gems speedup
# 0.003-0.30 ([4096,4096] 48.9ms, [10000,65536] 1860ms).
#
# Fix: when A is the imaginary view of a contiguous complex tensor, its logical
# element j reads raw float32 storage index (storage_offset + 2*j). A dedicated
# kernel with a large BLOCK reads that (stride-2) raw and writes a dense stride-1
# output, negating -> matches neg_func exactly but ~140x faster than the generic
# discrete path ([4096,4096] 48.9ms -> 0.34ms). Anything else falls back to
# neg_func (correctness preserved).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.neg import neg_func
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _neg_imag_kernel(fin, fout, so, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < N
    x = tl.load(fin + so + 2 * j, mask=m)
    tl.store(fout + j, -x, mask=m)


def _contig_strides(shape):
    s = []
    acc = 1
    for d in reversed(shape):
        s.append(acc)
        acc *= d
    return tuple(reversed(s))


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN RESOLVE_NEG")
    if not A.is_neg():
        return A
    # Fast path: A is the imaginary view of a contiguous complex64 tensor, i.e.
    # storage_offset 1 and strides = 2 * (contiguous strides of A.shape).
    fast = (
        A.dtype == torch.float32
        and A.storage_offset() == 1
        and A.stride() == tuple(2 * s for s in _contig_strides(A.shape))
    )
    if not fast:
        return neg_func(A)
    N = A.numel()
    total = A.untyped_storage().nbytes() // 4
    fin = torch.empty(0, dtype=torch.float32, device=A.device)
    fin.set_(A.untyped_storage(), 0, (total,), (1,))
    out = torch.empty(A.shape, dtype=torch.float32, device=A.device)
    fout = out.reshape(-1)
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    with torch_device_fn.device(A.device):
        _neg_imag_kernel[grid](
            fin, fout, A.storage_offset(), N, BLOCK=BLOCK, num_warps=8
        )
    return out
