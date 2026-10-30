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
#
# Fast-path kernel v2 (`_neg_imag_xor_kernel`): the stride-2 fp32 read of v1
# tops out at ~440 GB/s (discrete access on this XPU), while a contiguous copy
# reaches ~1015 GB/s. Each (real, imag) fp32 pair sits in one contiguous int64
# word (imag = high 32 bits), so we load CONTIGUOUS int64 words (full-bandwidth
# block DMA) and flip the sign bit of the extracted imag bits with an integer
# XOR (0x80000000) — no float negation in-kernel. Bit-exact vs v1 (verified).
# A float `-x` on the bitcast imag would be faster still on paper but
# miscompiles here, and `tl.split`-based pair extraction crashes the backend,
# so the int64+XOR form is the verified optimum.
#
# Config: BLOCK scales with N — 32768/nw=2 for N>=16384 (0.239ms on N=16M) and
# 512/nw=2 for tiny N (0.0055ms on N=4096). Official core benchmark (complex64,
# dtype-equal-weight mean): baseline 0.6695x -> post 0.7565x
# (64x64 1.110 / 1024x1024 0.678 / 4096x4096 0.619 / 64x512x512 0.619).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.neg import neg_func
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_SIGN_FLIP = tl.constexpr(0x80000000)


@triton.jit
def _neg_imag_xor_kernel(fin, fout, N, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr,
                         SIGN: tl.constexpr):
    # fin points at the raw storage viewed as int64 words (each word = one
    # (real, imag) fp32 pair). imag bits = high 32 of the word; -imag = flip
    # the sign bit. out is the same buffer viewed as int32/uint32.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        m = offs < N
        w = tl.load(fin + offs, mask=m, other=0)
    else:
        w = tl.load(fin + offs)
    hi = w.to(tl.uint64) >> 32
    lo = hi.to(tl.uint32) ^ SIGN
    if NEED_MASK:
        tl.store(fout + offs, lo, mask=m)
    else:
        tl.store(fout + offs, lo)


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
    fin = torch.empty(0, dtype=torch.int64, device=A.device)
    fin.set_(A.untyped_storage(), 0, (total // 2,), (1,))
    out = torch.empty(A.shape, dtype=torch.float32, device=A.device)
    # Same-size dtype view (bit reinterpretation), NOT `.to(torch.int32)` which
    # does a numeric conversion COPY and would leave `out` unwritten.
    fout = out.reshape(-1).view(torch.int32)
    # Tuned on the official core benchmark shapes ([64,64] N=4096, [1024,1024]
    # N=1M, [4096,4096] N=16M): BLOCK=32768/nw=2 -> 0.239ms on N=16M (vs 0.262ms
    # at 16384) and BLOCK=512/nw=2 -> 0.0055ms on N=4096 (vs 0.0088ms at 16384).
    # Large fixed BLOCK is launch-wasteful on tiny inputs, so scale BLOCK with N.
    if N < 16384:
        BLOCK, num_warps = 512, 2
    else:
        BLOCK, num_warps = 32768, 2
    grid = (triton.cdiv(N, BLOCK),)
    with torch_device_fn.device(A.device):
        _neg_imag_xor_kernel[grid](
            fin, fout, N, BLOCK=BLOCK, NEED_MASK=(N % BLOCK != 0),
            SIGN=_SIGN_FLIP, num_warps=num_warps,
        )
    return out
