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

"""GCU400 专属 acosh：contiguous 浮点输入走 1D flat kernel 快路径。"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# GCU400 override of the generic pointwise_dynamic implementation.  The
# previous flat kernel ran BLOCK=1024/num_warps=4 on all sizes, which leaves
# the memory pipeline far under-saturated: the 512M-element core shape took
# 7.4ms (fp16) vs the 1.5ms the same math body reaches with the swept tile
# ladder below (copy floor 0.68ms; native 3.46ms).  The body keeps the
# reference formula log(x + sqrt(x*x - 1)) in fp32; the ablation ladder shows
# sqrt is free in both widths while tl.log is the dominant ALU cost (fp16
# ~1.2ns/elem, fp32 ~3.6ns/elem), so the ladder trades tile size per dtype:
# fp32 peaks at BLOCK=4096/num_warps=4 on every measured bucket (512M: 2853
# vs 3287us @65536/2; 16.7M: 94.2 vs 118.1us), while 16-bit prefers wide
# tiles with num_warps=2 at the big buckets (512M: 1538us @131072/2 vs 1564
# @65536/2) and num_warps=4 at the small ones.


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def acosh_flat_kernel(X, Y, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(X + off, mask=mask, other=1.0).to(tl.float32)
        y = tl.log(x + tl.sqrt(x * x - 1.0))
        tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def acosh_func(x):
    # acosh(x) = log(x + sqrt(x*x - 1))
    # Domain: [1, inf), returns NaN for x < 1 (except x=1 returns 0)
    x_f32 = x.to(tl.float32)
    sqrt_term = tl.sqrt(x_f32 * x_f32 - 1.0)
    result = tl.log(x_f32 + sqrt_term)
    return result.to(x.dtype)


def _choose_block(N_total, dtype):
    # do_bench(median) sweep, kernel mode, vs torch.acosh on GCU400.  fp32 is
    # log-bound and peaks at BLOCK=4096/num_warps=4 on every bucket (512M:
    # 2853us, 16.7M: 94.2us, 0.5M: 7.1us, 65536: 4.2us).  16-bit is memory
    # bound up to ~16.7M and peaks at wide tiles: 65536/2 (16.7M: 57.9us) and
    # 131072/2 for the >2^28 bucket (512M: 1538 vs 1564us @65536); the small
    # buckets want num_warps=4 (65536 elems: 4.2us @4096/4 vs 4.7 @8192/2;
    # 0.5M: 6.0us @8192/4 vs 6.5 @8192/2).
    if dtype == torch.float32:
        if N_total <= 4096:
            return triton.next_power_of_2(N_total), 4
        return 4096, 4
    if N_total <= 4096:
        return triton.next_power_of_2(N_total), 4
    if N_total <= 65536:
        return 4096, 4
    if N_total <= 524288:
        return 8192, 4
    if N_total > (1 << 28):
        return 131072, 2
    return 65536, 2


def _launch_acosh_flat(inp, out, N_total, dtype):
    BLOCK, num_warps = _choose_block(N_total, dtype)
    grid_size = min(triton.cdiv(N_total, BLOCK), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; observed as converterResult=-2 and a misresolved
    # converter helper script during a host-level userland flake window);
    # retrying the launch recompiles and succeeds.  3 attempts keep
    # deterministic breakage loud while riding out the flake (sinh.py /
    # atan2_.py precedent for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                acosh_flat_kernel[(grid_size,)](
                    inp, out, N_total, BLOCK=BLOCK, num_warps=num_warps
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "acosh flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def acosh(A):
    logger.debug("GEMS_ENFLAME ACOSH")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return acosh_func(A)
    inp = A if A.is_contiguous() else A.contiguous()
    out = torch.empty_like(inp)
    N_total = inp.numel()
    if N_total == 0:
        return out
    _launch_acosh_flat(inp, out, N_total, inp.dtype)
    return out


def acosh_(A):
    logger.debug("GEMS_ENFLAME ACOSH_")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        acosh_func(A, out0=A)
        return A
    if not A.is_contiguous():
        A.copy_(acosh(A))
        return A
    N_total = A.numel()
    if N_total == 0:
        return A
    _launch_acosh_flat(A, A, N_total, A.dtype)
    return A
