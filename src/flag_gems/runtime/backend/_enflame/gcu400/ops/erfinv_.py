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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# Resident CTA count aligned with the SIP count; 24 CTAs perform no worse overall than 48 CTAs at 2x oversubscription,
# and avoid the repeated scheduling overhead on large shapes (2026-11-16 re-sweep: B4096/c24 ==
# B4096/c48 == B131072/c24 == B131072/c48, error <1%).
NUM_CTAS = 24

# Threshold for fp32 large shapes to use the deep-block streaming config: at 16M, measured B4096 211µs vs B131072 217µs;
# deep blocks take over from 32M on (416→399µs) and the gap grows monotonically with N (512M: 6541→6036µs, -7.7%).
# 16-bit triggers a compiler lowering cliff for any BLOCK > 4096 (+5~7x from B8192/c48 onward),
# so keep B4096/w4 at all times.
_FP32_DEEP_BLOCK_NUMEL = 1 << 25
_DEEP_BLOCK = 131072


@libentry()
@triton.jit(do_not_specialize=["N"])
def erfinv_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        q = 0.5 * x
        t = q * q
        p = 4.296451014e4
        p = p * t - 1.876153977e4
        p = p * t + 3.715949402e3
        p = p * t - 3.111872501e2
        p = p * t + 2.805659723e1
        p = p * t + 3.683845979e0
        p = p * t + 1.859579801e0
        p = p * t + 1.772448984e0
        y_center = q * p
        abx = tl.math.abs(x)
        u = 0.5 * (1.0 - abx)
        v = tl.math.sqrt(-tl.math.log(u)) - 1.6
        w = 7.645191756e-05
        w = w * v - 8.727375877e-04
        w = w * v + 4.518085617e-03
        w = w * v - 1.479333944e-02
        w = w * v + 3.740890474e-02
        w = w * v - 8.644987914e-02
        w = w * v + 1.207493122e0
        w = w * v + 1.006524121e0
        y_tail = tl.where(x < 0.0, -w, w)
        y = tl.where(tl.math.abs(q) <= 0.425, y_center, y_tail)
        tl.store(out_ptr + off, y, mask=mask)


def _choose_config(N, is_fp32):
    if N <= 1024:
        return 1024, 4
    if N <= 32768:
        return triton.next_power_of_2(N), 4
    if is_fp32 and N >= _FP32_DEEP_BLOCK_NUMEL:
        return _DEEP_BLOCK, 2
    return 4096, 4


def _launch(inp, out):
    N = inp.numel()
    block, warps = _choose_config(N, inp.dtype == torch.float32)
    grid_size = min(triton.cdiv(N, block), NUM_CTAS)
    with torch_device_fn.device(inp.device):
        erfinv_kernel[(grid_size,)](
            inp,
            out,
            N,
            BLOCK=block,
            num_warps=warps,
        )
    return out


def _erfinv_out(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


def erfinv(x):
    logger.debug("GEMS_ENFLAME ERFINV")
    return _erfinv_out(x)


def erfinv_(x):
    logger.debug("GEMS_ENFLAME ERFINV_")
    if not x.is_contiguous():
        x.copy_(_erfinv_out(x))
        return x
    return _launch(x, x)
