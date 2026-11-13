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


# GCU400 override of the generic pointwise_dynamic implementation: the generic
# path lowers expit via tl_extra_shim.exp2 (measured 2.4-3.9x slower than the
# native tl.exp on GCU400) and dispatches a block-pointer kernel on only 12
# CTAs (generic ENFLAME codegen config), which together cost ~15ms vs the
# ~0.7ms memory floor on the 512M-element core shape.  The flat grid-stride
# kernel below uses the native exp and saturates all 48 SIPS; math and the
# fp32 compute path are identical to sigmoid.py.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def special_expit_flat_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + off, out, mask=mask)


def _choose_block(N_total, itemsize):
    # do_bench(median) sweep, kernel mode, vs torch.special.expit on GCU400:
    #   tiny shapes want the exact next_pow2 tile (0.0036 vs 0.0041 ms @ N=4096);
    #   mid shapes peak at 65536 (fp32 16.7M 0.0494 vs 0.0541 ms @131072);
    #   the 512M bucket splits by dtype -- 2-byte elements gain ~2% from
    #   131072 (fp16 0.699 vs 0.715 ms, bf16 0.678 vs 0.689 ms) while fp32
    #   loses (1.368 vs 1.351 ms), so only 2-byte dtypes widen there.
    if N_total <= 8192:
        return triton.next_power_of_2(N_total)
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if N_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def special_expit(A):
    logger.debug("GEMS_ENFLAME SPECIAL_EXPIT_FORWARD")
    inp = A if A.is_contiguous() else A.contiguous()
    # aten 的 expit 对整数 / 布尔输入返回 float32；公共实现靠 INT_TO_FLOAT
    # promotion 保证该语义，手写 kernel 必须显式选择输出 dtype，
    # 否则 fp32 结果会按输入 dtype 截断存回整型张量。
    output_dtype = inp.dtype if inp.is_floating_point() else torch.float32
    out = torch.empty_like(inp, dtype=output_dtype)
    N_total = inp.numel()
    if N_total == 0:
        return out

    BLOCK = _choose_block(N_total, out.element_size())
    grid_size = min(triton.cdiv(N_total, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        special_expit_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=2,
        )
    return out
