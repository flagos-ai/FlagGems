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
def _amp2d_bwd_fused_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    hw_in,
    hw_out,
    ZBLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # One program per (n, c) plane: preload the scatter operands so the
    # idx/grad reads overlap the streaming zero writes, then zero the full
    # input plane, then atomic-add each pooled grad into its argmax location.
    plane = tl.program_id(0)
    out_base = plane * hw_in
    toffs = tl.arange(0, SBLOCK)
    tmask = toffs < hw_out
    g0 = tl.load(grad_ptr + plane * hw_out + toffs, mask=tmask, other=0.0)
    idx0 = tl.load(idx_ptr + plane * hw_out + toffs, mask=tmask, other=0)
    for base in range(0, hw_in, ZBLOCK):
        offs = base + tl.arange(0, ZBLOCK)
        mm = offs < hw_in
        tl.store(out_ptr + out_base + offs, tl.zeros((ZBLOCK,), dtype=DTYPE), mask=mm)
    tl.debug_barrier()
    tl.atomic_add(out_ptr + out_base + idx0, g0, mask=tmask)
    for sbase in range(SBLOCK, hw_out, SBLOCK):
        soffs = sbase + tl.arange(0, SBLOCK)
        sm = soffs < hw_out
        gg = tl.load(grad_ptr + plane * hw_out + soffs, mask=sm, other=0.0)
        ii = tl.load(idx_ptr + plane * hw_out + soffs, mask=sm, other=0)
        tl.atomic_add(out_ptr + out_base + ii, gg, mask=sm)


_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


def adaptive_max_pool2d_backward(grad_output, self, indices):
    out = torch.empty(self.shape, dtype=grad_output.dtype, device=grad_output.device)
    if self.dim() == 4:
        N, C, H_in, W_in = self.shape
    else:
        N, C, H_in, W_in = 1, self.shape[0], self.shape[1], self.shape[2]
    if grad_output.dim() == 4:
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]
    else:
        H_out, W_out = grad_output.shape[1], grad_output.shape[2]
    hw_in = H_in * W_in
    hw_out = H_out * W_out
    planes = N * C
    dt = _TL_DTYPES[grad_output.dtype]

    _amp2d_bwd_fused_kernel[(planes,)](
        grad_output,
        indices,
        out,
        hw_in,
        hw_out,
        ZBLOCK=2048,
        SBLOCK=1024,
        DTYPE=dt,
        num_warps=8,
    )
    return out
