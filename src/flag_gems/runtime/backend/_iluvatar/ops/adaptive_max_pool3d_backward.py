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
def _adaptive_max_pool3d_backward_scatter(
    grad_out_ptr,
    indices_ptr,
    out_ptr,
    out_plane_size,  # D_out * H_out * W_out
    in_plane_size,  # D_in * H_in * W_in
    BLOCK: tl.constexpr,
):
    pid_plane = tl.program_id(0)
    pid_elem = tl.program_id(1)
    offs = pid_elem * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_plane_size

    src_base = pid_plane * out_plane_size
    idx = tl.load(indices_ptr + src_base + offs, mask=mask, other=0)
    g = tl.load(grad_out_ptr + src_base + offs, mask=mask, other=0.0)

    dst = pid_plane * in_plane_size + idx
    tl.atomic_add(out_ptr + dst, g, mask=mask)


_BLOCK = 256


def run(grad_output, self_input, indices):
    out = torch.zeros_like(self_input)

    D_in, H_in, W_in = self_input.shape[-3:]
    D_out, H_out, W_out = indices.shape[-3:]
    in_plane_size = D_in * H_in * W_in
    out_plane_size = D_out * H_out * W_out
    planes = indices.numel() // out_plane_size

    grid = (planes, triton.cdiv(out_plane_size, _BLOCK))
    _adaptive_max_pool3d_backward_scatter[grid](
        grad_output,
        indices,
        out,
        out_plane_size,
        in_plane_size,
        BLOCK=_BLOCK,
    )
    return out
