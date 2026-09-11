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


@libentry()
@triton.jit(
    do_not_specialize=[
        "D_in",
        "H_in",
        "W_in",
        "pad_l",
        "pad_t",
        "pad_f",
        "stride_nc",
        "stride_xd",
        "stride_xh",
        "NCD_total",
    ],
)
def replicationpad3d_kernel(
    x_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    pad_l,
    pad_t,
    pad_f,
    stride_nc,
    stride_xd,
    stride_xh,
    NCD_total,
    NUM_HW_BLOCKS: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    HW_OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    for ncd in tl.range(pid, NCD_total, num_pids):
        d_out = ncd % D_out
        nc_idx = ncd // D_out

        iz = d_out - pad_f
        iz = tl.where(iz < 0, 0, iz)
        iz = tl.where(iz > D_in - 1, D_in - 1, iz)

        x_base = x_ptr + nc_idx * stride_nc + iz * stride_xd
        out_base = ncd * HW_OUT

        for hw_block in tl.range(0, NUM_HW_BLOCKS):
            hw_off = hw_block * BLOCK + arange
            mask = hw_off < HW_OUT

            h_out = hw_off // W_out
            w_out = hw_off % W_out

            iy = h_out - pad_t
            iy = tl.where(iy < 0, 0, iy)
            iy = tl.where(iy > H_in - 1, H_in - 1, iy)

            ix = w_out - pad_l
            ix = tl.where(ix < 0, 0, ix)
            ix = tl.where(ix > W_in - 1, W_in - 1, ix)

            x_offset = iy * stride_xh + ix
            vals = tl.load(x_base + x_offset, mask=mask)
            tl.store(out_ptr + out_base + hw_off, vals, mask=mask)


def replication_pad3d(x, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD3D")
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = pad_f = pad_ba = padding
    else:
        pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding

    is_4d = x.ndim == 4
    if is_4d:
        x = x.unsqueeze(0)

    x = x.contiguous()
    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in + pad_f + pad_ba
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    HW_out = H_out * W_out
    NCD_total = N * C * D_out

    BLOCK = triton.next_power_of_2(HW_out)
    if NCD_total <= NUM_SIPS * 2:
        BLOCK = min(BLOCK, 1024)
    elif HW_out > 8192:
        BLOCK = min(BLOCK, 4096)
    else:
        BLOCK = min(BLOCK, 2048)
    if BLOCK < 512:
        BLOCK = 512
    NUM_HW_BLOCKS = triton.cdiv(HW_out, BLOCK)

    stride_nc = x.stride(1)

    grid_size = min(NCD_total, NUM_SIPS * 2)

    with torch_device_fn.device(x.device):
        replicationpad3d_kernel[(grid_size,)](
            x,
            out,
            D_in,
            H_in,
            W_in,
            pad_l,
            pad_t,
            pad_f,
            stride_nc,
            x.stride(2),
            x.stride(3),
            NCD_total,
            NUM_HW_BLOCKS=NUM_HW_BLOCKS,
            D_out=D_out,
            H_out=H_out,
            W_out=W_out,
            HW_OUT=HW_out,
            BLOCK=BLOCK,
            num_warps=4,
        )

    return out.squeeze(0) if is_4d else out
