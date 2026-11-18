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

"""Nearest-neighbour 3D upsampling backward (gather form).

Two device architectures, both atomic-free and single-launch:

* ``_k2``: exact 2x-per-axis upsample fast path.  Each lane owns one input
  voxel ``(n, c, i, j, k)`` and reads the eight grad_output values of its
  2x2x2 output block as four (BLOCK, 2) contiguous w-pairs, so every load is
  a coalesced vector load and the number of load instructions is minimal.
* ``_kgeneric``: validated-candidate scan for arbitrary scales.  Each lane
  derives the output window of its voxel with the exact float32 ATen source
  index predicate and accumulates it.  Preimages partition the output grid,
  so each grad_output element is read exactly once.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.upsample_nearest3d_backward import (
    upsample_nearest3d_backward_grad_input as default_upsample_nearest3d_backward_grad_input,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _k2(
    go_ptr,
    gi_ptr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = tl.program_id(1)

    total = ID * IH * IW
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    iw = idx % IW
    t = idx // IW
    ih = t % IH
    id_ = t // IH

    # Contiguous 2x NCDHW layout: every stride is a compile-time constant.
    base = nc * (total * 8) + id_ * (IH * IW * 8) + ih * (IW * 4) + iw * 2
    tw = tl.arange(0, 2)
    acc = tl.zeros((BLOCK, 2), dtype=tl.float32)

    if EVEN:
        for kd in tl.static_range(2):
            for kh in tl.static_range(2):
                po = base + kd * (IH * IW * 4) + kh * (IW * 2)
                acc += tl.load(go_ptr + po[:, None] + tw[None, :])
        tl.store(gi_ptr + nc * total + idx, tl.sum(acc, axis=1))
    else:
        mask = idx < total
        for kd in tl.static_range(2):
            for kh in tl.static_range(2):
                po = base + kd * (IH * IW * 4) + kh * (IW * 2)
                acc += tl.load(
                    go_ptr + po[:, None] + tw[None, :], mask=mask[:, None], other=0.0
                )
        tl.store(gi_ptr + nc * total + idx, tl.sum(acc, axis=1), mask=mask)


@triton.jit
def _k2p(
    go_ptr,
    gi_ptr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Exact-2x path indexed by (nc, id) on grid.y and the (ih, iw) plane on
    grid.x, so no per-lane div/mod of the depth index is needed."""
    pid = tl.program_id(0)
    p1 = tl.program_id(1)  # nc * ID + id

    plane = pid * BLOCK + tl.arange(0, BLOCK)
    ih = plane // IW
    # 2x output offset of (ih, iw) inside its depth plane: 4*ih*IW + 2*iw
    base = p1 * (IH * IW * 8) + (plane + ih * IW) * 2
    tw = tl.arange(0, 2)
    acc = tl.zeros((BLOCK, 2), dtype=tl.float32)

    if EVEN:
        for kd in tl.static_range(2):
            for kh in tl.static_range(2):
                po = base + kd * (IH * IW * 4) + kh * (IW * 2)
                acc += tl.load(
                    go_ptr + po[:, None] + tw[None, :],
                    eviction_policy="evict_first",
                )
        tl.store(gi_ptr + p1 * (IH * IW) + plane, tl.sum(acc, axis=1))
    else:
        mask = plane < IH * IW
        for kd in tl.static_range(2):
            for kh in tl.static_range(2):
                po = base + kd * (IH * IW * 4) + kh * (IW * 2)
                acc += tl.load(
                    go_ptr + po[:, None] + tw[None, :],
                    mask=mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
        tl.store(gi_ptr + p1 * (IH * IW) + plane, tl.sum(acc, axis=1), mask=mask)


@triton.jit
def _kgeneric(
    go_ptr,
    gi_ptr,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    go_s0,
    go_s1,
    go_s2,
    go_s3,
    go_s4,
    gi_s0,
    gi_s1,
    gi_s2,
    gi_s3,
    gi_s4,
    r_d,
    r_h,
    r_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = tl.program_id(1)
    n = nc // C
    c = nc % C

    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < ID * IH * IW

    iw = idx % IW
    t = idx // IW
    ih = t % IH
    id_ = t // IH

    od0 = tl.maximum((id_.to(tl.float32) / r_d).to(tl.int32) - 1, 0)
    oh0 = tl.maximum((ih.to(tl.float32) / r_h).to(tl.int32) - 1, 0)
    ow0 = tl.maximum((iw.to(tl.float32) / r_w).to(tl.int32) - 1, 0)

    go_base = n * go_s0 + c * go_s1
    acc = tl.zeros((BLOCK,), dtype=ACC_DTYPE)

    for dd in tl.static_range(MAX_D):
        od = od0 + dd
        src_d = tl.minimum((od.to(tl.float32) * r_d).to(tl.int32), ID - 1)
        vd = (od < OD) & (src_d == id_)
        po = go_base + od * go_s2
        for hh in tl.static_range(MAX_H):
            oh = oh0 + hh
            src_h = tl.minimum((oh.to(tl.float32) * r_h).to(tl.int32), IH - 1)
            vh = vd & (oh < OH) & (src_h == ih)
            po2 = po + oh * go_s3
            for ww in tl.static_range(MAX_W):
                ow = ow0 + ww
                src_w = tl.minimum((ow.to(tl.float32) * r_w).to(tl.int32), IW - 1)
                vw = vh & (ow < OW) & (src_w == iw)
                acc += tl.load(go_ptr + po2 + ow * go_s4, mask=mask & vw, other=0.0)

    off_i = n * gi_s0 + c * gi_s1 + id_ * gi_s2 + ih * gi_s3 + iw * gi_s4
    tl.store(gi_ptr + off_i, acc, mask=mask)


@triton.jit
def _zero_kernel(ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < n)


def _window(o, i):
    if i <= 0:
        return 1
    return max(1, -(-o // i) + 1)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def upsample_nearest3d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    grad_input=None,
):
    logger.debug("GEMS_MTHREADS UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT")

    if not _use_triton_kernel(grad_output):
        return default_upsample_nearest3d_backward_grad_input(
            grad_output,
            output_size,
            input_size,
            scales_d,
            scales_h,
            scales_w,
            grad_input=grad_input,
        )

    with torch_device_fn.device(grad_output.device):
        return _launch_upsample_nearest3d_backward_grad_input(
            grad_output,
            output_size,
            input_size,
            scales_d,
            scales_h,
            scales_w,
            grad_input,
        )


def _launch_upsample_nearest3d_backward_grad_input(
    grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input
):
    if len(input_size) == 5:
        N = int(input_size[0])
        C = int(input_size[1])
        ID = int(input_size[2])
        IH = int(input_size[3])
        IW = int(input_size[4])
    else:
        ID = int(input_size[0])
        IH = int(input_size[1])
        IW = int(input_size[2])
        N = grad_output.shape[0]
        C = grad_output.shape[1]

    OD = int(output_size[0])
    OH = int(output_size[1])
    OW = int(output_size[2])

    if grad_input is None:
        grad_input = torch.empty(
            (N, C, ID, IH, IW), device=grad_output.device, dtype=grad_output.dtype
        )

    total_i = ID * IH * IW
    if total_i == 0:
        return grad_input
    if OD * OH * OW == 0:
        n = grad_input.numel()
        if n:
            _zero_kernel[((n + 1023) // 1024,)](grad_input, n, 1024)
        return grad_input

    if (
        OW == IW + IW
        and OH == IH + IH
        and OD == ID + ID
        and (scales_d is None or scales_d == 2.0)
        and (scales_h is None or scales_h == 2.0)
        and (scales_w is None or scales_w == 2.0)
        and grad_input.is_contiguous()
        and grad_output.is_contiguous()
        and grad_output.dtype != torch.float64
    ):
        blk = 64
        plane = IH * IW
        np1 = ID * N * C
        even = 1 if plane % blk == 0 else 0
        if np1 <= 65535:
            _k2p[((plane + blk - 1) // blk, np1)](
                grad_output,
                grad_input,
                ID,
                IH,
                IW,
                blk,
                even,
                num_warps=2,
            )
        else:
            _k2[((total_i + 127) // 128, N * C)](
                grad_output,
                grad_input,
                ID,
                IH,
                IW,
                128,
                1 if total_i % 128 == 0 else 0,
            )
        return grad_input

    if isinstance(scales_d, (tuple, list)):
        s = list(scales_d) + [None, None, None]
        scales_d, scales_h, scales_w = s[0], s[1], s[2]

    r_d = 1.0 / scales_d if (scales_d is not None and scales_d > 0) else ID / OD
    r_h = 1.0 / scales_h if (scales_h is not None and scales_h > 0) else IH / OH
    r_w = 1.0 / scales_w if (scales_w is not None and scales_w > 0) else IW / OW

    BLOCK = 256
    acc_dtype = tl.float64 if grad_output.dtype == torch.float64 else tl.float32

    _kgeneric[((total_i + BLOCK - 1) // BLOCK, N * C)](
        grad_output,
        grad_input,
        C,
        OD,
        OH,
        OW,
        ID,
        IH,
        IW,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_output.stride(4),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_input.stride(4),
        r_d,
        r_h,
        r_w,
        _window(OD, ID),
        _window(OH, IH),
        _window(OW, IW),
        BLOCK,
        acc_dtype,
    )

    return grad_input
