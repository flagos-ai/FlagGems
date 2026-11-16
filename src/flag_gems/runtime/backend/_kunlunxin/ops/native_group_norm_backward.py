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

"""native_group_norm_backward on Kunlunxin XPU3.

Semantics (aten::native_group_norm_backward): given per-(n, group) `mean` /
`rstd`, the forward was  xhat = (x - mean) * rstd,  y = xhat * weight + bias.
With L = group_size * HxW elements per group and dxhat = dY * weight:

    dX      = rstd * (dxhat - (sum dxhat + xhat * sum(dxhat * xhat)) / L)
    dweight = sum_{n,HW} dY * xhat
    dbias   = sum_{n,HW} dY

`dX` needs two passes over (dY, X): one to form the group sums, one to apply.
`dweight` / `dbias` need only sums already formed in the first pass, so they are
fused into it (a [N, C] f32 partial buffer + a tiny reduce over N, which is
skipped entirely when N == 1).  Net traffic: 4 reads + 1 write instead of the
previous 6 reads + 1 write.

The dominant cost on XPU3 is not the arithmetic but the MASKED memory path:
every load/store in the old kernel carried `idx < HW`, and the masked path
lowers to per-lane select / gather instead of a vector / block-DMA transfer
(measured 7.27 ms/call for (2, 64, 128, 128) fp32, ~7.7 GB/s).  When a chunk
size that divides HW is available (the common power-of-two case) the whole
kernel runs unmasked (`EVEN`), which restores wide vector loads.  Only shapes
whose HW cannot be tiled evenly fall back to the masked leg.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt

# Largest 1-D tile we use.  The old 1024 "cap" from KERNEL_OPT_EXPERIENCE is
# not universal: for this flat stream 128/256/512 are dramatically worse
# (measured 4.96/2.31/0.44 ms on the big shape) and 4096/8192 are best
# (0.091/0.079 ms); 16384 regresses.  `tl.sum` is exact through 8192 lanes.
_BLOCK_CAP = 8192
# `tl.arange(0, n)` is widened to >= 64 lanes by the backend, so an unmasked
# tile narrower than 64 would store past its bounds.  Anything below 64 lanes
# must therefore use the masked leg.
_MIN_EVEN = 64


def _pick_block(hw):
    """Return (block, even) for an HW-sized inner loop.

    `even` means `hw % block == 0`, i.e. no lane masks are needed at all.
    """
    if hw >= _MIN_EVEN:
        best = None
        b = _MIN_EVEN
        while b <= _BLOCK_CAP:
            if hw % b == 0:
                best = b
            b <<= 1
        if best is not None:
            return best, True
    return _MIN_EVEN, False


@libentry()
@triton.jit
def _gnb_dx_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    grad_x,
    PartDW,
    PartDB,
    DW,
    DB,
    num_groups,
    L,
    C,
    HW,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    HAS_W: tl.constexpr,
    DO_DX: tl.constexpr,
    DO_PART: tl.constexpr,
    WB_DIRECT: tl.constexpr,
    DO_DW: tl.constexpr,
    DO_DB: tl.constexpr,
):
    """One program per (n, group); flat 1-D chunks over the group's channels.

    Pass 1 forms the per-channel sums  sum(dY)  and  sum(dY * (x - mean)); the
    group sums needed by dX and the per-channel divisors needed by dW/dB are
    both linear combinations of exactly those two quantities, so a single
    memory pass serves all three outputs.  Pass 2 (only when DO_DX) re-reads
    (dY, X) and writes dX.
    """
    pid = ext.program_id(0)
    group = pid % num_groups
    base = pid * L
    ch_base = group * GROUP_SIZE
    rstd = tl.load(Rstd + pid).to(tl.float32)
    mean = tl.load(Mean + pid).to(tl.float32)

    dx_2 = 0.0
    dx_3 = 0.0
    for c in range(0, GROUP_SIZE):
        if HAS_W:
            w = tl.load(W + ch_base + c).to(tl.float32)
        else:
            w = 1.0
        cbase = base + c * HW
        p_dy = tl.zeros([BLOCK], dtype=tl.float32)
        p_dyx = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, HW, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            if EVEN:
                dy = tl.load(grad_y + cbase + idx).to(tl.float32)
                x = tl.load(X + cbase + idx).to(tl.float32)
            else:
                m = idx < HW
                dy = tl.load(grad_y + cbase + idx, mask=m, other=0.0).to(tl.float32)
                x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
                # OOB lanes may read back garbage under OTHER_SIM; force them
                # to zero so they cannot poison the (unmasked) reduction.
                dy = tl.where(m, dy, 0.0)
                x = tl.where(m, x, 0.0)
            p_dy += dy
            # Factored as dy*x (one mul) and corrected by -mean*s_dy after the
            # reduction, instead of the two-op dy*(x-mean) per lane.
            p_dyx += dy * x

        s_dy = tl.sum(p_dy)
        s_dyx = tl.sum(p_dyx) - mean * s_dy  # sum_k dY * (x - mean)
        dw_c = rstd * s_dyx  # sum_k dY * xhat  (per channel)
        if DO_PART:
            # pid * GROUP_SIZE == n * C + (group * GROUP_SIZE): the global
            # channel index, so the partials are a plain [N * C] buffer.
            tl.store(PartDW + pid * GROUP_SIZE + c, dw_c)
            tl.store(PartDB + pid * GROUP_SIZE + c, s_dy)
        if WB_DIRECT:
            # Only valid when N == 1: this program owns the channel outright.
            if DO_DW:
                tl.store(DW + ch_base + c, dw_c)
            if DO_DB:
                tl.store(DB + ch_base + c, s_dy)
        # group sums:  sum dxhat = sum_c w * sum dy,
        #              sum dxhat*xhat = sum_c w * (rstd * sum dy*(x-mean))
        dx_2 += w * s_dy
        dx_3 += w * dw_c

    if DO_DX:
        inv_l = 1.0 / L
        a = dx_2 * inv_l
        b = dx_3 * inv_l
        # dx = rstd*(w*dy - a - rstd*(x-mean)*b) folded to 2 FMA per element:
        #   dx = c1*dy + c4*x + c5
        rr = rstd * rstd
        for c in range(0, GROUP_SIZE):
            if HAS_W:
                w = tl.load(W + ch_base + c).to(tl.float32)
            else:
                w = 1.0
            c1 = rstd * w
            c4 = -rr * b
            c5 = rr * b * mean - rstd * a
            cbase = base + c * HW
            for off in range(0, HW, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                if EVEN:
                    dy = tl.load(grad_y + cbase + idx).to(tl.float32)
                    x = tl.load(X + cbase + idx).to(tl.float32)
                    dx = c1 * dy + c4 * x + c5
                    tl.store(grad_x + cbase + idx, dx)
                else:
                    m = idx < HW
                    dy = tl.load(
                        grad_y + cbase + idx, mask=m, other=0.0
                    ).to(tl.float32)
                    x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
                    dx = c1 * dy + c4 * x + c5
                    # Pointer clamp alongside the store mask: a masked vector
                    # store whose inactive lanes still point past the buffer
                    # corrupts active lanes on XPU3 (measured rel-err ~1 on
                    # HW=100).  The old kunlunxin kernel used the same -1
                    # trick, and it is required here too.
                    pidx = tl.where(m, cbase + idx, -1)
                    tl.store(grad_x + pidx, dx, m)


@libentry()
@triton.jit
def _gnb_wb_reduce_kernel(
    PartDW,
    PartDB,
    DW,
    DB,
    C,
    N,
    BLOCK_C: tl.constexpr,
    EVEN_C: tl.constexpr,
    DO_DW: tl.constexpr,
    DO_DB: tl.constexpr,
):
    """Fold the [N, C] f32 partials over N, one vector of channels per program.

    A purely SCALAR kernel (no `tl.arange` at all) costs ~0.76 ms on XPU3 even
    for 64 programs x 2 loads -- measured, and independent of grid shape.  A
    64-lane vector body drops the same work to ~0.015 ms, so the tile is kept
    vector even though the reduction itself is trivial.
    """
    pid = ext.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    acc_w = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_C], dtype=tl.float32)
    for n in range(0, N):
        off = n * C + c
        if EVEN_C:
            if DO_DW:
                acc_w += tl.load(PartDW + off)
            if DO_DB:
                acc_b += tl.load(PartDB + off)
        else:
            m = c < C
            if DO_DW:
                acc_w += tl.load(PartDW + off, mask=m, other=0.0)
            if DO_DB:
                acc_b += tl.load(PartDB + off, mask=m, other=0.0)
    if EVEN_C:
        if DO_DW:
            tl.store(DW + c, acc_w)
        if DO_DB:
            tl.store(DB + c, acc_b)
    else:
        m = c < C
        pidx = tl.where(m, c, -1)
        if DO_DW:
            tl.store(DW + pidx, acc_w, m)
        if DO_DB:
            tl.store(DB + pidx, acc_b, m)


def _set_mask_sim(on):
    """The XPU masked-memory path needs its simulator flags (as the previous
    kunlunxin group-norm implementation did).  They only affect compilation, so
    they are toggled around the masked launch and never for the EVEN leg."""
    import os

    if on:
        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
    else:
        os.environ.pop("TRITONXPU_OTHER_SIM", None)
        os.environ.pop("TRITONXPU_STORE_MASK_SIM", None)


def group_norm_backward(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    logger.debug("GEMS NATIVE_GROUP_NORM_BACKWARD (kunlunxin)")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()

    do_dx = bool(output_mask[0])
    do_dw = bool(output_mask[1])
    do_db = bool(output_mask[2])
    group_size = triton.cdiv(C, group) if group else 0

    grad_inp = torch.empty_like(input) if do_dx else None
    weight_grad = torch.empty_like(weight) if do_dw else None
    bias_grad = torch.empty_like(weight) if do_db else None

    if not do_dx and not do_dw and not do_db:
        return None, None, None

    L = group_size * HxW
    # Empty reduction domains -> an empty sum is exactly zero.  Nothing to
    # launch; dW/dB must be materialised as zeros, dX stays empty.
    if N == 0 or C == 0 or HxW == 0:
        if do_dw:
            weight_grad = torch.zeros_like(weight)
        if do_db:
            bias_grad = torch.zeros_like(weight)
        return grad_inp, weight_grad, bias_grad

    block, even = _pick_block(HxW)
    has_w = weight is not None
    n_prog = N * group

    # WB can be finished inside the dX kernel only when a single program owns
    # each channel completely (N == 1); otherwise the [N, C] partials need one
    # extra (tiny) reduction launch.
    wb_direct = (do_dw or do_db) and N == 1
    do_part = (do_dw or do_db) and not wb_direct

    part_dw = None
    part_db = None
    block_c = 0
    even_c = True
    if do_part:
        part_dw = torch.empty((N * C,), dtype=torch.float32, device=input.device)
        part_db = torch.empty((N * C,), dtype=torch.float32, device=input.device)
        block_c = max(_MIN_EVEN, min(triton.next_power_of_2(C), _BLOCK_CAP))
        even_c = C % block_c == 0

    with torch_device_fn.device(input.device):
        if not even or not even_c:
            _set_mask_sim(True)
        _gnb_dx_kernel[(n_prog,)](
            grad_out,
            input,
            weight,
            mean,
            rstd,
            grad_inp,
            part_dw,
            part_db,
            weight_grad,
            bias_grad,
            group,
            L,
            C,
            HxW,
            GROUP_SIZE=group_size,
            BLOCK=block,
            EVEN=even,
            HAS_W=has_w,
            DO_DX=do_dx,
            DO_PART=do_part,
            WB_DIRECT=wb_direct,
            DO_DW=do_dw,
            DO_DB=do_db,
            isCloseUnrollControl=True,
        )
        if do_part:
            _gnb_wb_reduce_kernel[(triton.cdiv(C, block_c),)](
                part_dw,
                part_db,
                weight_grad,
                bias_grad,
                C,
                N,
                BLOCK_C=block_c,
                EVEN_C=even_c,
                DO_DW=do_dw,
                DO_DB=do_db,
            )
        if not even or not even_c:
            _set_mask_sim(False)

    return grad_inp, weight_grad, bias_grad
