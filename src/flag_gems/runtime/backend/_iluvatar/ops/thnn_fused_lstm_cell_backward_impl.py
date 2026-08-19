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
def _lstm_cell_bwd_loop(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    ws_ptr,
    gig_ptr,
    gcx_ptr,
    gbias_ptr,
    batch,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    has_bias: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    acc_i = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_f = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in range(0, batch):
        row = b * H
        ws_row = b * 4 * H
        i = tl.load(ws_ptr + ws_row + offs, mask=mask, other=0.0).to(tl.float32)
        f = tl.load(ws_ptr + ws_row + H + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(ws_ptr + ws_row + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(ws_ptr + ws_row + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)
        cyv = tl.load(cy_ptr + row + offs, mask=mask, other=0.0).to(tl.float32)
        ghy = tl.load(grad_hy_ptr + row + offs, mask=mask, other=0.0).to(tl.float32)
        gcy = tl.load(grad_cy_ptr + row + offs, mask=mask, other=0.0).to(tl.float32)
        cxv = tl.load(cx_ptr + row + offs, mask=mask, other=0.0).to(tl.float32)

        tanh_cy = tl.extra.libdevice.tanh(cyv)
        dcy = gcy + ghy * o * (1.0 - tanh_cy * tanh_cy)

        dz_i = dcy * g * i * (1.0 - i)
        dz_f = dcy * cxv * f * (1.0 - f)
        dz_g = dcy * i * (1.0 - g * g)
        dz_o = ghy * tanh_cy * o * (1.0 - o)

        goff = ws_row + offs
        tl.store(gig_ptr + goff, dz_i.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gig_ptr + goff + H, dz_f.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gig_ptr + goff + 2 * H, dz_g.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gig_ptr + goff + 3 * H, dz_o.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(
            gcx_ptr + row + offs, (dcy * f).to(gcx_ptr.dtype.element_ty), mask=mask
        )

        if has_bias:
            acc_i += dz_i
            acc_f += dz_f
            acc_g += dz_g
            acc_o += dz_o

    if has_bias:
        tl.store(gbias_ptr + offs, acc_i.to(gbias_ptr.dtype.element_ty), mask=mask)
        tl.store(gbias_ptr + H + offs, acc_f.to(gbias_ptr.dtype.element_ty), mask=mask)
        tl.store(
            gbias_ptr + 2 * H + offs, acc_g.to(gbias_ptr.dtype.element_ty), mask=mask
        )
        tl.store(
            gbias_ptr + 3 * H + offs, acc_o.to(gbias_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _lstm_cell_bwd_vec(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    ws_ptr,
    gig_ptr,
    gcx_ptr,
    gbias_ptr,
    BATCH: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
    has_bias: tl.constexpr,
):
    BLOCK: tl.constexpr = CHUNK * H
    e = tl.arange(0, BLOCK)
    h = e % H
    r = e // H  # row index within a chunk
    acc_i = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_f = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in tl.static_range(0, BATCH, CHUNK):
        b = c + r
        mask = b < BATCH
        row = b * H + h
        ws_row = b * 4 * H + h

        i = tl.load(ws_ptr + ws_row, mask=mask, other=0.0).to(tl.float32)
        f = tl.load(ws_ptr + ws_row + H, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(ws_ptr + ws_row + 2 * H, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(ws_ptr + ws_row + 3 * H, mask=mask, other=0.0).to(tl.float32)
        cyv = tl.load(cy_ptr + row, mask=mask, other=0.0).to(tl.float32)
        ghy = tl.load(grad_hy_ptr + row, mask=mask, other=0.0).to(tl.float32)
        gcy = tl.load(grad_cy_ptr + row, mask=mask, other=0.0).to(tl.float32)
        cxv = tl.load(cx_ptr + row, mask=mask, other=0.0).to(tl.float32)

        tanh_cy = tl.extra.libdevice.tanh(cyv)
        dcy = gcy + ghy * o * (1.0 - tanh_cy * tanh_cy)

        dz_i = dcy * g * i * (1.0 - i)
        dz_f = dcy * cxv * f * (1.0 - f)
        dz_g = dcy * i * (1.0 - g * g)
        dz_o = ghy * tanh_cy * o * (1.0 - o)

        tl.store(gig_ptr + ws_row, dz_i.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gig_ptr + ws_row + H, dz_f.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gig_ptr + ws_row + 2 * H, dz_g.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gig_ptr + ws_row + 3 * H, dz_o.to(gig_ptr.dtype.element_ty), mask=mask)
        tl.store(gcx_ptr + row, (dcy * f).to(gcx_ptr.dtype.element_ty), mask=mask)

        if has_bias:
            acc_i += dz_i
            acc_f += dz_f
            acc_g += dz_g
            acc_o += dz_o

    if has_bias:
        bi = tl.sum(tl.reshape(acc_i, (CHUNK, H)), axis=0)
        bf = tl.sum(tl.reshape(acc_f, (CHUNK, H)), axis=0)
        bg = tl.sum(tl.reshape(acc_g, (CHUNK, H)), axis=0)
        bo = tl.sum(tl.reshape(acc_o, (CHUNK, H)), axis=0)
        boffs = tl.arange(0, H)
        bmask = boffs < H
        tl.store(gbias_ptr + boffs, bi.to(gbias_ptr.dtype.element_ty), mask=bmask)
        tl.store(gbias_ptr + H + boffs, bf.to(gbias_ptr.dtype.element_ty), mask=bmask)
        tl.store(
            gbias_ptr + 2 * H + boffs, bg.to(gbias_ptr.dtype.element_ty), mask=bmask
        )
        tl.store(
            gbias_ptr + 3 * H + boffs, bo.to(gbias_ptr.dtype.element_ty), mask=bmask
        )


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def run(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    batch = cx.shape[0]
    H = cx.shape[1]
    has_bias_bool = (
        bool(has_bias.item()) if torch.is_tensor(has_bias) else bool(has_bias)
    )

    gig = torch.empty_like(workspace)
    gcx = torch.empty_like(cx)
    gbias = torch.empty(4 * H, dtype=workspace.dtype, device=workspace.device)

    if _is_pow2(batch) and _is_pow2(H):
        CHUNK = max(1, min(batch, 256 // H))
        num_warps = max(1, min(16, (CHUNK * H) // 32))
        _lstm_cell_bwd_vec[(1,)](
            grad_hy,
            grad_cy,
            cx,
            cy,
            workspace,
            gig,
            gcx,
            gbias,
            BATCH=batch,
            H=H,
            CHUNK=CHUNK,
            has_bias=has_bias_bool,
            num_warps=num_warps,
        )
    else:
        BLOCK = triton.next_power_of_2(H)
        num_warps = 1 if BLOCK <= 32 else 2
        _lstm_cell_bwd_loop[(1,)](
            grad_hy,
            grad_cy,
            cx,
            cy,
            workspace,
            gig,
            gcx,
            gbias,
            batch,
            H=H,
            BLOCK=BLOCK,
            has_bias=has_bias_bool,
            num_warps=num_warps,
        )

    if not has_bias_bool:
        gbias = torch.empty(0, dtype=workspace.dtype, device=workspace.device)
    return gig, gcx, gbias
