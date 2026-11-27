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
def _engram_gate_fwd_kernel(
    x_ptr,
    kv_ptr,
    w_ptr,
    mask_ptr,
    out_ptr,
    dot_ptr,
    gate_ptr,
    rx_ptr,
    rk_ptr,
    T,
    H,
    inv_H,
    scalar,
    eps,
    clamp,
    HC: tl.constexpr,
    TOK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SAVE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    TAIL: tl.constexpr,
    EXACT: tl.constexpr,
    ACC: tl.constexpr,
):
    pid = tl.program_id(0)
    t0 = pid * TOK
    tt = tl.arange(0, TOK)
    ii = tl.arange(0, HC)
    tok = t0 + tt
    if TAIL:
        rmask = tok < T
        rm3 = rmask[:, None, None]
        rm2 = rmask[:, None]
    else:
        rm2 = None

    # flat element offsets of the first column of each (token, head) row.
    # Kept strictly affine in tok so the backend keeps the strided-DMA lowering.
    x_base = (tok[:, None] * HC + ii[None, :]) * H  # (TOK, HC)
    k_base = (tok[:, None] * (HC + 1) + ii[None, :]) * H  # (TOK, HC)
    w_base = ii * H  # (HC,)
    v_base = (tok * (HC + 1) + HC) * H  # (TOK,)

    acc_x = tl.zeros((TOK, HC), tl.float32)
    acc_k = tl.zeros((TOK, HC), tl.float32)
    acc_d = tl.zeros((TOK, HC), tl.float32)
    cmp_x = tl.zeros((TOK, HC), tl.float32)
    cmp_k = tl.zeros((TOK, HC), tl.float32)
    cmp_d = tl.zeros((TOK, HC), tl.float32)

    for off in range(0, H, BLOCK_H):
        c = off + tl.arange(0, BLOCK_H)
        x3 = x_base[:, :, None] + c[None, None, :]
        k3 = k_base[:, :, None] + c[None, None, :]
        if EXACT:
            if TAIL:
                x = tl.load(x_ptr + x3, mask=rm3, other=0.0).to(tl.float32)
                k = tl.load(kv_ptr + k3, mask=rm3, other=0.0).to(tl.float32)
            else:
                x = tl.load(x_ptr + x3).to(tl.float32)
                k = tl.load(kv_ptr + k3).to(tl.float32)
            w = tl.load(w_ptr + w_base[:, None] + c[None, :])
        else:
            cm = c < H
            if TAIL:
                m3 = rm3 & cm[None, None, :]
            else:
                m3 = cm[None, None, :]
            x = tl.load(x_ptr + x3, mask=m3, other=0.0).to(tl.float32)
            k = tl.load(kv_ptr + k3, mask=m3, other=0.0).to(tl.float32)
            w = tl.load(
                w_ptr + w_base[:, None] + c[None, :], mask=cm[None, :], other=0.0
            )
        if ACC:
            # Neumaier compensation: the cross-chunk chain is the dominant fp32
            # error term, so make it effectively exact.
            y = tl.sum(x * x, 2)
            t = acc_x + y
            cmp_x += tl.where(
                tl.abs(acc_x) >= tl.abs(y), (acc_x - t) + y, (y - t) + acc_x
            )
            acc_x = t
            y = tl.sum(k * k, 2)
            t = acc_k + y
            cmp_k += tl.where(
                tl.abs(acc_k) >= tl.abs(y), (acc_k - t) + y, (y - t) + acc_k
            )
            acc_k = t
            y = tl.sum(x * k * w[None, :, :], 2)
            t = acc_d + y
            cmp_d += tl.where(
                tl.abs(acc_d) >= tl.abs(y), (acc_d - t) + y, (y - t) + acc_d
            )
            acc_d = t
        else:
            acc_x += tl.sum(x * x, 2)
            acc_k += tl.sum(k * k, 2)
            acc_d += tl.sum(x * k * w[None, :, :], 2)

    if ACC:
        acc_x += cmp_x
        acc_k += cmp_k
        acc_d += cmp_d

    rstd_x = 1.0 / tl.sqrt(acc_x * inv_H + eps)
    rstd_k = 1.0 / tl.sqrt(acc_k * inv_H + eps)
    ndot = acc_d * rstd_x * rstd_k * scalar
    sr = tl.sqrt(tl.maximum(tl.abs(ndot), clamp))
    sr = tl.where(ndot >= 0.0, sr, -sr)
    gate = 1.0 / (tl.exp(-sr) + 1.0)

    if SAVE:
        so = tok[:, None] * HC + ii[None, :]
        tl.store(dot_ptr + so, acc_d, mask=rm2)
        tl.store(gate_ptr + so, gate, mask=rm2)
        tl.store(rx_ptr + so, rstd_x, mask=rm2)
        tl.store(rk_ptr + so, rstd_k, mask=rm2)

    if HAS_MASK:
        if TAIL:
            img = tl.load(mask_ptr + tok, mask=rmask, other=0) != 0
        else:
            img = tl.load(mask_ptr + tok) != 0

    for off in range(0, H, BLOCK_H):
        c = off + tl.arange(0, BLOCK_H)
        x3 = x_base[:, :, None] + c[None, None, :]
        v2 = v_base[:, None] + c[None, :]
        if EXACT:
            if TAIL:
                x = tl.load(x_ptr + x3, mask=rm3, other=0.0).to(tl.float32)
                v = tl.load(kv_ptr + v2, mask=rm2, other=0.0).to(tl.float32)
            else:
                x = tl.load(x_ptr + x3).to(tl.float32)
                v = tl.load(kv_ptr + v2).to(tl.float32)
            o = x + v[:, None, :] * gate[:, :, None]
            if HAS_MASK:
                o = tl.where(img[:, None, None], x, o)
            tl.store(out_ptr + x3, o.to(tl.bfloat16), mask=rm3 if TAIL else None)
        else:
            cm = c < H
            if TAIL:
                m3 = rm3 & cm[None, None, :]
                m2 = rm2 & cm[None, :]
            else:
                m3 = cm[None, None, :]
                m2 = cm[None, :]
            x = tl.load(x_ptr + x3, mask=m3, other=0.0).to(tl.float32)
            v = tl.load(kv_ptr + v2, mask=m2, other=0.0).to(tl.float32)
            o = x + v[:, None, :] * gate[:, :, None]
            if HAS_MASK:
                o = tl.where(img[:, None, None], x, o)
            tl.store(out_ptr + x3, o.to(tl.bfloat16), mask=m3)


def engram_gate_fwd(
    hidden_states,
    kv,
    weight_fused,
    eps,
    clamp_value,
    save_for_backward=True,
    image_token_mask=None,
    inplace=False,
):
    """Apply the fused Engram gate and optionally save backward intermediates."""
    tokens, streams, hidden = hidden_states.shape
    if (
        streams != 4
        or kv.shape != (tokens, streams + 1, hidden)
        or weight_fused.shape != (streams, hidden)
    ):
        raise ValueError("expected four hidden streams and a matching key/value table")
    if not all(t.is_contiguous() for t in (hidden_states, kv, weight_fused)):
        raise ValueError("expected contiguous gate inputs")
    num_tokens, hc, H = hidden_states.shape
    out = hidden_states if inplace else torch.empty_like(hidden_states)
    if save_for_backward:
        shape = (num_tokens, hc)
        dev = hidden_states.device
        dot = torch.empty(shape, dtype=torch.float32, device=dev)
        gate = torch.empty(shape, dtype=torch.float32, device=dev)
        rstd_x = torch.empty(shape, dtype=torch.float32, device=dev)
        rstd_k = torch.empty(shape, dtype=torch.float32, device=dev)
    else:
        dot = gate = rstd_x = rstd_k = None

    if num_tokens > 0:
        # every workload width is a multiple of 128; keep the H loop mask-free by
        # picking a chunk size that divides H (a masked path exists otherwise).
        block_h = 128 if H < 768 else 256
        if H % block_h != 0:
            block_h = 128
        # Short-wide shapes are dominated by the fixed per-program cost, which is
        # proportional to the grid size, so cut the program count there. Only
        # large grids are retiled: at num_tokens=512 the grid is already just 64
        # programs, so halving it would halve occupancy on 64 vector cores.
        tok = 16 if (H <= 384 and num_tokens >= 1024) else 8
        # Deep shapes: keep the proven 8192-element footprint and trade the tile
        # aspect ratio instead (64 rows x 128 rather than 32 rows x 256).
        if H >= 768 and num_tokens >= 1024 and H % 128 == 0:
            tok = 16
            block_h = 128
        launch(
            hidden_states,
            kv,
            weight_fused,
            eps,
            clamp_value,
            out,
            dot,
            gate,
            rstd_x,
            rstd_k,
            image_token_mask,
            save_for_backward,
            tok_per_block=tok,
            block_h=block_h,
            acc=1,
        )

    return out, dot, gate, rstd_x, rstd_k


def launch(
    hidden_states,
    kv,
    weight_fused,
    eps,
    clamp_value,
    out,
    dot,
    gate,
    rstd_x,
    rstd_k,
    image_token_mask,
    save_for_backward,
    tok_per_block=8,
    block_h=256,
    acc=0,
):
    num_tokens, hc, H = hidden_states.shape
    grid = ((num_tokens + tok_per_block - 1) // tok_per_block,)
    dummy = weight_fused
    m_arg = dummy if image_token_mask is None else image_token_mask.view(torch.int8)
    _engram_gate_fwd_kernel[grid](
        hidden_states,
        kv,
        weight_fused,
        m_arg,
        out,
        dummy if dot is None else dot,
        dummy if gate is None else gate,
        dummy if rstd_x is None else rstd_x,
        dummy if rstd_k is None else rstd_k,
        num_tokens,
        H,
        1.0 / H,
        H**-0.5,
        eps,
        clamp_value,
        HC=hc,
        TOK=tok_per_block,
        BLOCK_H=block_h,
        SAVE=save_for_backward,
        HAS_MASK=image_token_mask is not None,
        TAIL=(num_tokens % tok_per_block) != 0,
        EXACT=(H % block_h) == 0,
        ACC=acc,
    )
