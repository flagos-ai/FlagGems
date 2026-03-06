# Optimized gated delta rule implementation by FLA.
# Copyright (c) https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule
# Licensed under the MIT license.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from fla.utils import contiguous

# from torch.amp import custom_bwd, custom_fwd

try:
    from .wy_fast_fla import bwd_prepare_wy_repr, fwd_prepare_wy_repr, fwd_recompute_w_u
except ImportError:
    from wy_fast_fla import bwd_prepare_wy_repr, fwd_prepare_wy_repr, fwd_recompute_w_u

# import torch.nn.functional as F
from einops import rearrange
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.exp import safe_exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, tensor_cache


@triton.heuristics(
    {
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_G": lambda args: args["g"] is not None,
        "USE_DW": lambda args: args["dw"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=["BT", "BK", "BV", "USE_G", "USE_DW"],
)
@triton.jit
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dg,
    w,
    dv,
    dw,
    offsets,
    indices,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr,
    USE_DW: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_G:
        dg += i_k * B * H * T
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(
            indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    do += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    h += (i_bh * NT + i_t) * K * V if HEAD_FIRST else (i_tg * H + i_h) * K * V
    dh += (i_bh * NT + i_t) * K * V if HEAD_FIRST else (i_tg * H + i_h) * K * V
    q += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    k += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    dq += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    dk += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    s_qk = K if HEAD_FIRST else H * K
    s_vo = V if HEAD_FIRST else H * V
    s_g = 1 if HEAD_FIRST else H

    # for delta rule only
    if USE_DW:
        dw += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
        dv += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
        w += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = (
        tl.zeros(
            [
                1,
            ],
            dtype=tl.float32,
        )
        if USE_G
        else None
    )
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_do = tl.make_block_ptr(
            do, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
        )
        p_dh = tl.make_block_ptr(
            dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
        )
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += tl.sum(b_h * b_dh)
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(
                dv, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW and not USE_G:
        p_dw = tl.make_block_ptr(
            dw, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(
        q, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(
        dq, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_dk = tl.make_block_ptr(
        dk, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )

    if USE_G:
        b_dg = tl.zeros(
            [
                BT,
            ],
            dtype=tl.float32,
        )
        g += i_bh * T if HEAD_FIRST else bos * H + i_h
        dg += i_bh * T if HEAD_FIRST else bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (s_g,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * s_g)
        b_dg_last *= tl.exp(b_g_last)

        if USE_DW:
            p_w = tl.make_block_ptr(
                w, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_dw = tl.make_block_ptr(
                dw, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dw = b_dw * tl.exp(b_g)[:, None]
            tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
            b_dg -= tl.sum(b_w * b_dw, axis=1)

        b_dq = b_dq * tl.exp(b_g)[:, None] * scale
        b_dg += tl.sum(b_dq * b_q, axis=1)

        b_dk = b_dk * safe_exp(-b_g + b_g_last)[:, None]
        b_dg -= tl.sum(b_k * b_dk, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = (
            tl.where(
                o_i[:, None] >= o_i[None, :],
                b_ds * safe_exp(b_g[:, None] - b_g[None, :]),
                0,
            )
            * scale
        )
        b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
        b_dg += tl.sum(b_ds2, axis=1)
        b_dg -= tl.sum(b_ds2, axis=0)

        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        p_dg = tl.make_block_ptr(dg, (T,), (s_g,), (i_t * BT,), (BT,), (0,))
        # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
        # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
        b_dg = tl.where(o_i < min(BT, T - i_t * BT) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    else:
        b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    dv: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
    head_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = (
        torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device)
        if g is not None
        else None
    )
    dw = torch.empty_like(w) if w is not None else None
    grid = (NK, NT, B * H)

    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        dw=dw,
        dq=dq,
        dk=dk,
        dg=dg,
        offsets=offsets,
        indices=indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg


@triton.heuristics(
    {
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_G": lambda args: args["g"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["BT", "BK", "BV", "USE_G"],
)
@triton.jit
def chunk_gated_delta_rule_bwd_kernel_dhu(
    q,
    k,
    d,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    offsets,
    chunk_offsets,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(
            dht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_dh += tl.load(p_dht, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        if HEAD_FIRST:
            p_dh = tl.make_block_ptr(
                dh + (i_nh * NT + i_t) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
        else:
            p_dh = tl.make_block_ptr(
                dh + ((boh + i_t) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            if HEAD_FIRST:
                bg_last = tl.load(g + i_nh * T + last_idx)
            else:
                bg_last = tl.load(g + (bos + last_idx) * H + i_h)
        else:
            bg_last = None
            last_idx = None
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            if HEAD_FIRST:
                p_q = tl.make_block_ptr(
                    q + i_nh * T * K,
                    (K, T),
                    (1, K),
                    (i_k * BK, i_t * BT + i_c * BC),
                    (BK, BC),
                    (0, 1),
                )
                p_k = tl.make_block_ptr(
                    k + i_nh * T * K,
                    (T, K),
                    (K, 1),
                    (i_t * BT + i_c * BC, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_d = tl.make_block_ptr(
                    d + i_nh * T * K,
                    (K, T),
                    (1, K),
                    (i_k * BK, i_t * BT + i_c * BC),
                    (BK, BC),
                    (0, 1),
                )
                p_dv = tl.make_block_ptr(
                    dv + i_nh * T * V,
                    (T, V),
                    (V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_do = tl.make_block_ptr(
                    do + i_nh * T * V,
                    (T, V),
                    (V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_g = (
                    tl.make_block_ptr(
                        g + i_nh * T, (T,), (1,), (i_t * BT + i_c * BC,), (BC,), (0,)
                    )
                    if USE_G
                    else None
                )
                p_dv2 = tl.make_block_ptr(
                    dv2 + i_nh * T * V,
                    (T, V),
                    (V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
            else:
                p_q = tl.make_block_ptr(
                    q + (bos * H + i_h) * K,
                    (K, T),
                    (1, H * K),
                    (i_k * BK, i_t * BT + i_c * BC),
                    (BK, BC),
                    (0, 1),
                )
                p_k = tl.make_block_ptr(
                    k + (bos * H + i_h) * K,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT + i_c * BC, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_d = tl.make_block_ptr(
                    d + (bos * H + i_h) * K,
                    (K, T),
                    (1, H * K),
                    (i_k * BK, i_t * BT + i_c * BC),
                    (BK, BC),
                    (0, 1),
                )
                p_dv = tl.make_block_ptr(
                    dv + (bos * H + i_h) * V,
                    (T, V),
                    (H * V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_do = tl.make_block_ptr(
                    do + (bos * H + i_h) * V,
                    (T, V),
                    (H * V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_g = (
                    tl.make_block_ptr(
                        g + bos * H + i_h,
                        (T,),
                        (H,),
                        (i_t * BT + i_c * BC,),
                        (BC,),
                        (0,),
                    )
                    if USE_G
                    else None
                )
                p_dv2 = tl.make_block_ptr(
                    dv2 + (bos * H + i_h) * V,
                    (T, V),
                    (H * V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
            b_g = tl.load(p_g, boundary_check=(0,)) if USE_G else None
            # [BK, BT]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (
                (b_q * scale * tl.exp(b_g)[None, :]).to(b_q.dtype)
                if USE_G
                else (b_q * scale).to(b_q.dtype)
            )
            # [BT, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(bg_last - b_g)[:, None]).to(b_k.dtype) if USE_G else b_k
            b_d = (b_d * tl.exp(b_g)[None, :]).to(b_d.dtype) if USE_G else b_d
            # [BT, V]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv2 = b_dv + tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            tl.store(p_dv2, b_dv2.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # [BK, BV]
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)
            b_dh_tmp -= tl.dot(b_d, b_dv2.to(b_q.dtype), allow_tf32=False)
        b_dh *= tl.exp(bg_last) if USE_G else 1
        b_dh += b_dh_tmp

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(
            dh0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_G": lambda args: args["g"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4]],
    key=["BT", "BK", "BV", "USE_G"],
)
@triton.jit
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    do,
    dv,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(
            indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    k += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    do += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    dv += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    s_qk = K if HEAD_FIRST else H * K
    s_vo = V if HEAD_FIRST else H * V
    s_g = 1 if HEAD_FIRST else H

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_q = tl.make_block_ptr(
            q, (K, T), (1, s_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)

    if USE_G:
        g += (i_bh * T) if HEAD_FIRST else (bos * H + i_h)
        p_g = tl.make_block_ptr(g, (T,), (s_g,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

    mask = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    if USE_G:
        b_A = tl.where(mask, b_A * safe_exp(b_g[None, :] - b_g[:, None]) * scale, 0).to(
            do.dtype.element_ty
        )
    else:
        b_A = tl.where(mask, b_A * scale, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(
            do, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_dv = tl.make_block_ptr(
            dv, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *q.shape, do.shape[-1]
    else:
        B, T, H, K, V = *q.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_varlen_inputs(offsets, BT)
        NT = chunk_offsets[-1]

    BK = triton.next_power_of_2(K)
    assert (
        BK <= 256
    ), "current kernel does not support head dimension being larger than 256."
    # H100
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert (
        NK == 1
    ), "NK > 1 is not supported because it involves time-consuming synchronization"

    if head_first:
        dh = q.new_empty(B, H, NT, K, V)
    else:
        dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    grid = (NK, NV, N * H)
    chunk_gated_delta_rule_bwd_kernel_dhu[grid](
        q=q,
        k=k,
        d=w,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
    )
    return dh, dh0, dv2


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_bwd_kernel_dv_local[grid](
        q,
        k,
        g,
        do,
        dv,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
    )
    return dv


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_G": lambda args: args["g"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["BT", "BK", "BV", "USE_G"],
)
@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    v_new,
    g,
    h,
    h0,
    ht,
    offsets,
    chunk_offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(
                h + (i_nh * NT + i_t) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
        else:
            p_h = tl.make_block_ptr(
                h + ((boh + i_t) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            if HEAD_FIRST:
                b_g_last = tl.load(g + i_nh * T + last_idx)
            else:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        else:
            b_g_last = None
            last_idx = None
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden.
        # By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(
                    k + i_nh * T * K,
                    (K, T),
                    (1, K),
                    (i_k * BK, i_t * BT + i_c * BC),
                    (BK, BC),
                    (0, 1),
                )
                p_d = tl.make_block_ptr(
                    d + i_nh * T * K,
                    (T, K),
                    (K, 1),
                    (i_t * BT + i_c * BC, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_v = tl.make_block_ptr(
                    v + i_nh * T * V,
                    (T, V),
                    (V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_v_new = tl.make_block_ptr(
                    v_new + i_nh * T * V,
                    (T, V),
                    (V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_g = (
                    tl.make_block_ptr(
                        g + i_nh * T, (T,), (1,), (i_t * BT + i_c * BC,), (BC,), (0,)
                    )
                    if USE_G
                    else None
                )
            else:
                p_k = tl.make_block_ptr(
                    k + (bos * H + i_h) * K,
                    (K, T),
                    (1, H * K),
                    (i_k * BK, i_t * BT + i_c * BC),
                    (BK, BC),
                    (0, 1),
                )
                p_d = tl.make_block_ptr(
                    d + (bos * H + i_h) * K,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT + i_c * BC, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_v = tl.make_block_ptr(
                    v + (bos * H + i_h) * V,
                    (T, V),
                    (H * V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_v_new = tl.make_block_ptr(
                    v_new + (bos * H + i_h) * V,
                    (T, V),
                    (H * V, 1),
                    (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV),
                    (1, 0),
                )
                p_g = (
                    tl.make_block_ptr(
                        g + bos * H + i_h,
                        (T,),
                        (H,),
                        (i_t * BT + i_c * BC,),
                        (BC,),
                        (0,),
                    )
                    if USE_G
                    else None
                )
            b_g = tl.load(p_g, boundary_check=(0,)) if USE_G else None
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_k = (
                (b_k * tl.exp(b_g_last - b_g)[None, :]).to(b_k.dtype) if USE_G else b_k
            )
            # [BC, BK]
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_d = (b_d * tl.exp(b_g)[:, None]).to(b_d.dtype) if USE_G else b_d
            # [BC, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v2 = b_v - tl.dot(b_d, b_h.to(b_d.dtype))
            # [BK, BV]
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_h *= tl.exp(b_g_last) if USE_G else 1
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_G": lambda args: args["g"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
)
@triton.jit
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(
            indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    s_qk = K if HEAD_FIRST else H * K
    s_vo = V if HEAD_FIRST else H * V
    s_g = 1 if HEAD_FIRST else H
    # offset calculation
    q += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    k += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    v += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    o += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    h += ((i_bh * NT + i_t) * K * V) if HEAD_FIRST else ((i_tg * H + i_h) * K * V)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (s_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, s_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += (i_bh * T) if HEAD_FIRST else (bos * H + i_h)
        p_g = tl.make_block_ptr(g, (T,), (s_g,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        HEAD_FIRST=head_first,
    )
    return o


@tensor_cache
def prepare_varlen_inputs(
    offsets: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(
        [offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], chunk_size)]
    ).cumsum(-1)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_varlen_inputs(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert (
        NK == 1
    ), "NK > 1 is not supported because it involves time-consuming synchronization"

    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)

    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        d=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
):
    BT = chunk_size
    g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first)
    # obtain WY representation. u is actually the new v.
    w, u, Aw, Au = fwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        g=g,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT,
    )

    # obtain output
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    return g, o, Aw, Au, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(
        k=k,
        v=v,
        beta=beta,
        Aw=Aw,
        Au=Au,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        dh=None,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    dk2, dv, db, dg2 = bwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        g=g,
        Aw=Aw,
        Au=Au,
        dw=dw,
        du=dv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(
        dg, chunk_size, reverse=True, offsets=offsets, head_first=head_first
    )
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
    ):
        chunk_size = 64

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat(
                [
                    torch.arange(n)
                    for n in triton.cdiv(
                        offsets[1:] - offsets[:-1], chunk_size
                    ).tolist()
                ]
            )
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        g, o, Aw, Au, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, v, g, beta, Aw, Au, initial_state, offsets, indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.head_first = head_first
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, k, v, g, beta, Aw, Au, initial_state, offsets, indices = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            Aw=Aw,
            Au=Au,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
            head_first=ctx.head_first,
            chunk_size=ctx.chunk_size,
        )
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
        )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        offsets (Optional[torch.LongTensor]):
            Offsets of shape `[N+1]` defining the bos/eos positions of `N` variable-length
            sequences in the batch.
            For example,
            if `offsets` is `[0, 1, 3, 6, 10, 15]`, there are `N=5` sequences
            with lengths 1, 2, 3, 4 and 5 respectively.
            If provided, the inputs are concatenated and the batch size `B` is expected to be 1.
            Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(q,Ali Hatamizadeh
     output_final_state=True,
                                                   offsets=offsets,
                                                   head_first=False)
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, H, T] if head_first=True, or [B, T, H] if head_first=False."

    if offsets is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `offsets`."
                f"Please flatten variable-length inputs before processing."
            )
        if head_first:
            raise RuntimeError(
                "Sequences with variable lengths are not supported for head-first mode"
            )
        if initial_state is not None and initial_state.shape[0] != len(offsets) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(offsets) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state, offsets, head_first
    )
    return o, final_state


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int,  # chunk size
):
    # alias
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k**-0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).exp()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v
    attn = -((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    w = k_cumdecay = attn @ k_beta
    u = v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    return (
        rearrange(o, "b h n c d -> b h (n c) d"),
        rearrange(w, "b h n c d -> b h (n c) d"),
        rearrange(u, "b h n c d -> b h (n c) d"),
    )


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
):
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k**-0.5)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        S = S.clone() * g[:, :, i].exp()[..., None, None]
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q, S)
    return o
