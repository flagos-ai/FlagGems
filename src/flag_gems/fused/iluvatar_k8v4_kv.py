# SPDX-License-Identifier: Apache-2.0
"""Asymmetric K8 (CM8) + V4 (Centered-Mantissa 4-bit with BF16 center/scale) Paged KV Cache &
Attention for Iluvatar BI-V150."""

import torch
import triton
import triton.language as tl

from flag_gems.fused.iluvatar_cm8_kv import (
    _CM8_DECODE_WS,
    Z_CM8,
    _causal_cm8,
    _get_dummy_f32,
    _long_query_cm8,
    _quant_cm8_byte,
    _select_decode_plan,
    reserve_cm8_decode_workspace,
)
from flag_gems.fused.iluvatar_dense_prefill import _owner as _prefill_owner
from flag_gems.fused.iluvatar_dense_prefill import _reserved as _prefill_reserved
from flag_gems.fused.iluvatar_dense_prefill import reserve_prefill_workspace
from flag_gems.ops.flash_decode_splitkv import _decode_merge_kernel, _parallel_merge

Z_V4 = 67.75
K8V4_PAGE_BYTES = 6336
K8V4_PAYLOAD_PAGE_BYTES = 6144


def k8v4_cache_views(key_cache: torch.Tensor):
    """Return zero-copy (payload_bf16, scales_bf16) views over a layer's contiguous K8V4 KV buffer."""
    pages = key_cache.shape[0]
    flat = torch.as_strided(
        key_cache,
        size=(pages * K8V4_PAGE_BYTES,),
        stride=(1,),
        storage_offset=key_cache.storage_offset(),
    )
    pay_bytes = pages * K8V4_PAYLOAD_PAGE_BYTES
    return flat[:pay_bytes].view(torch.bfloat16), flat[pay_bytes:].view(torch.bfloat16)


@triton.jit
def _write_k8v4_kernel(
    K,
    V,
    PAY_BF16,
    SCALES_BF16,
    SLOTS,
    kr: tl.constexpr,
    kh: tl.constexpr,
    vr: tl.constexpr,
    vh: tl.constexpr,
    Z: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(SLOTS + token)
    if slot < 0:
        return
    page = (slot // 16).to(tl.int64)
    row = (slot % 16).to(tl.int64)
    c32 = tl.arange(0, 32)

    kb = K + token.to(tl.int64) * kr + head * kh
    k0 = tl.load(kb + c32).to(tl.float32)
    k1 = tl.load(kb + 32 + c32).to(tl.float32)
    k2 = tl.load(kb + 64 + c32).to(tl.float32)
    k3 = tl.load(kb + 96 + c32).to(tl.float32)
    k0 = tl.where(k0 != k0, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k0)))
    k1 = tl.where(k1 != k1, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k1)))
    k2 = tl.where(k2 != k2, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k2)))
    k3 = tl.where(k3 != k3, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k3)))
    kmin = tl.minimum(
        tl.minimum(tl.min(k0, 0), tl.min(k1, 0)),
        tl.minimum(tl.min(k2, 0), tl.min(k3, 0)),
    )
    kmax = tl.maximum(
        tl.maximum(tl.max(k0, 0), tl.max(k1, 0)),
        tl.maximum(tl.max(k2, 0), tl.max(k3, 0)),
    )
    sk_raw = tl.maximum((-kmin) / (Z - 64.0), kmax / (255.0 - Z))
    sk_bf = tl.where(sk_raw > 1e-12, sk_raw, 1.0).to(tl.bfloat16)
    inv_sk = 1.0 / sk_bf.to(tl.float32)
    kp0 = (
        _quant_cm8_byte(Z + k0 * inv_sk) | (_quant_cm8_byte(Z + k1 * inv_sk) << 8)
    ).to(tl.bfloat16, bitcast=True)
    kp1 = (
        _quant_cm8_byte(Z + k2 * inv_sk) | (_quant_cm8_byte(Z + k3 * inv_sk) << 8)
    ).to(tl.bfloat16, bitcast=True)

    vb = V + token.to(tl.int64) * vr + head * vh
    v0 = tl.load(vb + c32).to(tl.float32)
    v1 = tl.load(vb + 32 + c32).to(tl.float32)
    v2 = tl.load(vb + 64 + c32).to(tl.float32)
    v3 = tl.load(vb + 96 + c32).to(tl.float32)
    v0 = tl.where(v0 != v0, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v0)))
    v1 = tl.where(v1 != v1, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v1)))
    v2 = tl.where(v2 != v2, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v2)))
    v3 = tl.where(v3 != v3, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v3)))
    vmin = tl.minimum(
        tl.minimum(tl.min(v0, 0), tl.min(v1, 0)),
        tl.minimum(tl.min(v2, 0), tl.min(v3, 0)),
    )
    vmax = tl.maximum(
        tl.maximum(tl.max(v0, 0), tl.max(v1, 0)),
        tl.maximum(tl.max(v2, 0), tl.max(v3, 0)),
    )
    vc_bf = (0.5 * (vmin + vmax)).to(tl.bfloat16)
    vc = vc_bf.to(tl.float32)
    sv_raw = (vmax - vmin) / 7.5
    sv_bf = tl.where(sv_raw > 1e-12, sv_raw, 0.0).to(tl.bfloat16)
    sv = sv_bf.to(tl.float32)
    inv_step = tl.where(sv > 0.0, 2.0 / tl.where(sv > 0.0, sv, 1.0), 0.0)
    n0 = tl.maximum(
        0, tl.minimum(15, tl.floor((v0 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    n1 = tl.maximum(
        0, tl.minimum(15, tl.floor((v1 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    n2 = tl.maximum(
        0, tl.minimum(15, tl.floor((v2 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    n3 = tl.maximum(
        0, tl.minimum(15, tl.floor((v3 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    vp = (n0 | (n1 << 4) | (n2 << 8) | (n3 << 12)).to(tl.bfloat16, bitcast=True)

    p_off = page * 3072 + head * 1536 + row * 32 + c32
    tl.store(PAY_BF16 + p_off, kp0)
    tl.store(PAY_BF16 + p_off + 512, kp1)
    tl.store(PAY_BF16 + p_off + 1024, vp)
    meta = page * 96 + head * 48 + row
    tl.store(SCALES_BF16 + meta, sk_bf)
    tl.store(SCALES_BF16 + meta + 16, sv_bf)
    tl.store(SCALES_BF16 + meta + 32, vc_bf)


@triton.jit
def _write_k8v4_batched_kernel(
    K,
    V,
    PAY_BF16,
    SCALES_BF16,
    SLOTS,
    n_slots,
    kr: tl.constexpr,
    kh: tl.constexpr,
    vr: tl.constexpr,
    vh: tl.constexpr,
    Z: tl.constexpr,
    T: tl.constexpr,
):
    toks = tl.program_id(0) * T + tl.arange(0, T)
    head = tl.program_id(1)
    in_b = toks < n_slots
    slots = tl.load(SLOTS + toks, in_b, -1)
    valid = in_b & (slots >= 0)
    safe_slots = tl.where(valid, slots, 0)
    pages = (safe_slots // 16).to(tl.int64)
    rows = (safe_slots % 16).to(tl.int64)
    c32 = tl.arange(0, 32)
    vm = valid[:, None]

    kb = (toks.to(tl.int64) * kr + head * kh)[:, None] + c32[None, :]
    k0 = tl.load(K + kb, vm, 0.0).to(tl.float32)
    k1 = tl.load(K + kb + 32, vm, 0.0).to(tl.float32)
    k2 = tl.load(K + kb + 64, vm, 0.0).to(tl.float32)
    k3 = tl.load(K + kb + 96, vm, 0.0).to(tl.float32)
    k0 = tl.where(k0 != k0, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k0)))
    k1 = tl.where(k1 != k1, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k1)))
    k2 = tl.where(k2 != k2, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k2)))
    k3 = tl.where(k3 != k3, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, k3)))
    kmin = tl.minimum(
        tl.minimum(tl.min(k0, 1), tl.min(k1, 1)),
        tl.minimum(tl.min(k2, 1), tl.min(k3, 1)),
    )
    kmax = tl.maximum(
        tl.maximum(tl.max(k0, 1), tl.max(k1, 1)),
        tl.maximum(tl.max(k2, 1), tl.max(k3, 1)),
    )
    sk_raw = tl.maximum((-kmin) / (Z - 64.0), kmax / (255.0 - Z))
    sk_bf = tl.where(sk_raw > 1e-12, sk_raw, 1.0).to(tl.bfloat16)
    inv_sk = (1.0 / sk_bf.to(tl.float32))[:, None]
    kp0 = (
        _quant_cm8_byte(Z + k0 * inv_sk) | (_quant_cm8_byte(Z + k1 * inv_sk) << 8)
    ).to(tl.bfloat16, bitcast=True)
    kp1 = (
        _quant_cm8_byte(Z + k2 * inv_sk) | (_quant_cm8_byte(Z + k3 * inv_sk) << 8)
    ).to(tl.bfloat16, bitcast=True)

    vb = (toks.to(tl.int64) * vr + head * vh)[:, None] + c32[None, :]
    v0 = tl.load(V + vb, vm, 0.0).to(tl.float32)
    v1 = tl.load(V + vb + 32, vm, 0.0).to(tl.float32)
    v2 = tl.load(V + vb + 64, vm, 0.0).to(tl.float32)
    v3 = tl.load(V + vb + 96, vm, 0.0).to(tl.float32)
    v0 = tl.where(v0 != v0, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v0)))
    v1 = tl.where(v1 != v1, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v1)))
    v2 = tl.where(v2 != v2, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v2)))
    v3 = tl.where(v3 != v3, 0.0, tl.maximum(-65504.0, tl.minimum(65504.0, v3)))
    vmin = tl.minimum(
        tl.minimum(tl.min(v0, 1), tl.min(v1, 1)),
        tl.minimum(tl.min(v2, 1), tl.min(v3, 1)),
    )
    vmax = tl.maximum(
        tl.maximum(tl.max(v0, 1), tl.max(v1, 1)),
        tl.maximum(tl.max(v2, 1), tl.max(v3, 1)),
    )
    vc_bf = (0.5 * (vmin + vmax)).to(tl.bfloat16)
    vc = vc_bf.to(tl.float32)[:, None]
    sv_raw = (vmax - vmin) / 7.5
    sv_bf = tl.where(sv_raw > 1e-12, sv_raw, 0.0).to(tl.bfloat16)
    sv = sv_bf.to(tl.float32)
    inv_step = tl.where(sv > 0.0, 2.0 / tl.where(sv > 0.0, sv, 1.0), 0.0)[:, None]
    n0 = tl.maximum(
        0, tl.minimum(15, tl.floor((v0 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    n1 = tl.maximum(
        0, tl.minimum(15, tl.floor((v1 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    n2 = tl.maximum(
        0, tl.minimum(15, tl.floor((v2 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    n3 = tl.maximum(
        0, tl.minimum(15, tl.floor((v3 - vc) * inv_step + 8.0).to(tl.int32))
    ).to(tl.uint16)
    vp = (n0 | (n1 << 4) | (n2 << 8) | (n3 << 12)).to(tl.bfloat16, bitcast=True)

    p_off = (pages * 3072 + head * 1536 + rows * 32)[:, None] + c32[None, :]
    tl.store(PAY_BF16 + p_off, kp0, vm)
    tl.store(PAY_BF16 + p_off + 512, kp1, vm)
    tl.store(PAY_BF16 + p_off + 1024, vp, vm)
    meta = pages * 96 + head * 48 + rows
    tl.store(SCALES_BF16 + meta, sk_bf, valid)
    tl.store(SCALES_BF16 + meta + 16, sv_bf, valid)
    tl.store(SCALES_BF16 + meta + 32, vc_bf, valid)


def write_k8v4_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n_slots = slot_mapping.numel()
    if n_slots == 0:
        return
    if not slot_mapping.is_contiguous():
        slot_mapping = slot_mapping.contiguous()
    pay_bf16, scales_bf16 = k8v4_cache_views(key_cache)
    if n_slots <= 128:
        _write_k8v4_kernel[(n_slots, 2)](
            key,
            value,
            pay_bf16,
            scales_bf16,
            slot_mapping,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            Z_CM8,
            num_warps=1,
        )
    else:
        _write_k8v4_batched_kernel[(triton.cdiv(n_slots, 4), 2)](
            key,
            value,
            pay_bf16,
            scales_bf16,
            slot_mapping,
            n_slots,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            Z_CM8,
            4,
            num_warps=1,
        )


@triton.jit
def _decode_k8v4_main(
    Q,
    PAY_BF16,
    SCALES_BF16,
    BT,
    LENS,
    CU,
    PART,
    LSE,
    batch,
    qs: tl.constexpr,
    qh: tl.constexpr,
    ts,
    scale,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    Z: tl.constexpr,
    ZV: tl.constexpr,
    WIDE: tl.constexpr,
):
    split = tl.program_id(0)
    seq = tl.program_id(1)
    head = tl.program_id(2)
    row = tl.load(CU + seq).to(tl.int32)
    if tl.load(CU + seq + 1).to(tl.int32) - row != 1:
        return
    length = tl.load(LENS + seq).to(tl.int32)
    full_len = (length // N) * N
    per = tl.cdiv(full_len, N * SPLITS) * N
    start = split * per
    end = tl.minimum(start + per, full_len)

    g = tl.arange(0, 8)
    d32 = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 32), 32)
    qb = Q + row * qs + (head * 8 + g[:, None]) * qh
    q0 = tl.load(qb + d32[None, :])
    q1 = tl.load(qb + 32 + d32[None, :])
    q2 = tl.load(qb + 64 + d32[None, :])
    q3 = tl.load(qb + 96 + d32[None, :])
    log2_scale = scale * 1.4426950408889634
    q_bias = (
        tl.sum(q0.to(tl.float32), 1)
        + tl.sum(q1.to(tl.float32), 1)
        + tl.sum(q2.to(tl.float32), 1)
        + tl.sum(q3.to(tl.float32), 1)
    ) * Z

    peak = tl.full((8,), -float("inf"), tl.float32)
    norm = tl.zeros((8,), tl.float32)
    v_bias = tl.zeros((8,), tl.float32)
    a0 = tl.zeros((8, 32), tl.float32)
    a1 = tl.zeros((8, 32), tl.float32)
    a2 = tl.zeros((8, 32), tl.float32)
    a3 = tl.zeros((8, 32), tl.float32)

    if start < end:
        bt = BT + seq.to(tl.int64) * ts
        NPAGES: tl.constexpr = N // 16
        within = tl.arange(0, 16)
        for n0 in range(start, end, N):
            pages = n0 // 16 + tl.arange(0, NPAGES)
            pids = tl.load(bt + pages)
            if WIDE:
                pids = pids.to(tl.int64)
            row32 = tl.multiple_of(
                (pids[:, None] * 3072 + within[None, :] * 32).reshape(N) + head * 1536,
                32,
            )
            meta = tl.multiple_of(
                (pids[:, None] * 96 + within[None, :]).reshape(N) + head * 48, 16
            )
            sk = tl.load(SCALES_BF16 + meta).to(tl.float32) * log2_scale

            ku01 = tl.load(PAY_BF16 + row32[:, None] + d32[None, :]).to(
                tl.uint16, bitcast=True
            )
            s = tl.dot(
                q0,
                tl.trans(
                    ((ku01 & 0x00FF) + 0x4280)
                    .to(tl.uint16)
                    .to(tl.bfloat16, bitcast=True)
                ),
            )
            s = tl.dot(
                q1,
                tl.trans(
                    ((ku01 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                ),
                s,
            )

            ku23 = tl.load(PAY_BF16 + (row32 + 512)[:, None] + d32[None, :]).to(
                tl.uint16, bitcast=True
            )
            s = tl.dot(
                q2,
                tl.trans(
                    ((ku23 & 0x00FF) + 0x4280)
                    .to(tl.uint16)
                    .to(tl.bfloat16, bitcast=True)
                ),
                s,
            )
            s = tl.dot(
                q3,
                tl.trans(
                    ((ku23 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                ),
                s,
            )

            scores = (s - q_bias[:, None]) * sk[None, :]
            new_peak = tl.maximum(peak, tl.max(scores, 1))
            alpha = tl.exp2(peak - new_peak)
            p = tl.exp2(scores - new_peak[:, None])
            norm = norm * alpha + tl.sum(p, 1)

            sv = tl.load(SCALES_BF16 + meta + 16).to(tl.float32)
            vc = tl.load(SCALES_BF16 + meta + 32).to(tl.float32)
            voff = vc - ZV * sv
            v_bias = v_bias * alpha + tl.sum(p * voff[None, :], 1)
            ps = (p * sv[None, :]).to(tl.bfloat16)

            vu = tl.load(PAY_BF16 + (row32 + 1024)[:, None] + d32[None, :]).to(
                tl.uint16, bitcast=True
            )
            a0 = tl.dot(
                ps,
                ((vu & 0x000F) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True),
                a0 * alpha[:, None],
            )
            a1 = tl.dot(
                ps,
                (((vu >> 4) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True),
                a1 * alpha[:, None],
            )
            a2 = tl.dot(
                ps,
                (((vu >> 8) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True),
                a2 * alpha[:, None],
            )
            a3 = tl.dot(
                ps,
                ((vu >> 12) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True),
                a3 * alpha[:, None],
            )
            peak = new_peak

    has = norm > 0.0
    inv = tl.where(has, 1.0 / norm, 0.0)[:, None]
    vb = v_bias[:, None]
    base = ((split * batch + seq) * 2 + head) * 8
    pbase = PART + (base + g[:, None]) * 128
    tl.store(pbase + d32[None, :], (a0 + vb) * inv)
    tl.store(pbase + 32 + d32[None, :], (a1 + vb) * inv)
    tl.store(pbase + 64 + d32[None, :], (a2 + vb) * inv)
    tl.store(pbase + 96 + d32[None, :], (a3 + vb) * inv)
    tl.store(
        LSE + base + g,
        tl.where(
            has,
            peak * 0.6931471805599453 + tl.log(tl.where(has, norm, 1.0)),
            -float("inf"),
        ),
    )


@triton.jit
def _decode_k8v4_tail(
    Q,
    PAY_BF16,
    SCALES_BF16,
    BT,
    LENS,
    CU,
    PART,
    LSE,
    batch,
    qs: tl.constexpr,
    qh: tl.constexpr,
    ts,
    scale,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    Z: tl.constexpr,
    ZV: tl.constexpr,
    WIDE: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.load(CU + seq).to(tl.int32)
    if tl.load(CU + seq + 1).to(tl.int32) - row != 1:
        return
    length = tl.load(LENS + seq).to(tl.int32)
    full_len = (length // N) * N
    if full_len >= length:
        return

    g = tl.arange(0, 8)
    d32 = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 32), 32)
    qb = Q + row * qs + (head * 8 + g[:, None]) * qh
    q0 = tl.load(qb + d32[None, :])
    q1 = tl.load(qb + 32 + d32[None, :])
    q2 = tl.load(qb + 64 + d32[None, :])
    q3 = tl.load(qb + 96 + d32[None, :])
    log2_scale = scale * 1.4426950408889634
    q_bias = (
        tl.sum(q0.to(tl.float32), 1)
        + tl.sum(q1.to(tl.float32), 1)
        + tl.sum(q2.to(tl.float32), 1)
        + tl.sum(q3.to(tl.float32), 1)
    ) * Z

    bt = BT + seq.to(tl.int64) * ts
    NPAGES: tl.constexpr = N // 16
    within = tl.arange(0, 16)
    tokens = full_len + tl.arange(0, N)
    valid = tokens < length
    pages = full_len // 16 + tl.arange(0, NPAGES)
    pids = tl.load(bt + pages, pages * 16 < length, 0)
    if WIDE:
        pids = pids.to(tl.int64)
    row32 = tl.multiple_of(
        (pids[:, None] * 3072 + within[None, :] * 32).reshape(N) + head * 1536, 32
    )
    meta = (pids[:, None] * 96 + within[None, :]).reshape(N) + head * 48
    sk = tl.load(SCALES_BF16 + meta, valid, 0.0).to(tl.float32) * log2_scale

    ku01 = tl.load(PAY_BF16 + row32[:, None] + d32[None, :], valid[:, None], 0.0).to(
        tl.uint16, bitcast=True
    )
    s = tl.dot(
        q0,
        tl.trans(
            ((ku01 & 0x00FF) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        ),
    )
    s = tl.dot(
        q1,
        tl.trans(((ku01 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)),
        s,
    )
    ku23 = tl.load(
        PAY_BF16 + (row32 + 512)[:, None] + d32[None, :], valid[:, None], 0.0
    ).to(tl.uint16, bitcast=True)
    s = tl.dot(
        q2,
        tl.trans(
            ((ku23 & 0x00FF) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        ),
        s,
    )
    s = tl.dot(
        q3,
        tl.trans(((ku23 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)),
        s,
    )

    scores = tl.where(
        valid[None, :], (s - q_bias[:, None]) * sk[None, :], -float("inf")
    )
    peak = tl.max(scores, 1)
    p = tl.exp2(scores - peak[:, None])
    norm = tl.sum(p, 1)
    sv = tl.load(SCALES_BF16 + meta + 16, valid, 0.0).to(tl.float32)
    vc = tl.load(SCALES_BF16 + meta + 32, valid, 0.0).to(tl.float32)
    voff = vc - ZV * sv
    vb = tl.sum(p * voff[None, :], 1)[:, None]
    ps = (p * sv[None, :]).to(tl.bfloat16)

    vu = tl.load(
        PAY_BF16 + (row32 + 1024)[:, None] + d32[None, :], valid[:, None], 0.0
    ).to(tl.uint16, bitcast=True)
    a0 = tl.dot(
        ps, ((vu & 0x000F) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    )
    a1 = tl.dot(
        ps, (((vu >> 4) & 0x000F) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    )
    a2 = tl.dot(
        ps, (((vu >> 8) & 0x000F) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    )
    a3 = tl.dot(ps, ((vu >> 12) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True))

    inv = (1.0 / norm)[:, None]
    t0 = (a0 + vb) * inv
    t1 = (a1 + vb) * inv
    t2 = (a2 + vb) * inv
    t3 = (a3 + vb) * inv
    lse_t = peak * 0.6931471805599453 + tl.log(norm)

    base = (((SPLITS - 1) * batch + seq) * 2 + head) * 8
    pbase = PART + (base + g[:, None]) * 128
    lse_prev = tl.load(LSE + base + g)
    has_prev = lse_prev > -1e30
    lse_max = tl.maximum(lse_prev, lse_t)
    w_prev = tl.where(has_prev, tl.exp(lse_prev - lse_max), 0.0)
    w_t = tl.exp(lse_t - lse_max)
    w_sum = w_prev + w_t
    c_prev = (w_prev / w_sum)[:, None]
    c_t = (w_t / w_sum)[:, None]
    tl.store(
        pbase + d32[None, :],
        tl.where(
            has_prev[:, None], tl.load(pbase + d32[None, :]) * c_prev + t0 * c_t, t0
        ),
    )
    tl.store(
        pbase + 32 + d32[None, :],
        tl.where(
            has_prev[:, None],
            tl.load(pbase + 32 + d32[None, :]) * c_prev + t1 * c_t,
            t1,
        ),
    )
    tl.store(
        pbase + 64 + d32[None, :],
        tl.where(
            has_prev[:, None],
            tl.load(pbase + 64 + d32[None, :]) * c_prev + t2 * c_t,
            t2,
        ),
    )
    tl.store(
        pbase + 96 + d32[None, :],
        tl.where(
            has_prev[:, None],
            tl.load(pbase + 96 + d32[None, :]) * c_prev + t3 * c_t,
            t3,
        ),
    )
    tl.store(LSE + base + g, lse_max + tl.log(w_sum))


def k8v4_decode_splitkv(
    q: torch.Tensor,
    pay_bf16: torch.Tensor,
    scales_bf16: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    lse_out: torch.Tensor | None = None,
):
    batch = seqused_k.shape[0]
    total_q = q.shape[0]
    if batch == 0 or total_q == 0:
        return out
    splits, mw = _select_decode_plan(batch)
    idx = q.device.index if q.device.index is not None else torch.cuda.current_device()
    ws = _CM8_DECODE_WS.get(idx)
    if (
        ws is None
        and splits * batch <= 1024
        and not torch.cuda.is_current_stream_capturing()
    ):
        ws = reserve_cm8_decode_workspace(q.device)
    if ws is None or ws[0].numel() < splits * batch * 16 * 128:
        ws = (
            torch.empty(
                (splits, batch, 2, 8, 128), dtype=torch.float32, device=q.device
            ),
            torch.empty((splits, batch, 2, 8), dtype=torch.float32, device=q.device),
        )
    part, lse_buf = ws
    wide = pay_bf16.numel() >= (1 << 31)
    _decode_k8v4_main[(splits, batch, 2)](
        q,
        pay_bf16,
        scales_bf16,
        block_table,
        seqused_k,
        cu_seqlens_q,
        part,
        lse_buf,
        batch,
        q.stride(0),
        q.stride(1),
        block_table.stride(0),
        softmax_scale,
        32,
        splits,
        Z_CM8,
        Z_V4,
        wide,
        num_warps=1,
        num_stages=1,
    )
    _decode_k8v4_tail[(batch, 2)](
        q,
        pay_bf16,
        scales_bf16,
        block_table,
        seqused_k,
        cu_seqlens_q,
        part,
        lse_buf,
        batch,
        q.stride(0),
        q.stride(1),
        block_table.stride(0),
        softmax_scale,
        32,
        splits,
        Z_CM8,
        Z_V4,
        wide,
        num_warps=1,
        num_stages=1,
    )
    dummy_lse = lse_out if lse_out is not None else _get_dummy_f32(q.device)
    if mw:
        _parallel_merge[(batch, 16, 128 // mw)](
            part,
            lse_buf,
            out,
            cu_seqlens_q,
            dummy_lse,
            batch,
            out.stride(0),
            out.stride(1),
            16,
            128,
            splits,
            mw,
            total_q,
            lse_out is not None,
            num_warps=2,
        )
    else:
        _decode_merge_kernel[(batch, 2)](
            part,
            lse_buf,
            out,
            cu_seqlens_q,
            batch,
            H=16,
            HK=2,
            G=8,
            D=128,
            NUM_SPLITS=splits,
            o_row_stride=out.stride(0),
            o_head_stride=out.stride(1),
            final_lse_ptr=dummy_lse,
            total_q=total_q,
            STORE_LSE=lse_out is not None,
            num_warps=2,
        )
    return out


@triton.jit
def _pack_k8v4_4way(
    PAY_BF16,
    SCALES_BF16,
    BT,
    META,
    PK,
    PV,
    table_stride,
    capacity: tl.constexpr,
    Z: tl.constexpr,
    ZV: tl.constexpr,
    WIDE: tl.constexpr,
    N_PAD: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.load(META)
    if seq < 0:
        return
    length = tl.load(META + 3)
    padded_len = tl.cdiv(length, N_PAD) * N_PAD
    if tile * 32 >= padded_len:
        return
    pages = tile * 2 + tl.arange(0, 2)
    pids = tl.load(BT + seq.to(tl.int64) * table_stride + pages, pages * 16 < length, 0)
    if WIDE:
        pids = pids.to(tl.int64)
    within = tl.arange(0, 16)
    d32 = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 32), 32)
    tokens = tile * 32 + tl.arange(0, 32)
    valid = tokens < length

    row32 = tl.multiple_of(
        (pids[:, None] * 3072 + within[None, :] * 32).reshape(32) + head * 1536, 32
    )
    meta = (pids[:, None] * 96 + within[None, :]).reshape(32) + head * 48
    sk = tl.load(SCALES_BF16 + meta, valid, 0.0).to(tl.float32)
    sv = tl.load(SCALES_BF16 + meta + 16, valid, 0.0).to(tl.float32)
    vc = tl.load(SCALES_BF16 + meta + 32, valid, 0.0).to(tl.float32)

    ku01 = tl.load(PAY_BF16 + row32[:, None] + d32[None, :], valid[:, None], 0.0).to(
        tl.uint16, bitcast=True
    )
    ku23 = tl.load(
        PAY_BF16 + (row32 + 512)[:, None] + d32[None, :], valid[:, None], 0.0
    ).to(tl.uint16, bitcast=True)
    k0 = tl.where(
        valid[:, None],
        (
            ((ku01 & 0x00FF) + 0x4280)
            .to(tl.uint16)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
            - Z
        )
        * sk[:, None],
        0.0,
    ).to(tl.bfloat16)
    k1 = tl.where(
        valid[:, None],
        (
            ((ku01 >> 8) + 0x4280)
            .to(tl.uint16)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
            - Z
        )
        * sk[:, None],
        0.0,
    ).to(tl.bfloat16)
    k2 = tl.where(
        valid[:, None],
        (
            ((ku23 & 0x00FF) + 0x4280)
            .to(tl.uint16)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
            - Z
        )
        * sk[:, None],
        0.0,
    ).to(tl.bfloat16)
    k3 = tl.where(
        valid[:, None],
        (
            ((ku23 >> 8) + 0x4280)
            .to(tl.uint16)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
            - Z
        )
        * sk[:, None],
        0.0,
    ).to(tl.bfloat16)

    k_off = head * 128 * capacity + tokens[None, :]
    tl.store(
        PK + k_off + (0 + d32)[:, None] * capacity,
        tl.trans(k0),
        tokens[None, :] < capacity,
    )
    tl.store(
        PK + k_off + (32 + d32)[:, None] * capacity,
        tl.trans(k1),
        tokens[None, :] < capacity,
    )
    tl.store(
        PK + k_off + (64 + d32)[:, None] * capacity,
        tl.trans(k2),
        tokens[None, :] < capacity,
    )
    tl.store(
        PK + k_off + (96 + d32)[:, None] * capacity,
        tl.trans(k3),
        tokens[None, :] < capacity,
    )

    vu = tl.load(
        PAY_BF16 + (row32 + 1024)[:, None] + d32[None, :], valid[:, None], 0.0
    ).to(tl.uint16, bitcast=True)
    v0 = tl.where(
        valid[:, None],
        (
            (
                ((vu & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
                - ZV
            )
            * sv[:, None]
            + vc[:, None]
        ),
        0.0,
    ).to(tl.bfloat16)
    v1 = tl.where(
        valid[:, None],
        (
            (
                (((vu >> 4) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
                - ZV
            )
            * sv[:, None]
            + vc[:, None]
        ),
        0.0,
    ).to(tl.bfloat16)
    v2 = tl.where(
        valid[:, None],
        (
            (
                (((vu >> 8) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
                - ZV
            )
            * sv[:, None]
            + vc[:, None]
        ),
        0.0,
    ).to(tl.bfloat16)
    v3 = tl.where(
        valid[:, None],
        (
            (
                ((vu >> 12) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
                - ZV
            )
            * sv[:, None]
            + vc[:, None]
        ),
        0.0,
    ).to(tl.bfloat16)

    v_off = head * capacity * 128 + tokens[:, None] * 128
    tl.store(PV + v_off + d32[None, :], v0, tokens[:, None] < capacity)
    tl.store(PV + v_off + 32 + d32[None, :], v1, tokens[:, None] < capacity)
    tl.store(PV + v_off + 64 + d32[None, :], v2, tokens[:, None] < capacity)
    tl.store(PV + v_off + 96 + d32[None, :], v3, tokens[:, None] < capacity)


def k8v4_dense_long_query(
    q, pay_bf16, scales_bf16, out, lse, cu, lengths, table, max_q, max_k, scale
):
    capacity = triton.cdiv(max_k, 2048) * 2048
    size = 4 * capacity * 128
    owner = _prefill_owner(q.device)
    pair = _prefill_reserved.get(owner)
    if (
        pair is None
        and capacity <= 32768
        and not torch.cuda.is_current_stream_capturing()
    ):
        reserve_prefill_workspace(q.device)
        pair = _prefill_reserved[owner]
    if pair is None or pair[0].numel() < size:
        pair = (
            q.new_empty((size,)),
            torch.empty(8, device=q.device, dtype=torch.int32),
        )
    data, meta = pair
    pk, pv = data[:size].view(2, 2, capacity, 128).unbind(0)
    batch = lengths.numel()
    two = batch > 1 and max_q >= 128 and q.shape[0] - max_q >= 128
    _long_query_cm8[(1,)](
        cu, lengths, meta, batch, triton.next_power_of_2(batch), two, num_warps=1
    )
    m, warps = 256, 16
    n_tile = 256 if max_k > 2048 else 128
    wide = pay_bf16.numel() >= (1 << 31)
    dummy_lse = lse if lse is not None else _get_dummy_f32(q.device)
    for index in range(2 if two else 1):
        selected = meta[index * 4 :]
        _pack_k8v4_4way[(triton.cdiv(max_k, n_tile) * (n_tile // 32), 2)](
            pay_bf16,
            scales_bf16,
            table,
            selected,
            pk,
            pv,
            table.stride(0),
            capacity,
            Z_CM8,
            Z_V4,
            wide,
            n_tile,
            num_warps=4,
        )
        _causal_cm8[(triton.cdiv(max_q, m // 8), 2)](
            q,
            pk,
            pv,
            out,
            dummy_lse,
            selected,
            scale,
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            q.size(0),
            capacity,
            m,
            n_tile,
            128,
            lse is not None,
            num_warps=warps,
            num_stages=1,
        )
    return meta


@triton.jit
def _paged_prefill_k8v4_4way(
    Q,
    PAY_BF16,
    SCALES_BF16,
    OUT,
    LSE,
    CU,
    LENS,
    BT,
    META,
    scale,
    batch,
    total_q,
    q_rs: tl.constexpr,
    q_hs: tl.constexpr,
    o_rs: tl.constexpr,
    o_hs: tl.constexpr,
    bt_s,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH_W: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EXCLUDE_SELECTED: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    Z: tl.constexpr,
    ZV: tl.constexpr,
):
    hk = tl.program_id(0)
    tile = tl.program_id(1)
    Q_TILE: tl.constexpr = M // 8

    if BATCH_W > 0:
        seq_idx = tl.arange(0, BATCH_W)
        starts = tl.load(CU + seq_idx, mask=seq_idx < batch, other=0)
        ends = tl.load(CU + seq_idx + 1, mask=seq_idx < batch, other=0)
        lengths = ends - starts
        counts = tl.where(lengths > 1, tl.cdiv(lengths, Q_TILE), 0)
        if EXCLUDE_SELECTED:
            ex0 = tl.load(META)
            ex1 = tl.load(META + 4)
            counts = tl.where((seq_idx == ex0) | (seq_idx == ex1), 0, counts)
        tile_ends = tl.cumsum(counts, 0)
        total_tiles = tl.max(tile_ends, 0)
        if tile >= total_tiles:
            return
        bid = tl.sum((tile_ends <= tile).to(tl.int32), 0)
        m_block = tile - tl.sum(tl.where(seq_idx == bid, tile_ends - counts, 0), 0)
    else:
        m_block = tile
        bid = tl.program_id(2)

    q_bos = tl.load(CU + bid)
    q_eos = tl.load(CU + bid + 1)
    q_len = q_eos - q_bos
    if q_len <= 1:
        return
    if EXCLUDE_SELECTED:
        if (bid == tl.load(META)) | (bid == tl.load(META + 4)):
            return
    if m_block * Q_TILE >= q_len:
        return

    k_len = tl.load(LENS + bid)
    history = k_len - q_len
    lanes = tl.arange(0, M)
    pos = m_block * Q_TILE + lanes // 8
    head = hk * 8 + lanes % 8

    d32 = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 32), 32)
    qb = Q + ((q_bos + pos) * q_rs + head * q_hs)[:, None]
    p_valid = pos[:, None] < q_len
    q0 = tl.load(qb + d32[None, :], p_valid, 0.0)
    q1 = tl.load(qb + 32 + d32[None, :], p_valid, 0.0)
    q2 = tl.load(qb + 64 + d32[None, :], p_valid, 0.0)
    q3 = tl.load(qb + 96 + d32[None, :], p_valid, 0.0)
    log2_scale = scale * 1.4426950408889634
    q_bias = (
        tl.sum(q0.to(tl.float32), 1)
        + tl.sum(q1.to(tl.float32), 1)
        + tl.sum(q2.to(tl.float32), 1)
        + tl.sum(q3.to(tl.float32), 1)
    ) * Z

    m = tl.full((M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((M,), tl.float32)
    v_bias = tl.zeros((M,), tl.float32)
    a0 = tl.zeros((M, 32), tl.float32)
    a1 = tl.zeros((M, 32), tl.float32)
    a2 = tl.zeros((M, 32), tl.float32)
    a3 = tl.zeros((M, 32), tl.float32)

    if k_len > 0:
        max_k_needed = (
            tl.minimum(k_len, history + (m_block + 1) * Q_TILE) if IS_CAUSAL else k_len
        )
        interior = (
            (tl.maximum(0, history + m_block * Q_TILE + 1) // N)
            if IS_CAUSAL
            else (max_k_needed // N)
        )
        end_tiles = tl.cdiv(max_k_needed, N)
        bt = BT + bid.to(tl.int64) * bt_s
        NPAGES: tl.constexpr = N // 16
        within = tl.arange(0, 16)

        for t in range(interior):
            pages = t * NPAGES + tl.arange(0, NPAGES)
            pids = tl.load(bt + pages).to(tl.int64)
            row32 = tl.multiple_of(
                (pids[:, None] * 3072 + within[None, :] * 32).reshape(N) + hk * 1536, 32
            )
            meta = (pids[:, None] * 96 + within[None, :]).reshape(N) + hk * 48
            sk = tl.load(SCALES_BF16 + meta).to(tl.float32) * log2_scale

            ku01 = tl.load(PAY_BF16 + row32[:, None] + d32[None, :]).to(
                tl.uint16, bitcast=True
            )
            s = tl.dot(
                q0,
                tl.trans(
                    ((ku01 & 0x00FF) + 0x4280)
                    .to(tl.uint16)
                    .to(tl.bfloat16, bitcast=True)
                ),
            )
            s = tl.dot(
                q1,
                tl.trans(
                    ((ku01 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                ),
                s,
            )
            ku23 = tl.load(PAY_BF16 + (row32 + 512)[:, None] + d32[None, :]).to(
                tl.uint16, bitcast=True
            )
            s = tl.dot(
                q2,
                tl.trans(
                    ((ku23 & 0x00FF) + 0x4280)
                    .to(tl.uint16)
                    .to(tl.bfloat16, bitcast=True)
                ),
                s,
            )
            s = tl.dot(
                q3,
                tl.trans(
                    ((ku23 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                ),
                s,
            )

            scores = (s - q_bias[:, None]) * sk[None, :]
            new_m = tl.maximum(m, tl.max(scores, 1))
            safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
            alpha = tl.exp2(m - safe_m)
            p = tl.exp2(scores - safe_m[:, None])
            row_sum = row_sum * alpha + tl.sum(p, 1)

            sv = tl.load(SCALES_BF16 + meta + 16).to(tl.float32)
            vc = tl.load(SCALES_BF16 + meta + 32).to(tl.float32)
            voff = vc - ZV * sv
            v_bias = v_bias * alpha + tl.sum(p * voff[None, :], 1)
            ps = (p * sv[None, :]).to(tl.bfloat16)

            vu = tl.load(PAY_BF16 + (row32 + 1024)[:, None] + d32[None, :]).to(
                tl.uint16, bitcast=True
            )
            a0 = tl.dot(
                ps,
                ((vu & 0x000F) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True),
                a0 * alpha[:, None],
            )
            a1 = tl.dot(
                ps,
                (((vu >> 4) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True),
                a1 * alpha[:, None],
            )
            a2 = tl.dot(
                ps,
                (((vu >> 8) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True),
                a2 * alpha[:, None],
            )
            a3 = tl.dot(
                ps,
                ((vu >> 12) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True),
                a3 * alpha[:, None],
            )
            m = new_m

        for t in range(interior, end_tiles):
            tok = t * N + tl.arange(0, N)
            t_valid = tok < k_len
            pages = t * NPAGES + tl.arange(0, NPAGES)
            pids = tl.load(bt + pages, pages * 16 < k_len, 0).to(tl.int64)
            row32 = tl.multiple_of(
                (pids[:, None] * 3072 + within[None, :] * 32).reshape(N) + hk * 1536, 32
            )
            meta = (pids[:, None] * 96 + within[None, :]).reshape(N) + hk * 48
            sk = tl.load(SCALES_BF16 + meta, t_valid, 0.0).to(tl.float32) * log2_scale

            ku01 = tl.load(
                PAY_BF16 + row32[:, None] + d32[None, :], t_valid[:, None], 0.0
            ).to(tl.uint16, bitcast=True)
            s = tl.dot(
                q0,
                tl.trans(
                    ((ku01 & 0x00FF) + 0x4280)
                    .to(tl.uint16)
                    .to(tl.bfloat16, bitcast=True)
                ),
            )
            s = tl.dot(
                q1,
                tl.trans(
                    ((ku01 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                ),
                s,
            )
            ku23 = tl.load(
                PAY_BF16 + (row32 + 512)[:, None] + d32[None, :], t_valid[:, None], 0.0
            ).to(tl.uint16, bitcast=True)
            s = tl.dot(
                q2,
                tl.trans(
                    ((ku23 & 0x00FF) + 0x4280)
                    .to(tl.uint16)
                    .to(tl.bfloat16, bitcast=True)
                ),
                s,
            )
            s = tl.dot(
                q3,
                tl.trans(
                    ((ku23 >> 8) + 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                ),
                s,
            )

            mask = t_valid[None, :]
            if IS_CAUSAL:
                mask = mask & (tok[None, :] <= history + pos[:, None])
            scores = tl.where(mask, (s - q_bias[:, None]) * sk[None, :], -float("inf"))
            new_m = tl.maximum(m, tl.max(scores, 1))
            safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
            alpha = tl.exp2(m - safe_m)
            p = tl.exp2(scores - safe_m[:, None])
            row_sum = row_sum * alpha + tl.sum(p, 1)

            sv = tl.load(SCALES_BF16 + meta + 16, t_valid, 0.0).to(tl.float32)
            vc = tl.load(SCALES_BF16 + meta + 32, t_valid, 0.0).to(tl.float32)
            voff = vc - ZV * sv
            v_bias = v_bias * alpha + tl.sum(p * voff[None, :], 1)
            ps = (p * sv[None, :]).to(tl.bfloat16)

            vu = tl.load(
                PAY_BF16 + (row32 + 1024)[:, None] + d32[None, :], t_valid[:, None], 0.0
            ).to(tl.uint16, bitcast=True)
            a0 = tl.dot(
                ps,
                ((vu & 0x000F) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True),
                a0 * alpha[:, None],
            )
            a1 = tl.dot(
                ps,
                (((vu >> 4) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True),
                a1 * alpha[:, None],
            )
            a2 = tl.dot(
                ps,
                (((vu >> 8) & 0x000F) | 0x4280)
                .to(tl.uint16)
                .to(tl.bfloat16, bitcast=True),
                a2 * alpha[:, None],
            )
            a3 = tl.dot(
                ps,
                ((vu >> 12) | 0x4280).to(tl.uint16).to(tl.bfloat16, bitcast=True),
                a3 * alpha[:, None],
            )
            m = new_m

    vb = v_bias[:, None]
    has = row_sum > 0.0
    inv = tl.where(has, 1.0 / row_sum, 0.0)[:, None]
    ob = OUT + ((q_bos + pos) * o_rs + head * o_hs)[:, None]
    tl.store(ob + d32[None, :], (a0 + vb) * inv, p_valid)
    tl.store(ob + 32 + d32[None, :], (a1 + vb) * inv, p_valid)
    tl.store(ob + 64 + d32[None, :], (a2 + vb) * inv, p_valid)
    tl.store(ob + 96 + d32[None, :], (a3 + vb) * inv, p_valid)
    if WRITE_LSE:
        safe_l = tl.where(has, row_sum, 1.0)
        tl.store(
            LSE + head * total_q + q_bos + pos,
            tl.where(has, m * 0.6931471805599453 + tl.log(safe_l), -float("inf")),
            pos < q_len,
        )


def k8v4_varlen_fwd(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    out: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    softmax_scale: float | None,
    is_causal: bool,
    return_lse: bool,
):
    if out is None:
        out = torch.empty_like(q)
    total_q = q.shape[0]
    batch = cu_seqlens_q.numel() - 1
    lse = (
        torch.empty((16, total_q), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    max_seqlen_q = (
        int(max_seqlen_q.item()) if hasattr(max_seqlen_q, "item") else int(max_seqlen_q)
    )
    max_seqlen_k = (
        int(max_seqlen_k.item()) if hasattr(max_seqlen_k, "item") else int(max_seqlen_k)
    )
    if total_q == 0 or batch <= 0 or max_seqlen_q <= 0:
        return out, lse

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    pay_bf16, scales_bf16 = k8v4_cache_views(key_cache)

    if max_seqlen_q == 1:
        k8v4_decode_splitkv(
            q,
            pay_bf16,
            scales_bf16,
            out,
            cu_seqlens_q,
            seqused_k,
            block_table,
            softmax_scale,
            lse_out=lse,
        )
        return out, lse

    use_dense = (
        is_causal
        and 1 <= batch <= 64
        and 64 <= max_seqlen_q <= total_q
        and 512 <= max_seqlen_k <= 32768
        and softmax_scale > 0
        and cu_seqlens_q.stride(0) == 1
        and seqused_k.stride(0) == 1
        and all(
            sum((n - 1) * s for n, s in zip(t.shape, t.stride())) < (1 << 31)
            for t in (q, out)
        )
        and not torch.cuda.is_current_stream_capturing()
    )

    dummy_lse = lse if lse is not None else _get_dummy_f32(q.device)

    if use_dense:
        meta = k8v4_dense_long_query(
            q,
            pay_bf16,
            scales_bf16,
            out,
            lse,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
        )
        rem_q = total_q - max_seqlen_q
        if batch > 1 and rem_q >= 2:
            eff_max_q = min(max_seqlen_q, rem_q)
            m_p, nw = (128, 4) if eff_max_q < 256 else (256, 8)
            q_tile = m_p // 8
            tiles = min(
                triton.cdiv(eff_max_q, q_tile) * batch,
                triton.cdiv(rem_q, q_tile) + batch - 1,
            )
            bw = triton.next_power_of_2(max(2, batch))
            _paged_prefill_k8v4_4way[(2, tiles, 1)](
                q,
                pay_bf16,
                scales_bf16,
                out,
                dummy_lse,
                cu_seqlens_q,
                seqused_k,
                block_table,
                meta,
                softmax_scale,
                batch,
                total_q,
                q.stride(0),
                q.stride(1),
                out.stride(0),
                out.stride(1),
                block_table.stride(0),
                m_p,
                32,
                bw,
                is_causal,
                True,
                lse is not None,
                Z_CM8,
                Z_V4,
                num_warps=nw,
                num_stages=1,
            )
    else:
        m_p, nw = (128, 4) if max_seqlen_q < 256 else (256, 8)
        q_tile = m_p // 8
        if 1 <= batch <= 64:
            bw = triton.next_power_of_2(max(2, batch))
            tiles = min(
                triton.cdiv(max_seqlen_q, q_tile) * batch,
                triton.cdiv(total_q, q_tile) + batch,
            )
            grid = (2, tiles, 1)
        else:
            bw = 0
            grid = (2, triton.cdiv(max_seqlen_q, q_tile), batch)
        _paged_prefill_k8v4_4way[grid](
            q,
            pay_bf16,
            scales_bf16,
            out,
            dummy_lse,
            cu_seqlens_q,
            seqused_k,
            block_table,
            cu_seqlens_q,
            softmax_scale,
            batch,
            total_q,
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            block_table.stride(0),
            m_p,
            32,
            bw,
            is_causal,
            False,
            lse is not None,
            Z_CM8,
            Z_V4,
            num_warps=nw,
            num_stages=1,
        )

    if batch > 1:
        k8v4_decode_splitkv(
            q,
            pay_bf16,
            scales_bf16,
            out,
            cu_seqlens_q,
            seqused_k,
            block_table,
            softmax_scale,
            lse_out=lse,
        )

    return out, lse
