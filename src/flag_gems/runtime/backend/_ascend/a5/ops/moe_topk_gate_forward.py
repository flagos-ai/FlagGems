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
def _gate_kernel(
    x_ptr,
    x_s0,
    bias_ptr,
    imb_ptr,
    imask_ptr,
    mask_ptr,
    fr_ptr,
    fix_ptr,
    cnt_ptr,
    tmap_ptr,
    tmap_s0,
    idx32_ptr,
    idx32_s0,
    unm32_ptr,
    unm32_s0,
    w_ptr,
    w_s0,
    sc_ptr,
    sc_s0,
    rows,
    scale,
    ep_rank,
    E: tl.constexpr,
    EB: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    KB: tl.constexpr,
    NS: tl.constexpr,
    RPP: tl.constexpr,
    NFLAT: tl.constexpr,
    NLOG: tl.constexpr,
    CBLK: tl.constexpr,
    NDUP: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_FR: tl.constexpr,
    HAS_FIX: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SC: tl.constexpr,
    HAS_UNM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * RPP + tl.arange(0, RPP)
    offs_e = tl.arange(0, EB)
    ka = tl.arange(0, KB)
    rmask = offs_r < rows
    emask = offs_e < E
    kmask = ka < KP
    kl = ka < K
    # When the expert tile is exact (EB == E, true for every E=256 timing case)
    # the expert-axis predicate is statically all-true, so every construct built
    # from it is redundant work. Keep it only for the padded tile (E=72/E=144).
    if EB == E:
        lm = rmask[:, None]
    else:
        lm = rmask[:, None] & emask[None, :]

    x = tl.load(x_ptr + offs_r[:, None] * x_s0 + offs_e[None, :], mask=lm, other=0.0)
    ex = tl.exp(x)
    # log1p(ex) series, used only on the x < -4 branch. Truncated to 3 terms:
    # with e = exp(x) < exp(-4) = 0.0183156 the dropped terms are bounded by
    # e^4/4, i.e. a relative error of e^3/4 <= 1.53e-6 in log1p and <= 7.7e-7
    # after the sqrt, against a 1e-5 relative tolerance. The 4 saved full-tile
    # ops are a fifth of the score chain.
    s5 = ex * (1.0 + ex * (-0.5 + ex * 0.3333333432674408))
    sp = tl.where(x > 20.0, x, tl.where(x < -4.0, s5, tl.log(1.0 + ex)))
    s = tl.sqrt(sp)

    # The bias vectors are dense 1-D loads over the expert axis. When EB == E
    # (every E=256 workload) offs_e addresses exactly the E real experts, so the
    # expert-axis predicate here is redundant for the same reason it was for the
    # 2-D logits tile -- and once removed here, `emask` has no remaining user in
    # that specialization at all.
    if BIAS_MODE == 3:
        if EB == E:
            b0 = tl.load(bias_ptr + offs_e)
            b1 = tl.load(imb_ptr + offs_e)
        else:
            b0 = tl.load(bias_ptr + offs_e, mask=emask, other=0.0)
            b1 = tl.load(imb_ptr + offs_e, mask=emask, other=0.0)
        isimg = tl.load(imask_ptr + offs_r, mask=rmask, other=0) != 0
        c = s + tl.where(isimg[:, None], b1[None, :], b0[None, :])
    elif BIAS_MODE == 1:
        if EB == E:
            b0 = tl.load(bias_ptr + offs_e)
        else:
            b0 = tl.load(bias_ptr + offs_e, mask=emask, other=0.0)
        c = s + b0[None, :]
    elif BIAS_MODE == 2:
        if EB == E:
            b1 = tl.load(imb_ptr + offs_e)
        else:
            b1 = tl.load(imb_ptr + offs_e, mask=emask, other=0.0)
        isimg = tl.load(imask_ptr + offs_r, mask=rmask, other=0) != 0
        c = s + tl.where(isimg[:, None], b1[None, :], 0.0)
    else:
        c = s

    if HAS_MASK:
        active = tl.load(mask_ptr + offs_r, mask=rmask, other=0) != 0
    else:
        active = rmask

    # ---- deterministic top-k: greedy, ties resolve to the lowest expert index
    if EB == E:
        cs = c
    else:
        cs = tl.where(emask[None, :], c, -1e30)
    sel_t = tl.zeros((RPP, KB), tl.int32)
    for j in tl.static_range(K):
        v = tl.max(cs, axis=1)
        if EB == E:
            hit = cs >= v[:, None]
        else:
            hit = (cs >= v[:, None]) & emask[None, :]
        ji = tl.min(tl.where(hit, offs_e[None, :], E), axis=1)
        sel_t = tl.where(ka[None, :] == j, ji[:, None], sel_t)
        if j < K - 1:
            cs = tl.where(offs_e[None, :] == ji[:, None], -1e30, cs)

    if HAS_FIX:
        fx = tl.load(fix_ptr + offs_r, mask=rmask, other=0) != 0
        fu = tl.load(
            unm32_ptr + offs_r[:, None] * unm32_s0 + 2 * ka[None, :],
            mask=rmask[:, None] & kl[None, :],
            other=0,
        )
        sel_t = tl.where(fx[:, None], fu, sel_t)
    else:
        fx = offs_r < 0

    if HAS_FR:
        fr = tl.load(fr_ptr + offs_r, mask=rmask, other=0) != 0
        hh = offs_r[:, None] * 1013904223 + (ka[None, :] + 1) * 1597334677
        hh = hh ^ (hh >> 15)
        hh = hh * 1103515245 + 12345
        hh = hh ^ (hh >> 13)
        hh = hh * 2146121005 + 12345
        hh = hh ^ (hh >> 16)
        if HAS_MAP:
            # CBLK == NLOG now, so the load predicate is identically true.
            ca = tl.arange(0, CBLK)
            cnt_all = tl.load(cnt_ptr + ca)
            np_exp = tl.sum(cnt_all, axis=0)
            uu = (hh & 2147483647).to(tl.float32) * (
                np_exp.to(tl.float32) * 4.656612873077393e-10
            )
            rnd_idx = tl.minimum(uu.to(tl.int32), np_exp - 1)
        else:
            rnd_idx = (hh & 2147483647) % E
        hw = hh * 1664525 + 1013904223
        hw = hw ^ (hw >> 16)
        rnd_w = (hw & 1048575).to(tl.float32) * 9.5367431640625e-07 + 0.001
    else:
        fr = offs_r < 0

    # ---- value pass: one (RPP, KB) gather, score recomputed from the same bits
    # The clamp is only needed when fix_routing can inject indices straight from
    # the caller's unmapped_topk_idx. Without it, sel_t is produced solely by the
    # greedy min, which is masked by `hit` and therefore always lands in
    # [0, E-1] (lanes >= K keep their initial 0), so the two clamp ops are dead.
    if HAS_FIX:
        gidx = tl.maximum(tl.minimum(sel_t, E - 1), 0)
    else:
        gidx = sel_t
    gx = tl.load(x_ptr + offs_r[:, None] * x_s0 + gidx, mask=rmask[:, None], other=0.0)
    gex = tl.exp(gx)
    gs5 = gex * (1.0 + gex * (-0.5 + gex * 0.3333333432674408))
    gsp = tl.where(gx > 20.0, gx, tl.where(gx < -4.0, gs5, tl.log(1.0 + gex)))
    gg = tl.sqrt(gsp)
    gsum = tl.sum(tl.where(kl[None, :], gg, 0.0), axis=1)
    wt = gg / (gsum + 1e-20)[:, None] * scale

    log_t = tl.where(((ka >= K) & kmask)[None, :], (E + ka - K)[None, :], sel_t)
    wt = tl.where(((ka >= K) & kmask)[None, :], 1.0, wt)
    if HAS_FR:
        log_t = tl.where(fr[:, None], rnd_idx, log_t)
        wt = tl.where(fr[:, None], rnd_w, wt)
    wt = tl.where(active[:, None], wt, 0.0)

    nfr = fr == 0
    sel_b = active & nfr

    if HAS_MAP:
        am = sel_b[:, None] & kmask[None, :] & rmask[:, None]
        # Both per-lane indirect loads are issued on one flat (NFLAT,) index
        # tile. The last axis is only KB = 8..10 lanes, well under the 64-lane
        # vector, so the 2-D form makes the compiler emit a partially-filled
        # gather per row plus a 2-D buffer copy around it; flattened, the same
        # RPP*KB elements become full vectors. The values fetched are identical
        # (same tables, same indices) and the duplicate-index arithmetic runs on
        # those same values, so this is a code-shape change only. NFLAT is passed
        # from the host because this frontend validates reshape lengths as
        # constexpr[int] and rejects a length derived from other constexprs.
        am_f = tl.reshape(am, (NFLAT,))
        lt_f = tl.reshape(log_t, (NFLAT,))
        sd_f = tl.reshape(
            tl.broadcast_to((ep_rank + offs_r * 23333)[:, None], (RPP, KB)), (NFLAT,)
        )
        cntv = tl.maximum(tl.load(cnt_ptr + lt_f, mask=am_f, other=1), 1)
        if NDUP == 1:
            du = tl.zeros_like(cntv)
        elif NDUP == 2:
            du = tl.where(cntv == 2, sd_f & 1, 0)
        elif NDUP <= 4:
            t3 = (sd_f >> 16) + (sd_f & 65535)
            u3 = (t3 >> 8) + (t3 & 255)
            v3 = (u3 >> 4) + (u3 & 15)
            m3 = v3 - 3 * ((v3 * 43) >> 7)
            du1 = tl.where(cntv == 2, sd_f & 1, m3)
            du = tl.where(cntv == 4, sd_f & 3, du1)
            du = tl.where(cntv == 1, 0, du)
            du = tl.where(cntv == 3, m3, du)
        else:
            du = sd_f % cntv
        ph = tl.reshape(
            tl.load(tmap_ptr + lt_f * tmap_s0 + du, mask=am_f, other=-1), (RPP, KB)
        )
    else:
        ph = log_t
    res = tl.where(sel_b[:, None], ph, tl.where(fr[:, None], log_t, -1))

    m2 = rmask[:, None] & kmask[None, :]
    a_idx = idx32_ptr + offs_r[:, None] * idx32_s0 + 2 * ka[None, :]
    tl.store(a_idx, tl.where(active[:, None], res, -1), mask=m2)
    tl.store(a_idx + 1, tl.where(active[:, None], 0, -1), mask=m2)
    tl.store(w_ptr + offs_r[:, None] * w_s0 + ka[None, :], wt, mask=m2)

    if HAS_UNM:
        m3 = rmask[:, None] & kl[None, :]
        a_u = unm32_ptr + offs_r[:, None] * unm32_s0 + 2 * ka[None, :]
        tl.store(a_u, tl.where(sel_b[:, None], log_t, -1), mask=m3)
        tl.store(a_u + 1, tl.where(sel_b[:, None], 0, -1), mask=m3)

    if HAS_SC:
        tl.store(
            sc_ptr + offs_r[:, None] * sc_s0 + offs_e[None, :],
            tl.where(sel_b[:, None], s, 0.0),
            mask=lm,
        )


def moe_topk_gate_forward(
    logits,
    num_topk,
    use_shared_as_routed,
    num_shared_experts,
    routed_scaling_factor,
    ep_rank,
    scoring_func="sqrtsoftplus",
    mask=None,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    to_physical_map=None,
    logical_count=None,
    unmapped_topk_idx=None,
    force_random=None,
    out=None,
    scores=None,
):
    """Select experts with sqrt-softplus scores and stable top-k routing."""
    if scoring_func != "sqrtsoftplus":
        raise ValueError("only sqrtsoftplus scoring is supported")
    rows = logits.size(0)
    E = logits.size(1)
    K = int(num_topk)
    ns = int(num_shared_experts) if use_shared_as_routed else 0
    kp = K + ns

    if out is None:
        topk_idx = torch.empty((rows, kp), dtype=torch.int64, device=logits.device)
        topk_weights = torch.empty(
            (rows, kp), dtype=torch.float32, device=logits.device
        )
    else:
        topk_idx, topk_weights = out

    if rows == 0 or kp == 0:
        return topk_idx, topk_weights

    idx32 = topk_idx.view(torch.int32)
    has_unm = unmapped_topk_idx is not None
    if has_unm:
        unm32 = unmapped_topk_idx.view(torch.int32)
        unm_s0 = unmapped_topk_idx.stride(0) * 2
    else:
        unm32 = idx32
        unm_s0 = idx32.stride(0)

    has_mask = mask is not None
    has_fr = force_random is not None
    has_fix = fix_routing_mask is not None and has_unm
    has_map = to_physical_map is not None
    has_sc = scores is not None

    if bias is not None and image_bias is not None:
        bias_mode = 3
    elif bias is not None:
        bias_mode = 1
    elif image_bias is not None:
        bias_mode = 2
    else:
        bias_mode = 0

    i8 = torch.int8
    mask_i8 = mask.view(i8) if has_mask else logits
    fr_i8 = force_random.view(i8) if has_fr else logits
    fix_i8 = fix_routing_mask.view(i8) if has_fix else logits
    imask_i8 = image_token_mask.view(i8) if image_token_mask is not None else logits

    nlog = E + ns
    ndup = to_physical_map.size(1) if has_map else 1
    EB = triton.next_power_of_2(E)
    # The top-k tile was padded to a power of two. A compile probe on this
    # target confirmed tl.arange accepts an arbitrary length, so use the exact
    # lane count instead. This matters because the map epilogue issues two
    # indirect loads over this tile and they were measured at ~19-24 AIV vector
    # cycles PER SCHEDULED LANE: KB=16 scheduled 16 lanes for at most KP=10 live
    # ones, so ~37.5% of the gather work fetched dead lanes. With ns=0 (every
    # use_shared_as_routed=False workload) kp == K == 8 and KB is unchanged, so
    # this only touches the map specialization.
    KB = kp
    # Same reasoning as KB: the logical-expert count vector is loaded into a
    # (CBLK,) tile, which no longer needs power-of-two padding, so the load
    # predicate `ca < NLOG` disappears with it.
    CBLK = nlog
    # Block size derived from the vector-core count, not from a per-shape table:
    # aim for exactly one program per AIV core so the grid is a whole number of
    # rounds with no leftover tail.
    #   N_AIV = 64 comes from the target platform, not from timing:
    #   torch.npu.get_device_properties(0) reports
    #   _NPUDeviceProperties(name='Ascend950DT_9582', cube_core_num=32,
    #                        vector_core_num=64, multi_processor_count=64),
    #   i.e. the 32 "num_sm" entries in /status are cube cores, while the vector
    #   cores that execute this kernel are 64.
    # ceil(rows/64) is capped at 32: the kernel keeps several (RPP, EB) fp32
    # tiles live, so beyond 32 rows per program the live set outgrows UB and
    # spills. The cap only binds for rows > 2048, where ceil(rows/64) > 32.
    N_AIV = 64
    RPP_CAP = 32
    RPP = max(1, min(RPP_CAP, (rows + N_AIV - 1) // N_AIV))

    grid = ((rows + RPP - 1) // RPP,)
    _gate_kernel[grid](
        logits,
        logits.stride(0),
        bias if bias is not None else topk_weights,
        image_bias if image_bias is not None else topk_weights,
        imask_i8,
        mask_i8,
        fr_i8,
        fix_i8,
        logical_count if logical_count is not None else idx32,
        to_physical_map if has_map else idx32,
        to_physical_map.stride(0) if has_map else 0,
        idx32,
        idx32.stride(0),
        unm32,
        unm_s0,
        topk_weights,
        topk_weights.stride(0),
        scores if has_sc else logits,
        scores.stride(0) if has_sc else 0,
        rows,
        routed_scaling_factor,
        int(ep_rank),
        E=E,
        EB=EB,
        K=K,
        KP=kp,
        KB=KB,
        NS=ns,
        RPP=RPP,
        NFLAT=RPP * KB,
        NLOG=nlog,
        CBLK=CBLK,
        NDUP=ndup,
        BIAS_MODE=bias_mode,
        HAS_MASK=has_mask,
        HAS_FR=has_fr,
        HAS_FIX=has_fix,
        HAS_MAP=has_map,
        HAS_SC=has_sc,
        HAS_UNM=has_unm,
    )
    return topk_idx, topk_weights
