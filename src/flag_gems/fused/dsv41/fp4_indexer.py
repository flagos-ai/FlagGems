# Copyright 2023-2026 SGLang Team
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
# Adapted from CherryLemon/sglang, runtime 7a882e727d09ed7b896b44227b80b4e0b0b7481d.

"""DeepSeek V4.1 packed-FP4 indexer, ported from the validated V12 source.

DeepGEMM's fp8_fp4 mqa-logits kernels need SM100/SM120. This Triton kernel covers
the same decode step on SM90: one query token per request, scored against the
request's visible compressed positions read straight out of the fp4 indexer
pool (e2m1 payload + e8m0 per-32 block scales, page layout of
store_fp4_index_k_cache), summed over heads with relu and the per-head weights.

Numerics follow the torch reference path: bf16 dot, bf16 relu/weight product,
bf16 head reduction, fp32 logits out.
"""

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)


@triton.jit
def _e2m1_decode(code):
    # E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6. Their exact FP32
    # encodings avoid per-element exponentiation and floating-point decoding.
    u = code.to(tl.uint32)
    mag = u & 7
    bits = tl.where(
        mag == 0, 0, tl.where(mag == 1, 0x3F000000, 0x3F000000 + (mag << 22))
    )
    v = (bits | ((u & 8) << 28)).to(tl.float32, bitcast=True)
    # Legacy decoding canonicalizes FP4 -0 to +0. An explicit exact add keeps
    # that behavior without an extra predicate/live range in the score kernel.
    return tl.where(mag == 0, 0.0, v)


@triton.jit
def _fp4_index_logits_kernel(
    q_ptr,  # [B, H, D] bf16, fq4 queries (already rope'd)
    w_ptr,  # [B, H] bf16 head weights (softmax scale folded in)
    slots_ptr,  # [B, L] pool slots per (request, compressed position)
    req_to_token_ptr,  # [num_reqs, max_context_len] full-token pool slots
    req_ptr,  # [B] request-pool row for each query
    candidate_blocks_ptr,
    lens_ptr,  # [B] int64 visible compressed positions per request
    table_ptr,  # [num_pages, page_size * 64 + page_size * 4] uint8
    out_ptr,  # [B, L] fp32 logits, -inf beyond lens
    candidate_scores_ptr,
    candidate_lens_ptr,
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    stride_req,
    candidate_block_stride,
    stride_out,
    candidate_score_stride,
    H: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 == 64 nibble-pairs per row
    BLOCK_L: tl.constexpr,
    SKIP_INVALID: tl.constexpr,
    RATIO: tl.constexpr,
    USE_REQ_TO_TOKEN: tl.constexpr,
    USE_CANDIDATE_BLOCKS: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

    n_vis = tl.load(lens_ptr + b)
    if SKIP_INVALID and lb * BLOCK_L >= tl.minimum(n_vis, L):
        if WRITE_LOGITS:
            tl.store(out_ptr + b * stride_out + offs_l, -float("inf"), offs_l < L)
        if WRITE_CANDIDATES:
            nblocks = (L + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
            block_ids = lb * (BLOCK_L // CANDIDATE_BLOCK_SIZE) + tl.arange(
                0, BLOCK_L // CANDIDATE_BLOCK_SIZE
            )
            tl.store(
                candidate_scores_ptr + b * candidate_score_stride + block_ids,
                -float("inf"),
                block_ids < nblocks,
            )
            tl.store(candidate_lens_ptr + b, 0, mask=lb == 0)
        return
    valid = offs_l < tl.minimum(n_vis, L)
    if USE_CANDIDATE_BLOCKS:
        block_col = offs_l // CANDIDATE_BLOCK_SIZE
        within = offs_l % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + b * candidate_block_stride + block_col,
            mask=valid,
            other=0,
        )
        logical_position = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
        req = tl.load(req_ptr + b).to(tl.int64)
        slot = tl.load(
            req_to_token_ptr + req * stride_req + logical_position * RATIO,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = slot // RATIO
    elif USE_REQ_TO_TOKEN:
        req = tl.load(req_ptr + b).to(tl.int64)
        slot = tl.load(
            req_to_token_ptr + req * stride_req + offs_l.to(tl.int64) * RATIO,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = slot // RATIO
    else:
        slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(
            tl.int64
        )
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride

    # K payload: [BLOCK_L, HALF_D] uint8
    pay = tl.load(
        table_ptr + row_base[:, None] + off[:, None] * PAYLOAD_BYTES + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    low = _e2m1_decode(pay & 0x0F)
    high = _e2m1_decode((pay >> 4) & 0x0F)
    # e8m0 block scales: element j uses block j // 32 -> byte i uses block i // 16.
    sc_idx = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base[:, None]
        + page_size * PAYLOAD_BYTES
        + off[:, None] * SCALE_BYTES
        + sc_idx[None, :],
        mask=valid[:, None],
        other=127,
    )
    # UE8M0 reserves 255 for NaN; it is not a finite power-of-two scale.
    scale = tl.where(exps == 255, float("nan"), tl.exp2(exps.to(tl.float32) - 127.0))
    k_low = (low * scale).to(tl.bfloat16)  # [BLOCK_L, HALF_D] elements 2i
    k_high = (high * scale).to(tl.bfloat16)  # elements 2i+1

    # queries: even / odd elements, [H, HALF_D] bf16
    q_even = tl.load(
        q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :]
    )
    q_odd = tl.load(
        q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :] + 1
    )

    acc = tl.dot(q_even, tl.trans(k_low))  # [H, BLOCK_L] fp32
    acc += tl.dot(q_odd, tl.trans(k_high))
    # reference rounding points: bf16 dot -> relu -> * bf16 weight -> bf16 -> sum -> bf16
    s = acc.to(tl.bfloat16).to(tl.float32)
    # torch.relu preserves NaN from reserved UE8M0 exponent 255.
    s = tl.where(s != s, s, tl.maximum(s, 0.0))  # noqa: PLR0124
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    if WRITE_LOGITS:
        tl.store(out_ptr + b * stride_out + offs_l, logit, mask=offs_l < L)
    if WRITE_CANDIDATES:
        blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        block_scores = tl.reshape(logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE))
        block_scores = tl.max(block_scores, axis=1)
        block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
        last_block = (n_vis - 1) // CANDIDATE_BLOCK_SIZE
        block_scores = tl.where(
            (n_vis > 0) & (block_ids == last_block), float("inf"), block_scores
        )
        num_blocks = (L + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        tl.store(
            candidate_scores_ptr + b * candidate_score_stride + block_ids,
            block_scores,
            mask=block_ids < num_blocks,
        )
        tl.store(
            candidate_lens_ptr + b,
            (n_vis + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
            mask=lb == 0,
        )


# Each CTA shares a request's decoded K tile across six verify queries. Keep
# each query's original 32-head MMA and reduction layout: increasing the MMA M
# dimension can change floating-point reduction order. Request equality is
# checked on device on every replay; visible lengths and source writes remain
# per query. Compact candidate lists deliberately use the original kernel.
@triton.jit
def _fp4_group_load_keys(
    RT,
    TABLE,
    req,
    nvis,
    L,
    STRIDE_REQ,
    ROW_STRIDE,
    PAGE,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
):
    l = tl.program_id(1) * BL + tl.arange(0, BL)
    i = tl.arange(0, 64)
    valid = l < tl.minimum(nvis, L)
    slot = (
        tl.load(
            RT + req.to(tl.int64) * STRIDE_REQ + l.to(tl.int64) * RATIO, valid, other=0
        ).to(tl.int64)
        // RATIO
    )
    page = slot // PAGE
    off = slot % PAGE
    row = page * ROW_STRIDE
    pay = tl.load(
        TABLE + row[:, None] + off[:, None] * 64 + i[None, :], valid[:, None], other=0
    )
    low = _e2m1_decode(pay & 15)
    high = _e2m1_decode(pay >> 4 & 15)
    exps = tl.load(
        TABLE + row[:, None] + PAGE * 64 + off[:, None] * 4 + (i // 16)[None, :],
        valid[:, None],
        other=127,
    )
    # UE8M0 reserves 255 for NaN; it is not a finite power-of-two scale.
    scale = tl.where(exps == 255, float("nan"), tl.exp2(exps.to(tl.float32) - 127.0))
    return ((low * scale).to(tl.bfloat16), (high * scale).to(tl.bfloat16))


@triton.jit
def _fp4_group_invalid(
    OUT,
    CS,
    CL,
    b,
    L,
    SO,
    SCS,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    lb = tl.program_id(1)
    l = lb * BL + tl.arange(0, BL)
    if WRITE_LOGITS:
        tl.store(OUT + b * SO + l, -float("inf"), l < L)
    if WRITE_CANDIDATES:
        ids = lb * (BL // CB) + tl.arange(0, BL // CB)
        tl.store(CS + b * SCS + ids, -float("inf"), ids < tl.cdiv(L, CB))
        tl.store(CL + b, 0, lb == 0)


@triton.jit
def _fp4_group_score(
    Q,
    W,
    OUT,
    CS,
    CL,
    KL,
    KH,
    b,
    nvis,
    L,
    SQB,
    SQH,
    SWB,
    SO,
    SCS,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    h = tl.arange(0, 32)
    i = tl.arange(0, 64)
    lb = tl.program_id(1)
    l = lb * BL + tl.arange(0, BL)
    qe = tl.load(Q + b * SQB + h[:, None] * SQH + 2 * i[None, :])
    qo = tl.load(Q + b * SQB + h[:, None] * SQH + 2 * i[None, :] + 1)
    acc = tl.dot(qe, tl.trans(KL))
    acc += tl.dot(qo, tl.trans(KH))
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.where(s != s, s, tl.maximum(s, 0.0))  # noqa: PLR0124
    w = tl.load(W + b * SWB + h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, 0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(l < tl.minimum(nvis, L), logit, -float("inf"))
    if WRITE_LOGITS:
        tl.store(OUT + b * SO + l, logit, l < L)
    if WRITE_CANDIDATES:
        scores = tl.max(tl.reshape(logit, (BL // CB, CB)), 1)
        ids = lb * (BL // CB) + tl.arange(0, BL // CB)
        scores = tl.where((nvis > 0) & (ids == (nvis - 1) // CB), float("inf"), scores)
        tl.store(CS + b * SCS + ids, scores, ids < tl.cdiv(L, CB))
        tl.store(CL + b, tl.cdiv(nvis, CB), lb == 0)


@triton.jit
def _fp4_index_logits_grouped_kernel(
    Q,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    CL,
    B,
    L,
    PAGE,
    ROW_STRIDE,
    SQB,
    SQH,
    SWB,
    SR,
    SO,
    SCS,
    GROUP: tl.constexpr,
    PGROUP: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr = 64,
    CB: tl.constexpr = 8,
    WRITE_LOGITS: tl.constexpr = True,
    WRITE_CANDIDATES: tl.constexpr = False,
):
    first = tl.program_id(0) * GROUP
    lb = tl.program_id(1)
    offset = tl.arange(0, PGROUP)
    rows = first + offset
    mask = (offset < GROUP) & (rows < B)
    lens = tl.load(LENS + rows, mask, other=0)
    maxlens = tl.max(lens, 0)
    if lb * BL >= tl.minimum(maxlens, L):
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                _fp4_group_invalid(
                    OUT, CS, CL, b, L, SO, SCS, BL, CB, WRITE_LOGITS, WRITE_CANDIDATES
                )
        return
    req0 = tl.load(REQ + first)
    reqs = tl.load(REQ + rows, mask, other=req0)
    same = tl.sum((reqs != req0).to(tl.int32), 0) == 0
    if same:
        kl, kh = _fp4_group_load_keys(
            RT, TABLE, req0, maxlens, L, SR, ROW_STRIDE, PAGE, RATIO, BL
        )
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                nvis = tl.load(LENS + b)
                _fp4_group_score(
                    Q,
                    W,
                    OUT,
                    CS,
                    CL,
                    kl,
                    kh,
                    b,
                    nvis,
                    L,
                    SQB,
                    SQH,
                    SWB,
                    SO,
                    SCS,
                    BL,
                    CB,
                    WRITE_LOGITS,
                    WRITE_CANDIDATES,
                )
    else:
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                nvis = tl.load(LENS + b)
                if lb * BL < tl.minimum(nvis, L):
                    req = tl.load(REQ + b)
                    kl, kh = _fp4_group_load_keys(
                        RT, TABLE, req, nvis, L, SR, ROW_STRIDE, PAGE, RATIO, BL
                    )
                    _fp4_group_score(
                        Q,
                        W,
                        OUT,
                        CS,
                        CL,
                        kl,
                        kh,
                        b,
                        nvis,
                        L,
                        SQB,
                        SQH,
                        SWB,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    )
                else:
                    _fp4_group_invalid(
                        OUT,
                        CS,
                        CL,
                        b,
                        L,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    )


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """q [B, H, 128] bf16, weights [B, H], slots [B, L], lens [B] int64,
    table = the layer's fp4 index-K page buffer (uint8, 2D). Returns [B, L] fp32
    logits with -inf at positions >= lens."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    L = slots.shape[1]
    assert table.dtype == torch.uint8 and table.dim() == 2
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    out_storage = torch.empty(
        (B, triton.cdiv(L, 4) * 4), dtype=torch.float32, device=q.device
    )
    out = out_storage[:, :L]
    if B == 0 or L == 0:
        return out
    BLOCK_L = 64
    grid = (B, triton.cdiv(L, BLOCK_L))
    _fp4_index_logits_kernel[grid](
        q,
        weights,
        slots,
        slots,
        lens,
        slots,
        lens.to(torch.int64).contiguous(),
        table,
        out,
        out,
        lens,
        L,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        slots.stride(0),
        slots.stride(0),
        out.stride(0),
        out.stride(0),
        H=H,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=BLOCK_L,
        SKIP_INVALID=True,
        RATIO=1,
        USE_REQ_TO_TOKEN=False,
        USE_CANDIDATE_BLOCKS=False,
        CANDIDATE_BLOCK_SIZE=1,
        WRITE_CANDIDATES=False,
        WRITE_LOGITS=True,
        num_warps=4,
    )
    return out


def fp4_index_logits_req_to_token(
    q: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    width: int,
    candidate_block_size: int = 0,
    write_logits: bool = True,
    query_group_size: int = 1,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Score logical compressed positions without materializing their pool slots."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    assert req.shape == lens.shape == (B,)
    assert req_to_token.dim() == 2 and req_to_token.stride(1) == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert ratio in (1, 2)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req = req.to(torch.int64).contiguous()
    lens = lens.to(torch.int64).contiguous()
    output_width = triton.cdiv(width, 4) * 4 if write_logits else 4
    out_storage = torch.empty((B, output_width), dtype=torch.float32, device=q.device)
    out = out_storage[:, :width]
    block_l = 64
    if candidate_block_size:
        assert block_l % candidate_block_size == 0
        num_blocks = triton.cdiv(width, candidate_block_size)
        candidate_storage = torch.empty(
            (B, triton.cdiv(num_blocks, 4) * 4),
            dtype=torch.float32,
            device=q.device,
        )
        candidate_scores = candidate_storage[:, :num_blocks]
        candidate_lens = torch.empty(B, dtype=torch.int32, device=q.device)
    else:
        candidate_scores = out
        candidate_lens = lens
    if B == 0 or width == 0:
        if candidate_block_size:
            candidate_lens.zero_()
            if not write_logits:
                return candidate_scores, candidate_lens
            return out, candidate_scores, candidate_lens
        return out
    # The caller supplies a semantic target-verify hint, never a guess from B.
    # Unsupported heads, block layouts, and partial groups retain the old grid.
    use_grouped = (
        query_group_size == 6
        and q.device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(q.device) == (9, 0)
        # Validated even request buckets from 20 through 40 share six queries.
        # Other graph buckets (including partial groups) keep the original grid.
        and B in (120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240)
        and H == 32
        and candidate_block_size in (0, 8)
    )
    if use_grouped:
        _fp4_index_logits_grouped_kernel[(B // 6, triton.cdiv(width, block_l))](
            q,
            weights,
            req_to_token,
            req,
            lens,
            table,
            out,
            candidate_scores,
            candidate_lens,
            B,
            width,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            candidate_scores.stride(0),
            GROUP=6,
            PGROUP=8,
            RATIO=ratio,
            BL=block_l,
            CB=candidate_block_size or 1,
            WRITE_LOGITS=write_logits,
            WRITE_CANDIDATES=bool(candidate_block_size),
            num_warps=4,
        )
    else:
        grid = (B, triton.cdiv(width, block_l))
        _fp4_index_logits_kernel[grid](
            q,
            weights,
            req_to_token,
            req_to_token,
            req,
            req_to_token,
            lens,
            table,
            out,
            candidate_scores,
            candidate_lens,
            width,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            req_to_token.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            candidate_scores.stride(0),
            H=H,
            HALF_D=INDEX_HEAD_DIM // 2,
            BLOCK_L=block_l,
            SKIP_INVALID=True,
            RATIO=ratio,
            USE_REQ_TO_TOKEN=True,
            USE_CANDIDATE_BLOCKS=False,
            CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
            WRITE_CANDIDATES=bool(candidate_block_size),
            WRITE_LOGITS=write_logits,
            num_warps=4,
        )
    if not write_logits:
        assert candidate_block_size
        return candidate_scores, candidate_lens
    if candidate_block_size:
        return out, candidate_scores, candidate_lens
    return out


def fp4_index_logits_candidate_blocks(
    q: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    candidate_block_size: int,
) -> torch.Tensor:
    """Score compact candidate blocks while resolving physical slots in-kernel."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    batch, heads, _ = q.shape
    assert req.shape == candidate_lens.shape == (batch,)
    assert candidate_blocks.shape[0] == batch
    assert req_to_token.dim() == 2 and req_to_token.stride(1) == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert ratio in (1, 2)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req = req.to(torch.int64).contiguous()
    candidate_lens = candidate_lens.to(torch.int32).contiguous()
    candidate_blocks = candidate_blocks.to(torch.int32).contiguous()
    width = candidate_blocks.shape[1] * candidate_block_size
    out_storage = torch.empty(
        (batch, triton.cdiv(width, 4) * 4),
        dtype=torch.float32,
        device=q.device,
    )
    out = out_storage[:, :width]
    if batch == 0 or width == 0:
        return out
    block_l = 64
    _fp4_index_logits_kernel[(batch, triton.cdiv(width, block_l))](
        q,
        weights,
        candidate_blocks,
        req_to_token,
        req,
        candidate_blocks,
        candidate_lens,
        table,
        out,
        out,
        candidate_lens,
        width,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        req_to_token.stride(0),
        candidate_blocks.stride(0),
        out.stride(0),
        out.stride(0),
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=block_l,
        SKIP_INVALID=True,
        RATIO=ratio,
        USE_REQ_TO_TOKEN=False,
        USE_CANDIDATE_BLOCKS=True,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        WRITE_CANDIDATES=False,
        WRITE_LOGITS=True,
        num_warps=4,
    )
    return out


@triton.jit
def _unpack_fp4_index_keys_to_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Decode block-scaled E2M1 values directly into E4M3."""
    row = tl.program_id(0)
    offs_i = tl.arange(0, HALF_D)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(table_ptr + row_base + off * PAYLOAD_BYTES + offs_i)
    scale_block = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base
        + page_size * PAYLOAD_BYTES
        + off * SCALE_BYTES
        + scale_block
    )
    # UE8M0 reserves 255 for NaN; it is not a finite power-of-two scale.
    scale = tl.where(exps == 255, float("nan"), tl.exp2(exps.to(tl.float32) - 127.0))
    low = tl.clamp(_e2m1_decode(pay & 0x0F) * scale, -FP8_MAX, FP8_MAX).to(
        out_ptr.dtype.element_ty
    )
    high = tl.clamp(
        _e2m1_decode((pay >> 4) & 0x0F) * scale,
        -FP8_MAX,
        FP8_MAX,
    ).to(out_ptr.dtype.element_ty)
    out = out_ptr + row * (HALF_D * 2)
    tl.store(out + 2 * offs_i, low)
    tl.store(out + 2 * offs_i + 1, high)


def unpack_fp4_index_keys_to_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather FP4 index-K rows and decode them directly into E4M3."""
    assert slots.dim() == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    slots = slots.to(torch.int64).contiguous()
    values = torch.empty(
        (slots.shape[0], INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=slots.device,
    )
    if slots.numel() > 0:
        _unpack_fp4_index_keys_to_fp8_kernel[(slots.shape[0],)](
            slots,
            table,
            values,
            page_size,
            table.stride(0),
            HALF_D=INDEX_HEAD_DIM // 2,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=4,
        )
    return values


@triton.jit
def _quantize_bf16_index_queries_fp8_kernel(
    q_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :]
    ).to(tl.float32)
    q_fp8 = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :],
        q_fp8,
    )


def quantize_bf16_index_queries_fp8(
    q: torch.Tensor,
) -> torch.Tensor:
    """Cast each BF16 query head to E4M3 for one K=128 tensor-core dot."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    rows, heads, _ = q.shape
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if rows > 0:
        _quantize_bf16_index_queries_fp8_kernel[(rows,)](
            q,
            values,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            H=heads,
            D=INDEX_HEAD_DIM,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=8,
        )
    return values


@triton.jit
def _fp8_index_logits_prefill_kernel(
    q_ptr,  # [B, H, D] e4m3
    w_ptr,  # [B, H] bf16
    k_ptr,  # [L, D] e4m3
    lens_ptr,  # [B] int64
    out_ptr,  # [B, OUT_L] fp32
    candidate_blocks_ptr,
    stride_candidates,
    L,
    OUT_L,
    stride_qb,
    stride_qh,
    stride_kl,
    stride_wb,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    out_cols = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    if USE_CANDIDATES:
        blocks = tl.load(
            candidate_blocks_ptr
            + b * stride_candidates
            + out_cols // CANDIDATE_BLOCK_SIZE,
            mask=out_cols < OUT_L,
            other=-1,
        ).to(tl.int64)
        offs_l = blocks * CANDIDATE_BLOCK_SIZE + out_cols % CANDIDATE_BLOCK_SIZE
    else:
        offs_l = out_cols
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    n_vis = tl.load(lens_ptr + b)
    valid = (offs_l >= 0) & (offs_l < tl.minimum(n_vis, L)) & (out_cols < OUT_L)
    q = tl.load(q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :])
    k = tl.load(
        k_ptr + offs_l[:, None] * stride_kl + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    # Preserve the reference post-dot rounding and reduction points.
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.where(s != s, s, tl.maximum(s, 0.0))  # noqa: PLR0124
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * OUT_L + out_cols, logit, mask=out_cols < OUT_L)


def fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    *,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 8,
) -> torch.Tensor:
    """Score E4M3 queries against E4M3 keys with FP32 accumulation.

    Output rows are padded to four floats for the fused ragged top-k kernel;
    padding remains unreachable because it is initialized to ``-inf``.
    """
    assert q.dtype == torch.float8_e4m3fn and q.shape[-1] == INDEX_HEAD_DIM
    rows, heads, _ = q.shape
    width = keys.shape[0]
    assert keys.dtype == torch.float8_e4m3fn and keys.shape[1:] == (INDEX_HEAD_DIM,)
    assert weights.shape == (rows, heads)
    assert lens.shape == (rows,)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    keys = keys.contiguous()
    use_candidates = candidate_blocks is not None
    if use_candidates:
        assert candidate_blocks.ndim == 2 and candidate_blocks.shape[0] == rows
        assert candidate_blocks.dtype in (torch.int32, torch.int64)
        assert candidate_block_size % 4 == 0
        out_width = candidate_blocks.shape[1] * candidate_block_size
    else:
        out_width = ((width + 3) // 4) * 4
    out = torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    if rows == 0 or out_width == 0:
        return out
    # On H100, the 32-head source path is limited by repeated cached loads
    # and small CTAs, rather than HBM traffic. Widen only the key dimension:
    # grouping query heads would change the tensor-core accumulation path.
    # Candidate gathers need a smaller tile to retain enough resident warps.
    block_l = 64
    if heads == 32 and out_width >= 4096 and rows * out_width >= 262144:
        block_l = 128 if use_candidates else 256
    _fp8_index_logits_prefill_kernel[(rows, triton.cdiv(out_width, block_l))](
        q,
        weights,
        keys,
        lens.to(torch.int64).contiguous(),
        out,
        candidate_blocks if use_candidates else lens,
        candidate_blocks.stride(0) if use_candidates else 0,
        width,
        out_width,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        H=heads,
        D=INDEX_HEAD_DIM,
        BLOCK_L=block_l,
        USE_CANDIDATES=use_candidates,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        num_warps=4,
    )
    return out


FP8_E4M3_MAX = 448.0
