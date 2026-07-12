import torch
import triton
import triton.language as tl


@triton.jit
def mhc_split_sinkhorn_kernel_hcmult_4(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    num_tokens,
    BLOCK_N: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    """Vectorized kernel for HC_MULT=4."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < num_tokens
    base = offs * 24

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    m0 = tl.load(mixes_ptr + base + 0, mask=mask)
    m1 = tl.load(mixes_ptr + base + 1, mask=mask)
    m2 = tl.load(mixes_ptr + base + 2, mask=mask)
    m3 = tl.load(mixes_ptr + base + 3, mask=mask)
    m4 = tl.load(mixes_ptr + base + 4, mask=mask)
    m5 = tl.load(mixes_ptr + base + 5, mask=mask)
    m6 = tl.load(mixes_ptr + base + 6, mask=mask)
    m7 = tl.load(mixes_ptr + base + 7, mask=mask)

    b0 = tl.load(hc_base_ptr + 0)
    b1 = tl.load(hc_base_ptr + 1)
    b2 = tl.load(hc_base_ptr + 2)
    b3 = tl.load(hc_base_ptr + 3)
    b4 = tl.load(hc_base_ptr + 4)
    b5 = tl.load(hc_base_ptr + 5)
    b6 = tl.load(hc_base_ptr + 6)
    b7 = tl.load(hc_base_ptr + 7)

    pre_base = offs * 4
    tl.store(pre_ptr + pre_base + 0, tl.sigmoid(m0 * scale_0 + b0) + 1e-6, mask=mask)
    tl.store(pre_ptr + pre_base + 1, tl.sigmoid(m1 * scale_0 + b1) + 1e-6, mask=mask)
    tl.store(pre_ptr + pre_base + 2, tl.sigmoid(m2 * scale_0 + b2) + 1e-6, mask=mask)
    tl.store(pre_ptr + pre_base + 3, tl.sigmoid(m3 * scale_0 + b3) + 1e-6, mask=mask)

    post_base = offs * 4
    tl.store(post_ptr + post_base + 0, 2.0 * tl.sigmoid(m4 * scale_1 + b4), mask=mask)
    tl.store(post_ptr + post_base + 1, 2.0 * tl.sigmoid(m5 * scale_1 + b5), mask=mask)
    tl.store(post_ptr + post_base + 2, 2.0 * tl.sigmoid(m6 * scale_1 + b6), mask=mask)
    tl.store(post_ptr + post_base + 3, 2.0 * tl.sigmoid(m7 * scale_1 + b7), mask=mask)

    cb = 8
    b8 = tl.load(hc_base_ptr + cb + 0)
    b9 = tl.load(hc_base_ptr + cb + 1)
    b10 = tl.load(hc_base_ptr + cb + 2)
    b11 = tl.load(hc_base_ptr + cb + 3)
    b12 = tl.load(hc_base_ptr + cb + 4)
    b13 = tl.load(hc_base_ptr + cb + 5)
    b14 = tl.load(hc_base_ptr + cb + 6)
    b15 = tl.load(hc_base_ptr + cb + 7)
    b16 = tl.load(hc_base_ptr + cb + 8)
    b17 = tl.load(hc_base_ptr + cb + 9)
    b18 = tl.load(hc_base_ptr + cb + 10)
    b19 = tl.load(hc_base_ptr + cb + 11)
    b20 = tl.load(hc_base_ptr + cb + 12)
    b21 = tl.load(hc_base_ptr + cb + 13)
    b22 = tl.load(hc_base_ptr + cb + 14)
    b23 = tl.load(hc_base_ptr + cb + 15)

    cm_00 = tl.load(mixes_ptr + base + cb + 0, mask=mask) * scale_2 + b8
    cm_01 = tl.load(mixes_ptr + base + cb + 1, mask=mask) * scale_2 + b9
    cm_02 = tl.load(mixes_ptr + base + cb + 2, mask=mask) * scale_2 + b10
    cm_03 = tl.load(mixes_ptr + base + cb + 3, mask=mask) * scale_2 + b11
    cm_10 = tl.load(mixes_ptr + base + cb + 4, mask=mask) * scale_2 + b12
    cm_11 = tl.load(mixes_ptr + base + cb + 5, mask=mask) * scale_2 + b13
    cm_12 = tl.load(mixes_ptr + base + cb + 6, mask=mask) * scale_2 + b14
    cm_13 = tl.load(mixes_ptr + base + cb + 7, mask=mask) * scale_2 + b15
    cm_20 = tl.load(mixes_ptr + base + cb + 8, mask=mask) * scale_2 + b16
    cm_21 = tl.load(mixes_ptr + base + cb + 9, mask=mask) * scale_2 + b17
    cm_22 = tl.load(mixes_ptr + base + cb + 10, mask=mask) * scale_2 + b18
    cm_23 = tl.load(mixes_ptr + base + cb + 11, mask=mask) * scale_2 + b19
    cm_30 = tl.load(mixes_ptr + base + cb + 12, mask=mask) * scale_2 + b20
    cm_31 = tl.load(mixes_ptr + base + cb + 13, mask=mask) * scale_2 + b21
    cm_32 = tl.load(mixes_ptr + base + cb + 14, mask=mask) * scale_2 + b22
    cm_33 = tl.load(mixes_ptr + base + cb + 15, mask=mask) * scale_2 + b23

    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm)
    cm_01 = tl.exp(cm_01 - rm)
    cm_02 = tl.exp(cm_02 - rm)
    cm_03 = tl.exp(cm_03 - rm)
    inv_rs = 1.0 / (cm_00 + cm_01 + cm_02 + cm_03)
    cm_00 = cm_00 * inv_rs + 1e-6
    cm_01 = cm_01 * inv_rs + 1e-6
    cm_02 = cm_02 * inv_rs + 1e-6
    cm_03 = cm_03 * inv_rs + 1e-6

    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm)
    cm_11 = tl.exp(cm_11 - rm)
    cm_12 = tl.exp(cm_12 - rm)
    cm_13 = tl.exp(cm_13 - rm)
    inv_rs = 1.0 / (cm_10 + cm_11 + cm_12 + cm_13)
    cm_10 = cm_10 * inv_rs + 1e-6
    cm_11 = cm_11 * inv_rs + 1e-6
    cm_12 = cm_12 * inv_rs + 1e-6
    cm_13 = cm_13 * inv_rs + 1e-6

    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm)
    cm_21 = tl.exp(cm_21 - rm)
    cm_22 = tl.exp(cm_22 - rm)
    cm_23 = tl.exp(cm_23 - rm)
    inv_rs = 1.0 / (cm_20 + cm_21 + cm_22 + cm_23)
    cm_20 = cm_20 * inv_rs + 1e-6
    cm_21 = cm_21 * inv_rs + 1e-6
    cm_22 = cm_22 * inv_rs + 1e-6
    cm_23 = cm_23 * inv_rs + 1e-6

    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm)
    cm_31 = tl.exp(cm_31 - rm)
    cm_32 = tl.exp(cm_32 - rm)
    cm_33 = tl.exp(cm_33 - rm)
    inv_rs = 1.0 / (cm_30 + cm_31 + cm_32 + cm_33)
    cm_30 = cm_30 * inv_rs + 1e-6
    cm_31 = cm_31 * inv_rs + 1e-6
    cm_32 = cm_32 * inv_rs + 1e-6
    cm_33 = cm_33 * inv_rs + 1e-6

    inv_cs0 = 1.0 / (cm_00 + cm_10 + cm_20 + cm_30 + 1e-6)
    inv_cs1 = 1.0 / (cm_01 + cm_11 + cm_21 + cm_31 + 1e-6)
    inv_cs2 = 1.0 / (cm_02 + cm_12 + cm_22 + cm_32 + 1e-6)
    inv_cs3 = 1.0 / (cm_03 + cm_13 + cm_23 + cm_33 + 1e-6)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3

    for _ in range(SINKHORN_ITERS - 1):
        inv_rs0 = 1.0 / (cm_00 + cm_01 + cm_02 + cm_03 + 1e-6)
        inv_rs1 = 1.0 / (cm_10 + cm_11 + cm_12 + cm_13 + 1e-6)
        inv_rs2 = 1.0 / (cm_20 + cm_21 + cm_22 + cm_23 + 1e-6)
        inv_rs3 = 1.0 / (cm_30 + cm_31 + cm_32 + cm_33 + 1e-6)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3

        inv_cs0 = 1.0 / (cm_00 + cm_10 + cm_20 + cm_30 + 1e-6)
        inv_cs1 = 1.0 / (cm_01 + cm_11 + cm_21 + cm_31 + 1e-6)
        inv_cs2 = 1.0 / (cm_02 + cm_12 + cm_22 + cm_32 + 1e-6)
        inv_cs3 = 1.0 / (cm_03 + cm_13 + cm_23 + cm_33 + 1e-6)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3

    co = offs * 16
    tl.store(comb_ptr + co + 0, cm_00, mask=mask)
    tl.store(comb_ptr + co + 1, cm_01, mask=mask)
    tl.store(comb_ptr + co + 2, cm_02, mask=mask)
    tl.store(comb_ptr + co + 3, cm_03, mask=mask)
    tl.store(comb_ptr + co + 4, cm_10, mask=mask)
    tl.store(comb_ptr + co + 5, cm_11, mask=mask)
    tl.store(comb_ptr + co + 6, cm_12, mask=mask)
    tl.store(comb_ptr + co + 7, cm_13, mask=mask)
    tl.store(comb_ptr + co + 8, cm_20, mask=mask)
    tl.store(comb_ptr + co + 9, cm_21, mask=mask)
    tl.store(comb_ptr + co + 10, cm_22, mask=mask)
    tl.store(comb_ptr + co + 11, cm_23, mask=mask)
    tl.store(comb_ptr + co + 12, cm_30, mask=mask)
    tl.store(comb_ptr + co + 13, cm_31, mask=mask)
    tl.store(comb_ptr + co + 14, cm_32, mask=mask)
    tl.store(comb_ptr + co + 15, cm_33, mask=mask)


@triton.jit
def mhc_split_sinkhorn_kernel_generic(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    num_tokens,
    SINKHORN_ITERS: tl.constexpr,
    HC_MULT: tl.constexpr,
    MIX_HC: tl.constexpr,
):
    """Generic split+sinkhorn kernel for arbitrary HC_MULT (one token per program)."""
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    base = pid_n * MIX_HC
    pre_base = pid_n * HC_MULT
    post_base = pid_n * HC_MULT
    comb_base = pid_n * (HC_MULT * HC_MULT)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    for j in tl.static_range(HC_MULT):
        pre_idx = j
        post_idx = HC_MULT + j
        pre_m = tl.load(mixes_ptr + base + pre_idx)
        post_m = tl.load(mixes_ptr + base + post_idx)
        pre_b = tl.load(hc_base_ptr + pre_idx)
        post_b = tl.load(hc_base_ptr + post_idx)
        tl.store(pre_ptr + pre_base + j, tl.sigmoid(pre_m * scale_0 + pre_b) + 1e-6)
        tl.store(post_ptr + post_base + j, 2.0 * tl.sigmoid(post_m * scale_1 + post_b))

    comb_offset = 2 * HC_MULT

    for row in tl.static_range(HC_MULT):
        for col in tl.static_range(HC_MULT):
            idx = comb_offset + row * HC_MULT + col
            out_idx = row * HC_MULT + col
            m = tl.load(mixes_ptr + base + idx)
            b = tl.load(hc_base_ptr + idx)
            tl.store(comb_ptr + comb_base + out_idx, m * scale_2 + b)

    for row in tl.static_range(HC_MULT):
        row_ptr0 = comb_ptr + comb_base + row * HC_MULT
        row_max = tl.load(row_ptr0)
        for col in tl.static_range(HC_MULT):
            row_ptr = comb_ptr + comb_base + row * HC_MULT + col
            row_max = tl.maximum(row_max, tl.load(row_ptr))
        row_sum = 0.0
        for col in tl.static_range(HC_MULT):
            row_ptr = comb_ptr + comb_base + row * HC_MULT + col
            v = tl.exp(tl.load(row_ptr) - row_max)
            row_sum += v
            tl.store(row_ptr, v)
        inv_row_sum = 1.0 / row_sum
        for col in tl.static_range(HC_MULT):
            row_ptr = comb_ptr + comb_base + row * HC_MULT + col
            v = tl.load(row_ptr) * inv_row_sum + 1e-6
            tl.store(row_ptr, v)

    for col in tl.static_range(HC_MULT):
        col_sum = 0.0
        for row in tl.static_range(HC_MULT):
            ptr = comb_ptr + comb_base + row * HC_MULT + col
            col_sum += tl.load(ptr)
        inv_col_sum = 1.0 / (col_sum + 1e-6)
        for row in tl.static_range(HC_MULT):
            ptr = comb_ptr + comb_base + row * HC_MULT + col
            tl.store(ptr, tl.load(ptr) * inv_col_sum)

    for _ in range(SINKHORN_ITERS - 1):
        for row in tl.static_range(HC_MULT):
            row_sum = 0.0
            for col in tl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                row_sum += tl.load(ptr)
            inv_row_sum = 1.0 / (row_sum + 1e-6)
            for col in tl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                tl.store(ptr, tl.load(ptr) * inv_row_sum)

        for col in tl.static_range(HC_MULT):
            col_sum = 0.0
            for row in tl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                col_sum += tl.load(ptr)
            inv_col_sum = 1.0 / (col_sum + 1e-6)
            for row in tl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                tl.store(ptr, tl.load(ptr) * inv_col_sum)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_hc = (2 + hc_mult) * hc_mult
    assert mixes.shape[-1] == mix_hc
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (mix_hc,)

    if mixes.device.type == "cuda" and eps == 1e-6 and hc_mult >= 1:
        outer_shape = mixes.shape[:-1]
        mixes_flat = mixes.reshape(-1, mix_hc).contiguous()
        num_tokens = mixes_flat.shape[0]

        pre = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=mixes.device)
        post = torch.empty(
            num_tokens, hc_mult, dtype=torch.float32, device=mixes.device
        )
        comb = torch.empty(
            num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=mixes.device
        )

        if num_tokens <= 256:
            block_n = 16
            num_warps = 1
        elif num_tokens <= 2048:
            block_n = 32
            num_warps = 1
        elif num_tokens <= 16384:
            block_n = 128
            num_warps = 4
        else:
            block_n = 256
            num_warps = 8
        grid = (num_tokens + block_n - 1) // block_n

        if hc_mult == 4:
            mhc_split_sinkhorn_kernel_hcmult_4[(grid,)](
                mixes_flat,
                hc_scale,
                hc_base,
                pre,
                post,
                comb,
                num_tokens,
                BLOCK_N=block_n,
                SINKHORN_ITERS=sinkhorn_iters,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            if hc_mult <= 4:
                num_warps = 1
            elif hc_mult <= 8:
                num_warps = 2
            else:
                num_warps = 4

            mhc_split_sinkhorn_kernel_generic[(num_tokens,)](
                mixes_flat,
                hc_scale,
                hc_base,
                pre,
                post,
                comb,
                num_tokens,
                SINKHORN_ITERS=sinkhorn_iters,
                HC_MULT=hc_mult,
                MIX_HC=mix_hc,
                num_warps=num_warps,
                num_stages=1,
            )
    else:
        return mhc_split_sinkhorn_torch_ref(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )

    return (
        pre.view(*outer_shape, hc_mult),
        post.view(*outer_shape, hc_mult),
        comb.view(*outer_shape, hc_mult, hc_mult),
    )


def mhc_split_sinkhorn_torch_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outer_shape = mixes.shape[:-1]
    mix_hc = (2 + hc_mult) * hc_mult
    assert mixes.shape[-1] == mix_hc

    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*outer_shape, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = (comb - row_max).exp()
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
