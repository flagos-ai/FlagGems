import torch
import triton
import triton.language as tl


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 128, "NUM_D_TILES": 1, "BLOCK_H": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 128, "NUM_D_TILES": 1, "BLOCK_H": 16},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 128, "NUM_D_TILES": 1, "BLOCK_H": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 128, "BLOCK_D": 128, "NUM_D_TILES": 1, "BLOCK_H": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 128, "BLOCK_D": 128, "NUM_D_TILES": 1, "BLOCK_H": 16},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 32, "BLOCK_D": 128, "NUM_D_TILES": 1, "BLOCK_H": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 64, "NUM_D_TILES": 2, "BLOCK_H": 16},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 64, "NUM_D_TILES": 2, "BLOCK_H": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 128, "BLOCK_D": 64, "NUM_D_TILES": 2, "BLOCK_H": 16},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 64, "NUM_D_TILES": 1, "BLOCK_H": 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 64, "NUM_D_TILES": 1, "BLOCK_H": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 128, "BLOCK_D": 64, "NUM_D_TILES": 1, "BLOCK_H": 16},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["heads", "dim", "block_size"],
)
@triton.jit
def fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    logits_ptr,
    block_tables_ptr,
    context_lens_ptr,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kvblk,
    stride_kvpos,
    stride_kvone,
    stride_kvbyte,
    stride_wrow,
    stride_wh,
    stride_lrow,
    stride_lcol,
    stride_btb,
    stride_bts,
    next_n: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    max_model_len,
    dim_plus_4: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_kv_tile = tl.program_id(1)

    batch_idx = pid_row // next_n
    next_n_idx = pid_row % next_n

    context_len = tl.load(context_lens_ptr + batch_idx)
    query_seq_pos = context_len - next_n + next_n_idx

    kv_start = pid_kv_tile * BLOCK_KV
    if kv_start >= context_len:
        offs_kv = tl.arange(0, BLOCK_KV)
        kv_pos = kv_start + offs_kv
        out_mask = kv_pos < max_model_len
        out_ptrs = logits_ptr + pid_row * stride_lrow + kv_pos * stride_lcol
        tl.store(out_ptrs, float("-inf"), mask=out_mask)
        return

    offs_kv = tl.arange(0, BLOCK_KV)
    kv_global_pos = kv_start + offs_kv

    context_mask = kv_global_pos < context_len
    causal_mask = kv_global_pos <= query_seq_pos
    valid_mask = context_mask & causal_mask

    phys_block_idx = kv_global_pos // block_size
    intra_block_pos = kv_global_pos % block_size

    phys_block_ids = tl.load(
        block_tables_ptr + batch_idx * stride_btb + phys_block_idx * stride_bts,
        mask=valid_mask,
        other=0,
    )

    kv_base = phys_block_ids * stride_kvblk + intra_block_pos * stride_kvpos

    scale_addr = kv_base + dim * stride_kvbyte
    scale_ptr = (kv_ptr + scale_addr).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    scale_u32 = tl.load(scale_ptr, mask=valid_mask, other=0)
    scale_f32 = scale_u32.to(tl.float32, bitcast=True)

    logit_accum = tl.zeros([BLOCK_KV], dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = q_ptr + batch_idx * stride_qb + next_n_idx * stride_qn

    if NUM_D_TILES == 1:
        d_mask = offs_d < dim

        kv_byte_ptrs = kv_ptr + kv_base[:, None] + offs_d[None, :] * stride_kvbyte
        load_mask = valid_mask[:, None] & d_mask[None, :]
        kv_u8 = tl.load(kv_byte_ptrs, mask=load_mask, other=0)
        kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)
        kv_f32 = kv_fp8.to(tl.float32)

        for h_tile in tl.static_range(0, heads, BLOCK_H):
            offs_h = h_tile + tl.arange(0, BLOCK_H)
            h_mask = offs_h < heads

            q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
            q_vals = tl.load(
                q_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0
            ).to(tl.float32)
            weights = tl.load(
                weights_ptr + pid_row * stride_wrow + offs_h * stride_wh,
                mask=h_mask,
                other=0.0,
            )

            q_tile = tl.trans(q_vals)
            partial_dot = tl.dot(kv_f32, q_tile, out_dtype=tl.float32)
            partial_dot = partial_dot * scale_f32[:, None]
            partial_dot = tl.maximum(partial_dot, 0.0)
            logit_accum += tl.sum(partial_dot * weights[None, :], axis=1)

    else:
        d_offs0 = offs_d
        d_mask0 = d_offs0 < dim
        d_offs1 = BLOCK_D + offs_d
        d_mask1 = d_offs1 < dim

        kv_byte_ptrs0 = kv_ptr + kv_base[:, None] + d_offs0[None, :] * stride_kvbyte
        load_mask0 = valid_mask[:, None] & d_mask0[None, :]
        kv_u80 = tl.load(kv_byte_ptrs0, mask=load_mask0, other=0)
        kv_fp80 = kv_u80.to(tl.float8e4nv, bitcast=True)
        kv_f320 = kv_fp80.to(tl.float32)

        kv_byte_ptrs1 = kv_ptr + kv_base[:, None] + d_offs1[None, :] * stride_kvbyte
        load_mask1 = valid_mask[:, None] & d_mask1[None, :]
        kv_u81 = tl.load(kv_byte_ptrs1, mask=load_mask1, other=0)
        kv_fp81 = kv_u81.to(tl.float8e4nv, bitcast=True)
        kv_f321 = kv_fp81.to(tl.float32)

        for h_tile in tl.static_range(0, heads, BLOCK_H):
            offs_h = h_tile + tl.arange(0, BLOCK_H)
            h_mask = offs_h < heads

            q_ptrs0 = (
                q_base + offs_h[:, None] * stride_qh + d_offs0[None, :] * stride_qd
            )
            q_vals0 = tl.load(
                q_ptrs0, mask=h_mask[:, None] & d_mask0[None, :], other=0.0
            ).to(tl.float32)

            q_ptrs1 = (
                q_base + offs_h[:, None] * stride_qh + d_offs1[None, :] * stride_qd
            )
            q_vals1 = tl.load(
                q_ptrs1, mask=h_mask[:, None] & d_mask1[None, :], other=0.0
            ).to(tl.float32)

            weights = tl.load(
                weights_ptr + pid_row * stride_wrow + offs_h * stride_wh,
                mask=h_mask,
                other=0.0,
            )

            q_T0 = tl.trans(q_vals0)
            q_T1 = tl.trans(q_vals1)

            partial_dot = tl.dot(kv_f320, q_T0, out_dtype=tl.float32)
            partial_dot = tl.dot(kv_f321, q_T1, acc=partial_dot, out_dtype=tl.float32)

            partial_dot = partial_dot * scale_f32[:, None]
            partial_dot = tl.maximum(partial_dot, 0.0)
            logit_accum += tl.sum(partial_dot * weights[None, :], axis=1)

    out_vals = tl.where(valid_mask, logit_accum, float("-inf"))
    out_ptrs = logits_ptr + pid_row * stride_lrow + kv_global_pos * stride_lcol
    out_mask = valid_mask & (kv_global_pos < max_model_len)
    tl.store(out_ptrs, out_vals, mask=out_mask)


@triton.jit
def fill_neg_inf_kernel(
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(out_ptr + offs, float("-inf"), mask=mask)


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    assert q.is_cuda and kv_cache.is_cuda and weights.is_cuda
    assert context_lens.is_cuda and block_tables.is_cuda

    batch_size, next_n, heads, dim = q.size()
    num_blocks, block_size, one, dim_plus_4 = kv_cache.size()

    assert one == 1
    assert dim_plus_4 == dim + 4
    assert weights.shape == (batch_size * next_n, heads)
    assert kv_cache.dtype == torch.uint8
    assert context_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32

    q_contig = q.contiguous()
    kv_contig = kv_cache.contiguous()
    weights_contig = weights.contiguous()
    context_lens_contig = context_lens.contiguous()
    block_tables_contig = block_tables.contiguous()

    total_rows = batch_size * next_n

    logits = torch.empty(
        (total_rows, max_model_len),
        device=q.device,
        dtype=torch.float32,
    )
    n_elements = total_rows * max_model_len
    FILL_BLOCK = 1024
    fill_grid = (cdiv(n_elements, FILL_BLOCK),)
    fill_neg_inf_kernel[fill_grid](logits, n_elements, BLOCK=FILL_BLOCK)

    max_context = block_tables_contig.shape[1] * block_size

    def grid(meta):
        BLOCK_KV = meta["BLOCK_KV"]
        num_kv_tiles = cdiv(max_context, BLOCK_KV)
        return (total_rows, num_kv_tiles)

    fp8_paged_mqa_logits_kernel[grid](
        q_contig,
        kv_contig,
        weights_contig,
        logits,
        block_tables_contig,
        context_lens_contig,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        q_contig.stride(3),
        kv_contig.stride(0),
        kv_contig.stride(1),
        kv_contig.stride(2),
        kv_contig.stride(3),
        weights_contig.stride(0),
        weights_contig.stride(1),
        logits.stride(0),
        logits.stride(1),
        block_tables_contig.stride(0),
        block_tables_contig.stride(1),
        next_n,
        heads,
        dim,
        block_size,
        max_model_len,
        dim_plus_4,
    )

    return logits
