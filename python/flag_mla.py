import torch
import torch.library

from flag_gems import _C


# ---------------- 基础接口 ----------------
def flash_mla_prefill_alibi_causal(q, latent_k, latent_v, cum_seqlens, softmax_scale=None, alibi_slope=0.0):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out_latent = torch.empty((q.shape[0], q.shape[1], 512), dtype=q.dtype, device=q.device)
    _C.flash_mla_prefill_alibi_causal_bf16(
        q,
        latent_k,
        latent_v,
        cum_seqlens,
        out_latent,
        softmax_scale,
        alibi_slope,
        q.shape[1],
        cum_seqlens[-1],
    )
    return out_latent


def flash_mla_paged_decode(q, latent_k_cache, latent_v_cache, page_table, seq_lens, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out = torch.empty_like(q)
    _C.flash_mla_paged_decode(
        q,
        latent_k_cache,
        latent_v_cache,
        page_table,
        seq_lens,
        out,
        softmax_scale,
        q.shape[0],
        q.shape[1],
        page_table.shape[1],
    )
    return out


def flash_mla_decode_fused_wuv(q, lk_cache, lv_cache, w_uv, page_table, seq_lens, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out = torch.empty_like(q)
    _C.flash_mla_decode_fused_wuv_bf16(
        q,
        lk_cache,
        lv_cache,
        w_uv,
        page_table,
        seq_lens,
        out,
        softmax_scale,
        q.shape[0],
        q.shape[1],
        page_table.shape[1],
    )
    return out


# ---------------- Graph Capture custom op ----------------
torch.library.define(
    "flag_gems::flash_mla_paged_decode_bs32_h16_page64_bf16",
    "(Tensor q, Tensor lk_cache, Tensor lv_cache, Tensor page_table, Tensor seq_lens, float softmax_scale) -> Tensor",
)


@torch.library.impl("flag_gems::flash_mla_paged_decode_bs32_h16_page64_bf16", "cuda")
def impl_bs32_h16_p64(q, lk_cache, lv_cache, page_table, seq_lens, softmax_scale):
    out = torch.empty_like(q)
    _C.flash_mla_decode_bs32_h16_page64_bf16(
        q,
        lk_cache,
        lv_cache,
        page_table,
        seq_lens,
        out,
        softmax_scale,
        32,
        16,
        page_table.shape[1],
    )
    return out


@torch.library.impl_abstract("flag_gems::flash_mla_paged_decode_bs32_h16_page64_bf16")
def abs_bs32_h16_p64(q, lk_cache, lv_cache, page_table, seq_lens, softmax_scale):
    return torch.empty_like(q)


torch.library.define(
    "flag_gems::flash_mla_paged_decode_bs32_h16_page32_bf16",
    "(Tensor q, Tensor lk_cache, Tensor lv_cache, Tensor page_table, Tensor seq_lens, float softmax_scale) -> Tensor",
)


@torch.library.impl("flag_gems::flash_mla_paged_decode_bs32_h16_page32_bf16", "cuda")
def impl_bs32_h16_p32(q, k_cache, v_cache, page_table, kv_seq_len, sm_scale):
    """
    FlagGems 专用特化 MLA 分页解码 kernel
    固定规格: bs=32, num_heads=16, page_size=32, dtype=bfloat16, device=cuda
    Args:
        q: [bs, 1, h, d]  decode query, bf16
        k_cache: 分页 K cache 基存储
        v_cache: 分页 V cache 基存储
        page_table: [bs, max_pages] int64 page index table
        kv_seq_len: [bs] int, 每个序列当前kv有效长度
        sm_scale: float, softmax scale
    Returns:
        out: [bs, 1, h, d] bfloat16 attention output
    """
    # 静态断言，强制校验规格，防止错误 dispatch 进来
    assert q.size(0) == 32, "impl_bs32_h16_p32 only support bs=32"
    assert q.size(2) == 16, "impl_bs32_h16_p32 only support h=16"
    assert q.dtype == torch.bfloat16
    assert q.is_cuda

    out = torch.empty_like(q)

    # --------------------------
    # 调用底层 C++/CUDA raw kernel
    # 这里是 FlagGems 惯例：通过 triton / cuda raw launch 接口，你可替换为 cpp_extension 绑定函数
    # --------------------------
    torch.ops.flag_gems._flash_mla_paged_decode_raw(
        out,
        q,
        k_cache,
        v_cache,
        page_table,
        kv_seq_len,
        sm_scale,
        32,  # fixed bs
        16,  # fixed num_heads
        32,  # fixed page_size
    )
    return out


# 配套 fake impl (meta 实现，用于 torch.compile / dynamo 追踪)
@torch.library.impl_abstract("flag_gems::flash_mla_paged_decode_bs32_h16_page32_bf16")
def impl_abstract_bs32_h16_p32(q, k_cache, v_cache, page_table, kv_seq_len, sm_scale):
    return torch.empty_like(q)
