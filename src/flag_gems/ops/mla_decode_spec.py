import torch
import torch.library

# 注册算子定义
lib = torch.library.Library("flag_gems", "DEF")
lib.define(
    "flash_mla_paged_decode_bs32_h16_page32_bf16("
    "Tensor q, Tensor k_cache, Tensor v_cache, Tensor page_table, Tensor kv_seq_len, float sm_scale"
    ") -> Tensor"
)


# CUDA impl stub（占位，先不调用底层cuda kernel）
@torch.library.impl("flag_gems::flash_mla_paged_decode_bs32_h16_page32_bf16", "cuda")
def impl_bs32_h16_p32(q, k_cache, v_cache, page_table, kv_seq_len, sm_scale):
    # 静态校验规格，保证只有bs32 h16 page32 bf16进来
    assert q.shape[0] == 32, "only bs=32"
    assert q.shape[2] == 16, "only h=16"
    assert q.dtype == torch.bfloat16
    out = torch.empty_like(q)
    # TODO: 后续CUDA环境就绪后替换成torch.ops.flag_gems._flash_mla_paged_decode_raw
    return out


# abstract impl，给torch.compile/dynamo做meta追踪
@torch.library.impl_abstract("flag_gems::flash_mla_paged_decode_bs32_h16_page32_bf16")
def impl_abstract_bs32_h16_p32(q, k_cache, v_cache, page_table, kv_seq_len, sm_scale):
    return torch.empty_like(q)
