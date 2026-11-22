import torch

from flag_gems.spec import SpecBase


class FlashAttnVarlenSpec(SpecBase):
    @staticmethod
    def forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int):
        from flag_gems.ops.flash_attn_varlen import flash_attn_varlen

        return flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)

    @staticmethod
    def gen_inputs():
        # CI回归用的测试样例，按需修改shape、dtype
        dev = torch.device("privateuseone:0")
        q = torch.randn((128, 128), dtype=torch.float16, device=dev)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu_seqlens_q = torch.tensor([0, 128], dtype=torch.int32, device=dev)
        cu_seqlens_k = torch.tensor([0, 128], dtype=torch.int32, device=dev)
        return (q, k, v, cu_seqlens_q, cu_seqlens_k, 128, 128)
