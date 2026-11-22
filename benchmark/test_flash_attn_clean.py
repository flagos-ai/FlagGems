import sys

import torch

# 把src加入搜索路径
sys.path.insert(0, "/root/FlagOS/FlagGems/src")

# 只导入我们自己的算子注册文件，不加载flag_gems大包
from flag_gems.ops.flash_attn_varlen_spec import *


def main():
    nh = 16
    hd = 128
    total_q = 512
    total_k = 512
    device = torch.device("ix:0")

    q = torch.randn([total_q, nh, hd], dtype=torch.bfloat16, device=device)
    k = torch.randn([total_k, nh, hd], dtype=torch.bfloat16, device=device)
    v = torch.randn([total_k, nh, hd], dtype=torch.bfloat16, device=device)
    cu_q = torch.tensor([0, 128, 512], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 128, 512], dtype=torch.int32, device=device)

    softmax_scale = 1.0 / (hd**0.5)

    out = torch.ops.flag_gems.flash_attn_varlen_fun(
        q,
        k,
        v,
        cu_q,
        cu_k,
        softmax_scale,
        nh,
        max_seqlen_q=256,
        max_seqlen_k=256,
        causal=True,
    )
    print("out shape:", out.shape)
    print("✅ 算子注册链路跑通")


if __name__ == "__main__":
    main()
