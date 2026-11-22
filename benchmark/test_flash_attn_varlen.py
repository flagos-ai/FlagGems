import torch

import flag_gems

nh = 16
hd = 128
total_q = 512
total_k = 512

q = torch.randn([total_q, nh, hd], dtype=torch.bfloat16, device="ix")
k = torch.randn([total_k, nh, hd], dtype=torch.bfloat16, device="ix")
v = torch.randn([total_k, nh, hd], dtype=torch.bfloat16, device="ix")
cu_q = torch.tensor([0, 128, 512], dtype=torch.int32, device="ix")
cu_k = torch.tensor([0, 128, 512], dtype=torch.int32, device="ix")

out = flag_gems.flash_attn_varlen_fun(
    q,
    k,
    v,
    cu_q,
    cu_k,
    softmax_scale=1.0 / (hd**0.5),
    num_heads=nh,
    max_seqlen_q=256,
    max_seqlen_k=256,
    causal=True,
)
print("out shape:", out.shape)
print("ok")
