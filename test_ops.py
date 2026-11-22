import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from fused.addmm import addmm
from fused.bmm import bmm

# 直接从fused模块导入，不走flag_gems顶层包加载
from fused.mm import mm

torch.manual_seed(0)
M, N, K = 71, 53, 47
a = torch.randn(M, K, device="cuda")
b = torch.randn(K, N, device="cuda")
bias = torch.randn(N, device="cuda")

ref_mm = a @ b
o_mm = mm(a, b)
print("mm max diff:", torch.max(torch.abs(o_mm - ref_mm)))

ref_addmm = torch.addmm(bias, a, b, beta=0.5, alpha=2.0)
o_addmm = addmm(bias, a, b, beta=0.5, alpha=2.0)
print("addmm max diff:", torch.max(torch.abs(o_addmm - ref_addmm)))

B = 3
a3 = torch.randn(B, M, K, device="cuda")
b3 = torch.randn(B, K, N, device="cuda")
ref_bmm = torch.bmm(a3, b3)
o_bmm = bmm(a3, b3)
print("bmm max diff:", torch.max(torch.abs(o_bmm - ref_bmm)))
print("All kernel test finished")
