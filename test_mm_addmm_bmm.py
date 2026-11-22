import torch
from flag_gems.ops.triton_src.addmm import addmm
from flag_gems.ops.triton_src.bmm import bmm
from flag_gems.ops.triton_src.mm import mm

torch.manual_seed(0)
device = "cuda"
dtype = torch.float32

print("==== Test mm ====")
a = torch.randn(512, 256, device=device, dtype=dtype)
b = torch.randn(256, 1024, device=device, dtype=dtype)
ref = a @ b
tri = mm(a, b)
print("mm max diff:", (ref - tri).abs().max().item())

print("==== Test bmm ====")
a = torch.randn(4, 128, 64, device=device, dtype=dtype)
b = torch.randn(4, 64, 256, device=device, dtype=dtype)
ref = torch.bmm(a, b)
tri = bmm(a, b)
print("bmm max diff:", (ref - tri).abs().max().item())

print("==== Test addmm ====")
a = torch.randn(256, 128, device=device, dtype=dtype)
b = torch.randn(128, 512, device=device, dtype=dtype)
c = torch.randn(256, 512, device=device, dtype=dtype)
alpha = 0.5
beta = 2.0
ref = torch.addmm(c, a, b, alpha=alpha, beta=beta)
tri = addmm(a, b, c, alpha=alpha, beta=beta)
print("addmm max diff:", (ref - tri).abs().max().item())
