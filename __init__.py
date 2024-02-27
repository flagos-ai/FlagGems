import torch
from .matmul import matmul as mm
from .bmm import bmm
from .layernorm import layer_norm

aten_lib = torch.library.Library("aten", "IMPL")

aten_lib.impl('mm', mm, 'CUDA')
aten_lib.impl('bmm.out', bmm, 'CUDA')
aten_lib.impl('native_layer_norm', layer_norm, 'CUDA')

# test case

test_x = torch.randn((16, 16), device='cuda', dtype=torch.float16)
test_y = torch.randn((16, 16), device='cuda', dtype=torch.float16)
test_w = torch.randn((16, 16), device='cuda', dtype=torch.float16)
test_b = torch.randn((16, 16), device='cuda', dtype=torch.float16)

torch.mm(test_x, test_y)
torch.layer_norm(test_x, test_x.shape, test_w, test_b, 1e-5)

test_x = torch.randn((16, 16, 16), device='cuda', dtype=torch.float16)
test_y = torch.randn((16, 16, 16), device='cuda', dtype=torch.float16)
test_z = torch.empty_like(test_x)
torch.bmm(test_x, test_y, out=test_z)
