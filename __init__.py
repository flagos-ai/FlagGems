import torch
from .matmul import matmul as mm
from .bmm import bmm
from .softmax import softmax
from .layernorm import layer_norm
from .selu import selu
from .gelu import gelu
from .dropout import dropout
from .addmm import addmm

aten_lib = torch.library.Library("aten", "IMPL")

aten_lib.impl('mm', mm, 'CUDA')
aten_lib.impl('bmm', bmm, 'CUDA')
aten_lib.impl('bmm.out', bmm, 'CUDA')
aten_lib.impl('silu', selu, 'CUDA')
aten_lib.impl('silu.out', selu, 'CUDA')
aten_lib.impl('dropout', dropout, 'CUDA')
aten_lib.impl('native_layer_norm', layer_norm, 'CUDA')
# aten_lib.impl('softmax.int', softmax, 'CompositeImplicitAutograd')
aten_lib.impl('addmm', addmm, 'CUDA')

## TODO
# advanced softmax
# sort

# test case: copy the code below and run outside the FlagGems folder

# test_x = torch.randn((16, 16), device='cuda', dtype=torch.float16)
# test_y = torch.randn((16, 16), device='cuda', dtype=torch.float16)
# test_w = torch.randn((16, 16), device='cuda', dtype=torch.float16)
# test_b = torch.randn((16, 16), device='cuda', dtype=torch.float16)

# SiLU = torch.nn.SiLU()
# SiLU(test_x)
# GELU = torch.nn.GELU()
# GELU(test_x)
# DROPOUT = torch.nn.Dropout()
# DROPOUT(test_x, p=0.5)
# torch.mm(test_x, test_y)
# torch.softmax(test_x)
# torch.layer_norm(test_x, test_x.shape, test_w, test_b, 1e-5)


# test_x = torch.randn((16, 16, 16), device='cuda', dtype=torch.float16)
# test_y = torch.randn((16, 16, 16), device='cuda', dtype=torch.float16)
# test_z = torch.empty_like(test_x)
# torch.bmm(test_x, test_y)
# torch.bmm(test_x, test_y, out=test_z)
