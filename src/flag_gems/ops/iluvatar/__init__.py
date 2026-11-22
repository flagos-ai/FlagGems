import torch
from .iluvatar_mm import iluvatar_mm
from .iluvatar_addmm import iluvatar_addmm
from .iluvatar_bmm import iluvatar_bmm

__all__ = ["iluvatar_mm", "iluvatar_addmm", "iluvatar_bmm"]

LIB = torch.library.Library("flag_gems", "DEF")
LIB.define("iluvatar_mm(Tensor a, Tensor b) -> Tensor")
LIB.define("iluvatar_addmm(Tensor bias, Tensor mat1, Tensor mat2, float beta=1.0, float alpha=1.0) -> Tensor")
LIB.define("iluvatar_bmm(Tensor a, Tensor b) -> Tensor")

IMPL = torch.library.Library("flag_gems", "IMPL")
IMPL.impl("iluvatar_mm", iluvatar_mm, "PrivateUse1")
IMPL.impl("iluvatar_addmm", iluvatar_addmm, "PrivateUse1")
IMPL.impl("iluvatar_bmm", iluvatar_bmm, "PrivateUse1")

@torch.library.register_fake("flag_gems::iluvatar_mm")
def meta_mm(a, b):
    M, _ = a.shape
    _, N = b.shape
    return a.new_empty((M, N))

@torch.library.register_fake("flag_gems::iluvatar_addmm")
def meta_addmm(bias, mat1, mat2, beta, alpha):
    M, _ = mat1.shape
    _, N = mat2.shape
    return mat1.new_empty((M, N))

@torch.library.register_fake("flag_gems::iluvatar_bmm")
def meta_bmm(a, b):
    B, M, _ = a.shape
    _, _, N = b.shape
    return a.new_empty((B, M, N))
