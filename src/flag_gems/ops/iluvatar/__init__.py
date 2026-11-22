import torch
from .iluvatar_mm import iluvatar_mm
from .iluvatar_addmm import iluvatar_addmm
from .iluvatar_bmm import iluvatar_bmm
from .iluvatar_mm_conv_glu import iluvatar_mm_conv_glu

__all__ = [
    "iluvatar_mm",
    "iluvatar_addmm",
    "iluvatar_bmm",
    "iluvatar_mm_conv_glu",
    "register_iluvatar_ops",
]

_LIB_DEF = torch.library.Library("flag_gems", "DEF")
_LIB_DEF.define("iluvatar_mm(Tensor a, Tensor b) -> Tensor")
# 对齐原生 torch.addmm(bias, mat1, mat2, beta, alpha)
_LIB_DEF.define("iluvatar_addmm(Tensor bias, Tensor mat1, Tensor mat2, float beta=1.0, float alpha=1.0) -> Tensor")
_LIB_DEF.define("iluvatar_bmm(Tensor a, Tensor b) -> Tensor")
# mm-conv-glu: x @ w + b，w输出通道=2*C_out，内核融合GLU o = a * sigmoid(b_gate)
_LIB_DEF.define("iluvatar_mm_conv_glu(Tensor x, Tensor w, Tensor b) -> Tensor")

_LIB_IMPL = torch.library.Library("flag_gems", "IMPL")

# ---------------- Fake Meta ----------------
@torch.library.register_fake("flag_gems::iluvatar_mm")
def meta_iluvatar_mm(a, b):
    M, _ = a.shape
    _, N = b.shape
    return a.new_empty((M, N), dtype=a.dtype, device=a.device)

@torch.library.register_fake("flag_gems::iluvatar_addmm")
def meta_iluvatar_addmm(bias, mat1, mat2, beta=1.0, alpha=1.0):
    M, _ = mat1.shape
    _, N = mat2.shape
    return mat1.new_empty((M, N), dtype=mat1.dtype, device=mat1.device)

@torch.library.register_fake("flag_gems::iluvatar_bmm")
def meta_iluvatar_bmm(a, b):
    B, M, _ = a.shape
    _, _, N = b.shape
    return a.new_empty((B, M, N), dtype=a.dtype, device=a.device)

@torch.library.register_fake("flag_gems::iluvatar_mm_conv_glu")
def meta_iluvatar_mm_conv_glu(x, w, b):
    M, _ = x.shape
    Cout2 = w.shape[-1]
    C_out = Cout2 // 2
    return x.new_empty((M, C_out), dtype=x.dtype, device=x.device)

def register_iluvatar_ops():
    _LIB_IMPL.impl("iluvatar_mm", iluvatar_mm, "PrivateUse1")
    _LIB_IMPL.impl("iluvatar_addmm", iluvatar_addmm, "PrivateUse1")
    _LIB_IMPL.impl("iluvatar_bmm", iluvatar_bmm, "PrivateUse1")
    _LIB_IMPL.impl("iluvatar_mm_conv_glu", iluvatar_mm_conv_glu, "PrivateUse1")
