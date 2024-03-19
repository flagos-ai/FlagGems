import torch
from .addmm import addmm
from .bmm import bmm
from .cumsum import cumsum
from .dropout import dropout
from .gelu import gelu
from .layernorm import layer_norm
from .mm import mm
from .relu import relu
from .silu import silu
from .triu import triu
from .softmax import softmax

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("addmm", addmm, "CUDA")
    lib.impl("bmm", bmm, "CUDA")
    lib.impl("bmm.out", bmm, "CUDA")
    lib.impl("cumsum", cumsum, "CUDA")
    lib.impl("cumsum.out", cumsum, "CUDA")
    lib.impl("dropout", dropout, "CUDA")
    lib.impl("gelu", gelu, "CUDA")
    lib.impl("layer_norm", layer_norm, "CompositeImplicitAutograd")
    lib.impl("mm", mm, "CUDA")
    lib.impl("relu", relu, "CUDA")
    lib.impl("silu", silu, "CUDA")
    lib.impl("silu.out", silu, "CUDA")
    lib.impl("triu", triu, "CUDA")
    lib.impl("triu.out", triu, "CUDA")
    lib.impl("softmax.int", softmax, "CompositeImplicitAutograd")


class Context:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib
