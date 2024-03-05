from .addmm import addmm
from .bmm import bmm
from .cumsum import cumsum
from .dropout import dropout
from .gelu import gelu
from .layernorm import layer_norm
from .mm import mm
from .relu import relu
from .silu import silu
from .softmax import softmax

__all__ = [
    "addmm",
    "bmm",
    "cumsum",
    "dropout",
    "gelu",
    "layer_norm",
    "mm",
    "relu",
    "silu",
    "softmax",
]

import torch

aten_lib = torch.library.Library("aten", "IMPL")

aten_lib.impl("addmm", addmm, "CUDA")
aten_lib.impl("bmm", bmm, "CUDA")
aten_lib.impl("bmm.out", bmm, "CUDA")
aten_lib.impl("cumsum", cumsum, "CUDA")
aten_lib.impl("cumsum.out", cumsum, "CUDA")
aten_lib.impl("dropout", dropout, "CUDA")
aten_lib.impl("gelu", gelu, "CUDA")
aten_lib.impl("layer_norm", layer_norm, "CompositeImplicitAutograd")
aten_lib.impl("mm", mm, "CUDA")
aten_lib.impl("relu", relu, "CUDA")
aten_lib.impl("silu", silu, "CUDA")
aten_lib.impl("silu.out", silu, "CUDA")
aten_lib.impl("softmax.int", softmax, "CompositeImplicitAutograd")
