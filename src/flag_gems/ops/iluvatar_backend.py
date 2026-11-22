# Copyright (c) FlagOS
from flag_gems.ops import mm, addmm, bmm, glu

def register_iluvatar_ops():
    """
    Register Iluvatar backend kernels for mm/addmm/bmm/glu
    Removed stale div_mode / div_mode_ references
    """
    import torch
    lib = torch.library.Library("flag_gems", "DEF")
    lib.impl("flag_gems::mm", mm, "Iluvatar")
    lib.impl("flag_gems::addmm", addmm, "Iluvatar")
    lib.impl("flag_gems::bmm", bmm, "Iluvatar")
    lib.impl("flag_gems::glu", glu, "Iluvatar")
