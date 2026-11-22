import torch

aten_overrides_privateuse1 = {}

def register_privateuse1_overrides():
    """
    延迟注册：必须在 flag_gems.ops 导入完成之后调用
    """
    aten_overrides_privateuse1.clear()
    try:
        _ = torch.ops.flag_gems.iluvatar_mm
    except AttributeError:
        raise RuntimeError("flag_gems ops not registered! import flag_gems.ops first.")
    aten_overrides_privateuse1.update({
        torch.ops.aten.mm.default: torch.ops.flag_gems.iluvatar_mm.default,
        torch.ops.aten.addmm.default: torch.ops.flag_gems.iluvatar_addmm.default,
        torch.ops.aten.bmm.default: torch.ops.flag_gems.iluvatar_bmm.default,
    })
