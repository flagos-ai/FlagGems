# Iluvatar PrivateUse1 aten dispatch override
aten_overrides_privateuse1.update({
    torch.ops.aten.mm.default: torch.ops.flag_gems.iluvatar_mm,
    torch.ops.aten.addmm.default: torch.ops.flag_gems.iluvatar_addmm,
    torch.ops.aten.bmm.default: torch.ops.flag_gems.iluvatar_bmm,
})
