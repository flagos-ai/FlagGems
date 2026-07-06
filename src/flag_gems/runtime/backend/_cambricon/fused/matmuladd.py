from flag_gems.runtime.backend._cambricon.ops.addmm import addmm


def matmuladd(input, other, bias):
    return addmm(bias, input, other)
