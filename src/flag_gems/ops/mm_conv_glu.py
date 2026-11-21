import torch

from flag_gems.ops import conv2d, glu, mm


def mm_conv_glu(
    x: torch.Tensor,
    w_mm: torch.Tensor,
    w_conv: torch.Tensor,
    b_conv: torch.Tensor,
    stride,
    padding,
    dilation,
    groups,
) -> torch.Tensor:
    """
    mm -> conv2d -> glu 链式融合算子
    """
    y = mm(x, w_mm)
    y = conv2d(y, w_conv, b_conv, stride, padding, dilation, groups)
    y = glu(y)
    return y
