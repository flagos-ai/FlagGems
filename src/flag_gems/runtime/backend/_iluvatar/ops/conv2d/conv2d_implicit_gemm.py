import torch
import torch.nn.functional as F
import triton


def _ixdnn_supported(kernel, padding, dilation, bias):
    # 这里padding已经是数值tuple，不再是字符串
    if dilation[0] != 1 or dilation[1] != 1:
        return False
    return True


def _implicit_gemm_conv3x3_kernel(x, weight, bias, stride, padding, dilation, groups):
    # 占位实现，后续替换成真实kernel
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


def conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
):
    """
    Conv2d wrapper：优先3x3 stride1 implicit gemm；其余case fallback aten
    x: [N,Cin,H,W], weight: [Cout,Cin,KH,KW]
    Support "same"/"valid" string padding, convert to numeric padding first
    """
    kh, kw = weight.shape[-2], weight.shape[-1]

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # ========== 字符串padding转数值padding ==========
    if isinstance(padding, str):
        if padding == "valid":
            pad_h, pad_w = 0, 0
        elif padding == "same":
            _, _, h, w = x.shape
            sh, sw = stride
            dh, dw = dilation
            pad_h_total = (h - 1) * sh + (kh - 1) * dh - h + 1
            pad_w_total = (w - 1) * sw + (kw - 1) * dw - w + 1
            pad_h = pad_h_total // 2
            pad_w = pad_w_total // 2
        else:
            raise ValueError(f"Unsupported padding string: {padding}")
    elif isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding

    pad_tuple = (pad_h, pad_w)

    # 传入数值padding给supported判断
    if _ixdnn_supported(weight.shape, pad_tuple, dilation, bias):
        if (
            kh == 3
            and kw == 3
            and stride == (1, 1)
            and dilation == (1, 1)
            and groups == 1
        ):
            out = _implicit_gemm_conv3x3_kernel(
                x, weight, bias, stride, pad_tuple, dilation, groups
            )
        else:
            out = F.conv2d(x, weight, bias, stride, pad_tuple, dilation, groups)
    else:
        out = F.conv2d(x, weight, bias, stride, pad_tuple, dilation, groups)
    return out
