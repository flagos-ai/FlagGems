import logging
import math

from flag_gems.ops.conv2d import conv2d

logger = logging.getLogger(__name__)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS CONV1D")
    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(padding, str):
        if padding == "same":
            assert (
                stride == 1
            ), "Doesn't support any stride values other than 1 \
                in padding = 'same' mode, received stride value {stride}"
            il = input.shape[-1]
            kernel_size = weight.shape[-1]
            padding_width = math.ceil(
                (stride * (il - 1) + 1 + dilation * (kernel_size - 1) - il) / 2
            )
            ol = int(
                (il + 2 * padding_width - dilation * (kernel_size - 1) - 1) / stride + 1
            )
            return conv2d(
                input.unsqueeze(-1),
                weight.unsqueeze(-1),
                bias,
                (stride_width, 1),
                (padding_width, 0),
                dilation,
                groups,
            ).squeeze(-1)[..., (ol - il) :]
        elif padding == "valid":
            padding_width = 0
        else:
            raise ValueError(
                f"Unsupported padding mode: {padding}, only 'valid' or 'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding
    return conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        (stride_width, 1),
        (padding_width, 0),
        dilation,
        groups,
    ).squeeze(-1)
