import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


def _single(value):
    return [value] if isinstance(value, int) else value


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE1D")
    output_dtype = input.dtype
    needs_upcast = output_dtype in (torch.float16, torch.bfloat16)
    if needs_upcast:
        input = input.float()
        weight = weight.float()
        bias = None if bias is None else bias.float()

    output = torch.ops.aten.conv_transpose1d.default.redispatch(
        _FALLBACK_KEYSET,
        input,
        weight,
        bias,
        _single(stride),
        _single(padding),
        _single(output_padding),
        groups,
        _single(dilation),
    )
    return output.to(output_dtype) if needs_upcast else output
