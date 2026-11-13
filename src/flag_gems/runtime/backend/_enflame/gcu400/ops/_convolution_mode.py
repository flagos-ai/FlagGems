# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from .constant_pad_nd import constant_pad_nd
from .conv1d__opt import conv1d
from .conv2d__opt import conv2d
from .conv3d import conv3d

logger = logging.getLogger(__name__)


def _same_padding(input, weight, dilation):
    """Compute the asymmetric 'same' padding (PyTorch semantics) and apply it.

    For each spatial axis: total = dilation * (kernel - 1); left = total // 2;
    right = total - left. The FlagGems :func:`constant_pad_nd` op takes the pad
    list from the last dimension backward, as ``(left, right)`` pairs, so the
    leading side gets ``left`` and the trailing side gets ``right``, exactly
    matching ``torch._convolution_mode`` with ``padding='same'`` (which
    requires unit strides).
    """
    kernel_size = weight.shape[2:]
    # constant_pad_nd shares aten's pad order: last dimension backward, as
    # (left, right) pairs.
    pad = []
    for k, d in zip(reversed(kernel_size), reversed(dilation)):
        total = d * (k - 1)
        pad.append(total // 2)  # left
        pad.append(total - total // 2)  # right
    return constant_pad_nd(input, pad)


def _convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    """Computes a convolution using a string padding mode.

    This is the low-level dispatcher backing ``torch._convolution_mode`` (the
    function invoked by ``torch.nn.functional.conv1d/conv2d/conv3d`` when the
    ``padding`` argument is a string). ``padding`` is ``"valid"`` (no padding)
    or ``"same"`` (pad so the output spatial size matches the input; requires
    unit strides). The actual convolution is delegated to the gcu400-adapted
    FlagGems Triton convolution kernels (:func:`conv1d`, :func:`conv2d`,
    :func:`conv3d`) -- the same kernels registered for ``aten::conv1d/2d/3d` --
    selected by the spatial dimensionality inferred from the weight tensor.
    The generic (upstream) conv kernels miscompile masked fp32 tiles on this
    backend (``tl.dot`` K-tail lane leak), so the delegation must stay on the
    local copies. For ``"same"`` the asymmetric padding is applied with the FlagGems
    :func:`constant_pad_nd` op to match PyTorch's reference exactly, then the
    kernel runs with zero padding.
    """
    logger.debug("GEMS _CONVOLUTION_MODE")
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
    elif padding == "valid":
        pass
    else:
        raise ValueError(
            f"Invalid padding string: {padding!r}, should be one of {{'same', 'valid'}}"
        )

    # The spatial dimensionality is determined by the weight tensor:
    # weight shape is [out_channels, in_channels // groups, *kernel_size].
    spatial_ndim = weight.ndim - 2

    if padding == "same":
        input = _same_padding(input, weight, dilation)
        # After manual padding, run the kernel with zero padding and unit stride.
        if spatial_ndim == 1:
            return conv1d(input, weight, bias, stride, 0, dilation, groups)
        elif spatial_ndim == 2:
            return conv2d(input, weight, bias, stride, 0, dilation, groups)
        elif spatial_ndim == 3:
            return conv3d(input, weight, bias, stride, 0, dilation, groups)
    else:  # valid
        if spatial_ndim == 1:
            return conv1d(input, weight, bias, stride, "valid", dilation, groups)
        elif spatial_ndim == 2:
            return conv2d(input, weight, bias, stride, "valid", dilation, groups)
        elif spatial_ndim == 3:
            return conv3d(input, weight, bias, stride, "valid", dilation, groups)

    raise ValueError(f"Unsupported convolution with {spatial_ndim} spatial dimensions")
