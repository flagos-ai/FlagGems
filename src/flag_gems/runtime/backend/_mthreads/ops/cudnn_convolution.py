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

"""MTHREADS override of :func:`flag_gems.ops.cudnn_convolution`.

The generic operator already dispatches each rank to the shared Triton
``conv1d``/``conv2d``/``conv3d`` kernels. On MTHREADS the 2D kernel is the one
with a hand-tuned, backend-specific implementation (channels-last staging,
packed weights and its own autotune set), so this override routes *every* 2D
convolution through it:

* 2D goes straight to the MTHREADS ``conv2d`` kernel.
* 1D is lifted to 2D with a unit width, exactly as the shared ``conv1d`` does,
  so it runs on the MTHREADS 2D kernel instead of the generic one it would
  otherwise reach through ``flag_gems.ops.conv1d``.
* 3D has no MTHREADS kernel of its own and keeps the generic dispatch
  (shared ``conv3d`` plus its dedicated direct/depthwise/pointwise kernels).

``benchmark``, ``deterministic`` and ``allow_tf32`` are accepted for interface
compatibility with ``aten::cudnn_convolution``; like the generic operator they
do not select an algorithm.
"""

import logging

from flag_gems.ops.conv1d import conv1d as shared_conv1d
from flag_gems.ops.cudnn_convolution import _AABS_PRESENT, _aabs_disabled, _conv_forward

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def _to_list(param, ndim):
    """Broadcast a scalar spatial parameter to one entry per spatial dimension."""
    if isinstance(param, str):
        # "same"/"valid" are passed through to the kernels that understand them.
        return param
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """MTHREADS implementation of the bias-free cuDNN convolution entry point."""
    logger.debug("GEMS_MTHREADS CUDNN_CONVOLUTION")

    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )

    padding = _to_list(padding, ndim)
    stride = _to_list(stride, ndim)
    dilation = _to_list(dilation, ndim)

    if ndim == 2:
        # The MTHREADS kernel may return a channels-last tensor; the aten
        # operator returns a dense NCHW result, so normalize the layout.
        output = conv2d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        return output if output.is_contiguous() else output.contiguous()

    if ndim == 1:
        if isinstance(padding, str):
            # "same"/"valid" need conv1d's output-size trimming; the 2D lift
            # below only models numeric padding.
            return shared_conv1d(
                input,
                weight,
                bias=None,
                stride=stride[0],
                padding=padding,
                dilation=dilation[0],
                groups=groups,
            )
        # Same lift as flag_gems.ops.conv1d, but through the MTHREADS conv2d.
        output = conv2d(
            input.unsqueeze(-1),
            weight.unsqueeze(-1),
            bias=None,
            stride=(stride[0], 1),
            padding=(padding[0], 0),
            dilation=(dilation[0], 1),
            groups=groups,
        ).squeeze(-1)
        return output if output.is_contiguous() else output.contiguous()

    # 3D has no MTHREADS kernel, so fall through to the generic dispatch,
    # which selects the shared conv3d or one of its dedicated kernels
    # (depthwise / pointwise / small-C_in direct). The shared conv3d kernel
    # itself faults with a misaligned address on MTHREADS for shapes such as
    # (2, 4, 16, 16, 16), which the small-C_in direct kernel handles.
    if _AABS_PRESENT:
        with _aabs_disabled():
            return _conv_forward(input, weight, padding, stride, dilation, groups)
    return _conv_forward(input, weight, padding, stride, dilation, groups)
