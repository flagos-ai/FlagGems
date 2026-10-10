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

logger = logging.getLogger(__name__)


def _validate_conv_transpose1d_args(*args, **kwargs):
    """Passthrough to the generic 1D validation when present (keeps the
    original error semantics for invalid arguments)."""
    from flag_gems.ops import conv_transpose1d as _generic

    for name in ("_validate_conv_transpose1d_args", "_validate_cvt1d_args"):
        fn = getattr(_generic, name, None)
        if fn is not None:
            return fn(*args, **kwargs)
    return True


def _unsupported_conv_transpose1d(*args, **kwargs):
    from flag_gems.ops import conv_transpose1d as _generic
    from flag_gems.ops.conv_transpose1d import (  # noqa: F401
        conv_transpose1d_output_size,
    )

    for name in ("_unsupported_conv_transpose1d", "_unsupported_cvt1d"):
        fn = getattr(_generic, name, None)
        if fn is not None:
            return fn(*args, **kwargs)
    raise NotImplementedError(
        "flag_gems.conv_transpose1d does not support the given input"
    )


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
    from flag_gems.runtime.backend._kunlunxin.ops.conv_transpose2d import (
        conv_transpose2d as _klx_conv_transpose2d,
    )

    def _one(v):
        if isinstance(v, (list, tuple)):
            return int(v[0])
        return v

    stride = _one(stride)
    padding = _one(padding)
    output_padding = _one(output_padding)
    dilation = _one(dilation)

    _validate_conv_transpose1d_args(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    if groups > 1 and stride > 1 and padding == 0:
        return _klx_conv_transpose2d(
            input.unsqueeze(-1),
            weight.unsqueeze(-1),
            bias,
            (stride, 1),
            (padding, 0),
            (output_padding, 0),
            groups,
            (dilation, 1),
        ).squeeze(-1)

    return _klx_conv_transpose2d(
        input.unsqueeze(-2),
        weight.unsqueeze(-2),
        bias,
        (1, stride),
        (0, padding),
        (0, output_padding),
        groups,
        (1, dilation),
    ).squeeze(-2)
