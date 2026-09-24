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
#
# kunlunxin (XPU) re-implementation of conv_transpose1d.
#
# Same strategy as the 2D op (see _kunlunxin/ops/conv_transpose2d.py): the
# generic triton kernel hits the SDNN pipeline on XPU.  Here we lift 1D to 2D
# and reuse the vendor-binding 2D op.
#
# Orientation: with the length on the W axis (dummy H=1) the vendor's inner
# GEMM runs on the coalesced axis and measures ~2-5x faster than the naive
# length-on-H lift on the official matrix.  Grouped cases are kept on the
# H orientation: the vendor's grouped (groups>1) path mis-samples the output
# for some stride-2 shapes in the W orientation (observed as partially stale
# output regions), while the H orientation has been correct on the full
# grouped test matrix.
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
        # the aten schema passes int[1] args as length-1 lists
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
        # narrow workaround for the vendor's grouped+strided path: with
        # padding=0 in the W orientation it leaves stale output regions
        # (observed 2026-09-18, shape dependent); those cases keep the
        # long-proven H orientation (the L axis maps to H, dummy W=1).
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

    # default: the length rides the W axis (dummy H=1), which keeps the
    # vendor's GEMM on the coalesced axis and is markedly faster.
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
