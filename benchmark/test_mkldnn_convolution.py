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

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Native-valid oneDNN convolution rows: the original rank-4 NCHW/OIHW list, the
# rank-3 (NCL) and rank-5 (NCDHW) forms, and cheap non-contiguous or
# shifted-storage operands. The last field is the layout recipe applied to the
# operand it names; both operands are CPU tensors because the kernel is CPU-only.
MKLDNN_CONVOLUTION_ROWS = [
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((7, 16, 32, 29), (16, 16, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((2, 64, 320, 15), (64, 64, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((1, 8, 1024, 1024), (8, 8, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), (1, 1), 1, "contiguous"),
    ((8, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), (1, 1), 1, "contiguous"),
    ((16, 64, 112, 112), (64, 64, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((8, 256, 56, 56), (256, 256, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((4, 512, 28, 28), (512, 512, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((2, 1024, 14, 14), (1024, 1024, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((8, 32, 56, 56), (32, 1, 3, 3), (1, 1), (1, 1), (1, 1), 32, "contiguous"),
    ((2, 19, 7), (3, 19, 3), (1,), (1,), (1,), 1, "contiguous"),
    (
        (4, 7, 15, 29, 32),
        (8, 7, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        "contiguous",
    ),
    # oneDNN reorders strided, offset and channels-last CPU operands internally,
    # so these rows measure the same extents through a different memory layout.
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "channels-last"),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "input-offset"),
    ((2, 3, 10, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "input-transposed"),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "weight-strided"),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "weight-offset"),
]

_INPUT_LAYOUTS = frozenset({"channels-last", "input-offset", "input-transposed"})
_WEIGHT_LAYOUTS = frozenset({"weight-strided", "weight-offset"})


def _native_bias_dtype(dtype):
    # int8/uint8 bias descriptors are rejected natively; quantized inputs take a
    # float32 bias.
    return dtype if dtype.is_floating_point else torch.float32


def _cpu_tensor(shape, dtype):
    """CPU tensor built directly: utils.generate_tensor_input returns None for
    dtypes it cannot construct."""
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device="cpu")
    return torch.randint(-4, 5, shape, device="cpu").to(dtype)


def _laid_out_tensor(layout, shape, dtype):
    """Operand of the requested extents in the requested layout."""
    if layout == "channels-last":
        return _cpu_tensor(shape, dtype).contiguous(memory_format=torch.channels_last)
    if layout == "input-offset":
        base = (shape[0], shape[1], shape[2] + 2, shape[3] + 2)
        return _cpu_tensor(base, dtype)[:, :, 1 : shape[2] + 1, 1 : shape[3] + 1]
    if layout == "input-transposed":
        return _cpu_tensor((shape[0], shape[1], shape[3], shape[2]), dtype).transpose(
            2, 3
        )
    if layout == "weight-strided":
        return _cpu_tensor((shape[0] * 2,) + tuple(shape[1:]), dtype)[::2]
    if layout == "weight-offset":
        return _cpu_tensor((shape[0] + 1,) + tuple(shape[1:]), dtype)[1:]
    return _cpu_tensor(shape, dtype)


def _as_conv_row(shape):
    """Canonical convolution row for any configured shape.

    A curated row passes through. A bare shape (the shared DEFAULT_SHAPES grid,
    the base-class extras, or a shape-file entry without weight metadata) carries
    only an input extent, so it is read as a single-group convolution of the same
    extents: leading 1s reach the smallest legal rank, extra leading dims fold
    into the batch, and the weight keeps a real input-channel axis
    (out_channels, in_channels/groups, kernel...) with a modest output-channel
    count. No requested extent is dropped or capped.
    """
    if shape and isinstance(shape[0], (tuple, list)):
        row = tuple(shape)
        return row if len(row) == 7 else row + ("contiguous",)
    dims = [int(dim) for dim in shape] or [1]
    while len(dims) < 3:
        dims.insert(0, 1)
    if len(dims) > 5:
        batch = 1
        for dim in dims[:-3]:
            batch *= dim
        dims = [batch] + dims[-3:]
    channels = dims[1]
    spatial = dims[2:]
    return (
        tuple(dims),
        (min(8, channels), channels) + (3,) * len(spatial),
        (1,) * len(spatial),
        (1,) * len(spatial),
        (1,) * len(spatial),
        1,
        "contiguous",
    )


def _case_fn(shape, dtype):
    del dtype  # the dtype is carried by the case id
    input_shape, weight_shape, padding, stride, dilation, groups, layout = shape
    yield base.BenchmarkCasePlan(
        shape={
            "input": list(input_shape),
            "weight": list(weight_shape),
            "layout": layout,
        },
        params={
            "padding": list(padding),
            "stride": list(stride),
            "dilation": list(dilation),
            "groups": groups,
        },
        builder_args=(
            tuple(input_shape),
            tuple(weight_shape),
            tuple(padding),
            tuple(stride),
            tuple(dilation),
            groups,
            layout,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    del device  # the native kernel is CPU-only, so both ops take CPU tensors
    (
        input_shape,
        weight_shape,
        padding,
        stride,
        dilation,
        groups,
        layout,
    ) = plan.builder_args
    inp = _laid_out_tensor(
        layout if layout in _INPUT_LAYOUTS else "contiguous", input_shape, dtype
    )
    weight = _laid_out_tensor(
        layout if layout in _WEIGHT_LAYOUTS else "contiguous", weight_shape, dtype
    )
    bias = _cpu_tensor((weight_shape[0],), _native_bias_dtype(dtype))
    return (
        inp,
        weight,
        bias,
        list(padding),
        list(stride),
        list(dilation),
        groups,
        {},
    )


class MkldnnConvolutionBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Shape-file and shared-grid entries arrive here: bare extents become
        # convolution geometry of the same size, curated rows pass through.
        super().set_shapes(shape_file_path)
        self.shapes = [_as_conv_row(shape) for shape in self.shapes]
        for row in MKLDNN_CONVOLUTION_ROWS:
            if row not in self.shapes:
                self.shapes.append(row)

    def set_more_shapes(self):
        return list(super().set_more_shapes()) + MKLDNN_CONVOLUTION_ROWS


@pytest.mark.mkldnn_convolution
def test_mkldnn_convolution():
    bench = MkldnnConvolutionBenchmark(
        op_name="mkldnn_convolution",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_convolution,
        gems_op=getattr(flag_gems, "mkldnn_convolution", None),
        dtypes=consts.FLOAT_DTYPES + [torch.int8, torch.uint8],
    )
    bench.run()
